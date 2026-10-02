"""M14 storage. SQLite locally; Postgres (Neon) when DATABASE_URL is set, so a deploy with an ephemeral disk keeps results.

Both backends expose the same methods. The cache key is (analysis_id, signature): the signature changes when the
config or the models change, so a result made by the fallback model is never served once DeepGaze is available.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from app.config import data_dir

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS attention_analyses (
    id TEXT NOT NULL, signature TEXT NOT NULL, session_id TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    result TEXT NOT NULL, PRIMARY KEY (id, signature)
);
"""
PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS attention_analyses (
    id TEXT NOT NULL, signature TEXT NOT NULL, session_id TEXT, created_at TIMESTAMPTZ DEFAULT now(),
    result JSONB NOT NULL, PRIMARY KEY (id, signature)
);
"""


class SqliteStore:
    def __init__(self, root: Path | None = None):
        root = Path(root) if root else data_dir()
        root.mkdir(parents=True, exist_ok=True)
        self.db_path = root / "app.db"
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SQLITE_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get_analysis(self, analysis_id: str, signature: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT result FROM attention_analyses WHERE id = ? AND signature = ?", (analysis_id, signature)).fetchone()
        return None if row is None else json.loads(row["result"])

    def save_analysis(self, result: dict, signature: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO attention_analyses (id, signature, session_id, result) VALUES (?,?,?,?)",
                      (result["analysis_id"], signature, result.get("session_id"), json.dumps(result)))

    def close(self) -> None:
        pass


class PgStore:
    def __init__(self, url: str):
        from psycopg_pool import ConnectionPool

        # check= revalidates connections, since Neon drops idle ones when the compute scales to zero
        self.pool = ConnectionPool(url, min_size=1, max_size=5, kwargs={"autocommit": True},
                                   check=ConnectionPool.check_connection, open=True)
        with self.pool.connection() as c:
            c.execute(PG_SCHEMA)

    def get_analysis(self, analysis_id: str, signature: str) -> dict | None:
        with self.pool.connection() as c:
            row = c.execute("SELECT result FROM attention_analyses WHERE id = %s AND signature = %s", (analysis_id, signature)).fetchone()
        return None if row is None else row[0]

    def save_analysis(self, result: dict, signature: str) -> None:
        from psycopg.types.json import Jsonb

        with self.pool.connection() as c:
            c.execute(
                """INSERT INTO attention_analyses (id, signature, session_id, result) VALUES (%s,%s,%s,%s)
                   ON CONFLICT (id, signature) DO UPDATE SET session_id = EXCLUDED.session_id, result = EXCLUDED.result""",
                (result["analysis_id"], signature, result.get("session_id"), Jsonb(result)))

    def close(self) -> None:
        self.pool.close()


Store = SqliteStore | PgStore


def make_store() -> Store:
    url = os.environ.get("DATABASE_URL", "").strip()
    return PgStore(url) if url else SqliteStore()

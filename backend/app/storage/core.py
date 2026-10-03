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
CREATE TABLE IF NOT EXISTS attention_images (
    id TEXT PRIMARY KEY, png BLOB NOT NULL, layers TEXT
);
CREATE TABLE IF NOT EXISTS attention_versions (
    version_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, analysis_id TEXT NOT NULL, parent_id TEXT,
    label TEXT NOT NULL, kind TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS attention_versions_session ON attention_versions (session_id);
"""
PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS attention_analyses (
    id TEXT NOT NULL, signature TEXT NOT NULL, session_id TEXT, created_at TIMESTAMPTZ DEFAULT now(),
    result JSONB NOT NULL, PRIMARY KEY (id, signature)
);
CREATE TABLE IF NOT EXISTS attention_images (
    id TEXT PRIMARY KEY, png BYTEA NOT NULL, layers JSONB
);
CREATE TABLE IF NOT EXISTS attention_versions (
    version_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, analysis_id TEXT NOT NULL, parent_id TEXT,
    label TEXT NOT NULL, kind TEXT NOT NULL, created_at TIMESTAMPTZ DEFAULT now(), seq BIGSERIAL
);
CREATE INDEX IF NOT EXISTS attention_versions_session ON attention_versions (session_id);
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

    # ---- images (the normalised frame and the editor layers behind an analysis) ----
    def save_image(self, analysis_id: str, png: bytes, layers: list | None) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT OR IGNORE INTO attention_images (id, png, layers) VALUES (?,?,?)",
                      (analysis_id, png, None if layers is None else json.dumps(layers)))

    def get_image(self, analysis_id: str) -> tuple[bytes, list | None] | None:
        with self._conn() as c:
            row = c.execute("SELECT png, layers FROM attention_images WHERE id = ?", (analysis_id,)).fetchone()
        return None if row is None else (bytes(row["png"]), None if row["layers"] is None else json.loads(row["layers"]))

    # ---- versions ----
    def add_version(self, v: dict) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO attention_versions (version_id, session_id, analysis_id, parent_id, label, kind) VALUES (?,?,?,?,?,?)",
                      (v["version_id"], v["session_id"], v["analysis_id"], v.get("parent_id"), v["label"], v["kind"]))

    def list_versions(self, session_id: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT version_id, session_id, analysis_id, parent_id, label, kind, created_at FROM attention_versions "
                             "WHERE session_id = ? ORDER BY rowid", (session_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_version(self, version_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT version_id, session_id, analysis_id, parent_id, label, kind, created_at FROM attention_versions WHERE version_id = ?",
                            (version_id,)).fetchone()
        return None if row is None else dict(row)

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

    def save_image(self, analysis_id: str, png: bytes, layers: list | None) -> None:
        from psycopg.types.json import Jsonb

        with self.pool.connection() as c:
            c.execute("INSERT INTO attention_images (id, png, layers) VALUES (%s,%s,%s) ON CONFLICT (id) DO NOTHING",
                      (analysis_id, png, None if layers is None else Jsonb(layers)))

    def get_image(self, analysis_id: str) -> tuple[bytes, list | None] | None:
        with self.pool.connection() as c:
            row = c.execute("SELECT png, layers FROM attention_images WHERE id = %s", (analysis_id,)).fetchone()
        return None if row is None else (bytes(row[0]), row[1])

    _VCOLS = "version_id, session_id, analysis_id, parent_id, label, kind, created_at"

    def _vrow(self, r) -> dict:
        d = dict(zip(self._VCOLS.split(", "), r))
        d["created_at"] = d["created_at"].isoformat()
        return d

    def add_version(self, v: dict) -> None:
        with self.pool.connection() as c:
            c.execute("INSERT INTO attention_versions (version_id, session_id, analysis_id, parent_id, label, kind) VALUES (%s,%s,%s,%s,%s,%s)",
                      (v["version_id"], v["session_id"], v["analysis_id"], v.get("parent_id"), v["label"], v["kind"]))

    def list_versions(self, session_id: str) -> list[dict]:
        with self.pool.connection() as c:
            rows = c.execute(f"SELECT {self._VCOLS} FROM attention_versions WHERE session_id = %s ORDER BY seq", (session_id,)).fetchall()
        return [self._vrow(r) for r in rows]

    def get_version(self, version_id: str) -> dict | None:
        with self.pool.connection() as c:
            row = c.execute(f"SELECT {self._VCOLS} FROM attention_versions WHERE version_id = %s", (version_id,)).fetchone()
        return None if row is None else self._vrow(row)

    def close(self) -> None:
        self.pool.close()


Store = SqliteStore | PgStore


def make_store() -> Store:
    url = os.environ.get("DATABASE_URL", "").strip()
    return PgStore(url) if url else SqliteStore()

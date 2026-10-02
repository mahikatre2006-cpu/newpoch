"""M7. Explanation engine: measured features in, sentences out. Rules live in rules.yaml and may only use measured facts."""
from __future__ import annotations

import ast
import string
from pathlib import Path

import yaml

RULES_PATH = Path(__file__).with_name("rules.yaml")
APPLIES = {"any", "face", "text", "subject", "layer", "background", "frame"}
MAX_PER_ELEMENT, MAX_FRAME, MIN_PER_ELEMENT = 4, 3, 2

NUMERIC = {
    "attention_pct", "area_pct", "density", "predicted_rank", "intent_rank", "local_contrast", "saturation_pop", "saturation_pop_pct",
    "colour_distinctness", "dist_to_thirds", "dist_to_center", "safe_zone_overlap_pct", "text_contrast_ratio", "mobile_match_pct",
    "balance_x", "balance_y", "balance_x_pct", "balance_y_pct", "focal_points", "negative_space_pct", "palette_top_pct",
    "text_area_pct", "face_count",
}
OTHER_FACTS = {"label", "text", "type", "mobile_legible", "top_driver", "top_driver_phrase", "colour_name",
               "balance_side_x", "balance_side_y", "palette_top_name"}
KNOWN = NUMERIC | OTHER_FACTS
DECIMALS = {"local_contrast": 2, "saturation_pop": 2, "dist_to_thirds": 2, "dist_to_center": 2, "balance_x": 2, "balance_y": 2,
            "mobile_match_pct": 0, "colour_distinctness": 0, "focal_points": 0, "predicted_rank": 0, "intent_rank": 0, "face_count": 0}

_ALLOWED_NODES = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.Compare, ast.Eq, ast.NotEq, ast.Lt,
                  ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn, ast.Name, ast.Load, ast.Constant, ast.List,
                  ast.Tuple, ast.Call)
_ALLOWED_CALLS = {"abs": abs, "min": min, "max": max}


class RuleError(ValueError):
    pass


def _check_expr(expr: str) -> ast.Expression:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise RuleError(f"bad expression {expr!r}: {e.msg}") from e
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise RuleError(f"{type(node).__name__} is not allowed in {expr!r}")
        if isinstance(node, ast.Name) and node.id not in KNOWN and node.id not in _ALLOWED_CALLS and node.id not in {"true", "false"}:
            raise RuleError(f"unknown fact {node.id!r} in {expr!r}")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_CALLS):
            raise RuleError(f"only abs/min/max may be called in {expr!r}")
    return tree


def _eval(node, facts: dict):
    if isinstance(node, ast.Expression):
        return _eval(node.body, facts)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "true":
            return True
        if node.id == "false":
            return False
        return facts.get(node.id)
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e, facts) for e in node.elts]
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, facts)
        return (not v) if isinstance(node.op, ast.Not) else (None if v is None else -v)
    if isinstance(node, ast.BoolOp):
        vals = (_eval(v, facts) for v in node.values)
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.Call):
        args = [_eval(a, facts) for a in node.args]
        return None if any(a is None for a in args) else _ALLOWED_CALLS[node.func.id](*args)
    if isinstance(node, ast.Compare):
        left = _eval(node.left, facts)
        for op, comp in zip(node.ops, node.comparators):
            right = _eval(comp, facts)
            try:
                ok = {ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b, ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
                      ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b, ast.Is: lambda a, b: a is b, ast.IsNot: lambda a, b: a is not b,
                      ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}[type(op)](left, right)
            except TypeError:  # comparing a missing (None) number: the rule simply does not apply
                return False
            if not ok:
                return False
            left = right
        return True
    raise RuleError(f"cannot evaluate {type(node).__name__}")


def load_rules(path: Path = RULES_PATH) -> list[dict]:
    rules = yaml.safe_load(path.read_text(encoding="utf-8"))
    seen = set()
    for r in rules:
        for key in ("id", "applies_to", "when", "priority", "message"):
            if key not in r:
                raise RuleError(f"rule {r.get('id', '?')} is missing {key!r}")
        if r["id"] in seen:
            raise RuleError(f"duplicate rule id {r['id']}")
        seen.add(r["id"])
        if r["applies_to"] not in APPLIES:
            raise RuleError(f"{r['id']}: applies_to {r['applies_to']!r}")
        r["_tree"] = _check_expr(r["when"])
        fields = {f for _, f, _, _ in string.Formatter().parse(r["message"]) if f}
        unknown = fields - KNOWN
        if unknown:
            raise RuleError(f"{r['id']}: unknown placeholder(s) {sorted(unknown)}")
        if not fields & NUMERIC:
            raise RuleError(f"{r['id']}: message cites no measured number")
        r["_fields"] = fields
    return rules


def _fmt(key: str, v):
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    d = DECIMALS.get(key, 1)
    return f"{v:.{d}f}"


def render(rule: dict, facts: dict) -> str | None:
    """None if a number the message cites is missing (a message must never print 'None')."""
    vals = {}
    for f in rule["_fields"]:
        v = facts.get(f)
        if v is None:
            return None
        vals[f] = _fmt(f, v)
    return rule["message"].format(**vals)


def select(rules: list[dict], kind: str, facts: dict, limit: int, minimum: int = 0) -> list[str]:
    hits = [r for r in rules if r["applies_to"] in ("any", kind) and _eval(r["_tree"], facts)]
    primary = [r for r in hits if not r.get("fallback")]
    chosen = sorted(primary, key=lambda r: (-r["priority"], r["id"]))
    if len(chosen) < minimum:
        chosen += sorted((r for r in hits if r.get("fallback")), key=lambda r: (-r["priority"], r["id"]))
    out = []
    for r in chosen:
        text = render(r, facts)
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return out


# ---------- facts ----------
def top_driver(e: dict, f: dict) -> tuple[str | None, str]:
    """The feature with the largest normalised value among local contrast, saturation pop, colour distinctness, face and centrality."""
    cands: dict[str, float] = {}
    if f.get("local_contrast") is not None:
        cands["local_contrast"] = min(1.0, f["local_contrast"] / 0.45)
    if f.get("saturation_pop") is not None:
        cands["saturation"] = min(1.0, max(0.0, f["saturation_pop"]) / 0.30)
    if f.get("colour_distinctness") is not None:
        cands["colour_distinctness"] = min(1.0, f["colour_distinctness"] / 45.0)
    if e["type"] == "face":
        cands["face"] = 1.0
    if f.get("dist_to_center") is not None and e["type"] != "background":
        cands["centrality"] = max(0.0, 1.0 - f["dist_to_center"] / 0.6)
    if not cands:
        return None, "its position and size"
    key = max(cands, key=lambda k: (cands[k], k))
    if cands[key] < 0.25:
        return None, "a combination of position and size"
    phrase = {
        "local_contrast": f"strong local contrast ({f.get('local_contrast', 0):.2f})",
        "saturation": f"vivid colour ({f.get('saturation_pop', 0) * 100:+.0f}% saturation over the frame average)",
        "colour_distinctness": f"colour that stands apart from the frame (distance {f.get('colour_distinctness', 0):.0f})",
        "face": f"being a face ({e['area_pct']:.1f}% of the frame)",
        "centrality": f"its central position ({f.get('dist_to_center', 0):.2f} from the centre)",
    }[key]
    return key, phrase


def element_facts(e: dict, f: dict) -> dict:
    key, phrase = top_driver(e, f)
    sp = f.get("saturation_pop")
    return {
        "label": e["label"], "text": e.get("text"), "type": e["type"], "attention_pct": e["attention_pct"], "area_pct": e["area_pct"],
        "density": e["density"], "predicted_rank": e["predicted_rank"], "intent_rank": e.get("intent_rank"),
        "local_contrast": f.get("local_contrast"), "saturation_pop": sp, "saturation_pop_pct": None if sp is None else round(sp * 100, 1),
        "colour_distinctness": f.get("colour_distinctness"), "dist_to_thirds": f.get("dist_to_thirds"), "dist_to_center": f.get("dist_to_center"),
        "safe_zone_overlap_pct": f.get("safe_zone_overlap_pct"), "text_contrast_ratio": f.get("text_contrast_ratio"),
        "mobile_match_pct": f.get("mobile_match_pct"), "mobile_legible": f.get("mobile_legible"),
        "top_driver": key, "top_driver_phrase": phrase, "colour_name": f.get("colour_name"),
    }


def frame_facts(frame: dict, elements: list[dict]) -> dict:
    bx, by = (frame.get("balance_offset") or [None, None])
    pal = (frame.get("palette") or [{}])[0]
    return {
        "balance_x": bx, "balance_y": by,
        "balance_x_pct": None if bx is None else round(abs(bx) * 100, 1), "balance_y_pct": None if by is None else round(abs(by) * 100, 1),
        "balance_side_x": None if bx is None else ("right" if bx > 0 else "left"), "balance_side_y": None if by is None else ("bottom" if by > 0 else "top"),
        "focal_points": frame.get("focal_points"), "negative_space_pct": frame.get("negative_space_pct"),
        "palette_top_name": pal.get("name"), "palette_top_pct": None if "share" not in pal else round(pal["share"] * 100, 1),
        "text_area_pct": frame.get("text_area_pct"), "face_count": sum(1 for e in elements if e["type"] == "face"),
    }


class Explainer:
    def __init__(self, path: Path = RULES_PATH):
        self.rules = load_rules(path)

    def intent_message(self, label: str, intent_rank: int, predicted_rank: int) -> str | None:
        rule = next((r for r in self.rules if r["id"] == "hierarchy_mismatch"), None)
        return render(rule, {"label": label, "intent_rank": intent_rank, "predicted_rank": predicted_rank}) if rule else None

    def explain(self, elements: list[dict], features: dict[str, dict], frame: dict) -> None:
        """Fills `features`, `top_driver` and `explanations` on each element and `explanations` on the frame, in place."""
        for e in elements:
            f = features.get(e["id"], {})
            facts = element_facts(e, f)
            e["features"] = f
            e["top_driver"] = facts["top_driver"]
            e["explanations"] = select(self.rules, e["type"], facts, MAX_PER_ELEMENT, MIN_PER_ELEMENT)
        frame["explanations"] = select(self.rules, "frame", frame_facts(frame, elements), MAX_FRAME)

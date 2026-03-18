"""FactTable extractor — identifies key-value numerical facts from LDUs and persists to SQLite."""

import re
import sqlite3
from pathlib import Path
from typing import Any

from ledger.agents.refinery.models.ldu import LDU

_MONEY_RE = re.compile(
    r"(?P<key>[A-Za-z][A-Za-z &/,\-]{2,60}?)\s*[:=]?\s*"
    r"(?P<sign>[−\-]?)\s*"
    r"(?P<currency>[A-Z]{3}|ETB|USD|EUR|GBP|Birr|\$|€|£)?\s*"
    r"(?P<value>[\d,]+(?:\.\d+)?)\s*"
    r"(?P<unit>%|billion|million|thousand|mn|bn|m|k|pp)?"
    , re.IGNORECASE,
)

_NUMBER_KV_RE = re.compile(
    r"(?P<key>[A-Za-z][A-Za-z &/,\-]{2,60}?)\s*[:=|]\s*"
    r"(?P<sign>[−\-]?)\s*"
    r"(?P<value>[\d,]+(?:\.\d+)?)\s*"
    r"(?P<unit>%|billion|million|thousand|mn|bn|m|k|pp)?"
    , re.IGNORECASE,
)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT    NOT NULL,
    fact_key    TEXT    NOT NULL,
    fact_value  REAL    NOT NULL,
    unit        TEXT    DEFAULT '',
    page_ref    INTEGER DEFAULT 0,
    content_hash TEXT   DEFAULT '',
    source_ldu  TEXT    DEFAULT ''
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_facts_doc ON facts(doc_id);
"""


def _parse_number(raw: str) -> float | None:
    cleaned = raw.replace(",", "").strip()
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _extract_facts_from_text(text: str) -> list[dict[str, Any]]:
    """Extract key-value numerical facts from free text."""
    facts: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()

    for pattern in (_MONEY_RE, _NUMBER_KV_RE):
        for m in pattern.finditer(text):
            key = m.group("key").strip().rstrip(":")
            value = _parse_number(m.group("value"))
            if value is None:
                continue
            sign = m.group("sign") or ""
            if sign in ("−", "-"):
                value = -value
            unit = ""
            if "unit" in m.groupdict() and m.group("unit"):
                unit = m.group("unit").strip()
            if "currency" in m.groupdict() and m.group("currency"):
                unit = (m.group("currency") + " " + unit).strip()
            dedup = (key.lower(), value)
            if dedup in seen:
                continue
            seen.add(dedup)
            facts.append({"fact_key": key, "fact_value": value, "unit": unit})
    return facts


def _extract_facts_from_table_ldu(content: str) -> list[dict[str, Any]]:
    """Extract key-value pairs from table-type LDUs (tab-separated rows)."""
    facts: list[dict[str, Any]] = []
    lines = content.strip().split("\n")
    if len(lines) < 2:
        return _extract_facts_from_text(content)

    headers = [h.strip() for h in lines[0].split("\t")]
    for line in lines[1:]:
        cells = [c.strip() for c in line.split("\t")]
        if len(cells) < 2:
            continue
        key = cells[0]
        if not key or not any(c.isalpha() for c in key):
            continue
        for i, cell in enumerate(cells[1:], 1):
            val = _parse_number(cell)
            if val is not None:
                header_label = headers[i] if i < len(headers) else f"col_{i}"
                facts.append({
                    "fact_key": f"{key} — {header_label}",
                    "fact_value": val,
                    "unit": "",
                })
    return facts


class FactTableExtractor:
    """Extracts numerical key-value facts from LDUs and stores them in SQLite."""

    def __init__(self, db_path: Path | str | None = None):
        base = Path(__file__).resolve().parent.parent.parent
        self.db_path = Path(db_path) if db_path else base / ".refinery" / "facts.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.execute(_CREATE_INDEX_SQL)
        self._conn.commit()

    def extract_and_store(self, doc_id: str, ldus: list[LDU]) -> int:
        """Extract facts from LDUs and insert into SQLite. Returns count of facts inserted."""
        rows: list[tuple[str, str, float, str, int, str, str]] = []
        for ldu in ldus:
            if not ldu.content or not ldu.content.strip():
                continue
            if ldu.chunk_type == "table":
                facts = _extract_facts_from_table_ldu(ldu.content)
            else:
                facts = _extract_facts_from_text(ldu.content)
            page_ref = ldu.page_refs[0] if ldu.page_refs else 0
            for f in facts:
                rows.append((
                    doc_id,
                    f["fact_key"],
                    f["fact_value"],
                    f["unit"],
                    page_ref,
                    ldu.content_hash,
                    ldu.ldu_id,
                ))
        if rows:
            self._conn.executemany(
                "INSERT INTO facts (doc_id, fact_key, fact_value, unit, page_ref, content_hash, source_ldu) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()
        return len(rows)

    def query_facts(
        self,
        doc_id: str | None = None,
        sql_where: str | None = None,
        keyword: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Query the fact table. Supports doc_id filter, free SQL WHERE clause, or keyword search on fact_key."""
        conditions: list[str] = []
        params: list[Any] = []
        if doc_id:
            conditions.append("doc_id = ?")
            params.append(doc_id)
        if keyword:
            conditions.append("fact_key LIKE ?")
            params.append(f"%{keyword}%")
        if sql_where:
            conditions.append(f"({sql_where})")
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT * FROM facts WHERE {where_clause} ORDER BY id LIMIT ?"
        params.append(limit)
        cursor = self._conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def close(self) -> None:
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

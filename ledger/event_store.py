"""
ledger/event_store.py -- PostgreSQL-backed EventStore
"""
from __future__ import annotations
import json
from datetime import datetime
from typing import AsyncGenerator
from uuid import UUID
import asyncpg


class OptimisticConcurrencyError(Exception):
    """Raised when expected_version doesn't match current stream version."""
    def __init__(self, stream_id: str, expected: int, actual: int):
        self.stream_id = stream_id
        self.expected = expected
        self.actual = actual
        super().__init__(f"OCC on '{stream_id}': expected v{expected}, actual v{actual}")


class EventStore:
    """Append-only PostgreSQL event store. All agents and projections use this class."""

    def __init__(self, db_url: str, upcaster_registry=None):
        self.db_url = db_url
        self.upcasters = upcaster_registry
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(self.db_url, min_size=2, max_size=10)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    async def stream_version(self, stream_id: str) -> int:
        """Returns current version, or -1 if stream does not exist."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT current_version FROM event_streams WHERE stream_id = $1",
                stream_id)
            return row["current_version"] if row else -1

    async def append(
        self,
        stream_id: str,
        events: list[dict],
        expected_version: int,
        causation_id: str | None = None,
        metadata: dict | None = None,
    ) -> list[int]:
        """Appends events atomically with OCC. Returns list of positions assigned."""
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # 1. Lock the stream row to prevent concurrent appends
                row = await conn.fetchrow(
                    "SELECT current_version FROM event_streams "
                    "WHERE stream_id = $1 FOR UPDATE",
                    stream_id)

                # 2. OCC check
                current = row["current_version"] if row else -1
                if current != expected_version:
                    raise OptimisticConcurrencyError(stream_id, expected_version, current)

                # 3. Create stream row if new; base=0 so first event lands at position 1
                if row is None:
                    await conn.execute(
                        "INSERT INTO event_streams(stream_id, aggregate_type, current_version)"
                        " VALUES($1, $2, 0)",
                        stream_id, stream_id.split("-")[0])
                    base = 0
                else:
                    base = expected_version

                # 4. Insert each event at 1-indexed positions
                positions = []
                meta = {**(metadata or {})}
                if causation_id:
                    meta["causation_id"] = causation_id
                for i, event in enumerate(events):
                    pos = base + 1 + i
                    await conn.execute(
                        "INSERT INTO events(stream_id, stream_position, event_type,"
                        " event_version, payload, metadata, recorded_at)"
                        " VALUES($1,$2,$3,$4,$5::jsonb,$6::jsonb,$7)",
                        stream_id, pos,
                        event["event_type"], event.get("event_version", 1),
                        json.dumps(event["payload"]),
                        json.dumps(meta),
                        datetime.utcnow())
                    positions.append(pos)

                # 5. Update stream version to highest position assigned
                await conn.execute(
                    "UPDATE event_streams SET current_version=$1 WHERE stream_id=$2",
                    base + len(events), stream_id)
                return positions

    async def load_stream(
        self,
        stream_id: str,
        from_position: int = 0,
        to_position: int | None = None,
    ) -> list[dict]:
        """Loads events from a stream ordered by stream_position. Applies upcasters."""
        async with self._pool.acquire() as conn:
            q = ("SELECT event_id, stream_id, stream_position, event_type,"
                 " event_version, payload, metadata, recorded_at"
                 " FROM events WHERE stream_id=$1 AND stream_position>=$2")
            params: list = [stream_id, from_position]
            if to_position is not None:
                q += " AND stream_position<=$3"
                params.append(to_position)
            q += " ORDER BY stream_position ASC"
            rows = await conn.fetch(q, *params)
            result = []
            for row in rows:
                e = dict(row)
                if self.upcasters:
                    e = self.upcasters.upcast(e)
                result.append(e)
            return result

    async def load_all(
        self, from_position: int = 0, batch_size: int = 500
    ) -> AsyncGenerator[dict, None]:
        """Async generator yielding all events ordered by global_position."""
        async with self._pool.acquire() as conn:
            pos = from_position
            while True:
                rows = await conn.fetch(
                    "SELECT global_position, stream_id, stream_position,"
                    " event_type, event_version, payload, metadata, recorded_at"
                    " FROM events WHERE global_position > $1"
                    " ORDER BY global_position ASC LIMIT $2",
                    pos, batch_size)
                if not rows:
                    break
                for row in rows:
                    e = dict(row)
                    if self.upcasters:
                        e = self.upcasters.upcast(e)
                    yield e
                pos = rows[-1]["global_position"]
                if len(rows) < batch_size:
                    break

    async def get_event(self, event_id: UUID) -> dict | None:
        """Loads one event by UUID. Used for causation chain lookups."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM events WHERE event_id=$1", event_id)
            if not row:
                return None
            return dict(row)


# ---------------------------------------------------------------------------
# UPCASTER REGISTRY -- Phase 4
# ---------------------------------------------------------------------------

class UpcasterRegistry:
    """Transforms old event versions to current on load. Upcasters are pure functions."""

    def __init__(self):
        self._upcasters: dict[str, dict[int, callable]] = {}

    def upcaster(self, event_type: str, from_version: int, to_version: int):
        def decorator(fn):
            self._upcasters.setdefault(event_type, {})[from_version] = fn
            return fn
        return decorator

    def upcast(self, event: dict) -> dict:
        """Apply chain of upcasters until latest version reached."""
        et = event["event_type"]
        v = event.get("event_version", 1)
        chain = self._upcasters.get(et, {})
        while v in chain:
            event["payload"] = chain[v](dict(event["payload"]))
            v += 1
            event["event_version"] = v
        return event


# ---------------------------------------------------------------------------
# IN-MEMORY EVENT STORE -- for Phase 1 tests only
# ---------------------------------------------------------------------------

import asyncio as _asyncio
from collections import defaultdict as _defaultdict
from datetime import datetime as _datetime
from uuid import uuid4 as _uuid4


class InMemoryEventStore:
    """
    Thread-safe (asyncio-safe) in-memory event store.
    Used exclusively in Phase 1 tests and conftest fixtures.
    Same interface as EventStore -- swap one for the other with no code changes.
    """

    def __init__(self):
        self._streams: dict[str, list[dict]] = _defaultdict(list)
        self._versions: dict[str, int] = {}
        self._global: list[dict] = []
        self._checkpoints: dict[str, int] = {}
        self._locks: dict[str, _asyncio.Lock] = _defaultdict(_asyncio.Lock)

    async def stream_version(self, stream_id: str) -> int:
        return self._versions.get(stream_id, -1)

    async def append(
        self,
        stream_id: str,
        events: list[dict],
        expected_version: int,
        causation_id: str | None = None,
        metadata: dict | None = None,
    ) -> list[int]:
        async with self._locks[stream_id]:
            current = self._versions.get(stream_id, -1)
            if current != expected_version:
                raise OptimisticConcurrencyError(stream_id, expected_version, current)

            positions = []
            meta = {**(metadata or {})}
            if causation_id:
                meta["causation_id"] = causation_id

            for i, event in enumerate(events):
                pos = current + 1 + i
                stored = {
                    "event_id": str(_uuid4()),
                    "stream_id": stream_id,
                    "stream_position": pos,
                    "global_position": len(self._global),
                    "event_type": event["event_type"],
                    "event_version": event.get("event_version", 1),
                    "payload": dict(event.get("payload", {})),
                    "metadata": meta,
                    "recorded_at": _datetime.utcnow().isoformat(),
                }
                self._streams[stream_id].append(stored)
                self._global.append(stored)
                positions.append(pos)

            self._versions[stream_id] = current + len(events)
            return positions

    async def load_stream(
        self,
        stream_id: str,
        from_position: int = 0,
        to_position: int | None = None,
    ) -> list[dict]:
        events = [
            e for e in self._streams.get(stream_id, [])
            if e["stream_position"] >= from_position
            and (to_position is None or e["stream_position"] <= to_position)
        ]
        return sorted(events, key=lambda e: e["stream_position"])

    async def load_all(self, from_position: int = 0, batch_size: int = 500):
        for e in self._global:
            if e["global_position"] >= from_position:
                yield e

    async def get_event(self, event_id: str) -> dict | None:
        for e in self._global:
            if e["event_id"] == event_id:
                return e
        return None

    async def save_checkpoint(self, projection_name: str, position: int) -> None:
        self._checkpoints[projection_name] = position

    async def load_checkpoint(self, projection_name: str) -> int:
        return self._checkpoints.get(projection_name, 0)

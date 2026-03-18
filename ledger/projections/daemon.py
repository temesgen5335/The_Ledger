"""
ledger/projections/daemon.py — Asynchronous Projection Daemon
==============================================================
Phase 4: CQRS, Projections, and Schema Evolution

The ProjectionDaemon solves the "Read Problem" by maintaining denormalized
read models (application_summary, risk_dashboard) from the immutable event log.

Key Features:
- Polling loop reading from events table starting from last checkpoint
- Event handlers for ApplicationSubmitted, ExtractionCompleted, DecisionGenerated, ComplianceRuleFailed
- Checkpoint system: batch of 50 events, update projection_checkpoints
- Upcasting: all events run through UpcasterRegistry before processing
- --replay flag: TRUNCATE tables, reset checkpoints, re-process from beginning
- Idempotent writes: processing same event twice produces same state

Usage:
    python -m ledger.projections.daemon              # normal mode
    python -m ledger.projections.daemon --replay     # replay from beginning
"""
from __future__ import annotations
import asyncio
import asyncpg
import json
import sys
from datetime import datetime
from decimal import Decimal
from typing import Callable

# Projection names
PROJECTION_APPLICATION_SUMMARY = "application_summary"
PROJECTION_RISK_DASHBOARD = "risk_dashboard"

BATCH_SIZE = 50  # Checkpoint every 50 events


class ProjectionDaemon:
    """
    Asynchronous daemon that maintains read models from the event log.
    
    Architecture:
    - Reads events from global_position order (append-only log)
    - Applies upcasters to handle schema evolution
    - Routes events to handlers based on event_type
    - Checkpoints progress for crash recovery
    - All writes are idempotent (ON CONFLICT DO UPDATE)
    """
    
    def __init__(self, db_url: str, upcaster_registry=None):
        self.db_url = db_url
        self.upcasters = upcaster_registry
        self._pool: asyncpg.Pool | None = None
        self._handlers: dict[str, Callable] = {}
        self._register_handlers()
    
    async def connect(self) -> None:
        """Initialize connection pool and ensure schema exists."""
        self._pool = await asyncpg.create_pool(self.db_url, min_size=2, max_size=10)
        await self._ensure_schema()
    
    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
    
    async def _ensure_schema(self) -> None:
        """Create projection tables and checkpoint table if they don't exist."""
        async with self._pool.acquire() as conn:
            # application_summary table - create if not exists
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS application_summary (
                    application_id TEXT PRIMARY KEY,
                    applicant_name TEXT,
                    status TEXT DEFAULT 'PENDING',
                    total_revenue NUMERIC(15,2),
                    risk_score NUMERIC(5,2),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Add missing columns if they don't exist (for existing tables)
            columns_to_add = [
                ("applicant_id", "TEXT"),
                ("requested_amount_usd", "NUMERIC(15,2)"),
                ("fraud_score", "NUMERIC(5,2)"),
                ("compliance_blocked", "BOOLEAN DEFAULT FALSE"),
            ]
            
            for col_name, col_type in columns_to_add:
                try:
                    await conn.execute(f"""
                        ALTER TABLE application_summary
                        ADD COLUMN IF NOT EXISTS {col_name} {col_type}
                    """)
                except Exception:
                    pass  # Column might already exist
            
            # risk_dashboard table (single row aggregate)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_dashboard (
                    dashboard_id TEXT PRIMARY KEY DEFAULT 'global',
                    total_apps INTEGER DEFAULT 0,
                    total_approved_amount NUMERIC(15,2) DEFAULT 0,
                    total_declined_amount NUMERIC(15,2) DEFAULT 0,
                    avg_fraud_score NUMERIC(5,2) DEFAULT 0,
                    total_blocked_by_compliance INTEGER DEFAULT 0,
                    last_updated TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Initialize dashboard row if not exists
            await conn.execute("""
                INSERT INTO risk_dashboard (dashboard_id)
                VALUES ('global')
                ON CONFLICT (dashboard_id) DO NOTHING
            """)
            
            # projection_checkpoints table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS projection_checkpoints (
                    projection_name TEXT PRIMARY KEY,
                    last_position BIGINT DEFAULT 0
                )
            """)
            
            # Add last_updated column if it doesn't exist (for existing tables)
            try:
                await conn.execute("""
                    ALTER TABLE projection_checkpoints
                    ADD COLUMN IF NOT EXISTS last_updated TIMESTAMPTZ DEFAULT NOW()
                """)
            except Exception:
                pass  # Column might already exist in older PostgreSQL versions
            
            # Initialize checkpoints
            for proj in [PROJECTION_APPLICATION_SUMMARY, PROJECTION_RISK_DASHBOARD]:
                await conn.execute("""
                    INSERT INTO projection_checkpoints (projection_name, last_position)
                    VALUES ($1, 0)
                    ON CONFLICT (projection_name) DO NOTHING
                """, proj)
    
    def _register_handlers(self) -> None:
        """Map event types to handler methods."""
        self._handlers = {
            "ApplicationSubmitted": self._handle_application_submitted,
            "ExtractionCompleted": self._handle_extraction_completed,
            "DecisionGenerated": self._handle_decision_generated,
            "ComplianceRuleFailed": self._handle_compliance_rule_failed,
            "FraudScreeningCompleted": self._handle_fraud_screening_completed,
            "CreditAnalysisCompleted": self._handle_credit_analysis_completed,
        }
    
    async def _get_checkpoint(self, projection_name: str) -> int:
        """Get last processed global_position for a projection."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT last_position FROM projection_checkpoints WHERE projection_name = $1",
                projection_name
            )
            return row["last_position"] if row else 0
    
    async def _update_checkpoint(self, projection_name: str, position: int) -> None:
        """Update checkpoint for a projection."""
        async with self._pool.acquire() as conn:
            try:
                await conn.execute("""
                    UPDATE projection_checkpoints
                    SET last_position = $1, last_updated = NOW()
                    WHERE projection_name = $2
                """, position, projection_name)
            except Exception:
                # Fallback if last_updated column doesn't exist
                await conn.execute("""
                    UPDATE projection_checkpoints
                    SET last_position = $1
                    WHERE projection_name = $2
                """, position, projection_name)
    
    async def _handle_application_submitted(self, event: dict, conn: asyncpg.Connection) -> None:
        """Insert into application_summary (idempotent with ON CONFLICT)."""
        payload = event["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        
        await conn.execute("""
            INSERT INTO application_summary (
                application_id, applicant_name, applicant_id, 
                requested_amount_usd, status, updated_at
            )
            VALUES ($1, $2, $3, $4, 'PENDING', NOW())
            ON CONFLICT (application_id) DO UPDATE SET
                applicant_name = EXCLUDED.applicant_name,
                applicant_id = EXCLUDED.applicant_id,
                requested_amount_usd = EXCLUDED.requested_amount_usd,
                updated_at = NOW()
        """, 
            payload.get("application_id"),
            payload.get("contact_name", "Unknown"),
            payload.get("applicant_id"),
            Decimal(str(payload.get("requested_amount_usd", 0)))
        )
        
        # Update dashboard total_apps
        await conn.execute("""
            UPDATE risk_dashboard
            SET total_apps = (SELECT COUNT(*) FROM application_summary),
                last_updated = NOW()
            WHERE dashboard_id = 'global'
        """)
    
    async def _handle_extraction_completed(self, event: dict, conn: asyncpg.Connection) -> None:
        """Update total_revenue in application_summary from extracted facts."""
        payload = event["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        
        # Extract revenue from facts
        facts = payload.get("facts", {})
        if isinstance(facts, str):
            facts = json.loads(facts)
        
        total_revenue = facts.get("total_revenue")
        if total_revenue is None:
            return
        
        # Find application_id from package_id (format: pkg-{app_id})
        package_id = payload.get("package_id", "")
        if package_id.startswith("pkg-"):
            app_id = package_id[4:]
        else:
            # Try to find from stream_id
            stream_id = event.get("stream_id", "")
            if stream_id.startswith("docpkg-"):
                app_id = stream_id[7:]
            else:
                return
        
        await conn.execute("""
            UPDATE application_summary
            SET total_revenue = $1, updated_at = NOW()
            WHERE application_id = $2
        """, Decimal(str(total_revenue)), app_id)
    
    async def _handle_decision_generated(self, event: dict, conn: asyncpg.Connection) -> None:
        """Update status in application_summary based on decision."""
        payload = event["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        
        app_id = payload.get("application_id")
        decision = payload.get("decision", "PENDING")
        
        await conn.execute("""
            UPDATE application_summary
            SET status = $1, updated_at = NOW()
            WHERE application_id = $2
        """, decision, app_id)
        
        # Update dashboard approved/declined amounts
        if decision == "APPROVE":
            await conn.execute("""
                UPDATE risk_dashboard
                SET total_approved_amount = (
                    SELECT COALESCE(SUM(requested_amount_usd), 0)
                    FROM application_summary
                    WHERE status = 'APPROVE'
                ),
                last_updated = NOW()
                WHERE dashboard_id = 'global'
            """)
        elif decision == "DECLINE":
            await conn.execute("""
                UPDATE risk_dashboard
                SET total_declined_amount = (
                    SELECT COALESCE(SUM(requested_amount_usd), 0)
                    FROM application_summary
                    WHERE status = 'DECLINE'
                ),
                last_updated = NOW()
                WHERE dashboard_id = 'global'
            """)
    
    async def _handle_compliance_rule_failed(self, event: dict, conn: asyncpg.Connection) -> None:
        """Update risk_dashboard counters and mark application as blocked."""
        payload = event["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        
        app_id = payload.get("application_id")
        
        # Mark application as compliance blocked
        await conn.execute("""
            UPDATE application_summary
            SET compliance_blocked = TRUE, updated_at = NOW()
            WHERE application_id = $1
        """, app_id)
        
        # Update dashboard counter
        await conn.execute("""
            UPDATE risk_dashboard
            SET total_blocked_by_compliance = (
                SELECT COUNT(*) FROM application_summary WHERE compliance_blocked = TRUE
            ),
            last_updated = NOW()
            WHERE dashboard_id = 'global'
        """)
    
    async def _handle_fraud_screening_completed(self, event: dict, conn: asyncpg.Connection) -> None:
        """Update fraud_score in application_summary."""
        payload = event["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        
        app_id = payload.get("application_id")
        fraud_score = payload.get("fraud_score", 0)
        
        await conn.execute("""
            UPDATE application_summary
            SET fraud_score = $1, updated_at = NOW()
            WHERE application_id = $2
        """, Decimal(str(fraud_score)), app_id)
        
        # Update dashboard avg_fraud_score
        await conn.execute("""
            UPDATE risk_dashboard
            SET avg_fraud_score = (
                SELECT COALESCE(AVG(fraud_score), 0)
                FROM application_summary
                WHERE fraud_score IS NOT NULL
            ),
            last_updated = NOW()
            WHERE dashboard_id = 'global'
        """)
    
    async def _handle_credit_analysis_completed(self, event: dict, conn: asyncpg.Connection) -> None:
        """Update risk_score in application_summary."""
        payload = event["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        
        app_id = payload.get("application_id")
        risk_score = payload.get("risk_score")
        
        if risk_score is not None:
            await conn.execute("""
                UPDATE application_summary
                SET risk_score = $1, updated_at = NOW()
                WHERE application_id = $2
            """, Decimal(str(risk_score)), app_id)
    
    async def _process_event(self, event: dict, conn: asyncpg.Connection) -> None:
        """Route event to appropriate handler."""
        # Apply upcasters
        if self.upcasters:
            event = self.upcasters.upcast(event)
        
        event_type = event.get("event_type")
        handler = self._handlers.get(event_type)
        
        if handler:
            await handler(event, conn)
    
    async def run_once(self) -> int:
        """
        Process one batch of events. Returns number of events processed.
        Used for testing and controlled execution.
        """
        checkpoint = await self._get_checkpoint(PROJECTION_APPLICATION_SUMMARY)
        
        async with self._pool.acquire() as conn:
            # Fetch next batch
            rows = await conn.fetch("""
                SELECT global_position, stream_id, event_type, event_version, payload, metadata, recorded_at
                FROM events
                WHERE global_position > $1
                ORDER BY global_position ASC
                LIMIT $2
            """, checkpoint, BATCH_SIZE)
            
            if not rows:
                return 0
            
            # Process batch in transaction
            async with conn.transaction():
                for row in rows:
                    event = dict(row)
                    await self._process_event(event, conn)
                
                # Update checkpoint to highest position processed
                last_position = rows[-1]["global_position"]
                await self._update_checkpoint(PROJECTION_APPLICATION_SUMMARY, last_position)
                await self._update_checkpoint(PROJECTION_RISK_DASHBOARD, last_position)
            
            return len(rows)
    
    async def run_loop(self, poll_interval: float = 1.0) -> None:
        """
        Main polling loop. Runs continuously, processing batches and sleeping.
        """
        print(f"[ProjectionDaemon] Starting polling loop (interval={poll_interval}s)")
        
        while True:
            try:
                count = await self.run_once()
                if count > 0:
                    print(f"[ProjectionDaemon] Processed {count} events")
                else:
                    await asyncio.sleep(poll_interval)
            except KeyboardInterrupt:
                print("[ProjectionDaemon] Shutting down...")
                break
            except Exception as e:
                print(f"[ProjectionDaemon] Error: {e}")
                await asyncio.sleep(poll_interval)
    
    async def replay(self) -> None:
        """
        Replay mode: TRUNCATE projection tables, reset checkpoints, re-process from beginning.
        """
        print("[ProjectionDaemon] REPLAY MODE: Truncating projection tables...")
        
        async with self._pool.acquire() as conn:
            await conn.execute("TRUNCATE application_summary")
            await conn.execute("DELETE FROM risk_dashboard WHERE dashboard_id = 'global'")
            await conn.execute("""
                INSERT INTO risk_dashboard (dashboard_id)
                VALUES ('global')
            """)
            try:
                await conn.execute("""
                    UPDATE projection_checkpoints
                    SET last_position = 0, last_updated = NOW()
                    WHERE projection_name IN ($1, $2)
                """, PROJECTION_APPLICATION_SUMMARY, PROJECTION_RISK_DASHBOARD)
            except Exception:
                # Fallback if last_updated column doesn't exist
                await conn.execute("""
                    UPDATE projection_checkpoints
                    SET last_position = 0
                    WHERE projection_name IN ($1, $2)
                """, PROJECTION_APPLICATION_SUMMARY, PROJECTION_RISK_DASHBOARD)
        
        print("[ProjectionDaemon] Checkpoints reset to 0. Re-processing entire event log...")
        
        # Process all events
        total_processed = 0
        while True:
            count = await self.run_once()
            if count == 0:
                break
            total_processed += count
            print(f"[ProjectionDaemon] Replay progress: {total_processed} events processed")
        
        print(f"[ProjectionDaemon] Replay complete. Total events: {total_processed}")
        
        # Show summary
        async with self._pool.acquire() as conn:
            app_count = await conn.fetchval("SELECT COUNT(*) FROM application_summary")
            dashboard = await conn.fetchrow("SELECT * FROM risk_dashboard WHERE dashboard_id = 'global'")
            print(f"\n=== Projection Summary ===")
            print(f"Applications: {app_count}")
            print(f"Total Apps: {dashboard['total_apps']}")
            print(f"Approved Amount: ${dashboard['total_approved_amount']}")
            print(f"Declined Amount: ${dashboard['total_declined_amount']}")
            print(f"Avg Fraud Score: {dashboard['avg_fraud_score']}")
            print(f"Blocked by Compliance: {dashboard['total_blocked_by_compliance']}")


async def main():
    """CLI entry point."""
    import os
    from dotenv import load_dotenv
    from ledger.upcasters import UpcasterRegistry
    
    load_dotenv()
    
    db_url = os.environ.get("DATABASE_URL", "postgresql://postgres:apex@localhost/apex_ledger")
    replay_mode = "--replay" in sys.argv
    
    # Initialize upcaster registry
    upcasters = UpcasterRegistry()
    
    # Create daemon
    daemon = ProjectionDaemon(db_url, upcaster_registry=upcasters)
    await daemon.connect()
    
    try:
        if replay_mode:
            await daemon.replay()
        else:
            await daemon.run_loop()
    finally:
        await daemon.close()


if __name__ == "__main__":
    asyncio.run(main())

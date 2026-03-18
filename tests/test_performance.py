"""
tests/test_performance.py
==========================
Performance tests for The Ledger - "Week Standard" verification.

The "Week Standard": Decision history retrieval must complete in < 5 seconds.
(Program limit is 60s, but we aim for sub-second performance)

Run: pytest tests/test_performance.py -v -s
"""
import pytest
import time
import asyncpg
from ledger.event_store import EventStore

DB_URL = "postgresql://postgres:apex@localhost/apex_ledger"

# Performance thresholds
WEEK_STANDARD_THRESHOLD = 5.0  # seconds
OPTIMAL_THRESHOLD = 1.0  # seconds


@pytest.fixture
async def event_store():
    """Create and connect EventStore."""
    store = EventStore(DB_URL)
    await store.connect()
    yield store
    await store.close()


@pytest.fixture
async def db_pool():
    """Create database connection pool."""
    pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=5)
    yield pool
    await pool.close()


# ============================================================================
# TEST 1: Decision History Retrieval Performance
# ============================================================================

@pytest.mark.asyncio
async def test_decision_history_retrieval_performance(event_store, db_pool):
    """
    Test: Retrieve full decision history for an application.
    
    Week Standard: Must complete in < 5 seconds.
    Optimal: Should complete in < 1 second.
    """
    # Find an application with events
    async with db_pool.acquire() as conn:
        app_id = await conn.fetchval(
            "SELECT application_id FROM application_summary LIMIT 1"
        )
    
    if not app_id:
        pytest.skip("No applications in database to test")
    
    # Measure time to retrieve full decision history
    start_time = time.time()
    
    streams = [
        f"loan-{app_id}",
        f"docpkg-{app_id}",
        f"credit-{app_id}",
        f"fraud-{app_id}",
        f"compliance-{app_id}",
        f"decision-{app_id}"
    ]
    
    all_events = []
    for stream_id in streams:
        try:
            events = await event_store.load_stream(stream_id)
            all_events.extend(events)
        except Exception:
            # Stream might not exist
            continue
    
    # Sort by global_position
    all_events.sort(key=lambda e: e.get("global_position", 0))
    
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"\n{'='*70}")
    print(f"DECISION HISTORY RETRIEVAL PERFORMANCE TEST")
    print(f"{'='*70}")
    print(f"Application ID: {app_id}")
    print(f"Total Events Retrieved: {len(all_events)}")
    print(f"Streams Checked: {len(streams)}")
    print(f"Elapsed Time: {elapsed_time:.3f} seconds")
    print(f"{'='*70}")
    
    # Verify against thresholds
    if elapsed_time > WEEK_STANDARD_THRESHOLD:
        pytest.fail(
            f"❌ WEEK STANDARD VIOLATION: Query took {elapsed_time:.3f}s "
            f"(threshold: {WEEK_STANDARD_THRESHOLD}s)"
        )
    elif elapsed_time > OPTIMAL_THRESHOLD:
        print(f"⚠️  WARNING: Query took {elapsed_time:.3f}s (optimal: <{OPTIMAL_THRESHOLD}s)")
        print(f"✅ WEEK STANDARD MET: Query completed in <{WEEK_STANDARD_THRESHOLD}s")
    else:
        print(f"✅ OPTIMAL PERFORMANCE: Query completed in {elapsed_time:.3f}s")
    
    assert elapsed_time < WEEK_STANDARD_THRESHOLD, \
        f"Decision history retrieval exceeded Week Standard ({WEEK_STANDARD_THRESHOLD}s)"


# ============================================================================
# TEST 2: Projection Query Performance
# ============================================================================

@pytest.mark.asyncio
async def test_projection_query_performance(db_pool):
    """
    Test: Query application_summary projection table.
    
    Should be near-instantaneous (<100ms) since it's a denormalized read model.
    """
    start_time = time.time()
    
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT application_id, status, total_revenue, risk_score, fraud_score
            FROM application_summary
            ORDER BY updated_at DESC
            LIMIT 100
            """
        )
    
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"PROJECTION QUERY PERFORMANCE TEST")
    print(f"{'='*70}")
    print(f"Rows Retrieved: {len(rows)}")
    print(f"Elapsed Time: {elapsed_time*1000:.1f} ms")
    print(f"{'='*70}")
    
    # Projection queries should be very fast
    assert elapsed_time < 0.1, \
        f"Projection query too slow: {elapsed_time*1000:.1f}ms (expected <100ms)"
    
    print(f"✅ PROJECTION QUERY: {elapsed_time*1000:.1f}ms (excellent)")


# ============================================================================
# TEST 3: Event Store Aggregate Load Performance
# ============================================================================

@pytest.mark.asyncio
async def test_event_store_aggregate_load_performance(event_store, db_pool):
    """
    Test: Load and replay an aggregate from event stream.
    
    This simulates what agents do when they load application state.
    Should complete in <500ms for typical applications.
    """
    # Find an application with events
    async with db_pool.acquire() as conn:
        app_id = await conn.fetchval(
            "SELECT application_id FROM application_summary LIMIT 1"
        )
    
    if not app_id:
        pytest.skip("No applications in database to test")
    
    start_time = time.time()
    
    # Load loan stream (main aggregate)
    loan_events = await event_store.load_stream(f"loan-{app_id}")
    
    elapsed_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"AGGREGATE LOAD PERFORMANCE TEST")
    print(f"{'='*70}")
    print(f"Application ID: {app_id}")
    print(f"Events in Aggregate: {len(loan_events)}")
    print(f"Elapsed Time: {elapsed_time*1000:.1f} ms")
    print(f"{'='*70}")
    
    # Aggregate loads should be fast
    assert elapsed_time < 0.5, \
        f"Aggregate load too slow: {elapsed_time*1000:.1f}ms (expected <500ms)"
    
    if elapsed_time < 0.1:
        print(f"✅ AGGREGATE LOAD: {elapsed_time*1000:.1f}ms (excellent)")
    else:
        print(f"✅ AGGREGATE LOAD: {elapsed_time*1000:.1f}ms (acceptable)")


# ============================================================================
# TEST 4: Concurrent Query Performance
# ============================================================================

@pytest.mark.asyncio
async def test_concurrent_query_performance(event_store, db_pool):
    """
    Test: Simulate 10 concurrent decision history queries.
    
    Verifies that the system can handle multiple simultaneous queries
    without significant performance degradation.
    """
    import asyncio
    
    # Find multiple applications
    async with db_pool.acquire() as conn:
        app_ids = await conn.fetch(
            "SELECT application_id FROM application_summary LIMIT 10"
        )
    
    if len(app_ids) < 5:
        pytest.skip("Need at least 5 applications for concurrent test")
    
    app_ids = [row["application_id"] for row in app_ids]
    
    async def query_history(app_id: str):
        """Query history for a single application."""
        streams = [
            f"loan-{app_id}", f"docpkg-{app_id}", f"credit-{app_id}",
            f"fraud-{app_id}", f"compliance-{app_id}", f"decision-{app_id}"
        ]
        all_events = []
        for stream_id in streams:
            try:
                events = await event_store.load_stream(stream_id)
                all_events.extend(events)
            except Exception:
                continue
        return len(all_events)
    
    # Run concurrent queries
    start_time = time.time()
    results = await asyncio.gather(*[query_history(app_id) for app_id in app_ids])
    elapsed_time = time.time() - start_time
    
    total_events = sum(results)
    avg_time_per_query = elapsed_time / len(app_ids)
    
    print(f"\n{'='*70}")
    print(f"CONCURRENT QUERY PERFORMANCE TEST")
    print(f"{'='*70}")
    print(f"Concurrent Queries: {len(app_ids)}")
    print(f"Total Events Retrieved: {total_events}")
    print(f"Total Elapsed Time: {elapsed_time:.3f}s")
    print(f"Average Time Per Query: {avg_time_per_query:.3f}s")
    print(f"{'='*70}")
    
    # Average should still be under Week Standard
    assert avg_time_per_query < WEEK_STANDARD_THRESHOLD, \
        f"Concurrent queries too slow: {avg_time_per_query:.3f}s avg"
    
    print(f"✅ CONCURRENT PERFORMANCE: {avg_time_per_query:.3f}s avg per query")


# ============================================================================
# PERFORMANCE SUMMARY
# ============================================================================

@pytest.mark.asyncio
async def test_performance_summary(event_store, db_pool):
    """
    Generate a performance summary report.
    """
    print(f"\n{'='*70}")
    print(f"PERFORMANCE SUMMARY - THE LEDGER")
    print(f"{'='*70}")
    print(f"Week Standard Threshold: {WEEK_STANDARD_THRESHOLD}s")
    print(f"Optimal Threshold: {OPTIMAL_THRESHOLD}s")
    print(f"")
    print(f"Expected Performance:")
    print(f"  - Decision History Retrieval: <{WEEK_STANDARD_THRESHOLD}s (optimal: <{OPTIMAL_THRESHOLD}s)")
    print(f"  - Projection Queries: <100ms")
    print(f"  - Aggregate Loads: <500ms")
    print(f"  - Concurrent Queries: <{WEEK_STANDARD_THRESHOLD}s avg")
    print(f"{'='*70}")
    print(f"Run individual tests above to measure actual performance.")
    print(f"{'='*70}\n")

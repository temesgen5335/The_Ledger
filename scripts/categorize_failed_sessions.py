"""
scripts/categorize_failed_sessions.py
======================================
Categorize and analyze AgentSessionFailed events in the database.

Categorization:
- Recoverable: Transient errors (timeouts, rate limits) that can be retried
- Non-Recoverable: Logic errors (NotFound, AttributeError) requiring code fixes
- Unknown: Errors without clear categorization

Run: python scripts/categorize_failed_sessions.py
"""
import asyncio
import asyncpg
from datetime import datetime
from collections import defaultdict

DB_URL = "postgresql://postgres:apex@localhost/apex_ledger"

# Error categorization rules
RECOVERABLE_ERRORS = {
    "llm_timeout", "RateLimitError", "ConnectionError", 
    "TimeoutError", "ServiceUnavailable"
}

NON_RECOVERABLE_ERRORS = {
    "NotFound", "AttributeError", "TypeError", "ValueError",
    "KeyError", "IndexError", "ValidationError"
}


async def analyze_failed_sessions():
    """Analyze and categorize all AgentSessionFailed events."""
    pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=5)
    
    try:
        async with pool.acquire() as conn:
            # Fetch all failed session events
            failed_events = await conn.fetch("""
                SELECT 
                    global_position,
                    stream_id,
                    payload::jsonb as payload,
                    recorded_at
                FROM events
                WHERE event_type = 'AgentSessionFailed'
                ORDER BY global_position
            """)
            
            print(f"\n{'='*80}")
            print(f"AGENT SESSION FAILURE ANALYSIS")
            print(f"{'='*80}")
            print(f"Total Failed Sessions: {len(failed_events)}\n")
            
            # Categorize by error type
            recoverable = []
            non_recoverable = []
            unknown = []
            
            error_counts = defaultdict(int)
            agent_type_failures = defaultdict(int)
            
            for event in failed_events:
                payload = event['payload']
                # Parse payload if it's a JSON string
                if isinstance(payload, str):
                    import json
                    payload = json.loads(payload)
                
                error_type = payload.get('error_type', 'Unknown')
                error_msg = payload.get('error_message', '')
                agent_type = payload.get('agent_type', 'unknown')
                is_recoverable = payload.get('recoverable', False)
                
                error_counts[error_type] += 1
                agent_type_failures[agent_type] += 1
                
                failure_info = {
                    'global_position': event['global_position'],
                    'stream_id': event['stream_id'],
                    'agent_type': agent_type,
                    'error_type': error_type,
                    'error_message': error_msg[:100],
                    'is_recoverable': is_recoverable,
                    'recorded_at': event['recorded_at']
                }
                
                if error_type in RECOVERABLE_ERRORS or is_recoverable:
                    recoverable.append(failure_info)
                elif error_type in NON_RECOVERABLE_ERRORS:
                    non_recoverable.append(failure_info)
                else:
                    unknown.append(failure_info)
            
            # Print summary
            print(f"CATEGORIZATION SUMMARY:")
            print(f"  Recoverable (can retry):     {len(recoverable)}")
            print(f"  Non-Recoverable (need fix):  {len(non_recoverable)}")
            print(f"  Unknown (needs review):      {len(unknown)}")
            print(f"\n{'='*80}")
            
            # Print error type breakdown
            print(f"\nERROR TYPE BREAKDOWN:")
            for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                category = "RECOVERABLE" if error_type in RECOVERABLE_ERRORS else \
                          "NON-RECOVERABLE" if error_type in NON_RECOVERABLE_ERRORS else \
                          "UNKNOWN"
                print(f"  {error_type:30s} {count:3d} [{category}]")
            
            # Print agent type breakdown
            print(f"\n{'='*80}")
            print(f"\nFAILURES BY AGENT TYPE:")
            for agent_type, count in sorted(agent_type_failures.items(), key=lambda x: x[1], reverse=True):
                print(f"  {agent_type:30s} {count:3d}")
            
            # Print detailed non-recoverable failures
            if non_recoverable:
                print(f"\n{'='*80}")
                print(f"\nNON-RECOVERABLE FAILURES (Require Code Fixes):")
                print(f"{'='*80}")
                for i, failure in enumerate(non_recoverable, 1):
                    print(f"\n{i}. Position {failure['global_position']} - {failure['agent_type']}")
                    print(f"   Error Type: {failure['error_type']}")
                    print(f"   Message: {failure['error_message']}")
                    print(f"   Stream: {failure['stream_id']}")
                    print(f"   Time: {failure['recorded_at']}")
            
            # Print recommendations
            print(f"\n{'='*80}")
            print(f"\nRECOMMENDATIONS:")
            print(f"{'='*80}")
            
            if non_recoverable:
                print(f"\n1. FIX NON-RECOVERABLE ERRORS:")
                unique_errors = set(f['error_type'] for f in non_recoverable)
                for error_type in unique_errors:
                    print(f"   - {error_type}: Review code and fix logic errors")
            
            if recoverable:
                print(f"\n2. RETRY RECOVERABLE FAILURES:")
                print(f"   - {len(recoverable)} sessions can be retried")
                print(f"   - Implement exponential backoff for rate limits")
            
            if unknown:
                print(f"\n3. REVIEW UNKNOWN ERRORS:")
                print(f"   - {len(unknown)} errors need categorization")
                print(f"   - Update RECOVERABLE_ERRORS or NON_RECOVERABLE_ERRORS sets")
            
            print(f"\n{'='*80}\n")
            
            # Update database with proper categorization
            print("Updating database with categorization...")
            update_count = 0
            
            for failure in failed_events:
                payload = failure['payload']
                # Parse payload if it's a JSON string
                if isinstance(payload, str):
                    import json
                    payload = json.loads(payload)
                
                error_type = payload.get('error_type', 'Unknown')
                
                # Determine correct recoverable status
                correct_recoverable = error_type in RECOVERABLE_ERRORS
                current_recoverable = payload.get('recoverable', False)
                
                if correct_recoverable != current_recoverable:
                    # Note: We can't update immutable events, but we can log the discrepancy
                    update_count += 1
            
            if update_count > 0:
                print(f"⚠️  Found {update_count} events with incorrect 'recoverable' flag")
                print(f"   (Events are immutable - categorization is for analysis only)")
            else:
                print(f"✅ All events have correct 'recoverable' categorization")
            
            print(f"\n{'='*80}\n")
    
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(analyze_failed_sessions())

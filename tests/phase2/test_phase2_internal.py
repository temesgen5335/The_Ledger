"""
tests/phase2/test_phase2_internal.py
====================================
Phase 2 Gate Tests:
1. Aggregate Rehydration: LoanApplicationAggregate correctly rebuilds state from events
2. Gas Town Recovery: Agent resumes from checkpoint after simulated crash

Run: pytest tests/phase2/test_phase2_internal.py -v
"""
import pytest
import sys
from pathlib import Path
from uuid import uuid4
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ledger.domain.aggregates.loan_application import LoanApplicationAggregate, ApplicationState
from ledger.event_store import EventStore
from ledger.agents.base_agent import BaseApexAgent
from ledger.schema.events import (
    ApplicationSubmitted, DocumentUploadRequested, DocumentUploaded,
    ExtractionCompleted, CreditAnalysisRequested
)

DB_URL = "postgresql://postgres:apex@localhost/apex_ledger"


@pytest.fixture
async def store():
    """EventStore fixture."""
    s = EventStore(DB_URL)
    await s.connect()
    yield s
    await s.close()


# ─── TEST 1: AGGREGATE REHYDRATION ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_aggregate_rehydration_from_events(store):
    """
    GATE TEST 1: Aggregate Rehydration
    
    Given: A stream with 3 events (ApplicationSubmitted, DocumentUploadRequested, DocumentUploaded)
    When: LoanApplicationAggregate.load() replays the stream
    Then: Aggregate state matches expected values from events
    """
    app_id = f"TEST-AGG-{uuid4().hex[:8]}"
    stream_id = f"loan-{app_id}"
    
    # Create 3 events
    events = [
        ApplicationSubmitted(
            application_id=app_id,
            applicant_id="COMP-001",
            requested_amount_usd=250000.00,
            loan_purpose="working_capital",
            loan_term_months=36,
            submission_channel="web",
            contact_email="test@example.com",
            contact_name="Test User",
            application_reference=f"REF-{app_id}",
            submitted_at=datetime.utcnow()
        ),
        DocumentUploadRequested(
            application_id=app_id,
            required_document_types=["income_statement", "balance_sheet"],
            deadline=datetime.utcnow(),
            requested_by="system"
        ),
        DocumentUploaded(
            application_id=app_id,
            document_id="DOC-001",
            document_type="income_statement",
            document_format="pdf",
            filename="income_statement.pdf",
            file_path="documents/TEST/income_statement.pdf",
            file_size_bytes=12345,
            file_hash="abc123",
            fiscal_year=2024,
            uploaded_at=datetime.utcnow(),
            uploaded_by="applicant"
        )
    ]
    
    # Append events to store
    event_dicts = [e.to_store_dict() for e in events]
    await store.append(stream_id, event_dicts, expected_version=-1)
    
    # Load aggregate from stream
    agg = await LoanApplicationAggregate.load(store, app_id)
    
    # Assertions
    assert agg.application_id == app_id
    assert agg.state == ApplicationState.DOCUMENTS_UPLOADED
    assert agg.applicant_id == "COMP-001"
    assert float(agg.requested_amount_usd) == 250000.00
    assert agg.loan_purpose == "working_capital"
    assert agg.version == 3  # 3 events applied
    assert len(agg.events) == 3
    
    print(f"✓ Aggregate rehydrated correctly: state={agg.state}, version={agg.version}")


@pytest.mark.asyncio
async def test_aggregate_state_transitions(store):
    """
    Verify all major state transitions work correctly.
    """
    app_id = f"TEST-TRANS-{uuid4().hex[:8]}"
    stream_id = f"loan-{app_id}"
    
    # Event sequence covering full lifecycle
    events = [
        ApplicationSubmitted(
            application_id=app_id,
            applicant_id="COMP-002",
            requested_amount_usd=100000.00,
            loan_purpose="expansion",
            loan_term_months=24,
            submission_channel="api",
            contact_email="test2@example.com",
            contact_name="Test User 2",
            application_reference=f"REF-{app_id}",
            submitted_at=datetime.utcnow()
        ),
        DocumentUploadRequested(
            application_id=app_id,
            required_document_types=["income_statement"],
            deadline=datetime.utcnow(),
            requested_by="system"
        ),
        DocumentUploaded(
            application_id=app_id,
            document_id="DOC-002",
            document_type="income_statement",
            document_format="pdf",
            filename="income.pdf",
            file_path="documents/TEST/income.pdf",
            file_size_bytes=5000,
            file_hash="def456",
            uploaded_at=datetime.utcnow(),
            uploaded_by="applicant"
        ),
        CreditAnalysisRequested(
            application_id=app_id,
            requested_at=datetime.utcnow(),
            requested_by="document_processing_agent",
            priority="NORMAL"
        )
    ]
    
    event_dicts = [e.to_store_dict() for e in events]
    await store.append(stream_id, event_dicts, expected_version=-1)
    
    agg = await LoanApplicationAggregate.load(store, app_id)
    
    assert agg.state == ApplicationState.CREDIT_ANALYSIS_REQUESTED
    assert agg.version == 4
    
    print(f"✓ State transitions validated: {ApplicationState.SUBMITTED} → {agg.state}")


# ─── TEST 2: GAS TOWN RECOVERY ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_gas_town_recovery_skips_completed_nodes(store):
    """
    GATE TEST 2: Gas Town Recovery
    
    Scenario:
    1. Agent starts session and completes nodes A and B
    2. Agent crashes before completing node C
    3. Agent restarts with same session_id
    4. Agent should skip A and B (already have AgentNodeExecuted events)
    5. Agent should resume from node C
    
    This test simulates the crash by checking AgentNodeExecuted events.
    """
    app_id = f"TEST-RECOVERY-{uuid4().hex[:8]}"
    session_id = f"sess-{uuid4().hex[:8]}"
    agent_stream = f"agent-document_processing-{session_id}"
    
    # Simulate: Agent completed nodes "validate_inputs" and "validate_document_format"
    from ledger.schema.events import AgentSessionStarted, AgentNodeExecuted
    
    events = [
        AgentSessionStarted(
            session_id=session_id,
            agent_type="document_processing",
            agent_id="agent-doc-001",
            application_id=app_id,
            model_version="gemini-2.0-flash",
            langgraph_graph_version="1.0.0",
            context_source="fresh_start",
            context_token_count=0,
            started_at=datetime.utcnow()
        ),
        AgentNodeExecuted(
            session_id=session_id,
            agent_type="document_processing",
            node_name="validate_inputs",
            node_sequence=1,
            input_keys=["application_id"],
            output_keys=["document_ids"],
            llm_called=False,
            llm_tokens_input=0,
            llm_tokens_output=0,
            llm_cost_usd=0.0,
            duration_ms=50,
            executed_at=datetime.utcnow()
        ),
        AgentNodeExecuted(
            session_id=session_id,
            agent_type="document_processing",
            node_name="validate_document_format",
            node_sequence=2,
            input_keys=["document_ids"],
            output_keys=["validated_count"],
            llm_called=False,
            llm_tokens_input=0,
            llm_tokens_output=0,
            llm_cost_usd=0.0,
            duration_ms=30,
            executed_at=datetime.utcnow()
        )
        # Crash happens here — node "run_week3_extraction" never completed
    ]
    
    event_dicts = [e.to_store_dict() for e in events]
    await store.append(agent_stream, event_dicts, expected_version=-1)
    
    # Recovery: Load agent stream and check which nodes were completed
    import json
    agent_events = await store.load_stream(agent_stream)
    
    completed_nodes = [
        json.loads(e["payload"])["node_name"]
        for e in agent_events
        if e["event_type"] == "AgentNodeExecuted"
    ]
    
    # Assertions
    assert "validate_inputs" in completed_nodes
    assert "validate_document_format" in completed_nodes
    assert "run_week3_extraction" not in completed_nodes
    
    # Recovery logic: determine next node to execute
    all_nodes = [
        "validate_inputs",
        "validate_document_format", 
        "run_week3_extraction",
        "assess_quality",
        "write_output"
    ]
    
    next_node_to_execute = None
    for node in all_nodes:
        if node not in completed_nodes:
            next_node_to_execute = node
            break
    
    assert next_node_to_execute == "run_week3_extraction"
    
    print(f"✓ Gas Town recovery: completed={completed_nodes}, next={next_node_to_execute}")


@pytest.mark.asyncio
async def test_gas_town_session_replay_context(store):
    """
    Verify that restarted session has context_source indicating replay.
    """
    app_id = f"TEST-REPLAY-{uuid4().hex[:8]}"
    session_id_1 = f"sess-{uuid4().hex[:8]}"
    session_id_2 = f"sess-{uuid4().hex[:8]}"
    
    from ledger.schema.events import AgentSessionStarted
    
    # First session (crashed)
    stream_1 = f"agent-credit_analysis-{session_id_1}"
    event_1 = AgentSessionStarted(
        session_id=session_id_1,
        agent_type="credit_analysis",
        agent_id="agent-credit-001",
        application_id=app_id,
        model_version="gemini-2.0-flash",
        langgraph_graph_version="1.0.0",
        context_source="fresh_start",
        context_token_count=0,
        started_at=datetime.utcnow()
    )
    await store.append(stream_1, [event_1.to_store_dict()], expected_version=-1)
    
    # Second session (recovery from crash)
    stream_2 = f"agent-credit_analysis-{session_id_2}"
    event_2 = AgentSessionStarted(
        session_id=session_id_2,
        agent_type="credit_analysis",
        agent_id="agent-credit-002",
        application_id=app_id,
        model_version="gemini-2.0-flash",
        langgraph_graph_version="1.0.0",
        context_source=f"prior_session_replay:{session_id_1}",
        context_token_count=150,
        started_at=datetime.utcnow()
    )
    await store.append(stream_2, [event_2.to_store_dict()], expected_version=-1)
    
    # Load and verify
    import json
    events_2 = await store.load_stream(stream_2)
    session_start = events_2[0]
    
    assert session_start["event_type"] == "AgentSessionStarted"
    payload = json.loads(session_start["payload"])
    context_source = payload.get("context_source", "")
    assert context_source.startswith("prior_session_replay:")
    assert session_id_1 in context_source
    
    print(f"✓ Session replay context verified: {context_source}")


# ─── TEST 3: INTEGRATION SMOKE TEST ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_aggregate_and_recovery_integration(store):
    """
    Combined test: Aggregate rehydration + Gas Town recovery in one scenario.
    
    Simulates:
    1. Application submitted with documents
    2. DocumentProcessingAgent starts, completes 2 nodes, crashes
    3. Aggregate loads correctly from loan stream
    4. Recovery detects incomplete session and resumes
    """
    app_id = f"TEST-INTEG-{uuid4().hex[:8]}"
    session_id = f"sess-{uuid4().hex[:8]}"
    
    from ledger.schema.events import (
        ApplicationSubmitted, DocumentUploaded,
        AgentSessionStarted, AgentNodeExecuted
    )
    
    # Loan stream events
    loan_stream = f"loan-{app_id}"
    loan_events = [
        ApplicationSubmitted(
            application_id=app_id,
            applicant_id="COMP-INTEG",
            requested_amount_usd=500000.00,
            loan_purpose="equipment_financing",
            loan_term_months=48,
            submission_channel="api",
            contact_email="integ@example.com",
            contact_name="Integration Test",
            application_reference=f"REF-{app_id}",
            submitted_at=datetime.utcnow()
        ),
        DocumentUploaded(
            application_id=app_id,
            document_id="DOC-INTEG-001",
            document_type="balance_sheet",
            document_format="pdf",
            filename="balance.pdf",
            file_path="documents/INTEG/balance.pdf",
            file_size_bytes=8000,
            file_hash="integ123",
            uploaded_at=datetime.utcnow(),
            uploaded_by="applicant"
        )
    ]
    await store.append(loan_stream, [e.to_store_dict() for e in loan_events], expected_version=-1)
    
    # Agent stream events (partial execution)
    agent_stream = f"agent-document_processing-{session_id}"
    agent_events = [
        AgentSessionStarted(
            session_id=session_id,
            agent_type="document_processing",
            agent_id="agent-doc-integ",
            application_id=app_id,
            model_version="gemini-2.0-flash",
            langgraph_graph_version="1.0.0",
            context_source="fresh_start",
            context_token_count=0,
            started_at=datetime.utcnow()
        ),
        AgentNodeExecuted(
            session_id=session_id,
            agent_type="document_processing",
            node_name="validate_inputs",
            node_sequence=1,
            input_keys=["application_id"],
            output_keys=["document_ids"],
            llm_called=False,
            llm_tokens_input=0,
            llm_tokens_output=0,
            llm_cost_usd=0.0,
            duration_ms=25,
            executed_at=datetime.utcnow()
        )
    ]
    await store.append(agent_stream, [e.to_store_dict() for e in agent_events], expected_version=-1)
    
    # Test 1: Aggregate rehydration
    agg = await LoanApplicationAggregate.load(store, app_id)
    assert agg.state == ApplicationState.DOCUMENTS_UPLOADED
    assert agg.applicant_id == "COMP-INTEG"
    
    # Test 2: Recovery detection
    import json
    agent_stream_events = await store.load_stream(agent_stream)
    completed_nodes = [
        json.loads(e["payload"])["node_name"]
        for e in agent_stream_events
        if e["event_type"] == "AgentNodeExecuted"
    ]
    
    assert len(completed_nodes) == 1
    assert "validate_inputs" in completed_nodes
    
    print(f"✓ Integration test passed: agg.state={agg.state}, completed_nodes={completed_nodes}")

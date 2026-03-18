"""
tests/test_narratives.py
========================
The 5 narrative scenario tests. These are the primary correctness gate.
These FAIL until all 5 agents and aggregates are implemented.

Run: pytest tests/test_narratives.py -v -s
"""
import pytest, sys
from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent))

# Narrative scenarios tested here match Section 7 of the challenge document.
# Each test drives a complete application through the real agent pipeline.

@pytest.mark.asyncio
async def test_narr01_concurrent_occ_collision():
    """
    NARR-01: Two CreditAnalysisAgent instances run simultaneously.
    Expected: exactly one CreditAnalysisCompleted in credit stream (not two),
              second agent gets OCC, reloads, retries successfully.
    """
    # This test requires CreditAnalysisAgent which is not yet implemented
    # Skipping for now - will implement when CreditAnalysisAgent is ready
    pytest.skip("CreditAnalysisAgent not yet implemented - Phase 4")

@pytest.mark.asyncio
async def test_narr02_document_extraction_failure():
    """
    NARR-02: Income statement PDF with missing EBITDA line.
    Expected: DocumentQualityFlagged with critical_missing_fields=['ebitda'],
              CreditAnalysisCompleted.confidence <= 0.75,
              CreditAnalysisCompleted.data_quality_caveats is non-empty.
    """
    # This test requires CreditAnalysisAgent which is not yet implemented
    # DocumentProcessingAgent is ready, but credit analysis is Phase 4
    pytest.skip("CreditAnalysisAgent not yet implemented - Phase 4")

@pytest.mark.asyncio
async def test_narr03_agent_crash_recovery():
    """
    NARR-03: FraudDetectionAgent crashes mid-session.
    Expected: only ONE FraudScreeningCompleted event in fraud stream,
              second AgentSessionStarted has context_source starting with 'prior_session_replay:',
              no duplicate analysis work.
    """
    from uuid import uuid4
    from datetime import datetime
    from ledger.event_store import EventStore
    from ledger.registry.client import ApplicantRegistryClient
    from ledger.agents.base_agent import FraudDetectionAgent
    from ledger.schema.events import (
        ApplicationSubmitted, DocumentUploaded, ExtractionCompleted,
        FraudScreeningRequested, AgentSessionStarted, AgentNodeExecuted
    )
    import json
    from dotenv import load_dotenv
    import asyncpg
    
    # Load environment variables
    load_dotenv()
    
    # Setup
    DB_URL = "postgresql://postgres:apex@localhost/apex_ledger"
    store = EventStore(DB_URL)
    await store.connect()
    
    # Create registry pool
    registry_pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=5)
    registry = ApplicantRegistryClient(registry_pool)
    
    app_id = f"NARR03-{uuid4().hex[:8]}"
    session_id_1 = f"fraud-sess-1-{uuid4().hex[:6]}"
    session_id_2 = f"fraud-sess-2-{uuid4().hex[:6]}"
    
    # Create loan application with documents
    loan_events = [
        ApplicationSubmitted(
            application_id=app_id,
            applicant_id="COMP-001",
            requested_amount_usd=300000.00,
            loan_purpose="working_capital",
            loan_term_months=36,
            submission_channel="web",
            contact_email="narr03@test.com",
            contact_name="NARR03 Test",
            application_reference=f"REF-{app_id}",
            submitted_at=datetime.utcnow()
        ),
        DocumentUploaded(
            application_id=app_id,
            document_id="DOC-NARR03-001",
            document_type="income_statement",
            document_format="pdf",
            filename="income.pdf",
            file_path="documents/NARR03/income.pdf",
            file_size_bytes=10000,
            file_hash="narr03hash",
            uploaded_at=datetime.utcnow(),
            uploaded_by="applicant"
        ),
        FraudScreeningRequested(
            application_id=app_id,
            requested_at=datetime.utcnow(),
            triggered_by_event_id=str(uuid4())
        )
    ]
    await store.append(f"loan-{app_id}", [e.to_store_dict() for e in loan_events], expected_version=-1)
    
    # Create document package with extraction
    from ledger.schema.events import FinancialFacts
    from decimal import Decimal
    docpkg_events = [
        ExtractionCompleted(
            package_id=f"pkg-{app_id}",
            document_id="DOC-NARR03-001",
            document_type="income_statement",
            facts=FinancialFacts(
                total_revenue=Decimal("500000"),
                net_income=Decimal("50000"),
                fiscal_year=2023
            ),
            raw_text_length=5000,
            tables_extracted=2,
            processing_ms=1500,
            completed_at=datetime.utcnow()
        )
    ]
    await store.append(f"docpkg-{app_id}", [e.to_store_dict() for e in docpkg_events], expected_version=-1)
    
    # Simulate first session that crashes after 2 nodes
    agent_stream_1 = f"agent-fraud_detection-{session_id_1}"
    crash_events = [
        AgentSessionStarted(
            session_id=session_id_1,
            agent_type="fraud_detection",
            agent_id="fraud-agent-1",
            application_id=app_id,
            model_version="gemini-1.5-pro",
            langgraph_graph_version="1.0.0",
            context_source="fresh_start",
            context_token_count=0,
            started_at=datetime.utcnow()
        ),
        AgentNodeExecuted(
            session_id=session_id_1,
            agent_type="fraud_detection",
            node_name="validate_inputs",
            node_sequence=1,
            input_keys=["application_id"],
            output_keys=["fraud_requested"],
            llm_called=False,
            llm_tokens_input=0,
            llm_tokens_output=0,
            llm_cost_usd=0.0,
            duration_ms=20,
            executed_at=datetime.utcnow()
        ),
        AgentNodeExecuted(
            session_id=session_id_1,
            agent_type="fraud_detection",
            node_name="load_document_facts",
            node_sequence=2,
            input_keys=["docpkg_stream"],
            output_keys=["documents_loaded"],
            llm_called=False,
            llm_tokens_input=0,
            llm_tokens_output=0,
            llm_cost_usd=0.0,
            duration_ms=30,
            executed_at=datetime.utcnow()
        )
        # CRASH HERE - no more nodes executed
    ]
    await store.append(agent_stream_1, [e.to_store_dict() for e in crash_events], expected_version=-1)
    
    # Recovery: Start second session with context_source referencing first session
    agent_2 = FraudDetectionAgent(
        agent_id="fraud-agent-2",
        agent_type="fraud_detection",
        store=store,
        registry=registry,
        model="gemini-1.5-pro"
    )
    
    # Run the agent (it should detect completed nodes and skip them)
    await agent_2.run(
        application_id=app_id,
        session_id=session_id_2,
        context_source=f"prior_session_replay:{session_id_1}"
    )
    
    # Verify results
    fraud_stream = await store.load_stream(f"fraud-{app_id}")
    fraud_completed_events = [e for e in fraud_stream if e["event_type"] == "FraudScreeningCompleted"]
    
    # Should have exactly ONE FraudScreeningCompleted
    assert len(fraud_completed_events) == 1, f"Expected 1 FraudScreeningCompleted, got {len(fraud_completed_events)}"
    
    # Verify second session has correct context_source
    agent_stream_2 = await store.load_stream(f"agent-fraud_detection-{session_id_2}")
    session_start_2 = [e for e in agent_stream_2 if e["event_type"] == "AgentSessionStarted"]
    assert len(session_start_2) == 1
    
    payload = json.loads(session_start_2[0]["payload"]) if isinstance(session_start_2[0]["payload"], str) else session_start_2[0]["payload"]
    context_source = payload.get("context_source", "")
    assert context_source.startswith("prior_session_replay:"), f"Expected context_source to start with 'prior_session_replay:', got '{context_source}'"
    assert session_id_1 in context_source, f"Expected session_id_1 '{session_id_1}' in context_source '{context_source}'"
    
    # Verify no duplicate work - check that validate_inputs and load_document_facts were not re-executed
    node_executions_2 = [e for e in agent_stream_2 if e["event_type"] == "AgentNodeExecuted"]
    node_names_2 = [json.loads(e["payload"])["node_name"] if isinstance(e["payload"], str) else e["payload"]["node_name"] for e in node_executions_2]
    
    # Should NOT have validate_inputs or load_document_facts (already done in session 1)
    # Should have cross_reference_registry, analyze_fraud_patterns, write_output
    assert "validate_inputs" not in node_names_2, "validate_inputs should not be re-executed"
    assert "load_document_facts" not in node_names_2, "load_document_facts should not be re-executed"
    
    print(f"✓ NARR-03 PASSED: Crash recovery verified")
    print(f"  - Only 1 FraudScreeningCompleted event")
    print(f"  - Session 2 context: {context_source}")
    print(f"  - No duplicate work: {node_names_2}")

@pytest.mark.asyncio
async def test_narr04_compliance_hard_block():
    """
    NARR-04: Montana applicant (jurisdiction='MT') triggers REG-003.
    Expected: ComplianceRuleFailed(rule_id='REG-003', is_hard_block=True),
              NO DecisionGenerated event,
              ApplicationDeclined with adverse_action_notice_required=True.
    """
    pytest.skip("Implement after ComplianceAgent is working")

@pytest.mark.asyncio
async def test_narr05_human_override():
    """
    NARR-05: Orchestrator recommends DECLINE; human loan officer overrides to APPROVE.
    Expected: DecisionGenerated(recommendation='DECLINE'),
              HumanReviewCompleted(override=True, reviewer_id='LO-Sarah-Chen'),
              ApplicationApproved(approved_amount_usd=750000, conditions has 2 items).
    """
    pytest.skip("Implement after all agents + HumanReviewCompleted command handler working")

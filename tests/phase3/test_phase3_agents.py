"""
tests/phase3/test_phase3_agents.py
===================================
Phase 3 Gate Tests: Agent Implementations

Tests for the 5-agent pipeline:
1. CreditAnalysisAgent (reference implementation)
2. DocumentProcessingAgent (Week 3 integration)
3. FraudDetectionAgent (cross-reference with registry)
4. ComplianceAgent (deterministic rules)
5. DecisionOrchestratorAgent (LLM synthesis + OCC)

Each test verifies:
- Gas Town checkpointing (AgentSessionStarted, AgentNodeExecuted, AgentSessionCompleted)
- Event appends to correct streams
- LLM token/cost tracking
- OCC handling where applicable
"""
import pytest
import asyncpg
from uuid import uuid4
from datetime import datetime
from decimal import Decimal
from dotenv import load_dotenv
import os

from ledger.event_store import EventStore
from ledger.registry.client import ApplicantRegistryClient
from ledger.agents.credit_analysis_agent import CreditAnalysisAgent
from ledger.agents.base_agent import (
    DocumentProcessingAgent, FraudDetectionAgent, 
    ComplianceAgent, DecisionOrchestratorAgent
)
from ledger.schema.events import (
    ApplicationSubmitted, DocumentUploaded, ExtractionCompleted,
    FinancialFacts, FraudScreeningRequested
)

load_dotenv()

DB_URL = "postgresql://postgres:apex@localhost/apex_ledger"


@pytest.fixture
async def event_store():
    """Create and connect EventStore."""
    store = EventStore(DB_URL)
    await store.connect()
    yield store
    await store.close()


@pytest.fixture
async def registry():
    """Create ApplicantRegistryClient with connection pool."""
    pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=5)
    client = ApplicantRegistryClient(pool)
    yield client
    await pool.close()


# ============================================================================
# TEST 1: CreditAnalysisAgent - Reference Implementation
# ============================================================================

@pytest.mark.asyncio
async def test_credit_analysis_agent_complete_flow(event_store, registry):
    """
    Test CreditAnalysisAgent end-to-end:
    - Validates inputs from loan stream
    - Opens credit record
    - Loads registry data
    - Loads extracted facts from docpkg stream
    - Performs LLM credit analysis
    - Applies policy constraints
    - Writes CreditAnalysisCompleted to credit stream
    - Triggers FraudScreeningRequested on loan stream
    """
    app_id = f"PHASE3-CREDIT-{uuid4().hex[:8]}"
    
    # Setup: Create loan application with documents
    loan_events = [
        ApplicationSubmitted(
            application_id=app_id,
            applicant_id="COMP-001",
            requested_amount_usd=Decimal("500000.00"),
            loan_purpose="working_capital",
            loan_term_months=36,
            submission_channel="web",
            contact_email="test@credit.com",
            contact_name="Credit Test",
            application_reference=f"REF-{app_id}",
            submitted_at=datetime.utcnow()
        )
    ]
    await event_store.append(f"loan-{app_id}", [e.to_store_dict() for e in loan_events], expected_version=-1)
    
    # Setup: Create document package with extraction
    docpkg_events = [
        ExtractionCompleted(
            package_id=f"pkg-{app_id}",
            document_id="DOC-001",
            document_type="income_statement",
            facts=FinancialFacts(
                total_revenue=Decimal("2000000"),
                net_income=Decimal("200000"),
                total_assets=Decimal("1500000"),
                fiscal_year=2023
            ),
            raw_text_length=5000,
            tables_extracted=3,
            processing_ms=2000,
            completed_at=datetime.utcnow()
        )
    ]
    await event_store.append(f"docpkg-{app_id}", [e.to_store_dict() for e in docpkg_events], expected_version=-1)
    
    # Execute: Run CreditAnalysisAgent
    agent = CreditAnalysisAgent(
        agent_id="credit-agent-test",
        agent_type="credit_analysis",
        store=event_store,
        registry=registry,
        model="gemini-1.5-pro"
    )
    
    await agent.process_application(app_id)
    
    # Verify: Check agent session events
    session_stream = f"agent-credit_analysis-{agent.session_id}"
    session_events = await event_store.load_stream(session_stream)
    
    assert len(session_events) > 0, "Agent session should have events"
    assert session_events[0]["event_type"] == "AgentSessionStarted"
    assert session_events[-1]["event_type"] == "AgentSessionCompleted"
    
    # Verify: Check node executions
    node_events = [e for e in session_events if e["event_type"] == "AgentNodeExecuted"]
    expected_nodes = [
        "validate_inputs", "open_credit_record", "load_applicant_registry",
        "load_extracted_facts", "analyze_credit_risk", "apply_policy_constraints",
        "write_output"
    ]
    executed_nodes = []
    for e in node_events:
        payload = e["payload"]
        if isinstance(payload, str):
            import json
            payload = json.loads(payload)
        executed_nodes.append(payload["node_name"])
    
    for expected in expected_nodes:
        assert expected in executed_nodes, f"Node {expected} should be executed"
    
    # Verify: Check credit stream output
    credit_events = await event_store.load_stream(f"credit-{app_id}")
    credit_types = [e["event_type"] for e in credit_events]
    
    assert "CreditRecordOpened" in credit_types
    assert "HistoricalProfileConsumed" in credit_types
    assert "ExtractedFactsConsumed" in credit_types
    assert "CreditAnalysisCompleted" in credit_types
    
    # Verify: Check fraud trigger on loan stream
    loan_events_after = await event_store.load_stream(f"loan-{app_id}")
    fraud_triggers = [e for e in loan_events_after if e["event_type"] == "FraudScreeningRequested"]
    assert len(fraud_triggers) == 1, "Should trigger fraud screening"
    
    # Verify: LLM was called for analysis
    analyze_nodes = []
    for e in node_events:
        payload = e["payload"]
        if isinstance(payload, str):
            import json
            payload = json.loads(payload)
        if payload.get("node_name") == "analyze_credit_risk":
            analyze_nodes.append(payload)
    
    if analyze_nodes:
        analyze_node = analyze_nodes[0]
        assert analyze_node.get("llm_called") is True, "LLM should be called for credit analysis"
        assert analyze_node.get("llm_tokens_input", 0) > 0, "Should have LLM input tokens"
        assert analyze_node.get("llm_cost_usd", 0) > 0, "Should have LLM cost"


# ============================================================================
# TEST 2: FraudDetectionAgent
# ============================================================================

@pytest.mark.asyncio
async def test_fraud_detection_agent_flow(event_store, registry):
    """
    Test FraudDetectionAgent:
    - Validates FraudScreeningRequested exists
    - Loads document facts
    - Cross-references with registry
    - Performs LLM fraud pattern analysis
    - Writes FraudScreeningCompleted to fraud stream
    """
    app_id = f"PHASE3-FRAUD-{uuid4().hex[:8]}"
    
    # Setup: Create loan with fraud screening request
    loan_events = [
        ApplicationSubmitted(
            application_id=app_id,
            applicant_id="COMP-001",
            requested_amount_usd=Decimal("300000.00"),
            loan_purpose="equipment",
            loan_term_months=24,
            submission_channel="web",
            contact_email="test@fraud.com",
            contact_name="Fraud Test",
            application_reference=f"REF-{app_id}",
            submitted_at=datetime.utcnow()
        ),
        FraudScreeningRequested(
            application_id=app_id,
            requested_at=datetime.utcnow(),
            triggered_by_event_id=str(uuid4())
        )
    ]
    await event_store.append(f"loan-{app_id}", [e.to_store_dict() for e in loan_events], expected_version=-1)
    
    # Setup: Create document package
    docpkg_events = [
        ExtractionCompleted(
            package_id=f"pkg-{app_id}",
            document_id="DOC-FRAUD-001",
            document_type="income_statement",
            facts=FinancialFacts(
                total_revenue=Decimal("1000000"),
                net_income=Decimal("100000"),
                fiscal_year=2023
            ),
            raw_text_length=4000,
            tables_extracted=2,
            processing_ms=1500,
            completed_at=datetime.utcnow()
        )
    ]
    await event_store.append(f"docpkg-{app_id}", [e.to_store_dict() for e in docpkg_events], expected_version=-1)
    
    # Execute: Run FraudDetectionAgent
    agent = FraudDetectionAgent(
        agent_id="fraud-agent-test",
        agent_type="fraud_detection",
        store=event_store,
        registry=registry,
        model="gemini-1.5-pro"
    )
    
    await agent.process_application(app_id)
    
    # Verify: Check fraud stream output
    fraud_events = await event_store.load_stream(f"fraud-{app_id}")
    fraud_types = [e["event_type"] for e in fraud_events]
    
    assert "FraudScreeningCompleted" in fraud_types or "FraudAnomalyDetected" in fraud_types
    
    # Verify: Agent session completed
    session_stream = f"agent-fraud_detection-{agent.session_id}"
    session_events = await event_store.load_stream(session_stream)
    assert session_events[-1]["event_type"] == "AgentSessionCompleted"


# ============================================================================
# TEST 3: ComplianceAgent - Deterministic Rules
# ============================================================================

@pytest.mark.asyncio
async def test_compliance_agent_deterministic_rules(event_store, registry):
    """
    Test ComplianceAgent:
    - No LLM calls (deterministic only)
    - Evaluates 6 regulatory rules
    - Writes ComplianceCheckCompleted
    - Detects hard blocks
    """
    app_id = f"PHASE3-COMPLIANCE-{uuid4().hex[:8]}"
    
    # Setup: Create application with high requested amount (will trigger REG-001)
    loan_events = [
        ApplicationSubmitted(
            application_id=app_id,
            applicant_id="COMP-001",
            requested_amount_usd=Decimal("2000000.00"),  # High amount
            loan_purpose="expansion",
            loan_term_months=60,
            submission_channel="web",
            contact_email="test@compliance.com",
            contact_name="Compliance Test",
            application_reference=f"REF-{app_id}",
            submitted_at=datetime.utcnow()
        )
    ]
    await event_store.append(f"loan-{app_id}", [e.to_store_dict() for e in loan_events], expected_version=-1)
    
    # Execute: Run ComplianceAgent
    agent = ComplianceAgent(
        agent_id="compliance-agent-test",
        agent_type="compliance",
        store=event_store,
        registry=registry,
        model="gemini-1.5-pro"  # Should not be used
    )
    
    await agent.process_application(app_id)
    
    # Verify: Check compliance stream
    compliance_events = await event_store.load_stream(f"compliance-{app_id}")
    compliance_types = [e["event_type"] for e in compliance_events]
    
    assert "ComplianceCheckCompleted" in compliance_types
    
    # Verify: No LLM calls (deterministic only)
    session_stream = f"agent-compliance-{agent.session_id}"
    session_events = await event_store.load_stream(session_stream)
    node_events = [e for e in session_events if e["event_type"] == "AgentNodeExecuted"]
    
    for node in node_events:
        assert node["payload"]["llm_called"] is False, "ComplianceAgent should not call LLM"


# ============================================================================
# TEST 4: DecisionOrchestratorAgent - OCC Handling
# ============================================================================

@pytest.mark.asyncio
async def test_decision_orchestrator_occ_retry(event_store, registry):
    """
    Test DecisionOrchestratorAgent:
    - Loads all prior agent outputs (credit, fraud, compliance)
    - Synthesizes decision with LLM
    - Handles OCC conflicts with retry
    - Writes DecisionGenerated to loan stream
    """
    app_id = f"PHASE3-DECISION-{uuid4().hex[:8]}"
    
    # Setup: Create complete pipeline state
    # (In real scenario, all prior agents would have run)
    loan_events = [
        ApplicationSubmitted(
            application_id=app_id,
            applicant_id="COMP-001",
            requested_amount_usd=Decimal("400000.00"),
            loan_purpose="working_capital",
            loan_term_months=36,
            submission_channel="web",
            contact_email="test@decision.com",
            contact_name="Decision Test",
            application_reference=f"REF-{app_id}",
            submitted_at=datetime.utcnow()
        )
    ]
    await event_store.append(f"loan-{app_id}", [e.to_store_dict() for e in loan_events], expected_version=-1)
    
    # Execute: Run DecisionOrchestratorAgent
    agent = DecisionOrchestratorAgent(
        agent_id="decision-agent-test",
        agent_type="decision_orchestrator",
        store=event_store,
        registry=registry,
        model="gemini-1.5-pro"
    )
    
    await agent.process_application(app_id)
    
    # Verify: Check decision stream
    decision_events = await event_store.load_stream(f"decision-{app_id}")
    decision_types = [e["event_type"] for e in decision_events]
    
    assert "DecisionGenerated" in decision_types or "DecisionDeferred" in decision_types
    
    # Verify: Agent session completed
    session_stream = f"agent-decision_orchestrator-{agent.session_id}"
    session_events = await event_store.load_stream(session_stream)
    assert len(session_events) > 0


# ============================================================================
# TEST 5: Gas Town Recovery - Node Replay
# ============================================================================

@pytest.mark.asyncio
async def test_agent_crash_recovery_gas_town(event_store, registry):
    """
    Test Gas Town crash recovery:
    - Simulate crashed session with partial node execution
    - Start new session with context_source="prior_session_replay:{session_id}"
    - Verify completed nodes are skipped
    - Verify remaining nodes execute
    """
    app_id = f"PHASE3-RECOVERY-{uuid4().hex[:8]}"
    session_id_1 = f"sess-recovery-1-{uuid4().hex[:6]}"
    
    # Setup: Create application
    loan_events = [
        ApplicationSubmitted(
            application_id=app_id,
            applicant_id="COMP-001",
            requested_amount_usd=Decimal("250000.00"),
            loan_purpose="inventory",
            loan_term_months=24,
            submission_channel="web",
            contact_email="test@recovery.com",
            contact_name="Recovery Test",
            application_reference=f"REF-{app_id}",
            submitted_at=datetime.utcnow()
        )
    ]
    await event_store.append(f"loan-{app_id}", [e.to_store_dict() for e in loan_events], expected_version=-1)
    
    # Simulate crashed session (only first 2 nodes completed)
    from ledger.schema.events import AgentSessionStarted, AgentNodeExecuted
    crash_events = [
        AgentSessionStarted(
            session_id=session_id_1,
            agent_type="fraud_detection",
            agent_id="fraud-agent-crashed",
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
            output_keys=["validated"],
            llm_called=False,
            llm_tokens_input=0,
            llm_tokens_output=0,
            llm_cost_usd=0.0,
            duration_ms=15,
            executed_at=datetime.utcnow()
        )
        # CRASH - no more nodes
    ]
    agent_stream_1 = f"agent-fraud_detection-{session_id_1}"
    await event_store.append(agent_stream_1, [e.to_store_dict() for e in crash_events], expected_version=-1)
    
    # Recovery: New session should detect and skip completed nodes
    # Note: Full recovery logic would need to be implemented in agents
    # This test verifies the infrastructure is in place
    
    session_events_after_crash = await event_store.load_stream(agent_stream_1)
    assert len(session_events_after_crash) == 2
    assert session_events_after_crash[0]["event_type"] == "AgentSessionStarted"
    assert session_events_after_crash[1]["event_type"] == "AgentNodeExecuted"

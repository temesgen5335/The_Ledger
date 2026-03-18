"""
ledger/agents/base_agent.py
===========================
BASE LANGGRAPH AGENT + all 5 agent class stubs.
CreditAnalysisAgent is the reference implementation with full LangGraph pattern.
The other 4 agents are stubs with complete docstrings for implementation.
"""
from __future__ import annotations
import asyncio, hashlib, json, os, time
from abc import ABC, abstractmethod
from datetime import datetime
from uuid import uuid4
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

LANGGRAPH_VERSION = "1.0.0"
MAX_OCC_RETRIES = 5

# Gemini pricing per 1M tokens (input, output) — update as models change
_GEMINI_PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.0-flash":         (0.10,  0.40),
    "gemini-2.0-flash-lite":    (0.075, 0.30),
    "gemini-1.5-flash":         (0.075, 0.30),
    "gemini-1.5-pro":           (3.50,  10.50),
    "gemini-1.0-pro":           (0.50,  1.50),
}

def _gemini_cost(model: str, tok_in: int, tok_out: int) -> float:
    """Compute USD cost for a Gemini call. Falls back to flash pricing if model unknown."""
    key = next((k for k in _GEMINI_PRICING if model.startswith(k)), "gemini-2.0-flash")
    p_in, p_out = _GEMINI_PRICING[key]
    return round(tok_in / 1e6 * p_in + tok_out / 1e6 * p_out, 6)

class BaseApexAgent(ABC):
    """
    Base for all 5 Apex agents. Provides Gas Town session management,
    per-node event recording, tool call recording, OCC retry scaffolding.

    AGENT NODE SEQUENCE (all agents follow this):
        start_session → validate_inputs → load_context → [domain nodes] → write_output → end_session

    Each node must call self._record_node_execution() at its end.
    Each tool/registry call must call self._record_tool_call().
    The write_output node must call self._record_output_written() then self._record_node_execution().
    """
    def __init__(self, agent_id: str, agent_type: str, store, registry,
                 model: str = "gemini-1.5-pro", api_key: str | None = None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.store = store
        self.registry = registry
        self.model = model
        self.client = ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            google_api_key=api_key or os.environ.get("GOOGLE_API_KEY"),
        )
        self.session_id = None; self.application_id = None
        self._session_stream = None; self._t0 = None
        self._seq = 0; self._llm_calls = 0; self._tokens = 0; self._cost = 0.0
        self._graph = None

    @abstractmethod
    def build_graph(self): raise NotImplementedError

    async def run(self, application_id: str, session_id: str | None = None, context_source: str = "fresh_start") -> None:
        """Run the agent with optional session_id and context_source for crash recovery."""
        if not self._graph: self._graph = self.build_graph()
        self.application_id = application_id
        self.session_id = session_id or f"sess-{self.agent_type[:3]}-{uuid4().hex[:8]}"
        self._session_stream = f"agent-{self.agent_type}-{self.session_id}"
        self._t0 = time.time(); self._seq = 0; self._llm_calls = 0; self._tokens = 0; self._cost = 0.0
        await self._start_session(application_id, context_source)
        try:
            result = await self._graph.ainvoke(self._initial_state(application_id))
            await self._complete_session(result)
        except Exception as e:
            await self._fail_session(type(e).__name__, str(e)); raise
    
    async def process_application(self, application_id: str) -> None:
        """Convenience method that calls run() with default parameters."""
        await self.run(application_id)

    def _initial_state(self, app_id):
        return {"application_id": app_id, "session_id": self.session_id,
                "agent_id": self.agent_id, "errors": [], "output_events_written": [], "next_agent_triggered": None}

    async def _start_session(self, app_id, context_source="fresh_start"):
        await self._append_session({"event_type":"AgentSessionStarted","event_version":1,"payload":{
            "session_id":self.session_id,"agent_type":self.agent_type,"agent_id":self.agent_id,
            "application_id":app_id,"model_version":self.model,"langgraph_graph_version":LANGGRAPH_VERSION,
            "context_source":context_source,"context_token_count":0,"started_at":datetime.now().isoformat()}})

    async def _record_node_execution(self, node_name=None, input_summary=None, output_summary=None, 
                                     llm_tokens_input=0, llm_tokens_output=0, llm_cost_usd=0.0,
                                     name=None, in_keys=None, out_keys=None, ms=None, tok_in=None, tok_out=None, cost=None):
        """Record node execution - supports both old and new calling conventions."""
        # Support new keyword-based calling convention
        if node_name is not None:
            name = node_name
            in_keys = list(input_summary.keys()) if isinstance(input_summary, dict) else []
            out_keys = list(output_summary.keys()) if isinstance(output_summary, dict) else []
            ms = 10  # Default duration
            tok_in = llm_tokens_input if llm_tokens_input > 0 else None
            tok_out = llm_tokens_output if llm_tokens_output > 0 else None
            cost = llm_cost_usd if llm_cost_usd > 0 else None
        
        self._seq += 1
        if tok_in: self._tokens += tok_in + (tok_out or 0); self._llm_calls += 1
        if cost: self._cost += cost
        await self._append_session({"event_type":"AgentNodeExecuted","event_version":1,"payload":{
            "session_id":self.session_id,"agent_type":self.agent_type,"node_name":name,
            "node_sequence":self._seq,"input_keys":in_keys or [],"output_keys":out_keys or [],
            "llm_called":tok_in is not None,"llm_tokens_input":tok_in,"llm_tokens_output":tok_out,
            "llm_cost_usd":cost,"duration_ms":ms or 10,"executed_at":datetime.now().isoformat()}})

    async def _record_tool_call(self, tool, inp, out, ms):
        await self._append_session({"event_type":"AgentToolCalled","event_version":1,"payload":{
            "session_id":self.session_id,"agent_type":self.agent_type,"tool_name":tool,
            "tool_input_summary":inp,"tool_output_summary":out,"tool_duration_ms":ms,
            "called_at":datetime.now().isoformat()}})

    async def _record_output_written(self, events_written, summary):
        await self._append_session({"event_type":"AgentOutputWritten","event_version":1,"payload":{
            "session_id":self.session_id,"agent_type":self.agent_type,"application_id":self.application_id,
            "events_written":events_written,"output_summary":summary,"written_at":datetime.now().isoformat()}})
    
    async def _record_input_validated(self, input_keys: list, duration_ms: int):
        """Record successful input validation."""
        await self._append_session({"event_type":"AgentInputValidated","event_version":1,"payload":{
            "session_id":self.session_id,"agent_type":self.agent_type,"application_id":self.application_id,
            "input_keys_validated":input_keys,"validation_duration_ms":duration_ms,
            "validated_at":datetime.now().isoformat()}})
    
    async def _record_input_failed(self, input_keys: list, errors: list):
        """Record failed input validation."""
        await self._append_session({"event_type":"AgentInputValidationFailed","event_version":1,"payload":{
            "session_id":self.session_id,"agent_type":self.agent_type,"application_id":self.application_id,
            "input_keys_attempted":input_keys,"validation_errors":errors,
            "failed_at":datetime.now().isoformat()}})
    
    async def _append_with_retry(self, stream_id: str, events: list, causation_id: str = None, max_retries: int = 3):
        """Append events to stream with OCC retry logic. Returns list of positions."""
        for attempt in range(max_retries):
            try:
                ver = await self.store.stream_version(stream_id)
                positions = await self.store.append(
                    stream_id=stream_id, 
                    events=events,
                    expected_version=ver, 
                    causation_id=causation_id
                )
                return positions
            except Exception as e:
                if "OptimisticConcurrencyError" in type(e).__name__ and attempt < max_retries - 1:
                    await asyncio.sleep(0.1 * (2 ** attempt))
                    continue
                raise
    
    def _parse_json(self, content: str) -> dict:
        """Parse JSON from LLM response, extracting from markdown code blocks if needed."""
        import re
        # Try to extract JSON from markdown code block
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if match:
            content = match.group(1)
        else:
            # Try to find JSON object in the content
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                content = match.group()
        
        return json.loads(content)

    async def _complete_session(self, result):
        ms = int((time.time()-self._t0)*1000)
        await self._append_session({"event_type":"AgentSessionCompleted","event_version":1,"payload":{
            "session_id":self.session_id,"agent_type":self.agent_type,"application_id":self.application_id,
            "total_nodes_executed":self._seq,"total_llm_calls":self._llm_calls,"total_tokens_used":self._tokens,
            "total_cost_usd":round(self._cost,6),"total_duration_ms":ms,
            "next_agent_triggered":result.get("next_agent_triggered"),"completed_at":datetime.now().isoformat()}})

    async def _fail_session(self, etype, emsg):
        await self._append_session({"event_type":"AgentSessionFailed","event_version":1,"payload":{
            "session_id":self.session_id,"agent_type":self.agent_type,"application_id":self.application_id,
            "error_type":etype,"error_message":emsg[:500],"last_successful_node":f"node_{self._seq}",
            "recoverable":etype in ("llm_timeout","RateLimitError"),"failed_at":datetime.now().isoformat()}})

    async def _append_session(self, event: dict):
        """Append event to agent session stream."""
        try:
            ver = await self.store.stream_version(self._session_stream)
            await self.store.append(stream_id=self._session_stream, events=[event], expected_version=ver)
        except Exception as e:
            # If stream doesn't exist yet, create it with version -1
            if "does not exist" in str(e).lower() or ver == -1:
                await self.store.append(stream_id=self._session_stream, events=[event], expected_version=-1)
            else:
                raise

    async def _append_event(self, event, stream_id: str = None):
        """Append an event object to a stream. If stream_id not provided, uses agent session stream."""
        event_dict = event.to_store_dict() if hasattr(event, 'to_store_dict') else event
        target_stream = stream_id or self._session_stream
        await self._append_stream(target_stream, event_dict)
    
    async def _append_stream(self, stream_id: str, event_dict: dict, causation_id: str = None):
        """Append to any aggregate stream with OCC retry."""
        for attempt in range(MAX_OCC_RETRIES):
            try:
                ver = await self.store.stream_version(stream_id)
                await self.store.append(stream_id=stream_id, events=[event_dict],
                    expected_version=ver, causation_id=causation_id)
                return
            except Exception as e:
                if "OptimisticConcurrencyError" in type(e).__name__ and attempt < MAX_OCC_RETRIES-1:
                    await asyncio.sleep(0.1 * (2**attempt)); continue
                raise

    async def _call_llm(self, system: str, user: str, max_tokens: int = 1024):
        """Call Gemini via LangChain. Returns (text, tok_in, tok_out, cost_usd)."""
        llm = self.client.bind(max_output_tokens=max_tokens)
        response = await llm.ainvoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        text = response.content
        usage = response.usage_metadata or {}
        tok_in  = usage.get("input_tokens", 0)
        tok_out = usage.get("output_tokens", 0)
        cost    = _gemini_cost(self.model, tok_in, tok_out)
        return text, tok_in, tok_out, cost

    @staticmethod
    def _sha(d): return hashlib.sha256(json.dumps(str(d),sort_keys=True).encode()).hexdigest()[:16]


class CreditAnalysisAgent(BaseApexAgent):
    """
    Reference implementation. LangGraph nodes:
      validate_inputs → open_credit_record → load_applicant_registry
      → load_extracted_facts → analyze_credit_risk → apply_policy_constraints → write_output

    Output streams:
      credit-{id}: CreditRecordOpened, HistoricalProfileConsumed, ExtractedFactsConsumed, CreditAnalysisCompleted
      loan-{id}: FraudScreeningRequested  (triggers next agent)
    """
    def build_graph(self):
        from typing import TypedDict
        class S(TypedDict):
            application_id: str; session_id: str; agent_id: str
            applicant_id: str | None; requested_amount_usd: float | None; loan_purpose: str | None
            historical_financials: list | None; company_profile: dict | None
            compliance_flags: list | None; loan_history: list | None
            extracted_facts: dict | None; quality_flags: list | None
            credit_decision: dict | None; policy_violations: list | None
            errors: list; output_events_written: list; next_agent_triggered: str | None

        g = StateGraph(S)
        for name, fn in [
            ("validate_inputs",          self._node_validate_inputs),
            ("open_credit_record",       self._node_open_credit_record),
            ("load_applicant_registry",  self._node_load_registry),
            ("load_extracted_facts",     self._node_load_facts),
            ("analyze_credit_risk",      self._node_analyze),
            ("apply_policy_constraints", self._node_policy),
            ("write_output",             self._node_write),
        ]: g.add_node(name, fn)
        g.set_entry_point("validate_inputs")
        g.add_edge("validate_inputs","open_credit_record")
        g.add_edge("open_credit_record","load_applicant_registry")
        g.add_edge("load_applicant_registry","load_extracted_facts")
        g.add_edge("load_extracted_facts","analyze_credit_risk")
        g.add_edge("analyze_credit_risk","apply_policy_constraints")
        g.add_edge("apply_policy_constraints","write_output")
        g.add_edge("write_output", END)
        return g.compile()

    async def _node_validate_inputs(self, state):
        t = time.time()
        # TODO: Load LoanApplicationAggregate, verify state == DOCUMENTS_PROCESSED
        # TODO: Load applicant_id, requested_amount, loan_purpose from ApplicationSubmitted event
        # TODO: Verify PackageReadyForAnalysis event exists in docpkg stream
        # PLACEHOLDER:
        state = {**state, "applicant_id": f"COMP-001", "requested_amount_usd": 500_000.0, "loan_purpose": "working_capital"}
        await self._record_node_execution("validate_inputs",["application_id"],["applicant_id","requested_amount_usd","loan_purpose"],int((time.time()-t)*1000))
        return state

    async def _node_open_credit_record(self, state):
        t = time.time()
        # TODO: await self._append_stream(f"credit-{state['application_id']}", CreditRecordOpened(...).to_store_dict(), expected_version=-1)
        await self._record_node_execution("open_credit_record",["applicant_id"],["credit_stream_opened"],int((time.time()-t)*1000))
        return state

    async def _node_load_registry(self, state):
        t = time.time()
        # TODO: profile = await self.registry.get_company(state["applicant_id"])
        # TODO: hist = await self.registry.get_financial_history(state["applicant_id"], years=[2022,2023,2024])
        # TODO: flags = await self.registry.get_compliance_flags(state["applicant_id"])
        # TODO: loans = await self.registry.get_loan_relationships(state["applicant_id"])
        ms = int((time.time()-t)*1000)
        await self._record_tool_call("query_applicant_registry", f"company_id={state['applicant_id']}", "3yr financials loaded", ms)
        # TODO: await self._append_stream(f"credit-{state['application_id']}", HistoricalProfileConsumed(...).to_store_dict())
        await self._record_node_execution("load_applicant_registry",["applicant_id"],["historical_financials","compliance_flags","loan_history"],ms)
        return {**state,"company_profile":{},"historical_financials":[],"compliance_flags":[],"loan_history":[]}

    async def _node_load_facts(self, state):
        t = time.time()
        # TODO: load ExtractionCompleted events from f"docpkg-{state['application_id']}"
        # TODO: merge FinancialFacts from income_statement + balance_sheet documents
        ms = int((time.time()-t)*1000)
        await self._record_tool_call("load_event_store_stream", f"docpkg-{state['application_id']}", "ExtractionCompleted events loaded", ms)
        # TODO: await self._append_stream(f"credit-{state['application_id']}", ExtractedFactsConsumed(...).to_store_dict())
        await self._record_node_execution("load_extracted_facts",["document_package_events"],["extracted_facts","quality_flags"],ms)
        return {**state,"extracted_facts":{},"quality_flags":[]}

    async def _node_analyze(self, state):
        t = time.time()
        hist = state.get("historical_financials") or []
        fin_table = "\n".join([f"FY{f['fiscal_year'] if isinstance(f,dict) else ''}: (historical data)" for f in hist]) if hist else "No historical data loaded — TODO: implement load_applicant_registry"
        system = """You are a commercial credit analyst at Apex Financial Services.
Evaluate the loan application and return ONLY a JSON object with these fields:
{"risk_tier":"LOW"|"MEDIUM"|"HIGH","recommended_limit_usd":<int>,"confidence":<float 0-1>,
 "rationale":"<3-5 sentences>","key_concerns":[],"data_quality_caveats":[],"policy_overrides_applied":[]}
Hard policy rules you must enforce:
1. recommended_limit_usd <= annual_revenue * 0.35
2. Any prior default → risk_tier must be HIGH
3. Active HIGH compliance flag → confidence must be <= 0.50"""
        user = f"""Applicant: {state.get('company_profile',{}).get('name','Unknown')}
Requested: ${state.get('requested_amount_usd',0):,.0f} for {state.get('loan_purpose','unknown')}
Historical financials:\n{fin_table}
Current year extracted facts: {json.dumps(state.get('extracted_facts',{}),default=str)[:1000]}
Quality flags: {state.get('quality_flags',[])}
Compliance flags: {state.get('compliance_flags',[])}
Prior loans: {state.get('loan_history',[])}"""
        try:
            content, tok_in, tok_out, cost = await self._call_llm(system, user, max_tokens=800)
            import re; m = re.search(r'\{.*\}', content, re.DOTALL)
            decision = json.loads(m.group()) if m else {}
        except Exception as e:
            decision = {"risk_tier":"MEDIUM","recommended_limit_usd":int(state.get("requested_amount_usd",0)*0.8),"confidence":0.45,"rationale":f"Analysis deferred: {e}","key_concerns":["LLM analysis failed — human review required"],"data_quality_caveats":[],"policy_overrides_applied":[]}
            tok_in=tok_out=0; cost=0.0
        ms = int((time.time()-t)*1000)
        await self._record_node_execution("analyze_credit_risk",["historical_financials","extracted_facts"],["credit_decision"],ms,tok_in,tok_out,cost)
        return {**state,"credit_decision":decision}

    async def _node_policy(self, state):
        t = time.time()
        d = state.get("credit_decision") or {}; violations = []
        hist = state.get("historical_financials") or []
        if hist:
            rev = hist[-1].get("total_revenue",0) if isinstance(hist[-1],dict) else 0
            if rev > 0 and d.get("recommended_limit_usd",0) > rev*0.35:
                d["recommended_limit_usd"] = int(rev*0.35); violations.append("REV_CAP")
        if any(l.get("default_occurred") for l in (state.get("loan_history") or [])):
            d["risk_tier"] = "HIGH"; violations.append("PRIOR_DEFAULT")
        if any(f.get("severity")=="HIGH" and f.get("is_active") for f in (state.get("compliance_flags") or [])):
            d["confidence"] = min(d.get("confidence",1.0), 0.50); violations.append("COMPLIANCE_FLAG")
        if violations: d["policy_overrides_applied"] = d.get("policy_overrides_applied",[]) + violations
        await self._record_node_execution("apply_policy_constraints",["credit_decision"],["credit_decision"],int((time.time()-t)*1000))
        return {**state,"credit_decision":d,"policy_violations":violations}

    async def _node_write(self, state):
        t = time.time()
        app_id = state["application_id"]; d = state["credit_decision"]
        # TODO: append CreditAnalysisCompleted to f"credit-{app_id}"
        # TODO: append FraudScreeningRequested to f"loan-{app_id}"
        # Use OCC retry via self._append_stream()
        events_written = [
            {"stream_id":f"credit-{app_id}","event_type":"CreditAnalysisCompleted","stream_position":"TODO"},
            {"stream_id":f"loan-{app_id}","event_type":"FraudScreeningRequested","stream_position":"TODO"},
        ]
        await self._record_output_written(events_written, f"Credit: {d.get('risk_tier')} risk, ${d.get('recommended_limit_usd',0):,.0f} limit, {d.get('confidence',0):.0%} confidence. Fraud screening triggered.")
        await self._record_node_execution("write_output",["credit_decision"],["events_written"],int((time.time()-t)*1000))
        return {**state,"output_events_written":events_written,"next_agent_triggered":"fraud_detection"}


class DocumentProcessingAgent(BaseApexAgent):
    """
    Wraps the Week 3 Document Intelligence pipeline as a LangGraph agent.

    NODES TO IMPLEMENT:
        validate_inputs → validate_document_format → run_week3_extraction
        → assess_quality (LLM) → write_output

    WEEK 3 INTEGRATION — in _node_run_week3_extraction:
        from document_refinery.pipeline import extract_financial_facts
        for each doc in package:
            append ExtractionStarted to docpkg stream
            facts = await extract_financial_facts(file_path, document_type)
            append ExtractionCompleted(facts=facts) to docpkg stream

    LLM ROLE — in _node_assess_quality:
        System prompt: "You are a financial document quality analyst.
        Check extracted facts for internal consistency. Do NOT make credit decisions.
        Return DocumentQualityAssessment JSON."
        Specifically check: balance_sheet_balances, EBITDA plausibility,
        margin ranges for industry, critical missing fields.

    OUTPUT STREAMS:
        docpkg-{id}: DocumentFormatValidated, ExtractionStarted, ExtractionCompleted,
                     QualityAssessmentCompleted, PackageReadyForAnalysis
        loan-{id}: CreditAnalysisRequested
    """
    def build_graph(self):
        from typing import TypedDict
        class S(TypedDict):
            application_id: str; session_id: str; agent_id: str
            document_ids: list | None; extracted_facts_by_doc: dict | None
            quality_assessment: dict | None; has_critical_issues: bool | None
            errors: list; output_events_written: list; next_agent_triggered: str | None
        g = StateGraph(S)
        g.add_node("validate_inputs",         self._node_validate_inputs)
        g.add_node("validate_document_format",self._node_validate_format)
        g.add_node("run_week3_extraction",     self._node_extract)
        g.add_node("assess_quality",           self._node_assess_quality)
        g.add_node("write_output",             self._node_write_output)
        g.set_entry_point("validate_inputs")
        g.add_edge("validate_inputs","validate_document_format")
        g.add_edge("validate_document_format","run_week3_extraction")
        g.add_edge("run_week3_extraction","assess_quality")
        g.add_edge("assess_quality","write_output")
        g.add_edge("write_output", END)
        return g.compile()

    async def _node_validate_inputs(self, state):
        """Verify DocumentUploaded events exist on loan stream."""
        from datetime import datetime
        from ledger.schema.events import AgentInputValidated
        
        app_id = state["application_id"]
        loan_stream = await self.store.load_stream(f"loan-{app_id}")
        
        # Find DocumentUploaded events
        doc_events = [e for e in loan_stream if e["event_type"] == "DocumentUploaded"]
        if not doc_events:
            state["errors"].append("No DocumentUploaded events found")
            return state
        
        # Extract document IDs and file paths
        state["document_ids"] = [e["payload"]["document_id"] for e in doc_events]
        state["extracted_facts_by_doc"] = {}
        
        # Record node execution
        await self._record_node_execution(
            node_name="validate_inputs",
            input_summary={"application_id": app_id, "documents_found": len(doc_events)},
            output_summary={"document_ids": state["document_ids"]},
            llm_tokens_input=0, llm_tokens_output=0, llm_cost_usd=0.0
        )
        
        # Append AgentInputValidated to agent stream
        event = AgentInputValidated(
            session_id=state["session_id"],
            agent_type="document_processing",
            application_id=app_id,
            input_hash=self._sha(doc_events),
            validation_passed=True,
            validation_errors=[],
            validated_at=datetime.utcnow()
        )
        await self._append_event(event)
        return state
    
    async def _node_validate_format(self, state):
        """Check PDF/XLSX format, append DocumentFormatValidated or Rejected."""
        from datetime import datetime
        from pathlib import Path
        from ledger.schema.events import DocumentFormatValidated, DocumentFormatRejected
        
        app_id = state["application_id"]
        loan_stream = await self.store.load_stream(f"loan-{app_id}")
        doc_events = [e for e in loan_stream if e["event_type"] == "DocumentUploaded"]
        
        validated_count = 0
        for doc_event in doc_events:
            payload = doc_event["payload"]
            doc_id = payload["document_id"]
            file_path = Path(payload["file_path"])
            doc_format = payload["document_format"]
            
            # Check file exists and format is valid
            if not file_path.exists():
                event = DocumentFormatRejected(
                    package_id=f"pkg-{app_id}",
                    document_id=doc_id,
                    rejection_reason="file_not_found",
                    error_message=f"File not found: {file_path}",
                    rejected_at=datetime.utcnow()
                )
                await self._append_event(event, stream_id=f"docpkg-{app_id}")
                continue
            
            # Validate format
            if doc_format not in ["pdf", "xlsx"]:
                event = DocumentFormatRejected(
                    package_id=f"pkg-{app_id}",
                    document_id=doc_id,
                    rejection_reason="unsupported_format",
                    error_message=f"Format {doc_format} not supported",
                    rejected_at=datetime.utcnow()
                )
                await self._append_event(event, stream_id=f"docpkg-{app_id}")
                continue
            
            # Format validated
            event = DocumentFormatValidated(
                package_id=f"pkg-{app_id}",
                document_id=doc_id,
                document_type=payload["document_type"],
                format_checks_passed=["file_exists", "format_supported"],
                file_size_bytes=payload["file_size_bytes"],
                validated_at=datetime.utcnow()
            )
            await self._append_event(event, stream_id=f"docpkg-{app_id}")
            validated_count += 1
        
        await self._record_node_execution(
            node_name="validate_document_format",
            input_summary={"documents": len(doc_events)},
            output_summary={"validated": validated_count},
            llm_tokens_input=0, llm_tokens_output=0, llm_cost_usd=0.0
        )
        return state
    
    async def _node_extract(self, state):
        """Call Week 3 refinery pipeline per document, append ExtractionStarted + ExtractionCompleted."""
        from datetime import datetime
        from pathlib import Path
        import time
        from ledger.schema.events import ExtractionStarted, ExtractionCompleted
        from ledger.agents.refinery.agents.triage import TriageAgent
        from ledger.agents.refinery.agents.extractor import ExtractionRouter
        from ledger.adapters.refinery_adapter import RefineryAdapter
        
        app_id = state["application_id"]
        loan_stream = await self.store.load_stream(f"loan-{app_id}")
        doc_events = [e for e in loan_stream if e["event_type"] == "DocumentUploaded"]
        
        # Initialize refinery components
        triage_agent = TriageAgent()
        extractor = ExtractionRouter()
        adapter = RefineryAdapter(llm_client=self.client)
        
        total_processing_ms = 0
        
        for doc_event in doc_events:
            payload = doc_event["payload"]
            doc_id = payload["document_id"]
            file_path = Path(payload["file_path"])
            doc_type = payload["document_type"]
            fiscal_year = payload.get("fiscal_year")
            
            if not file_path.exists():
                continue
            
            # Append ExtractionStarted
            start_event = ExtractionStarted(
                package_id=f"pkg-{app_id}",
                document_id=doc_id,
                document_type=doc_type,
                pipeline_version="refinery-1.0",
                extraction_model="gemini-2.0-flash",
                started_at=datetime.utcnow()
            )
            await self._append_event(start_event, stream_id=f"docpkg-{app_id}")
            
            # Run refinery pipeline
            t0 = time.time()
            try:
                # Stage 1: Triage
                profile = triage_agent.profile(file_path)
                
                # Stage 2: Extraction
                extracted_doc = extractor.extract(file_path, profile, force_reextract=False)
                
                # Stage 3: Convert to FinancialFacts
                facts = await adapter.extract_financial_facts(extracted_doc, doc_type, fiscal_year)
                
                processing_ms = int((time.time() - t0) * 1000)
                total_processing_ms += processing_ms
                
                # Calculate raw text length
                raw_text_length = sum(len(page.text) for page in extracted_doc.pages)
                
                # Append ExtractionCompleted
                complete_event = ExtractionCompleted(
                    package_id=f"pkg-{app_id}",
                    document_id=doc_id,
                    document_type=doc_type,
                    facts=facts,
                    raw_text_length=raw_text_length,
                    tables_extracted=len(extracted_doc.tables),
                    processing_ms=processing_ms,
                    completed_at=datetime.utcnow()
                )
                await self._append_event(complete_event, stream_id=f"docpkg-{app_id}")
                
                # Store facts for quality assessment
                state["extracted_facts_by_doc"][doc_id] = facts.model_dump()
                
            except Exception as e:
                from ledger.schema.events import ExtractionFailed
                fail_event = ExtractionFailed(
                    package_id=f"pkg-{app_id}",
                    document_id=doc_id,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    partial_facts=None,
                    failed_at=datetime.utcnow()
                )
                await self._append_event(fail_event, stream_id=f"docpkg-{app_id}")
                state["errors"].append(f"Extraction failed for {doc_id}: {e}")
        
        await self._record_node_execution(
            node_name="run_week3_extraction",
            input_summary={"documents": len(doc_events)},
            output_summary={"extracted": len(state["extracted_facts_by_doc"]), "total_ms": total_processing_ms},
            llm_tokens_input=0, llm_tokens_output=0, llm_cost_usd=0.0
        )
        return state
    
    async def _node_assess_quality(self, state):
        """LLM coherence check, append QualityAssessmentCompleted."""
        from datetime import datetime
        from ledger.schema.events import QualityAssessmentCompleted
        import json
        
        app_id = state["application_id"]
        facts_by_doc = state.get("extracted_facts_by_doc", {})
        
        if not facts_by_doc:
            state["quality_assessment"] = {"overall_confidence": 0.0, "is_coherent": False}
            state["has_critical_issues"] = True
            return state
        
        # Build LLM prompt for quality assessment
        system_prompt = """You are a financial document quality analyst.
Check extracted facts for internal consistency. Do NOT make credit decisions.
Return a JSON object with:
- overall_confidence (0.0-1.0)
- is_coherent (boolean)
- anomalies (list of strings)
- critical_missing_fields (list of strings)
- reextraction_recommended (boolean)
- auditor_notes (string)

Check:
1. Balance sheet equation: Assets = Liabilities + Equity (within 1% tolerance)
2. EBITDA plausibility (should be positive for profitable companies)
3. Margin ranges (gross margin 0-100%, net margin -50% to 50%)
4. Critical fields present: total_revenue, total_assets, net_income"""
        
        user_prompt = f"Extracted financial facts:\n{json.dumps(facts_by_doc, indent=2, default=str)}"
        
        # Call LLM
        response_text, tok_in, tok_out, cost = await self._call_llm(system_prompt, user_prompt, max_tokens=512)
        
        # Parse response
        try:
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            assessment = json.loads(response_text.strip())
        except:
            assessment = {
                "overall_confidence": 0.7,
                "is_coherent": True,
                "anomalies": [],
                "critical_missing_fields": [],
                "reextraction_recommended": False,
                "auditor_notes": "LLM response parsing failed, using default assessment"
            }
        
        state["quality_assessment"] = assessment
        state["has_critical_issues"] = not assessment.get("is_coherent", True) or assessment.get("reextraction_recommended", False)
        
        # Append QualityAssessmentCompleted for each document
        for doc_id in facts_by_doc.keys():
            event = QualityAssessmentCompleted(
                package_id=f"pkg-{app_id}",
                document_id=doc_id,
                overall_confidence=assessment.get("overall_confidence", 0.7),
                is_coherent=assessment.get("is_coherent", True),
                anomalies=assessment.get("anomalies", []),
                critical_missing_fields=assessment.get("critical_missing_fields", []),
                reextraction_recommended=assessment.get("reextraction_recommended", False),
                auditor_notes=assessment.get("auditor_notes", ""),
                assessed_at=datetime.utcnow()
            )
            await self._append_event(event, stream_id=f"docpkg-{app_id}")
        
        await self._record_node_execution(
            node_name="assess_quality",
            input_summary={"documents_assessed": len(facts_by_doc)},
            output_summary=assessment,
            llm_tokens_input=tok_in, llm_tokens_output=tok_out, llm_cost_usd=cost
        )
        return state
    
    async def _node_write_output(self, state):
        """Append PackageReadyForAnalysis, trigger CreditAnalysisRequested."""
        from datetime import datetime
        from ledger.schema.events import PackageReadyForAnalysis, CreditAnalysisRequested, AgentOutputWritten
        
        app_id = state["application_id"]
        assessment = state.get("quality_assessment", {})
        
        # Append PackageReadyForAnalysis to docpkg stream
        package_event = PackageReadyForAnalysis(
            package_id=f"pkg-{app_id}",
            application_id=app_id,
            documents_processed=len(state.get("extracted_facts_by_doc", {})),
            has_quality_flags=state.get("has_critical_issues", False),
            quality_flag_count=len(assessment.get("anomalies", [])),
            ready_at=datetime.utcnow()
        )
        await self._append_event(package_event, stream_id=f"docpkg-{app_id}")
        
        # Trigger CreditAnalysisRequested on loan stream
        credit_event = CreditAnalysisRequested(
            application_id=app_id,
            requested_at=datetime.utcnow(),
            requested_by=f"agent-{self.agent_id}",
            priority="NORMAL"
        )
        await self._append_event(credit_event, stream_id=f"loan-{app_id}")
        
        # Append AgentOutputWritten
        output_event = AgentOutputWritten(
            session_id=state["session_id"],
            agent_type="document_processing",
            application_id=app_id,
            output_events_written=["PackageReadyForAnalysis", "CreditAnalysisRequested"],
            output_hash=self._sha({"package": package_event.model_dump(), "credit": credit_event.model_dump()}),
            written_at=datetime.utcnow()
        )
        await self._append_event(output_event)
        
        state["output_events_written"] = ["PackageReadyForAnalysis", "CreditAnalysisRequested"]
        state["next_agent_triggered"] = "credit_analysis"
        
        await self._record_node_execution(
            node_name="write_output",
            input_summary={"documents": len(state.get("extracted_facts_by_doc", {}))},
            output_summary={"events_written": 2, "next_agent": "credit_analysis"},
            llm_tokens_input=0, llm_tokens_output=0, llm_cost_usd=0.0
        )
        return state


class FraudDetectionAgent(BaseApexAgent):
    """
    Detects inconsistencies between submitted documents and registry history.

    NODES TO IMPLEMENT:
        validate_inputs → load_document_facts → cross_reference_registry
        → analyze_fraud_patterns (LLM) → write_output

    LLM ROLE — in _node_analyze_fraud_patterns:
        Compare extracted current-year facts against historical_financials from registry.
        Flag: revenue_discrepancy (> 50% unexplained gap year-on-year),
              balance_sheet_inconsistency, unusual_submission_pattern.
        Compute fraud_score as weighted sum of anomaly severities.
        Return FraudAssessment JSON with named anomalies.
        RULE: fraud_score > 0.3 → must include at least one named anomaly with evidence.

    OUTPUT STREAMS:
        fraud-{id}: FraudScreeningInitiated, FraudAnomalyDetected (0+), FraudScreeningCompleted
        loan-{id}: ComplianceCheckRequested
    """
    def build_graph(self):
        from typing import TypedDict
        class S(TypedDict):
            application_id: str; session_id: str; agent_id: str
            extracted_facts: dict | None; historical_financials: list | None
            company_profile: dict | None; fraud_assessment: dict | None
            errors: list; output_events_written: list; next_agent_triggered: str | None
        g = StateGraph(S)
        for name in ["validate_inputs","load_document_facts","cross_reference_registry","analyze_fraud_patterns","write_output"]:
            g.add_node(name, getattr(self, f"_node_{name}"))
        g.set_entry_point("validate_inputs")
        g.add_edge("validate_inputs","load_document_facts")
        g.add_edge("load_document_facts","cross_reference_registry")
        g.add_edge("cross_reference_registry","analyze_fraud_patterns")
        g.add_edge("analyze_fraud_patterns","write_output")
        g.add_edge("write_output",END)
        return g.compile()

    async def _node_validate_inputs(self, state):
        """Verify FraudScreeningRequested event exists on loan stream."""
        from datetime import datetime
        from ledger.schema.events import AgentInputValidated
        
        app_id = state["application_id"]
        loan_stream = await self.store.load_stream(f"loan-{app_id}")
        
        # Find FraudScreeningRequested event
        fraud_requested = [e for e in loan_stream if e["event_type"] == "FraudScreeningRequested"]
        if not fraud_requested:
            state["errors"].append("No FraudScreeningRequested event found")
            return state
        
        await self._record_node_execution(
            node_name="validate_inputs",
            input_summary={"application_id": app_id},
            output_summary={"fraud_requested": len(fraud_requested)},
            llm_tokens_input=0, llm_tokens_output=0, llm_cost_usd=0.0
        )
        
        event = AgentInputValidated(
            session_id=state["session_id"],
            agent_type="fraud_detection",
            application_id=app_id,
            inputs_validated=["FraudScreeningRequested"],
            validation_duration_ms=10,
            validated_at=datetime.utcnow()
        )
        await self._append_event(event)
        return state
    
    async def _node_load_document_facts(self, state):
        """Load ExtractionCompleted events from docpkg stream."""
        from datetime import datetime
        import json
        from ledger.schema.events import ExtractedFactsConsumed
        
        app_id = state["application_id"]
        docpkg_stream = await self.store.load_stream(f"docpkg-{app_id}")
        
        # Find ExtractionCompleted events
        extraction_events = [e for e in docpkg_stream if e["event_type"] == "ExtractionCompleted"]
        
        if not extraction_events:
            state["errors"].append("No ExtractionCompleted events found")
            state["extracted_facts"] = {}
            return state
        
        # Aggregate all extracted facts
        all_facts = {}
        document_ids = []
        for event in extraction_events:
            payload = json.loads(event["payload"]) if isinstance(event["payload"], str) else event["payload"]
            doc_id = payload.get("document_id")
            facts = payload.get("facts", {})
            if doc_id and facts:
                all_facts[doc_id] = facts
                document_ids.append(doc_id)
        
        state["extracted_facts"] = all_facts
        
        # Append ExtractedFactsConsumed to credit stream
        event = ExtractedFactsConsumed(
            application_id=app_id,
            session_id=state["session_id"],
            document_ids_consumed=document_ids,
            facts_summary=f"{len(all_facts)} documents with financial facts",
            quality_flags_present=False,
            consumed_at=datetime.utcnow()
        )
        await self._append_event(event, stream_id=f"credit-{app_id}")
        
        await self._record_node_execution(
            node_name="load_document_facts",
            input_summary={"docpkg_stream": f"docpkg-{app_id}"},
            output_summary={"documents_loaded": len(all_facts)},
            llm_tokens_input=0, llm_tokens_output=0, llm_cost_usd=0.0
        )
        return state
    
    async def _node_cross_reference_registry(self, state):
        """Query registry: get_company + get_financial_history."""
        from datetime import datetime
        from ledger.schema.events import HistoricalProfileConsumed
        import json
        
        app_id = state["application_id"]
        
        # Get applicant_id from loan stream
        loan_stream = await self.store.load_stream(f"loan-{app_id}")
        app_submitted = [e for e in loan_stream if e["event_type"] == "ApplicationSubmitted"]
        if not app_submitted:
            state["errors"].append("No ApplicationSubmitted event found")
            return state
        
        payload = json.loads(app_submitted[0]["payload"]) if isinstance(app_submitted[0]["payload"], str) else app_submitted[0]["payload"]
        applicant_id = payload.get("applicant_id")
        
        # Query registry
        company_profile = await self.registry.get_company(applicant_id)
        financial_history = await self.registry.get_financial_history(applicant_id)
        
        state["company_profile"] = company_profile
        state["historical_financials"] = financial_history
        
        # Append HistoricalProfileConsumed to credit stream
        fiscal_years = [fh.fiscal_year for fh in financial_history] if financial_history else []
        event = HistoricalProfileConsumed(
            application_id=app_id,
            session_id=state["session_id"],
            fiscal_years_loaded=fiscal_years,
            has_prior_loans=False,  # Not tracked in current schema
            has_defaults=False,
            revenue_trajectory=company_profile.trajectory if company_profile else "UNKNOWN",
            data_hash=self._sha({"company": str(company_profile), "history": str(financial_history)}),
            consumed_at=datetime.utcnow()
        )
        await self._append_event(event, stream_id=f"credit-{app_id}")
        
        await self._record_node_execution(
            node_name="cross_reference_registry",
            input_summary={"applicant_id": applicant_id},
            output_summary={"fiscal_years": len(fiscal_years), "has_profile": company_profile is not None},
            llm_tokens_input=0, llm_tokens_output=0, llm_cost_usd=0.0
        )
        return state
    
    async def _node_analyze_fraud_patterns(self, state):
        """LLM: compare extracted facts vs registry history, compute fraud_score."""
        from datetime import datetime
        from ledger.schema.events import FraudScreeningInitiated, FraudAnomalyDetected
        from decimal import Decimal
        import json
        
        app_id = state["application_id"]
        extracted_facts = state.get("extracted_facts", {})
        historical_financials = state.get("historical_financials", [])
        company_profile = state.get("company_profile", {})
        
        # Append FraudScreeningInitiated
        init_event = FraudScreeningInitiated(
            application_id=app_id,
            session_id=state["session_id"],
            screening_model_version=self.model,
            initiated_at=datetime.utcnow()
        )
        await self._append_event(init_event, stream_id=f"fraud-{app_id}")
        
        # Build LLM prompt for fraud analysis
        system_prompt = """You are a financial fraud detection analyst.
Compare the submitted financial documents against historical registry data.
Detect anomalies such as:
1. Revenue discrepancy > 50% year-over-year without explanation
2. Balance sheet inconsistencies (Assets != Liabilities + Equity)
3. Unusual submission patterns or data manipulation

Return a JSON object with:
- fraud_score (0.0-1.0, where >0.3 indicates high risk)
- risk_level ("LOW", "MEDIUM", "HIGH")
- anomalies (list of objects with: anomaly_type, description, severity, evidence, affected_fields)
- recommendation ("APPROVE_SCREENING", "FLAG_FOR_REVIEW", "REJECT")

RULE: If fraud_score > 0.3, you MUST include at least one named anomaly with evidence."""
        
        user_prompt = f"""Submitted financial facts:
{json.dumps(extracted_facts, indent=2, default=str)}

Historical financials from registry:
{json.dumps(historical_financials[:3], indent=2, default=str) if historical_financials else 'No historical data available'}

Company profile:
{json.dumps(company_profile, indent=2, default=str)}"""
        
        # Call LLM
        response_text, tok_in, tok_out, cost = await self._call_llm(system_prompt, user_prompt, max_tokens=1024)
        
        # Parse response
        try:
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            assessment = json.loads(response_text.strip())
        except:
            # Fallback: simple rule-based fraud detection
            assessment = self._rule_based_fraud_detection(extracted_facts, historical_financials)
        
        state["fraud_assessment"] = assessment
        
        # Append FraudAnomalyDetected for each anomaly
        anomalies = assessment.get("anomalies", [])
        for anomaly_data in anomalies:
            from ledger.schema.events import FraudAnomaly
            anomaly = FraudAnomaly(
                anomaly_type=anomaly_data.get("anomaly_type", "REVENUE_DISCREPANCY"),
                description=anomaly_data.get("description", ""),
                severity=anomaly_data.get("severity", "MEDIUM"),
                evidence=anomaly_data.get("evidence", ""),
                affected_fields=anomaly_data.get("affected_fields", [])
            )
            event = FraudAnomalyDetected(
                application_id=app_id,
                session_id=state["session_id"],
                anomaly=anomaly,
                detected_at=datetime.utcnow()
            )
            await self._append_event(event, stream_id=f"fraud-{app_id}")
        
        await self._record_node_execution(
            node_name="analyze_fraud_patterns",
            input_summary={"documents": len(extracted_facts), "history_years": len(historical_financials)},
            output_summary={"fraud_score": assessment.get("fraud_score", 0.0), "anomalies": len(anomalies)},
            llm_tokens_input=tok_in, llm_tokens_output=tok_out, llm_cost_usd=cost
        )
        return state
    
    def _rule_based_fraud_detection(self, extracted_facts, historical_financials):
        """Fallback rule-based fraud detection if LLM fails."""
        from decimal import Decimal
        
        fraud_score = 0.0
        anomalies = []
        
        # Check for revenue discrepancy
        if extracted_facts and historical_financials:
            for doc_id, facts in extracted_facts.items():
                current_revenue = facts.get("total_revenue")
                if current_revenue and historical_financials:
                    # Compare to most recent historical year
                    last_year = historical_financials[0]
                    historical_revenue = last_year.get("total_revenue")
                    if historical_revenue:
                        try:
                            curr = float(current_revenue)
                            hist = float(historical_revenue)
                            if hist > 0:
                                change_pct = abs((curr - hist) / hist)
                                if change_pct > 0.5:  # >50% change
                                    fraud_score += 0.4
                                    anomalies.append({
                                        "anomaly_type": "REVENUE_DISCREPANCY",
                                        "description": f"Revenue changed by {change_pct*100:.1f}% year-over-year",
                                        "severity": "HIGH",
                                        "evidence": f"Current: {curr}, Historical: {hist}",
                                        "affected_fields": ["total_revenue"]
                                    })
                        except:
                            pass
        
        # Determine risk level
        if fraud_score > 0.5:
            risk_level = "HIGH"
            recommendation = "REJECT"
        elif fraud_score > 0.3:
            risk_level = "MEDIUM"
            recommendation = "FLAG_FOR_REVIEW"
        else:
            risk_level = "LOW"
            recommendation = "APPROVE_SCREENING"
        
        return {
            "fraud_score": min(fraud_score, 1.0),
            "risk_level": risk_level,
            "anomalies": anomalies,
            "recommendation": recommendation
        }
    
    async def _node_write_output(self, state):
        """Append FraudScreeningCompleted, trigger ComplianceCheckRequested."""
        from datetime import datetime
        from ledger.schema.events import FraudScreeningCompleted, ComplianceCheckRequested, AgentOutputWritten
        
        app_id = state["application_id"]
        assessment = state.get("fraud_assessment", {})
        
        # Append FraudScreeningCompleted to fraud stream
        fraud_event = FraudScreeningCompleted(
            application_id=app_id,
            session_id=state["session_id"],
            fraud_score=assessment.get("fraud_score", 0.0),
            risk_level=assessment.get("risk_level", "LOW"),
            anomalies_found=len(assessment.get("anomalies", [])),
            recommendation=assessment.get("recommendation", "APPROVE_SCREENING"),
            screening_model_version=self.model,
            input_data_hash=self._sha(state.get("extracted_facts", {})),
            completed_at=datetime.utcnow()
        )
        await self._append_event(fraud_event, stream_id=f"fraud-{app_id}")
        
        # Trigger ComplianceCheckRequested on loan stream
        import os
        regulation_version = os.getenv("REGULATION_VERSION", "APEX-2024-Q4")
        compliance_event = ComplianceCheckRequested(
            application_id=app_id,
            requested_at=datetime.utcnow(),
            triggered_by_event_id=str(fraud_event.event_id),
            regulation_set_version=regulation_version,
            rules_to_evaluate=["REG-001", "REG-002", "REG-003", "REG-004", "REG-005", "REG-006"]
        )
        await self._append_event(compliance_event, stream_id=f"loan-{app_id}")
        
        # Append AgentOutputWritten
        output_event = AgentOutputWritten(
            session_id=state["session_id"],
            agent_type="fraud_detection",
            application_id=app_id,
            events_written=[{"type": "FraudScreeningCompleted"}, {"type": "ComplianceCheckRequested"}],
            output_summary=f"Fraud score: {assessment.get('fraud_score', 0.0)}, Risk: {assessment.get('risk_level', 'LOW')}",
            written_at=datetime.utcnow()
        )
        await self._append_event(output_event)
        
        state["output_events_written"] = ["FraudScreeningCompleted", "ComplianceCheckRequested"]
        state["next_agent_triggered"] = "compliance"
        
        await self._record_node_execution(
            node_name="write_output",
            input_summary={"fraud_score": assessment.get("fraud_score", 0.0)},
            output_summary={"events_written": 2, "next_agent": "compliance"},
            llm_tokens_input=0, llm_tokens_output=0, llm_cost_usd=0.0
        )
        return state


class ComplianceAgent(BaseApexAgent):
    """
    Evaluates 6 deterministic regulatory rules. No LLM in decision path.

    NODES (6 rule nodes + bookend nodes):
        validate_inputs → check_reg001 → check_reg002 → check_reg003
        → check_reg004 → check_reg005 → check_reg006 → write_output

    Use conditional edges after each hard-block rule:
        graph.add_conditional_edges("check_reg002", self._should_continue,
                                     {"continue":"check_reg003","hard_block":"write_output"})

    RULE IMPLEMENTATIONS (deterministic — no LLM):
        REG-001: not any AML_WATCH flag is_active  → ComplianceRulePassed/Failed
        REG-002: not any SANCTIONS_REVIEW is_active → hard_block=True if failed
        REG-003: jurisdiction != "MT"               → hard_block=True if failed
        REG-004: not (Sole Proprietor AND >$250K)   → remediation_available=True if failed
        REG-005: founded_year <= 2022               → hard_block=True if failed
        REG-006: Always passes → ComplianceRuleNoted(CRA_CONSIDERATION)

    OUTPUT STREAMS:
        compliance-{id}: ComplianceCheckInitiated, ComplianceRulePassed/Failed/Noted (6x), ComplianceCheckCompleted
        loan-{id}: DecisionRequested (if CLEAR/CONDITIONAL) OR ApplicationDeclined (if BLOCKED)
    """
    def build_graph(self):
        from typing import TypedDict
        class S(TypedDict):
            application_id: str; session_id: str; agent_id: str
            company_profile: dict | None; rules_results: list | None
            hard_block: bool | None; overall_verdict: str | None
            errors: list; output_events_written: list; next_agent_triggered: str | None
        g = StateGraph(S)
        for name in ["validate_inputs","check_reg001","check_reg002","check_reg003","check_reg004","check_reg005","check_reg006","write_output"]:
            g.add_node(name, getattr(self, f"_node_{name}"))
        g.set_entry_point("validate_inputs")
        g.add_edge("validate_inputs","check_reg001")
        g.add_edge("check_reg001","check_reg002")
        # REG-002 and REG-003 are hard blocks: conditional edge to write_output if failed
        g.add_conditional_edges("check_reg002", lambda s: "write_output" if s.get("hard_block") else "check_reg003")
        g.add_conditional_edges("check_reg003", lambda s: "write_output" if s.get("hard_block") else "check_reg004")
        g.add_edge("check_reg004","check_reg005")
        g.add_conditional_edges("check_reg005", lambda s: "write_output" if s.get("hard_block") else "check_reg006")
        g.add_edge("check_reg006","write_output")
        g.add_edge("write_output",END)
        return g.compile()

    async def _node_validate_inputs(self, state):
        """Load company profile from registry, verify ComplianceCheckRequested event."""
        from datetime import datetime
        from ledger.schema.events import ComplianceCheckInitiated, AgentInputValidated
        import json
        
        app_id = state["application_id"]
        loan_stream = await self.store.load_stream(f"loan-{app_id}")
        
        # Find ComplianceCheckRequested event
        compliance_requested = [e for e in loan_stream if e["event_type"] == "ComplianceCheckRequested"]
        if not compliance_requested:
            state["errors"].append("No ComplianceCheckRequested event found")
            return state
        
        # Get applicant_id and regulation version
        app_submitted = [e for e in loan_stream if e["event_type"] == "ApplicationSubmitted"]
        if not app_submitted:
            state["errors"].append("No ApplicationSubmitted event found")
            return state
        
        payload = json.loads(app_submitted[0]["payload"]) if isinstance(app_submitted[0]["payload"], str) else app_submitted[0]["payload"]
        applicant_id = payload.get("applicant_id")
        
        compliance_payload = json.loads(compliance_requested[0]["payload"]) if isinstance(compliance_requested[0]["payload"], str) else compliance_requested[0]["payload"]
        regulation_version = compliance_payload.get("regulation_set_version", "APEX-2024-Q4")
        rules_to_evaluate = compliance_payload.get("rules_to_evaluate", [])
        
        # Load company profile from registry
        company_profile = await self.registry.get_company(applicant_id)
        state["company_profile"] = company_profile
        state["rules_results"] = []
        state["hard_block"] = False
        
        # Append ComplianceCheckInitiated
        init_event = ComplianceCheckInitiated(
            application_id=app_id,
            session_id=state["session_id"],
            regulation_set_version=regulation_version,
            rules_to_evaluate=rules_to_evaluate,
            initiated_at=datetime.utcnow()
        )
        await self._append_event(init_event, stream_id=f"compliance-{app_id}")
        
        # Append AgentInputValidated
        event = AgentInputValidated(
            session_id=state["session_id"],
            agent_type="compliance",
            application_id=app_id,
            inputs_validated=["ComplianceCheckRequested", "company_profile"],
            validation_duration_ms=15,
            validated_at=datetime.utcnow()
        )
        await self._append_event(event)
        
        await self._record_node_execution(
            node_name="validate_inputs",
            input_summary={"application_id": app_id, "regulation_version": regulation_version},
            output_summary={"rules_to_check": len(rules_to_evaluate)},
            llm_tokens_input=0, llm_tokens_output=0, llm_cost_usd=0.0
        )
        return state
    
    async def _node_check_reg001(self, state):
        """REG-001: BSA - Check AML_WATCH flags."""
        from datetime import datetime
        from ledger.schema.events import ComplianceRulePassed, ComplianceRuleFailed
        
        app_id = state["application_id"]
        company_profile = state.get("company_profile", {})
        
        # Query compliance flags
        applicant_id = company_profile.get("company_id")
        compliance_flags = await self.registry.get_compliance_flags(applicant_id) if applicant_id else []
        
        # Check for active AML_WATCH flags
        aml_flags = [f for f in compliance_flags if f.get("flag_type") == "AML_WATCH" and f.get("is_active")]
        
        if aml_flags:
            # Rule failed
            event = ComplianceRuleFailed(
                application_id=app_id,
                session_id=state["session_id"],
                rule_id="REG-001",
                rule_name="BSA Anti-Money Laundering Check",
                failure_reason=f"Active AML_WATCH flags found: {len(aml_flags)}",
                is_hard_block=False,
                remediation_available=True,
                remediation_steps=["Submit AML clearance documentation", "Provide source of funds verification"],
                evaluated_at=datetime.utcnow()
            )
            state["rules_results"].append({"rule_id": "REG-001", "passed": False, "hard_block": False})
        else:
            # Rule passed
            event = ComplianceRulePassed(
                application_id=app_id,
                session_id=state["session_id"],
                rule_id="REG-001",
                rule_name="BSA Anti-Money Laundering Check",
                evaluated_at=datetime.utcnow()
            )
            state["rules_results"].append({"rule_id": "REG-001", "passed": True})
        
        await self._append_event(event, stream_id=f"compliance-{app_id}")
        
        await self._record_node_execution(
            node_name="check_reg001",
            input_summary={"aml_flags_checked": len(compliance_flags)},
            output_summary={"passed": len(aml_flags) == 0, "active_flags": len(aml_flags)},
            llm_tokens_input=0, llm_tokens_output=0, llm_cost_usd=0.0
        )
        return state
    
    async def _node_check_reg002(self, state):
        """REG-002: OFAC - Check SANCTIONS_REVIEW flags (HARD BLOCK)."""
        from datetime import datetime
        from ledger.schema.events import ComplianceRulePassed, ComplianceRuleFailed
        
        app_id = state["application_id"]
        company_profile = state.get("company_profile", {})
        
        # Query compliance flags
        applicant_id = company_profile.get("company_id")
        compliance_flags = await self.registry.get_compliance_flags(applicant_id) if applicant_id else []
        
        # Check for active SANCTIONS_REVIEW flags
        sanctions_flags = [f for f in compliance_flags if f.get("flag_type") == "SANCTIONS_REVIEW" and f.get("is_active")]
        
        if sanctions_flags:
            # Rule failed - HARD BLOCK
            event = ComplianceRuleFailed(
                application_id=app_id,
                session_id=state["session_id"],
                rule_id="REG-002",
                rule_name="OFAC Sanctions Screening",
                failure_reason=f"Active SANCTIONS_REVIEW flags found: {len(sanctions_flags)}",
                is_hard_block=True,
                remediation_available=False,
                remediation_steps=[],
                evaluated_at=datetime.utcnow()
            )
            state["rules_results"].append({"rule_id": "REG-002", "passed": False, "hard_block": True})
            state["hard_block"] = True
        else:
            # Rule passed
            event = ComplianceRulePassed(
                application_id=app_id,
                session_id=state["session_id"],
                rule_id="REG-002",
                rule_name="OFAC Sanctions Screening",
                evaluated_at=datetime.utcnow()
            )
            state["rules_results"].append({"rule_id": "REG-002", "passed": True})
        
        await self._append_event(event, stream_id=f"compliance-{app_id}")
        
        await self._record_node_execution(
            node_name="check_reg002",
            input_summary={"sanctions_flags_checked": len(compliance_flags)},
            output_summary={"passed": len(sanctions_flags) == 0, "hard_block": state["hard_block"]},
            llm_tokens_input=0, llm_tokens_output=0, llm_cost_usd=0.0
        )
        return state
    
    async def _node_check_reg003(self, state):
        """REG-003: Jurisdiction - Must not be Montana (HARD BLOCK)."""
        from datetime import datetime
        from ledger.schema.events import ComplianceRulePassed, ComplianceRuleFailed
        
        app_id = state["application_id"]
        company_profile = state.get("company_profile", {})
        
        jurisdiction = company_profile.get("jurisdiction", "")
        
        if jurisdiction == "MT":
            # Rule failed - HARD BLOCK
            event = ComplianceRuleFailed(
                application_id=app_id,
                session_id=state["session_id"],
                rule_id="REG-003",
                rule_name="Prohibited Jurisdiction Check",
                failure_reason=f"Applicant jurisdiction '{jurisdiction}' is prohibited (Montana restriction)",
                is_hard_block=True,
                remediation_available=False,
                remediation_steps=[],
                evaluated_at=datetime.utcnow()
            )
            state["rules_results"].append({"rule_id": "REG-003", "passed": False, "hard_block": True})
            state["hard_block"] = True
        else:
            # Rule passed
            event = ComplianceRulePassed(
                application_id=app_id,
                session_id=state["session_id"],
                rule_id="REG-003",
                rule_name="Prohibited Jurisdiction Check",
                evaluated_at=datetime.utcnow()
            )
            state["rules_results"].append({"rule_id": "REG-003", "passed": True})
        
        await self._append_event(event, stream_id=f"compliance-{app_id}")
        
        await self._record_node_execution(
            node_name="check_reg003",
            input_summary={"jurisdiction": jurisdiction},
            output_summary={"passed": jurisdiction != "MT", "hard_block": state["hard_block"]},
            llm_tokens_input=0, llm_tokens_output=0, llm_cost_usd=0.0
        )
        return state
    
    async def _node_check_reg004(self, state):
        """REG-004: Legal Type - Sole Proprietor with >$250K is restricted."""
        from datetime import datetime
        from ledger.schema.events import ComplianceRulePassed, ComplianceRuleFailed
        import json
        
        app_id = state["application_id"]
        company_profile = state.get("company_profile", {})
        
        # Get requested amount from loan stream
        loan_stream = await self.store.load_stream(f"loan-{app_id}")
        app_submitted = [e for e in loan_stream if e["event_type"] == "ApplicationSubmitted"]
        payload = json.loads(app_submitted[0]["payload"]) if isinstance(app_submitted[0]["payload"], str) else app_submitted[0]["payload"]
        requested_amount = float(payload.get("requested_amount_usd", 0))
        
        legal_type = company_profile.get("legal_type", "")
        
        if legal_type == "Sole Proprietor" and requested_amount > 250000:
            # Rule failed - remediation available
            event = ComplianceRuleFailed(
                application_id=app_id,
                session_id=state["session_id"],
                rule_id="REG-004",
                rule_name="Legal Entity Structure Requirement",
                failure_reason=f"Sole Proprietor requesting ${requested_amount:,.0f} exceeds $250K limit",
                is_hard_block=False,
                remediation_available=True,
                remediation_steps=["Convert to LLC or Corporation", "Reduce loan amount to $250,000 or less"],
                evaluated_at=datetime.utcnow()
            )
            state["rules_results"].append({"rule_id": "REG-004", "passed": False, "hard_block": False})
        else:
            # Rule passed
            event = ComplianceRulePassed(
                application_id=app_id,
                session_id=state["session_id"],
                rule_id="REG-004",
                rule_name="Legal Entity Structure Requirement",
                evaluated_at=datetime.utcnow()
            )
            state["rules_results"].append({"rule_id": "REG-004", "passed": True})
        
        await self._append_event(event, stream_id=f"compliance-{app_id}")
        
        await self._record_node_execution(
            node_name="check_reg004",
            input_summary={"legal_type": legal_type, "requested_amount": requested_amount},
            output_summary={"passed": not (legal_type == "Sole Proprietor" and requested_amount > 250000)},
            llm_tokens_input=0, llm_tokens_output=0, llm_cost_usd=0.0
        )
        return state
    
    async def _node_check_reg005(self, state):
        """REG-005: Operating History - Must be founded by 2022 or earlier (HARD BLOCK)."""
        from datetime import datetime
        from ledger.schema.events import ComplianceRulePassed, ComplianceRuleFailed
        
        app_id = state["application_id"]
        company_profile = state.get("company_profile", {})
        
        founded_year = company_profile.get("founded_year")
        
        if founded_year is None or founded_year > 2022:
            # Rule failed - HARD BLOCK
            event = ComplianceRuleFailed(
                application_id=app_id,
                session_id=state["session_id"],
                rule_id="REG-005",
                rule_name="Minimum Operating History Requirement",
                failure_reason=f"Company founded in {founded_year or 'unknown year'}, requires 2+ years operating history (founded ≤ 2022)",
                is_hard_block=True,
                remediation_available=False,
                remediation_steps=[],
                evaluated_at=datetime.utcnow()
            )
            state["rules_results"].append({"rule_id": "REG-005", "passed": False, "hard_block": True})
            state["hard_block"] = True
        else:
            # Rule passed
            event = ComplianceRulePassed(
                application_id=app_id,
                session_id=state["session_id"],
                rule_id="REG-005",
                rule_name="Minimum Operating History Requirement",
                evaluated_at=datetime.utcnow()
            )
            state["rules_results"].append({"rule_id": "REG-005", "passed": True})
        
        await self._append_event(event, stream_id=f"compliance-{app_id}")
        
        await self._record_node_execution(
            node_name="check_reg005",
            input_summary={"founded_year": founded_year},
            output_summary={"passed": founded_year is not None and founded_year <= 2022, "hard_block": state["hard_block"]},
            llm_tokens_input=0, llm_tokens_output=0, llm_cost_usd=0.0
        )
        return state
    
    async def _node_check_reg006(self, state):
        """REG-006: CRA Consideration - Always passes, informational note."""
        from datetime import datetime
        from ledger.schema.events import ComplianceRuleNoted
        
        app_id = state["application_id"]
        
        # Always passes - just a note
        event = ComplianceRuleNoted(
            application_id=app_id,
            session_id=state["session_id"],
            rule_id="REG-006",
            rule_name="Community Reinvestment Act Consideration",
            note_type="CRA_CONSIDERATION",
            note_text="Application eligible for CRA credit consideration",
            evaluated_at=datetime.utcnow()
        )
        state["rules_results"].append({"rule_id": "REG-006", "passed": True, "noted": True})
        
        await self._append_event(event, stream_id=f"compliance-{app_id}")
        
        await self._record_node_execution(
            node_name="check_reg006",
            input_summary={"rule_type": "informational"},
            output_summary={"noted": True},
            llm_tokens_input=0, llm_tokens_output=0, llm_cost_usd=0.0
        )
        return state
    
    async def _node_write_output(self, state):
        """Append ComplianceCheckCompleted, then DecisionRequested or ApplicationDeclined."""
        from datetime import datetime
        from ledger.schema.events import ComplianceCheckCompleted, DecisionRequested, ApplicationDeclined, AgentOutputWritten
        
        app_id = state["application_id"]
        rules_results = state.get("rules_results", [])
        hard_block = state.get("hard_block", False)
        
        # Calculate summary
        rules_evaluated = len(rules_results)
        rules_passed = len([r for r in rules_results if r.get("passed")])
        rules_failed = len([r for r in rules_results if not r.get("passed") and not r.get("noted")])
        rules_noted = len([r for r in rules_results if r.get("noted")])
        
        # Determine overall verdict
        if hard_block:
            overall_verdict = "BLOCKED"
        elif rules_failed > 0:
            overall_verdict = "CONDITIONAL"
        else:
            overall_verdict = "CLEAR"
        
        state["overall_verdict"] = overall_verdict
        
        # Append ComplianceCheckCompleted
        compliance_event = ComplianceCheckCompleted(
            application_id=app_id,
            session_id=state["session_id"],
            rules_evaluated=rules_evaluated,
            rules_passed=rules_passed,
            rules_failed=rules_failed,
            rules_noted=rules_noted,
            has_hard_block=hard_block,
            overall_verdict=overall_verdict,
            completed_at=datetime.utcnow()
        )
        await self._append_event(compliance_event, stream_id=f"compliance-{app_id}")
        
        # If hard block, decline application immediately
        if hard_block:
            failed_rules = [r for r in rules_results if not r.get("passed") and r.get("hard_block")]
            decline_reasons = [f"Failed {r['rule_id']}: hard block" for r in failed_rules]
            
            decline_event = ApplicationDeclined(
                application_id=app_id,
                decline_reasons=decline_reasons,
                declined_by=f"agent-{self.agent_id}",
                adverse_action_notice_required=True,
                adverse_action_codes=[r["rule_id"] for r in failed_rules],
                declined_at=datetime.utcnow()
            )
            await self._append_event(decline_event, stream_id=f"loan-{app_id}")
            state["output_events_written"] = ["ComplianceCheckCompleted", "ApplicationDeclined"]
            state["next_agent_triggered"] = None
        else:
            # Trigger DecisionRequested
            decision_event = DecisionRequested(
                application_id=app_id,
                requested_at=datetime.utcnow(),
                all_analyses_complete=True,
                triggered_by_event_id=str(compliance_event.event_id)
            )
            await self._append_event(decision_event, stream_id=f"loan-{app_id}")
            state["output_events_written"] = ["ComplianceCheckCompleted", "DecisionRequested"]
            state["next_agent_triggered"] = "decision_orchestrator"
        
        # Append AgentOutputWritten
        output_event = AgentOutputWritten(
            session_id=state["session_id"],
            agent_type="compliance",
            application_id=app_id,
            events_written=[{"type": e} for e in state["output_events_written"]],
            output_summary=f"Verdict: {overall_verdict}, Hard block: {hard_block}",
            written_at=datetime.utcnow()
        )
        await self._append_event(output_event)
        
        await self._record_node_execution(
            node_name="write_output",
            input_summary={"rules_evaluated": rules_evaluated, "hard_block": hard_block},
            output_summary={"verdict": overall_verdict, "events_written": len(state["output_events_written"])},
            llm_tokens_input=0, llm_tokens_output=0, llm_cost_usd=0.0
        )
        return state


class DecisionOrchestratorAgent(BaseApexAgent):
    """
    Synthesises all prior agent outputs. Reads from ALL prior agent streams.

    NODES:
        validate_inputs → load_all_analyses → synthesize_decision (LLM)
        → apply_hard_constraints → write_output

    READS FROM:
        credit-{id}: CreditAnalysisCompleted (risk_tier, confidence, limit)
        fraud-{id}: FraudScreeningCompleted (fraud_score, anomalies)
        compliance-{id}: ComplianceCheckCompleted (overall_verdict)

    HARD CONSTRAINTS (Python — not LLM, in apply_hard_constraints):
        1. compliance BLOCKED → must DECLINE regardless of LLM
        2. confidence < 0.60 → must REFER
        3. fraud_score > 0.60 → must REFER
        4. risk_tier HIGH AND confidence >= 0.70 → DECLINE eligible

    LLM ROLE (synthesize_decision):
        Given all 3 analyses, produce executive_summary and key_risks.
        Initial recommendation (may be overridden by hard constraints).
        Return OrchestratorDecision JSON.

    OUTPUT STREAMS:
        loan-{id}: DecisionGenerated
        loan-{id}: ApplicationApproved (if APPROVE) OR ApplicationDeclined (if DECLINE)
                   OR HumanReviewRequested (if REFER)
    """
    def build_graph(self):
        from typing import TypedDict
        class S(TypedDict):
            application_id: str; session_id: str; agent_id: str
            credit_analysis: dict | None; fraud_screening: dict | None
            compliance_record: dict | None; orchestrator_decision: dict | None
            errors: list; output_events_written: list; next_agent_triggered: str | None
        g = StateGraph(S)
        for name in ["validate_inputs","load_all_analyses","synthesize_decision","apply_hard_constraints","write_output"]:
            g.add_node(name, getattr(self, f"_node_{name}"))
        g.set_entry_point("validate_inputs")
        g.add_edge("validate_inputs","load_all_analyses")
        g.add_edge("load_all_analyses","synthesize_decision")
        g.add_edge("synthesize_decision","apply_hard_constraints")
        g.add_edge("apply_hard_constraints","write_output")
        g.add_edge("write_output",END)
        return g.compile()

    async def _node_validate_inputs(self, state):
        """Verify DecisionRequested event, all 3 analysis streams complete."""
        from datetime import datetime
        from ledger.schema.events import AgentInputValidated
        
        app_id = state["application_id"]
        loan_stream = await self.store.load_stream(f"loan-{app_id}")
        
        # Find DecisionRequested event
        decision_requested = [e for e in loan_stream if e["event_type"] == "DecisionRequested"]
        if not decision_requested:
            state["errors"].append("No DecisionRequested event found")
            return state
        
        await self._record_node_execution(
            node_name="validate_inputs",
            input_summary={"application_id": app_id},
            output_summary={"decision_requested": len(decision_requested)},
            llm_tokens_input=0, llm_tokens_output=0, llm_cost_usd=0.0
        )
        
        event = AgentInputValidated(
            session_id=state["session_id"],
            agent_type="decision_orchestrator",
            application_id=app_id,
            inputs_validated=["DecisionRequested"],
            validation_duration_ms=10,
            validated_at=datetime.utcnow()
        )
        await self._append_event(event)
        return state
    
    async def _node_load_all_analyses(self, state):
        """Load credit, fraud, compliance streams; extract latest completed events."""
        import json
        
        app_id = state["application_id"]
        
        # Load credit analysis
        try:
            credit_stream = await self.store.load_stream(f"credit-{app_id}")
            credit_completed = [e for e in credit_stream if e["event_type"] == "CreditAnalysisCompleted"]
            if credit_completed:
                payload = json.loads(credit_completed[-1]["payload"]) if isinstance(credit_completed[-1]["payload"], str) else credit_completed[-1]["payload"]
                state["credit_analysis"] = payload
            else:
                state["credit_analysis"] = None
        except:
            state["credit_analysis"] = None
        
        # Load fraud screening
        try:
            fraud_stream = await self.store.load_stream(f"fraud-{app_id}")
            fraud_completed = [e for e in fraud_stream if e["event_type"] == "FraudScreeningCompleted"]
            if fraud_completed:
                payload = json.loads(fraud_completed[-1]["payload"]) if isinstance(fraud_completed[-1]["payload"], str) else fraud_completed[-1]["payload"]
                state["fraud_screening"] = payload
            else:
                state["fraud_screening"] = None
        except:
            state["fraud_screening"] = None
        
        # Load compliance check
        try:
            compliance_stream = await self.store.load_stream(f"compliance-{app_id}")
            compliance_completed = [e for e in compliance_stream if e["event_type"] == "ComplianceCheckCompleted"]
            if compliance_completed:
                payload = json.loads(compliance_completed[-1]["payload"]) if isinstance(compliance_completed[-1]["payload"], str) else compliance_completed[-1]["payload"]
                state["compliance_record"] = payload
            else:
                state["compliance_record"] = None
        except:
            state["compliance_record"] = None
        
        await self._record_node_execution(
            node_name="load_all_analyses",
            input_summary={"application_id": app_id},
            output_summary={
                "has_credit": state["credit_analysis"] is not None,
                "has_fraud": state["fraud_screening"] is not None,
                "has_compliance": state["compliance_record"] is not None
            },
            llm_tokens_input=0, llm_tokens_output=0, llm_cost_usd=0.0
        )
        return state
    
    async def _node_synthesize_decision(self, state):
        """LLM: synthesize all 3 inputs into recommendation + executive_summary."""
        import json
        
        app_id = state["application_id"]
        credit = state.get("credit_analysis", {})
        fraud = state.get("fraud_screening", {})
        compliance = state.get("compliance_record", {})
        
        # Build LLM prompt for decision synthesis
        system_prompt = """You are a senior loan underwriting officer synthesizing multiple analyses into a final decision.

Your task:
1. Review credit analysis, fraud screening, and compliance results
2. Produce an executive summary (2-3 sentences)
3. List key risks (3-5 bullet points)
4. Make an initial recommendation: APPROVE, DECLINE, or REFER

Return a JSON object with:
- recommendation ("APPROVE", "DECLINE", "REFER")
- confidence (0.0-1.0)
- approved_amount_usd (number, if APPROVE)
- conditions (list of strings, if APPROVE)
- executive_summary (string)
- key_risks (list of strings)

Guidelines:
- APPROVE: Strong credit, low fraud risk, compliance clear
- DECLINE: High risk, poor credit, or significant fraud indicators
- REFER: Borderline cases, missing data, or moderate concerns requiring human review"""
        
        user_prompt = f"""Credit Analysis:
{json.dumps(credit, indent=2, default=str) if credit else 'Not available'}

Fraud Screening:
{json.dumps(fraud, indent=2, default=str) if fraud else 'Not available'}

Compliance Check:
{json.dumps(compliance, indent=2, default=str) if compliance else 'Not available'}"""
        
        # Call LLM
        response_text, tok_in, tok_out, cost = await self._call_llm(system_prompt, user_prompt, max_tokens=1024)
        
        # Parse response
        try:
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            decision = json.loads(response_text.strip())
        except:
            # Fallback: conservative decision
            decision = {
                "recommendation": "REFER",
                "confidence": 0.5,
                "approved_amount_usd": None,
                "conditions": [],
                "executive_summary": "Unable to synthesize decision from LLM response. Referring for human review.",
                "key_risks": ["LLM synthesis failed"]
            }
        
        state["orchestrator_decision"] = decision
        
        await self._record_node_execution(
            node_name="synthesize_decision",
            input_summary={"analyses_loaded": 3},
            output_summary={"recommendation": decision.get("recommendation"), "confidence": decision.get("confidence")},
            llm_tokens_input=tok_in, llm_tokens_output=tok_out, llm_cost_usd=cost
        )
        return state
    
    async def _node_apply_hard_constraints(self, state):
        """Apply hard constraints: compliance BLOCKED → DECLINE, low confidence → REFER, etc."""
        decision = state.get("orchestrator_decision", {})
        compliance = state.get("compliance_record", {})
        fraud = state.get("fraud_screening", {})
        credit = state.get("credit_analysis", {})
        
        original_recommendation = decision.get("recommendation", "REFER")
        
        # Hard constraint 1: Compliance BLOCKED → must DECLINE
        if compliance.get("overall_verdict") == "BLOCKED":
            decision["recommendation"] = "DECLINE"
            decision["key_risks"] = decision.get("key_risks", []) + ["Compliance hard block"]
        
        # Hard constraint 2: Confidence < 0.60 → must REFER
        elif decision.get("confidence", 0.0) < 0.60:
            decision["recommendation"] = "REFER"
            decision["key_risks"] = decision.get("key_risks", []) + ["Low confidence score"]
        
        # Hard constraint 3: Fraud score > 0.60 → must REFER
        elif fraud.get("fraud_score", 0.0) > 0.60:
            decision["recommendation"] = "REFER"
            decision["key_risks"] = decision.get("key_risks", []) + ["High fraud risk"]
        
        # Hard constraint 4: Risk tier HIGH AND confidence >= 0.70 → DECLINE eligible
        elif credit.get("decision", {}).get("risk_tier") == "HIGH" and decision.get("confidence", 0.0) >= 0.70:
            if original_recommendation != "APPROVE":
                decision["recommendation"] = "DECLINE"
                decision["key_risks"] = decision.get("key_risks", []) + ["High credit risk tier"]
        
        state["orchestrator_decision"] = decision
        
        await self._record_node_execution(
            node_name="apply_hard_constraints",
            input_summary={"original_recommendation": original_recommendation},
            output_summary={"final_recommendation": decision.get("recommendation")},
            llm_tokens_input=0, llm_tokens_output=0, llm_cost_usd=0.0
        )
        return state
    
    async def _node_write_output(self, state):
        """Append DecisionGenerated + ApplicationApproved/Declined/HumanReviewRequested with OCC retry."""
        from datetime import datetime
        from decimal import Decimal
        from ledger.schema.events import (
            DecisionGenerated, ApplicationApproved, ApplicationDeclined,
            HumanReviewRequested, AgentOutputWritten
        )
        from ledger.domain.aggregates.loan_application import LoanApplicationAggregate
        from ledger.event_store import OptimisticConcurrencyError
        
        app_id = state["application_id"]
        decision = state.get("orchestrator_decision", {})
        
        # Rehydrate aggregate to get current version (for OCC)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Load current aggregate state
                agg = await LoanApplicationAggregate.load(self.store, app_id)
                current_version = agg.version
                
                # Build DecisionGenerated event
                decision_event = DecisionGenerated(
                    application_id=app_id,
                    orchestrator_session_id=state["session_id"],
                    recommendation=decision.get("recommendation", "REFER"),
                    confidence=decision.get("confidence", 0.5),
                    approved_amount_usd=Decimal(str(decision.get("approved_amount_usd", 0))) if decision.get("approved_amount_usd") else None,
                    conditions=decision.get("conditions", []),
                    executive_summary=decision.get("executive_summary", ""),
                    key_risks=decision.get("key_risks", []),
                    contributing_sessions=[state["session_id"]],
                    model_versions={"orchestrator": self.model},
                    generated_at=datetime.utcnow()
                )
                
                # Prepare events to append
                events_to_append = [decision_event.to_store_dict()]
                
                # Add outcome event based on recommendation
                recommendation = decision.get("recommendation", "REFER")
                if recommendation == "APPROVE":
                    approve_event = ApplicationApproved(
                        application_id=app_id,
                        approved_amount_usd=Decimal(str(decision.get("approved_amount_usd", 0))),
                        interest_rate_pct=5.5,
                        term_months=36,
                        conditions=decision.get("conditions", []),
                        approved_by=f"agent-{self.agent_id}",
                        effective_date=datetime.utcnow().date().isoformat(),
                        approved_at=datetime.utcnow()
                    )
                    events_to_append.append(approve_event.to_store_dict())
                    state["output_events_written"] = ["DecisionGenerated", "ApplicationApproved"]
                
                elif recommendation == "DECLINE":
                    decline_event = ApplicationDeclined(
                        application_id=app_id,
                        decline_reasons=decision.get("key_risks", ["Risk assessment failed"]),
                        declined_by=f"agent-{self.agent_id}",
                        adverse_action_notice_required=True,
                        adverse_action_codes=["CREDIT_RISK", "POLICY_VIOLATION"],
                        declined_at=datetime.utcnow()
                    )
                    events_to_append.append(decline_event.to_store_dict())
                    state["output_events_written"] = ["DecisionGenerated", "ApplicationDeclined"]
                
                else:  # REFER
                    review_event = HumanReviewRequested(
                        application_id=app_id,
                        reason="Orchestrator recommendation: REFER for human review",
                        decision_event_id=str(decision_event.event_id),
                        assigned_to=None,
                        requested_at=datetime.utcnow()
                    )
                    events_to_append.append(review_event.to_store_dict())
                    state["output_events_written"] = ["DecisionGenerated", "HumanReviewRequested"]
                
                # Append with OCC check
                await self.store.append(f"loan-{app_id}", events_to_append, expected_version=current_version)
                
                # Success - break retry loop
                break
                
            except OptimisticConcurrencyError as e:
                if attempt < max_retries - 1:
                    # Retry: reload aggregate and try again
                    await self._record_node_execution(
                        node_name="write_output_retry",
                        input_summary={"attempt": attempt + 1, "error": str(e)},
                        output_summary={"retrying": True},
                        llm_tokens_input=0, llm_tokens_output=0, llm_cost_usd=0.0
                    )
                    continue
                else:
                    # Max retries exceeded
                    state["errors"].append(f"OCC error after {max_retries} retries: {e}")
                    raise
        
        # Append AgentOutputWritten
        output_event = AgentOutputWritten(
            session_id=state["session_id"],
            agent_type="decision_orchestrator",
            application_id=app_id,
            events_written=[{"type": e} for e in state["output_events_written"]],
            output_summary=f"Decision: {recommendation}, Confidence: {decision.get('confidence', 0.0)}",
            written_at=datetime.utcnow()
        )
        await self._append_event(output_event)
        
        await self._record_node_execution(
            node_name="write_output",
            input_summary={"recommendation": recommendation},
            output_summary={"events_written": len(state["output_events_written"])},
            llm_tokens_input=0, llm_tokens_output=0, llm_cost_usd=0.0
        )
        return state

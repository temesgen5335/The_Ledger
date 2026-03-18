"""
ledger/upcasters.py — UpcasterRegistry
=======================================
COMPLETION STATUS: STUB — implement upcast() for two event versions.

Upcasters transform old event versions to the current version ON READ.
They NEVER write to the events table. Immutability is non-negotiable.

IMPLEMENT:
  CreditAnalysisCompleted v1 → v2: add model_versions={} if absent
  DecisionGenerated v1 → v2: add model_versions={} if absent

RULE: if event_version == current version, return unchanged.
      if event_version < current version, apply the chain of upcasters.
"""
from __future__ import annotations

class UpcasterRegistry:
    """
    Apply on load_stream() — never on append().
    
    Upcasters transform old event versions to current version ON READ.
    They NEVER write to the events table. Immutability is non-negotiable.
    
    Phase 4 Addition: Currency field upcasting for ExtractionCompleted events.
    """
    def upcast(self, event: dict) -> dict:
        et = event.get("event_type"); ver = event.get("event_version", 1)
        
        # CreditAnalysisCompleted v1 → v2: add regulatory_basis if absent
        if et == "CreditAnalysisCompleted" and ver < 2:
            event = dict(event); event["event_version"] = 2
            p = dict(event.get("payload", {}))
            p.setdefault("regulatory_basis", [])
            event["payload"] = p
        
        # DecisionGenerated v1 → v2: add model_versions if absent
        if et == "DecisionGenerated" and ver < 2:
            event = dict(event); event["event_version"] = 2
            p = dict(event.get("payload", {}))
            p.setdefault("model_versions", {})
            event["payload"] = p
        
        # ExtractionCompleted: inject currency=USD if missing in facts
        # This handles old events before currency field was added
        if et in ("ExtractionCompleted", "DocumentExtracted"):
            import json
            event = dict(event)
            
            # Parse payload if it's a JSON string
            payload = event.get("payload", {})
            if isinstance(payload, str):
                payload = json.loads(payload)
            
            p = dict(payload) if isinstance(payload, dict) else {}
            
            # Handle facts field (could be dict or JSON string)
            facts = p.get("facts", {})
            if isinstance(facts, str):
                facts = json.loads(facts)
            
            if isinstance(facts, dict) and "currency" not in facts:
                facts["currency"] = ""
                p["facts"] = facts
                event["payload"] = p
        
        return event

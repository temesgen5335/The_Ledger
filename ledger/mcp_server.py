"""
ledger/mcp_server.py
====================
FastMCP Server for The Ledger - Exposes Command and Query interfaces via MCP protocol.

COMMAND SIDE (Tools):
  - trigger_analysis(app_id): Invoke orchestrator to run agents for an application
  - append_manual_event(app_id, event_type, payload): Human-in-the-loop event injection

QUERY SIDE (Resources):
  - resource("summary://{app_id}"): Flat row from application_summary table
  - resource("history://{app_id}"): Full replayed event stream from events table

Run: python -m ledger.mcp_server
"""
import asyncio
import json
from datetime import datetime
from typing import Any
import asyncpg
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, Tool, TextContent

from ledger.event_store import EventStore
from ledger.registry.client import ApplicantRegistryClient
from ledger.agents.credit_analysis_agent import CreditAnalysisAgent
from ledger.agents.base_agent import (
    DocumentProcessingAgent, FraudDetectionAgent,
    ComplianceAgent, DecisionOrchestratorAgent
)

# Database configuration
DB_URL = "postgresql://postgres:apex@localhost/apex_ledger"

# Global connections (initialized on server start)
event_store: EventStore = None
registry_pool: asyncpg.Pool = None
registry_client: ApplicantRegistryClient = None


# ============================================================================
# SERVER INITIALIZATION
# ============================================================================

async def initialize_connections():
    """Initialize database connections for event store and registry."""
    global event_store, registry_pool, registry_client
    
    event_store = EventStore(DB_URL)
    await event_store.connect()
    
    registry_pool = await asyncpg.create_pool(DB_URL, min_size=2, max_size=10)
    registry_client = ApplicantRegistryClient(registry_pool)
    
    print("[MCP Server] Database connections initialized", flush=True)


async def cleanup_connections():
    """Cleanup database connections on server shutdown."""
    global event_store, registry_pool
    
    if event_store:
        await event_store.close()
    if registry_pool:
        await registry_pool.close()
    
    print("[MCP Server] Database connections closed", flush=True)


# ============================================================================
# COMMAND SIDE - TOOLS
# ============================================================================

async def trigger_analysis(app_id: str, agent_type: str = "all") -> dict:
    """
    Trigger agent analysis for a specific application.
    
    Args:
        app_id: Application ID to analyze
        agent_type: Which agent to run - "all", "credit", "fraud", "compliance", "decision"
    
    Returns:
        dict with status, agents_run, and events_written
    """
    agents_run = []
    events_written = []
    
    try:
        # Run agents based on agent_type
        if agent_type in ("all", "credit"):
            credit_agent = CreditAnalysisAgent(
                agent_id=f"mcp-credit-{datetime.utcnow().timestamp()}",
                agent_type="credit_analysis",
                store=event_store,
                registry=registry_client,
                model="gemini-1.5-pro"
            )
            await credit_agent.process_application(app_id)
            agents_run.append("credit_analysis")
        
        if agent_type in ("all", "fraud"):
            fraud_agent = FraudDetectionAgent(
                agent_id=f"mcp-fraud-{datetime.utcnow().timestamp()}",
                agent_type="fraud_detection",
                store=event_store,
                registry=registry_client,
                model="gemini-1.5-pro"
            )
            await fraud_agent.process_application(app_id)
            agents_run.append("fraud_detection")
        
        if agent_type in ("all", "compliance"):
            compliance_agent = ComplianceAgent(
                agent_id=f"mcp-compliance-{datetime.utcnow().timestamp()}",
                agent_type="compliance",
                store=event_store,
                registry=registry_client,
                model="gemini-1.5-pro"
            )
            await compliance_agent.process_application(app_id)
            agents_run.append("compliance")
        
        if agent_type in ("all", "decision"):
            decision_agent = DecisionOrchestratorAgent(
                agent_id=f"mcp-decision-{datetime.utcnow().timestamp()}",
                agent_type="decision_orchestrator",
                store=event_store,
                registry=registry_client,
                model="gemini-1.5-pro"
            )
            await decision_agent.process_application(app_id)
            agents_run.append("decision_orchestrator")
        
        return {
            "status": "success",
            "application_id": app_id,
            "agents_run": agents_run,
            "message": f"Analysis triggered for {app_id} with agents: {', '.join(agents_run)}"
        }
    
    except Exception as e:
        return {
            "status": "error",
            "application_id": app_id,
            "agents_run": agents_run,
            "error": str(e),
            "message": f"Analysis failed: {str(e)}"
        }


async def append_manual_event(app_id: str, event_type: str, payload: dict) -> dict:
    """
    Append a manual event to an application stream (human-in-the-loop override).
    
    Args:
        app_id: Application ID
        event_type: Event type to append
        payload: Event payload as dict
    
    Returns:
        dict with status and event details
    """
    try:
        # Determine target stream based on event type
        if event_type.startswith("Credit"):
            stream_id = f"credit-{app_id}"
        elif event_type.startswith("Fraud"):
            stream_id = f"fraud-{app_id}"
        elif event_type.startswith("Compliance"):
            stream_id = f"compliance-{app_id}"
        elif event_type.startswith("Decision"):
            stream_id = f"decision-{app_id}"
        elif event_type.startswith("Document") or event_type.startswith("Extraction"):
            stream_id = f"docpkg-{app_id}"
        else:
            stream_id = f"loan-{app_id}"
        
        # Create event dict
        event = {
            "event_type": event_type,
            "event_version": 1,
            "payload": payload,
            "metadata": {
                "source": "mcp_manual_override",
                "injected_at": datetime.utcnow().isoformat(),
                "injected_by": "human_operator"
            }
        }
        
        # Append to stream
        ver = await event_store.stream_version(stream_id)
        await event_store.append(stream_id, [event], expected_version=ver)
        
        return {
            "status": "success",
            "application_id": app_id,
            "stream_id": stream_id,
            "event_type": event_type,
            "message": f"Manual event {event_type} appended to {stream_id}"
        }
    
    except Exception as e:
        return {
            "status": "error",
            "application_id": app_id,
            "event_type": event_type,
            "error": str(e),
            "message": f"Failed to append manual event: {str(e)}"
        }


# ============================================================================
# QUERY SIDE - RESOURCES
# ============================================================================

async def get_application_summary(app_id: str) -> dict:
    """
    Get application summary from projection table.
    
    Args:
        app_id: Application ID
    
    Returns:
        dict with application summary data
    """
    async with registry_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM application_summary WHERE application_id = $1",
            app_id
        )
        
        if not row:
            return {
                "status": "not_found",
                "application_id": app_id,
                "message": f"No summary found for {app_id}"
            }
        
        return {
            "status": "success",
            "application_id": app_id,
            "data": dict(row)
        }


async def get_application_history(app_id: str) -> dict:
    """
    Get full event history for an application by replaying all related streams.
    
    Args:
        app_id: Application ID
    
    Returns:
        dict with full event history across all streams
    """
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
            for event in events:
                event["_stream_id"] = stream_id
                all_events.append(event)
        except Exception:
            # Stream might not exist yet
            continue
    
    # Sort by global_position
    all_events.sort(key=lambda e: e.get("global_position", 0))
    
    return {
        "status": "success",
        "application_id": app_id,
        "total_events": len(all_events),
        "streams_checked": streams,
        "events": all_events
    }


# ============================================================================
# MCP SERVER SETUP
# ============================================================================

app = Server("ledger-mcp-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools (command side)."""
    return [
        Tool(
            name="trigger_analysis",
            description="Trigger agent analysis for a specific application. Runs credit, fraud, compliance, and decision agents.",
            inputSchema={
                "type": "object",
                "properties": {
                    "app_id": {
                        "type": "string",
                        "description": "Application ID to analyze"
                    },
                    "agent_type": {
                        "type": "string",
                        "enum": ["all", "credit", "fraud", "compliance", "decision"],
                        "description": "Which agent(s) to run",
                        "default": "all"
                    }
                },
                "required": ["app_id"]
            }
        ),
        Tool(
            name="append_manual_event",
            description="Append a manual event to an application stream for human-in-the-loop override",
            inputSchema={
                "type": "object",
                "properties": {
                    "app_id": {
                        "type": "string",
                        "description": "Application ID"
                    },
                    "event_type": {
                        "type": "string",
                        "description": "Event type to append (e.g., CreditAnalysisCompleted, DecisionGenerated)"
                    },
                    "payload": {
                        "type": "object",
                        "description": "Event payload as JSON object"
                    }
                },
                "required": ["app_id", "event_type", "payload"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls (command side)."""
    if name == "trigger_analysis":
        result = await trigger_analysis(
            app_id=arguments["app_id"],
            agent_type=arguments.get("agent_type", "all")
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    elif name == "append_manual_event":
        result = await append_manual_event(
            app_id=arguments["app_id"],
            event_type=arguments["event_type"],
            payload=arguments["payload"]
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    
    else:
        return [TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": f"Unknown tool: {name}"
        }))]


@app.list_resources()
async def list_resources() -> list[Resource]:
    """List available MCP resources (query side)."""
    return [
        Resource(
            uri="summary://",
            name="Application Summary",
            description="Get application summary from projection table. Use summary://{app_id}",
            mimeType="application/json"
        ),
        Resource(
            uri="history://",
            name="Application History",
            description="Get full event history for an application. Use history://{app_id}",
            mimeType="application/json"
        )
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Handle resource reads (query side)."""
    if uri.startswith("summary://"):
        app_id = uri.replace("summary://", "")
        result = await get_application_summary(app_id)
        return json.dumps(result, indent=2, default=str)
    
    elif uri.startswith("history://"):
        app_id = uri.replace("history://", "")
        result = await get_application_history(app_id)
        return json.dumps(result, indent=2, default=str)
    
    else:
        return json.dumps({
            "status": "error",
            "message": f"Unknown resource URI: {uri}"
        })


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """Run the MCP server with stdio transport."""
    await initialize_connections()
    
    try:
        async with stdio_server() as (read_stream, write_stream):
            print("[MCP Server] Starting on stdio transport...", flush=True)
            await app.run(read_stream, write_stream, app.create_initialization_options())
    finally:
        await cleanup_connections()


if __name__ == "__main__":
    asyncio.run(main())

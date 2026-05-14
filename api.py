"""
Enterprise AI Support Platform — FastAPI Backend
Session 3 of 12 — The ReAct Architecture
"""

import json
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from support_agent import (
    run_ticket, stream_ticket,
    run_session_verification,
    TOOLS, MAX_ITERATIONS
)

app = FastAPI(title="Enterprise AI Support Platform", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TicketRequest(BaseModel):
    ticket: str


@app.post("/api/run")
def run(req: TicketRequest):
    result = run_ticket(req.ticket)

    # Extract tool call log
    tool_calls_log = []
    for msg in result.get('messages', []):
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_log.append({
                    'tool_name': tc['name'],
                    'args':      tc['args'],
                    'call_id':   tc['id'],
                })

    # Match each call_id to a ToolMessage result
    tool_results_map = {}
    for msg in result.get('messages', []):
        if hasattr(msg, 'tool_call_id'):
            try:
                tool_results_map[msg.tool_call_id] = json.loads(msg.content)
            except Exception:
                tool_results_map[msg.tool_call_id] = msg.content

    for entry in tool_calls_log:
        entry['result'] = tool_results_map.get(entry['call_id'], {})

    iterations_used = result.get('iteration_count', 0)

    return {
        "category":              result.get("category", ""),
        "final_response":        result.get("final_response", ""),
        "is_safe":               result.get("is_safe", True),
        "pii_detected":          result.get("pii_detected", False),
        "iteration_count":       iterations_used,
        "raw_input":             result.get("raw_input", ""),
        "tool_calls_log":        tool_calls_log,
        "circuit_breaker_fired": iterations_used > MAX_ITERATIONS - 1,
        "iterations_used":       iterations_used,
        "max_iterations":        MAX_ITERATIONS,
    }


@app.post("/api/stream")
async def stream(req: TicketRequest):
    def generate():
        for node_name, snapshot in stream_ticket(req.ticket):
            payload = {
                "node":     node_name,
                "category": snapshot.get("category", ""),
                "response": snapshot.get("final_response", ""),
            }

            # Enrich agent_node events with tool call info and iteration data
            if node_name == 'agent_node':
                msgs = snapshot.get('messages', [])
                if msgs:
                    last_msg = msgs[-1]
                    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                        payload['tool_calls'] = [
                            {'name': tc['name'], 'args': tc['args']}
                            for tc in last_msg.tool_calls
                        ]
                payload['iteration']      = snapshot.get('iteration_count', 0)
                payload['max_iterations'] = MAX_ITERATIONS

            # Enrich tool_node events with tool results
            if node_name == 'tool_node':
                msgs = snapshot.get('messages', [])
                tool_results = []
                for msg in msgs:
                    if hasattr(msg, 'tool_call_id'):
                        tool_results.append({
                            'tool_name': getattr(msg, 'name', ''),
                            'content':   msg.content,
                        })
                if tool_results:
                    payload['tool_results'] = tool_results

            yield f"data: {json.dumps(payload)}\n\n"
        yield 'data: {"done": true}\n\n'

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/verify")
def verify():
    result = run_session_verification()
    return result


@app.get("/health")
def health():
    return {
        "status":         "ok",
        "session":        3,
        "tools":          len(TOOLS),
        "max_iterations": MAX_ITERATIONS,
    }


# Serve frontend
app.mount("/", StaticFiles(directory=".", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
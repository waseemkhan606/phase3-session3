# Enterprise AI Support Platform
### Session 3 of 12 — The ReAct Architecture

A production-grade AI customer support agent built with LangGraph and Gemini 2.5 Flash. Each session in this 12-part series adds a new capability layer. Session 3 introduces the full **ReAct (Reason + Act) loop** with three safety mechanisms that prevent the agent from looping forever, blowing up the context window, or repeating failed tool calls.

---

## What This Session Builds

Session 2 gave the agent tools. Session 3 teaches it to use them safely across multiple iterations.

| Layer | What it does |
|---|---|
| Circuit breaker | Hard-stops the loop at `MAX_ITERATIONS=5` and returns a graceful escalation |
| Context trim | Keeps only the original message + last 11 messages before every LLM call |
| Duplicate detection | Fingerprints every tool call; escalates immediately if the same call is attempted twice |

The fraud handler is also upgraded from a stub into a real tool-calling node using `check_fraud_signals`.

---

## Architecture

```
User Ticket
    │
    ▼
classify_node          ← Gemini classifies into: technical | billing | fraud | general
    │
    ├── technical ──┐
    ├── billing ────┼──► agent_node (ReAct loop)
    │               │       │
    │               │       ├── [has tool calls] ──► tool_node ──► agent_node (loop)
    │               │       └── [no tool calls]  ──► respond_node ──► END
    │               │
    ├── fraud ──────┴──► fraud_handler ──► END
    └── general ───────► general_handler ──► END
```

**agent_node** runs the ReAct loop with 3 safety layers on every iteration:
1. Check iteration count — escalate if over limit
2. Trim context to rolling window
3. Fingerprint tool calls — escalate if duplicate detected

---

## Project Structure

```
session3/
├── support_agent.py   # Agent logic, tools, graph definition
├── api.py             # FastAPI server (REST + SSE streaming)
├── index.html         # Single-file frontend
├── requirements.txt   # Python dependencies
└── .env               # API keys (not committed)
```

### `support_agent.py` — Key sections

| Section | What it contains |
|---|---|
| State schema (`SupportState`) | 17-field TypedDict spanning all 12 sessions |
| `MOCK_CRM` | 3 customer records with billing data |
| `MOCK_KB` | 6 knowledge base articles (api, login, billing, update, 2fa, sdk) |
| `MOCK_FRAUD_DB` | 3 fraud profiles (ACC-F001 high risk, ACC-F002 clean, ACC-F003 medium) |
| `get_customer_details` | CRM lookup tool — 3-layer: validate → lookup → filter noisy fields |
| `search_knowledge_base` | KB search tool — keyword matching across 6 articles |
| `check_fraud_signals` | Fraud tool — 3-layer: validate format → lookup → inject account_id |
| `AGENT_SYSTEM_PROMPT` | Module-level constant — not rebuilt on every iteration |
| `build_escalation_response` | Graceful escalation with tool findings summary + reference ID |
| `trim_context` | Rolling window — keeps `messages[0]` + last N messages |
| `get_tool_fingerprint` | `name::sorted_args_json` for duplicate detection |
| `agent_node` | Full ReAct loop with 3 safety layers |
| `fraud_handler` | Single-pass fraud analysis — calls `check_fraud_signals` directly |
| `build_graph()` | Wires 8 nodes into the LangGraph state machine |
| `run_session_verification()` | 5-check automated test suite |

---

## Setup

### Prerequisites
- Python 3.10+
- A Google AI Studio API key ([get one here](https://aistudio.google.com))

### Install

```bash
git clone https://github.com/waseemkhan606/phase3-session3
cd phase3-session3
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your-key-here
```

### Run the server

```bash
python api.py
```

Open `http://localhost:8000` in your browser.

### Run the CLI test harness

```bash
python support_agent.py
```

This runs 6 tests and then the full 5-check verification suite.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `GET /health` | GET | Returns session info and tool count |
| `POST /api/run` | POST | Runs a ticket synchronously, returns full result |
| `POST /api/stream` | POST | Streams graph execution as SSE events |
| `POST /api/verify` | POST | Runs the automated verification suite |

### `POST /api/run` — Request

```json
{ "ticket": "My account C-1002 is past due." }
```

### `POST /api/run` — Response

```json
{
  "category": "billing",
  "final_response": "Your account C-1002 has an outstanding balance of $998...",
  "is_safe": true,
  "pii_detected": false,
  "iterations_used": 2,
  "max_iterations": 5,
  "circuit_breaker_fired": false,
  "tool_calls_log": [
    {
      "tool_name": "get_customer_details",
      "args": { "customer_id": "C-1002" },
      "result": { "name": "Arjun Mehta", "billing_status": "Past Due", ... }
    }
  ]
}
```

### SSE Stream events

Each `data:` line is a JSON object. Agent node events include iteration info:

```json
{ "node": "agent_node", "iteration": 2, "max_iterations": 5, "tool_calls": [...] }
{ "node": "tool_node",  "tool_results": [...] }
{ "node": "respond_node", "response": "..." }
{ "done": true }
```

---

## Tools

### `get_customer_details(customer_id)`
Fetches billing status, subscription tier, last payment, and recent transactions from the mock CRM.
- Format: `C-XXXX` e.g. `C-1001`
- Returns error dict if not found or wrong format

### `search_knowledge_base(query)`
Keyword-matches query against 6 internal articles (api, login, billing, update, 2fa, sdk).
- Returns matched articles or fallback guidance

### `check_fraud_signals(account_id)`
Looks up fraud risk score, flagged behavioral patterns, and recommended action.
- Format: `ACC-FXXX` e.g. `ACC-F001`
- Returns risk score (0.0–1.0): >0.7 high, 0.4–0.7 medium, <0.4 low

---

## Test Accounts

### CRM accounts

| ID | Name | Status | Tier | Balance |
|---|---|---|---|---|
| C-1001 | Priya Sharma | Active | Enterprise | $0 |
| C-1002 | Arjun Mehta | Past Due | Pro | $998 |
| C-1003 | Kavya Nair | Active | Starter | $0 |

### Fraud accounts

| ID | Risk Score | Recommendation |
|---|---|---|
| ACC-F001 | 0.91 (high) | freeze_account |
| ACC-F002 | 0.23 (low) | no_action |
| ACC-F003 | 0.67 (medium) | manual_review |

---

## Sample Tickets to Try

```
# Multi-tool (billing + technical)
Account C-1002 is past due and I cannot log in after the recent update.

# Fraud — high risk account
I did not authorize charges on account ACC-F001.

# Fraud — unknown account (error handling)
Suspicious activity on account ACC-F999.

# Clean billing lookup
What is my current subscription plan? Account C-1001.

# Technical — SDK issue
Getting 401 auth errors on every API call after SDK v3 update.
```

---

## Verification

Click **Session 3 — Verification Test** in the UI, or hit the API directly:

```bash
curl -s -X POST http://localhost:8000/api/verify | python3 -m json.tool
```

All 5 checks must pass before advancing to Session 4:

- Multi-tool ticket → 2 distinct tools called
- Circuit breaker fires at `MAX_ITERATIONS=1`, returns escalation
- Fraud valid ID → `check_fraud_signals` called
- Fraud invalid ID → error handled gracefully, no crash
- Context trim → activates on 14-message history

---

## Session Roadmap

| Session | Title | Status |
|---|---|---|
| 1 | The Blueprint | ✅ Complete |
| 2 | Tool Binding & Execution | ✅ Complete |
| **3** | **The ReAct Architecture** | ✅ **Complete** |
| 4 | Persistence & Threading | 🔒 Next |
| 5 | Memory Compression | 🔒 |
| 6 | The Shield (PII + Injection) | 🔒 |
| 7–12 | Multi-agent, Human-in-loop, Audit | 🔒 |

---

## What Session 4 Adds

- SQLite checkpointer via `langgraph.checkpoint.sqlite`
- `thread_id` support in `run_ticket()` and `stream_ticket()`
- `GET /api/threads` — list active thread IDs
- `GET /api/history/{tid}` — full checkpoint history
- Conversation history panel in the UI
- All Session 3 safety layers remain unchanged and permanent
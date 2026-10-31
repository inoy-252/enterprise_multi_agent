# Enterprise Multi-Agent Collaboration Engine

A production-grade multi-agent orchestration architecture built with LangGraph, implementing the Supervisor-Worker design pattern, domain-isolated tool execution, inter-agent causal dependency routing, fan-in executive synthesis, and durable multi-turn state persistence with SQLite.

---

## 1. Architectural Overview

Monolithic single-agent systems degrade as tool definitions and domain responsibilities scale. Large toolsets cause prompt dilution, tool selection collisions, and persona inconsistency. 

This repository replaces monolithic execution with a decoupled **Supervisor-Worker Pattern**:
- **Supervisor (Cognitive Orchestrator):** A pure router with zero direct tool access. It evaluates conversation state and delegates tasks using structured schema outputs.
- **Specialist Workers:** Domain-bounded agents operating under the Principle of Least Privilege (PoLP). Each worker possesses only the tools and persona required for its domain.
- **Synthesizer (Fan-In Aggregator):** Consolidates findings from all departments into a structured, fully grounded executive briefing once all requirements are satisfied.
- **State Checkpointing:** Backed by `SqliteSaver`, allowing conversational turns to persist across processes and enabling delta-processing on subsequent requests.

---

## 2. System Topology

```
                       [START]
                          |
                          v
                +--------------------+
                |     SUPERVISOR     |<-------------+
                | (Cognitive Router) |              |
                +---------+----------+              |
                          |                         |
               (Structured Routing)                 |
                          |                         |
         +----------------+----------------+        |
         |                |                |        |
         v                v                v        |
+----------------+ +----------------+ +----------------+
|  travel_worker | |  policy_worker | | finance_worker |
| (Weather/Dest) | | (Rules/Limits) | |  (Budget Math) |
+--------+-------+ +--------+-------+ +--------+-------+
         |                  |                  |
         +------------------+------------------+
                            |
                   (Tagged HumanMessage)
                            |
                            v
                   [Back to Supervisor]
                            |
                     (next == FINISH)
                            |
                            v
                 +--------------------+
                 |    SYNTHESIZER     |
                 | (Executive Report) |
                 +---------+----------+
                            |
                            v
                          [END]
```

---

## 3. Core Engineering Principles

### Domain-Isolated Toolsets (PoLP)
Tools are partitioned strictly across departments to prevent tool collisions and eliminate unauthorized execution:
- `travel_worker`: `get_current_weather`
- `policy_worker`: `search_company_policy` (with stop-word filtering and keyword scoring)
- `finance_worker`: `calculate_trip_budget`

### Scope Boundary Guardrails
Workers are configured with negative persona constraints forbidding them from answering questions outside their operational domain. This prevents unconstrained workers from hallucinating answers and prematurely terminating the supervisor loop.

### Inter-Agent Causal Dependencies
When a request contains cross-departmental dependencies (e.g., calculating a trip budget based on internal per-diem limits rather than user-provided figures):
1. The Supervisor identifies the dependency and routes to `policy_worker` first.
2. `policy_worker` queries company runbooks and establishes the allowance rate.
3. `finance_worker` inspects the conversation state, extracts the policy figure, and parameterizes its calculation tool.

### Fan-In Executive Synthesis
Rather than terminating immediately when workers finish, the graph routes `FINISH` to `synthesizer_node`. This node aggregates all specialist outputs into a cohesive executive report with standardized sections:
- Executive Summary
- Destination Logistics
- Compliance and Policy Verification
- Audited Financial Breakdown
- Actionable Next Steps

### Multi-Turn Delta Processing (`SqliteSaver`)
State transitions are checkpointer-backed in SQLite and partitioned by `thread_id`. On subsequent turns (e.g., modifying trip length from 4 to 7 days), the Supervisor checks state memory, skips already-verified logistics and policy lookups, and delegates exclusively to `finance_worker` for delta recalculation.

---

## 4. Repository Structure

```
enterprise_multi_agent/
├── multi_agent_supervisor.py   # Complete multi-agent state machine and test runner
├── requirements.txt            # Pinned dependencies
├── .gitignore                  # Git hygiene and secret protection
└── README.md                   # System documentation
```

---

## 5. Quickstart

### Prerequisites
- Python 3.10+
- Google Gemini API Key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/inoy-252/enterprise_multi_agent.git
cd enterprise_multi_agent
```

2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### Execution

Run the multi-turn persistent orchestration demonstration:
```bash
python multi_agent_supervisor.py
```

---

## 6. Execution Verification

The orchestration test validates two consecutive turns under `thread_id="exec-trip-tokyo-101"`:

- **Turn 1 (Initial Multi-Department Planning):**
  - Input: Plan 4-day Tokyo trip ($1,200 flights, company per-diem rate).
  - Sequence: `travel_worker` (16°C clear) -> `policy_worker` (SEC-401: $100/day) -> `finance_worker` ($1,600.00 total) -> `synthesizer` (Executive Briefing).
- **Turn 2 (Incremental Modification):**
  - Input: Extend stay from 4 days to 7 days.
  - Sequence: Supervisor recognizes unchanged weather/policy context, skips redundant workers, routes directly to `finance_worker` (7 days @ $100 = $1,900.00 total) -> `synthesizer` (Updated Briefing).

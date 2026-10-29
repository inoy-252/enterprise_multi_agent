# Project 5 — Chunk 1: The Supervisor-Worker Multi-Agent Architecture

## 1. The Core Problem: The Single-Agent Ceiling
* **Context Dilution & Tool Bloat:** When a monolithic agent is given 20+ tools and a massive multi-page system prompt, attention degrades and the model selects incorrect tools.
* **Persona Conflict:** A creative travel planner persona directly conflicts with a conservative financial auditor or compliance guard.
* **Blast Radius:** If a single agent fails in a 10-step chain, debugging which tool or prompt instruction caused the hallucination is extremely difficult.

---

## 2. The Solution: Hub-and-Spoke Supervisor-Worker Pattern
* **The Supervisor (Hub / Orchestrator):** Has ZERO domain tools. Its sole responsibility is cognitive routing: analyzing conversational state, deciding which specialist acts next, or choosing `FINISH`.
* **The Workers (Spokes / Specialists):**
  1. `travel_worker`: Equipped *only* with destination and weather tools (`get_current_weather`).
  2. `finance_worker`: Equipped *only* with budget calculation and math tools (`calculate_trip_budget`).
  3. `policy_worker`: Equipped *only* with enterprise policy retrieval tools (`search_company_policy`).
* **State Handoff:** Workers convert their final conclusions to `HumanMessage(content=..., name="worker_name")` so the Supervisor clearly distinguishes between its own thoughts and specialist reports.

---

## 3. The Over-Eager Worker Trap & Persona Guardrails
* **The Failure Mode:** Without strict boundary instructions, an LLM worker tries to answer all questions in the prompt, hallucinating numbers outside its domain tools.
* **The Domino Effect:** The Supervisor sees the hallucinated answers in state and prematurely routes to `FINISH`, starving other specialists of execution.
* **The Fix:** Explicitly instruct workers with negative constraints (*"CRITICAL: Do NOT calculate budgets or search policies. ONLY answer weather."*).

---

## 4. Multi-Agent State Machine (`EnterpriseState`)
```python
class EnterpriseState(MessagesState):
    next: str

class RouteResponse(BaseModel):
    next: Literal["travel_worker", "finance_worker", "policy_worker", "FINISH"]
```
* **Cyclical Routing Loop:**
  `START` -> `supervisor` -> `conditional_edge(state['next'])` -> `worker` -> `supervisor` -> ... -> `FINISH` -> `END`.

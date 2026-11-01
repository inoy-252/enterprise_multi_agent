"""
Project 5: Enterprise Multi-Agent Collaboration
Architecture: The Supervisor-Worker Pattern
Phase 1: Defining Specialist Workers and Domain-Isolated Tools
"""

import sqlite3
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


@tool
def get_current_weather(city: str) -> str:
    """Fetch the current weather forecast for a destination city."""
    city_lower = city.strip().lower()
    mock_weather = {
        "tokyo": "16°C, Clear skies with a gentle breeze. Ideal travel conditions.",
        "london": "11°C, Overcast with light drizzle. Umbrella recommended.",
        "new york": "22°C, Sunny and mild.",
        "paris": "18°C, Partly cloudy.",
    }
    return mock_weather.get(
        city_lower, f"15°C, Moderate conditions reported for {city}."
    )


@tool
def calculate_trip_budget(daily_allowance: float, days: int, flight_cost: float) -> str:
    """Calculate the exact total cost of a business trip based on per-diem and flights."""
    total_per_diem = daily_allowance * days
    grand_total = total_per_diem + flight_cost
    return (
        f"Financial Audit Breakdown:\n"
        f"  - Flights: {flight_cost:,.2f} USD\n"
        f"  - Per Diem ({days} days @ {daily_allowance:,.2f}/day): {total_per_diem:,.2f} USD\n"
        f"  - Total Projected Expense: {grand_total:,.2f} USD"
    )


COMPANY_POLICIES = {
    "meal_allowance": (
        "Policy Rule SEC-401 (Meals & Per Diem): Tokyo max per diem is $100 USD/day. "
        "London and New York max is $120 USD/day. All other destinations cap at $75 USD/day."
    ),
    "flight_policy": (
        "Policy Rule SEC-402 (Flight Bookings): Flights under 6 hours must be Economy. "
        "Flights exceeding 6 hours allow Premium Economy. Business class requires VP sign-off."
    ),
    "lodging_caps": (
        "Policy Rule SEC-403 (Lodging): Standard maximum hotel allowance is $250 USD/night inclusive of tax."
    ),
    "expense_threshold": (
        "Corporate General Travel Rule: Any trip whose grand total exceeds $2,500 USD "
        "requires explicit written approval from the Department Finance Lead."
    ),
}


@tool
def search_company_policy(query: str) -> str:
    """Search internal corporate travel policies, per-diem limits, and approval rules."""
    stop_words = {
        "what",
        "is",
        "the",
        "for",
        "in",
        "a",
        "an",
        "to",
        "of",
        "and",
        "our",
        "company",
        "policy",
        "travel",
        "rules",
    }
    query_words = [w for w in query.lower().split() if w not in stop_words]
    scored_matches = []
    for category, text in COMPANY_POLICIES.items():
        score = sum(
            1 for w in query_words if w in text.lower() or w in category.lower()
        )
        if score > 0:
            scored_matches.append((score, f"[{category.upper()}]: {text}"))

    scored_matches.sort(key=lambda x: x[0], reverse=True)
    if scored_matches:
        return "\n\n".join(rule for _, rule in scored_matches)

    return "No matching corporate policy found for this query."


def get_trimmed_history(messages: list) -> list:
    """Retains system prompt and caps conversation history to the last 4 turns."""
    return list(
        trim_messages(
            messages,
            max_tokens=4,
            token_counter=len,
            strategy="last",
            start_on="human",
            include_system=True,
            allow_partial=False,
        )
    )


travel_agent = create_react_agent(
    llm,
    tools=[get_current_weather],
    prompt=(
        "You are a specialized Corporate Travel Planning agent.\n"
        "Your ONLY job is to research destination weather and logistical travel conditions.\n"
        "CRITICAL RULE: Do NOT calculate budgets, and do NOT answer company policy or per-diem questions.\n"
        "If the user asked about budgets or policies, IGNORE those parts—other specialists will handle them.\n"
        "Only report destination weather using your tool."
    ),
)


def travel_node(state: MessagesState) -> dict:
    """Executes the travel agent and returns its findings tagged with its name."""
    trimmed_state = {"messages": get_trimmed_history(state["messages"])}
    result = travel_agent.invoke(trimmed_state)
    last_message = result["messages"][-1]
    return {
        "messages": [HumanMessage(content=last_message.content, name="travel_worker")]
    }


finance_agent = create_react_agent(
    llm,
    tools=[calculate_trip_budget],
    prompt=(
        "You are a specialized Corporate Financial Auditor agent.\n"
        "Your ONLY job is to calculate trip budgets and audit projected expenses using your tool.\n"
        "CROSS-AGENT COLLABORATION RULE:\n"
        "- Carefully inspect the conversation history. If 'policy_worker' has retrieved a daily per-diem "
        "or meal allowance limit, extract that exact amount as your daily_allowance.\n"
        "- If flight costs are specified by the user, use them in flight_cost.\n"
        "CRITICAL RULE: Do NOT answer weather questions or search policies. "
        "Only calculate financial totals using your calculate_trip_budget tool."
    ),
)


def finance_node(state: MessagesState) -> dict:
    """Executes the finance agent and returns its findings tagged with its name."""
    trimmed_state = {"messages": get_trimmed_history(state["messages"])}
    result = finance_agent.invoke(trimmed_state)
    last_message = result["messages"][-1]
    return {
        "messages": [HumanMessage(content=last_message.content, name="finance_worker")]
    }


policy_agent = create_react_agent(
    llm,
    tools=[search_company_policy],
    prompt=(
        "You are a specialized Corporate Travel Compliance Guard.\n"
        "Your ONLY job is to verify company travel guidelines and per-diem caps using your tool.\n"
        "CRITICAL RULE: Do NOT answer weather questions and do NOT calculate total trip budgets.\n"
        "Only look up corporate policies using search_company_policy."
        "Do NOT write an executive summary or next steps. "
        "Report ONLY the raw financial audit figures (flights, per-diem days, and grand total)."
    ),
)


def policy_node(state: MessagesState) -> dict:
    """Executes the policy agent and returns its findings tagged with its name."""
    trimmed_state = {"messages": get_trimmed_history(state["messages"])}
    result = policy_agent.invoke(trimmed_state)
    last_message = result["messages"][-1]
    return {
        "messages": [HumanMessage(content=last_message.content, name="policy_worker")]
    }


class EnterpriseState(MessagesState):
    """Global multi-agent state tracking both message history and active routing."""

    next: str
    trip_profile: dict


class RouteResponse(BaseModel):
    """Deterministic routing schema for worker handoff."""

    next: Literal["travel_worker", "finance_worker", "policy_worker", "FINISH"]


SUPERVISOR_PROMPT = (
    "You are the executive supervisor managing a team of specialized workers:\n"
    "1. 'travel_worker': Researches destination weather forecasts and travel logistics.\n"
    "2. 'policy_worker': Checks company travel rules, per-diem limits, and approval policies.\n"
    "3. 'finance_worker': Calculates trip budgets, per-diem totals, and expense math.\n\n"
    "Order of Operations & Delegation Rules:\n"
    "- If destination weather is requested and hasn't been provided, select 'travel_worker'.\n"
    "- If company policies/limits must be checked and haven't been provided, select 'policy_worker'.\n"
    "- DEPENDENCY RULE: If the user needs a budget calculated that depends on company per-diem policies, "
    "ensure 'policy_worker' runs FIRST so 'finance_worker' has the necessary figures.\n"
    "- Once policy data and trip parameters are available, select 'finance_worker' to perform the calculation.\n"
    "- Once ALL requested aspects of the user inquiry have been fully answered by the specialists, select 'FINISH'."
)

supervisor_chain = llm.with_structured_output(RouteResponse)


def supervisor_node(state: EnterpriseState) -> dict:
    """Evaluates conversation state and chooses the next worker or FINISH."""
    trimmed = get_trimmed_history(state["messages"])

    profile = state.get("trip_profile", {})

    profile_context = (
        f"\n\nCURRENT TRIP PROFILE (VERIFIED FACTS):\n{profile}" if profile else ""
    )
    system_instruction = SystemMessage(content=SUPERVISOR_PROMPT + profile_context)

    messages = [system_instruction] + trimmed
    decision = supervisor_chain.invoke(messages)

    if isinstance(decision, dict):
        next_step = decision.get("next", "FINISH")
    else:
        next_step = getattr(decision, "next", "FINISH")
    return {"next": next_step}


SYNTHESIZER_PROMPT = (
    "You are the Executive Enterprise Concierge.\n"
    "Your specialist workers (Travel, Compliance/Policy, and Finance) have completed their tasks.\n"
    "Carefully review the entire dialogue history and synthesize a cohesive, professional "
    "executive briefing for the user.\n\n"
    "Formatting Requirements:\n"
    "1. Start with an executive summary paragraph.\n"
    "2. Provide clear sections for Destination Logistics, Compliance & Policy, and Financial Audit.\n"
    "3. Conclude with actionable next steps.\n"
    "4. Ground all numbers strictly in the reports provided by the specialist workers. Do NOT invent new figures."
)


def synthesizer_node(state: EnterpriseState) -> dict:
    """Consolidates findings from all specialist workers into a unified final briefing."""
    trimmed = get_trimmed_history(state["messages"])
    profile = state.get("trip_profile", {})
    profile_context = f"\n\nVERIFIED TRIP PROFILE:\n{profile}" if profile else ""

    messages = [SystemMessage(content=SYNTHESIZER_PROMPT + profile_context)] + trimmed
    response = llm.invoke(messages)
    return {
        "messages": [AIMessage(content=response.content, name="executive_briefing")]
    }


builder = StateGraph(EnterpriseState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("travel_worker", travel_node)
builder.add_node("finance_worker", finance_node)
builder.add_node("policy_worker", policy_node)
builder.add_node("synthesizer", synthesizer_node)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "travel_worker": "travel_worker",
        "finance_worker": "finance_worker",
        "policy_worker": "policy_worker",
        "FINISH": "synthesizer",
    },
)

builder.add_edge("travel_worker", "supervisor")
builder.add_edge("finance_worker", "supervisor")
builder.add_edge("policy_worker", "supervisor")
builder.add_edge("synthesizer", END)

DB_PATH = "multi_agent_memory.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)
graph = builder.compile(checkpointer=checkpointer)


def get_clean_text(content) -> str:
    """Helper to safely extract pure string text from Gemini message payloads."""
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return str(content)


if __name__ == "__main__":
    print("=" * 70)
    print("ENTERPRISE MULTI-AGENT COLLABORATION: MULTI-TURN PERSISTENCE")
    print("=" * 70)

    session_config = {"configurable": {"thread_id": "exec-trip-tokyo-101"}}

    def stream_turn(prompt: str):
        """Helper to run conversation turn and stream active node handoffs."""
        print(f"\nUSER REQUEST:\n{prompt}\n")
        print("-" * 70)

        turn_state: EnterpriseState = {
            "messages": [HumanMessage(content=prompt)],
            "next": "",
            "trip_profile": {},
        }

        for step_chunk in graph.stream(
            turn_state, config=session_config, stream_mode="updates"
        ):
            for node_name, state_update in step_chunk.items():
                print(f"\n[ACTIVE NODE: {node_name.upper()}]")
                if node_name == "supervisor":
                    print(
                        f"Supervisor Decision: Next worker -> {state_update.get('next')}"
                    )
                elif node_name == "synthesizer":
                    last_msg = state_update["messages"][-1]
                    clean_text = get_clean_text(last_msg.content)
                    print(f"\nFINAL EXECUTIVE BRIEFING:\n{clean_text}\n")
                else:
                    last_msg = state_update["messages"][-1]
                    clean_text = get_clean_text(last_msg.content)
                    print(f"Worker Output ({last_msg.name}):\n{clean_text}")

    print("\n>>> EXECUTING TURN 1 (Initial Planning) <<<")
    turn_1_query = (
        "I need to plan a 4-day corporate business trip to Tokyo. "
        "Check the weather in Tokyo, look up our corporate meal allowance policy for Tokyo, "
        "and calculate my total projected budget assuming flights cost 1200 USD and using whatever "
        "daily per-diem rate our company policy allows."
    )

    stream_turn(turn_1_query)

    print("\n" + "=" * 70)
    print(" >>> EXECUTING TURN 2 (Delta Modification via Checkpoint Memory) <<<")
    print("=" * 70)
    turn_2_query = (
        "Good news! The conference got extended. I will be in Tokyo for 7 days instead of 4 days. "
        "Please recalculate my total budget."
    )
    stream_turn(turn_2_query)

    print("\n" + "=" * 70)
    print(" MULTI-TURN PERSISTENT COLLABORATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

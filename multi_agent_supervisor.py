"""
Project 5: Enterprise Multi-Agent Collaboration
Architecture: The Supervisor-Worker Pattern
Phase 1: Defining Specialist Workers and Domain-Isolated Tools
"""

from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
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
    result = travel_agent.invoke(state)
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
        "CRITICAL RULE: Do NOT answer weather questions or search policies.\n"
        "Only calculate financial totals using your calculate_trip_budget tool."
    ),
)


def finance_node(state: MessagesState) -> dict:
    """Executes the finance agent and returns its findings tagged with its name."""
    result = finance_agent.invoke(state)
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
    ),
)


def policy_node(state: MessagesState) -> dict:
    """Executes the policy agent and returns its findings tagged with its name."""
    result = policy_agent.invoke(state)
    last_message = result["messages"][-1]
    return {
        "messages": [HumanMessage(content=last_message.content, name="policy_worker")]
    }


class EnterpriseState(MessagesState):
    """Global multi-agent state tracking both message history and active routing."""

    next: str


class RouteResponse(BaseModel):
    """Deterministic routing schema for worker handoff."""

    next: Literal["travel_worker", "finance_worker", "policy_worker", "FINISH"]


SUPERVISOR_PROMPT = (
    "You are the executive supervisor managing a team of specialized workers:\n"
    "1. 'travel_worker': Researches destination weather forecasts and travel logistics.\n"
    "2. 'finance_worker': Calculates trip budgets, per-diem totals, and expense math.\n"
    "3. 'policy_worker': Checks company travel rules, per-diem limits, and approval policies.\n\n"
    "Carefully review the user prompt and the conversation history:\n"
    "- If destination weather is needed and hasn't been provided, select 'travel_worker'.\n"
    "- If company policies/limits must be checked and haven't been provided, select 'policy_worker'.\n"
    "- If budget or cost math needs to be computed, select 'finance_worker'.\n"
    "- Once ALL requested aspects of the user's inquiry have been fully answered by the workers, select 'FINISH'."
)

supervisor_chain = llm.with_structured_output(RouteResponse)


def supervisor_node(state: EnterpriseState) -> dict:
    """Evaluates conversation state and chooses the next worker or FINISH."""
    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
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
    messages = [SystemMessage(content=SYNTHESIZER_PROMPT)] + state["messages"]
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
graph = builder.compile()


def get_clean_text(content) -> str:
    """Helper to safely extract pure string text from Gemini message payloads."""
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return str(content)


if __name__ == "__main__":
    print("=" * 70)
    print("ENTERPRISE MULTI-AGENT COLLABORATION: LIVE ORCHESTRATION")
    print("=" * 70)
    user_query = (
        "I need to plan a 4-day corporate business trip to Tokyo. "
        "Check the weather in Tokyo, look up our corporate meal allowance policy for Tokyo, "
        "and calculate my total projected budget assuming flights cost 1200 USD and daily per-diem is 100 USD."
    )
    print(f"\nUSER REQUEST:\n{user_query}\n")
    print("-" * 70)
    initial_state: EnterpriseState = {
        "messages": [HumanMessage(content=user_query)],
        "next": "",
    }

    for step_chunk in graph.stream(initial_state, stream_mode="updates"):
        for node_name, state_update in step_chunk.items():
            print(f"\n[ACTIVE NODE: {node_name.upper()}]")

            if node_name == "supervisor":
                print(f"Supervisor Decision: Next worker -> {state_update.get('next')}")
            elif node_name == "synthesizer":
                last_msg = state_update["messages"][-1]
                clean_text = get_clean_text(last_msg.content)
                print(f"\nFINAL EXECUTIVE BRIEFING:\n{clean_text}\n")
            else:
                last_msg = state_update["messages"][-1]
                clean_text = get_clean_text(last_msg.content)
                print(f"Worker Output ({last_msg.name}):\n{clean_text}")
    print("MULTI-AGENT ORCHESTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

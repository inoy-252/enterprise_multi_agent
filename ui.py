import time
import uuid

import streamlit as st
from langchain_core.runnables import RunnableConfig

from multi_agent_supervisor import EnterpriseState, HumanMessage, get_clean_text, graph
from semantic_cache import SemanticCache

st.set_page_config(page_title="Enterprise Multi-Agent Swarm", layout="wide")


@st.cache_resource
def load_cache():
    return SemanticCache(db_path="semantic_cache.db", threshold=0.90)


cache = load_cache()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"exex-{uuid.uuid4().hex[:6]}"

with st.sidebar:
    st.header("System Telemetry")
    st.subheader("Streamlit Cache")
    stats = cache.get_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Cached Queries", stats["total_entries"])
    with col2:
        st.metric("Cache Hits", stats["total_hits"])
    st.caption(f"Similarity Threshold: **{stats['similarity_threshold'] * 100:.0f}%**")
    st.markdown("---")

    st.subheader("Conversation State")
    current_thread = st.text_input(
        "Active Thread ID",
        value=st.session_state.thread_id,
        help="Conversation context is partitioned in SQLite by this thread ID.",
    )
    if current_thread != st.session_state.thread_id:
        st.session_state.thread_id = current_thread
        st.session_state.messages = []
        st.rerun()

    if st.button("Reset Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = f"exec-{uuid.uuid4().hex[:6]}"
        st.rerun()

    st.markdown("---")

    st.subheader("Active Agent Swarm")
    st.markdown("""
    - ** `travel_worker`**: Logistics & Weather
    - ** `policy_worker`**: Corporate Runbooks (SEC-401)
    - ** `finance_worker`**: Audited Per-Diem Math
    - ** `synthesizer`**: Executive Briefing Aggregator
            """)

st.title("Enterprise Multi-Agent Collaboration Engine")
st.markdown(
    "Autonomous LangGraph Supervisor-Worker architecture with **Semantic Caching** and **SQLite State Persistence**."
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("cached"):
            st.caption(
                f" Served from Semantic Cache ({msg.get('similarity', 0) * 100:.1f}% match) in {msg.get('latency', 0)}s · $0.00 LLM Cost"
            )
        elif "latency" in msg:
            st.caption(f"Full multi-agent swarm execution in {msg['latency']}s")

if prompt := st.chat_input(
    "E.g., Plan a 4-day corporate trip to Tokyo with $1200 flight budget..."
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        start_time = time.time()

        hit = cache.lookup(prompt)
        if hit:
            latency = round(time.time() - start_time, 4)
            st.markdown(hit["response"])
            st.success(
                f" **Semantic Cache Hit!** Intent match: **{hit['similarity'] * 100:.1f}%** · Served in **{latency}s** · **$0.00 Marginal Cost**"
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": hit["response"],
                    "cached": True,
                    "similarity": hit["similarity"],
                    "latency": latency,
                }
            )
            st.rerun()
        session_config: RunnableConfig = {
            "configurable": {"thread_id": st.session_state.thread_id}
        }
        turn_state: EnterpriseState = {
            "messages": [HumanMessage(content=prompt)],
            "next": "",
            "trip_profile": {},
        }
        final_briefing = ""
        with st.status("**Agent Swarm Collaborating...**", expanded=True) as status:
            try:
                for step_chunk in graph.stream(
                    turn_state, config=session_config, stream_mode="updates"
                ):
                    for node_name, state_update in step_chunk.items():
                        if node_name == "supervisor":
                            next_target = state_update.get("next")
                            st.write(
                                f"**Supervisor:** Routing control flow to `{next_target}`"
                            )
                        elif node_name == "synthesizer":
                            last_msg = state_update["messages"][-1]
                            final_briefing = get_clean_text(last_msg.content)
                            st.write(
                                "**Synthesizer:** Compiling domain findings into executive briefing..."
                            )
                        else:
                            st.write(
                                f"**{node_name}:** Executed assigned domain tools successfully."
                            )

                status.update(
                    label="**Swarm Execution Complete!**",
                    state="complete",
                    expanded=False,
                )
            except Exception as e:
                status.update(
                    label="**Swarm Execution Failed**", state="error", expanded=True
                )
                st.error(f"Multi-Agent execution error: {e}")
        if final_briefing:
            st.markdown(final_briefing)
            cache.store(prompt, final_briefing)
            latency = round(time.time() - start_time, 2)
            st.caption(f"Full Multi-Agent swarm execution completed in {latency}s")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_briefing,
                    "cached": False,
                    "latency": latency,
                }
            )
            st.rerun()

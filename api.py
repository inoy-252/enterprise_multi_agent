import time

from fastapi import FastAPI, HTTPException
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from multi_agent_supervisor import EnterpriseState, HumanMessage, get_clean_text, graph
from qdrant_manager import QdrantSemanticCache

app = FastAPI(
    title="Enterprise Multi-Agent Collaboration API",
    description="Production-grade Supervisor-Worker orchestration with Semantic Caching and State Persistence",
    version="1.0.0",
)

cache = QdrantSemanticCache(collection_name="enterprise_cache", threshold=0.85)


class ChatRequest(BaseModel):
    message: str = Field(
        ..., description="The user query or prompt for the multi-agent system"
    )
    thread_id: str = Field(
        default="default-thread",
        description="Unique session thread ID for conversation memory",
    )


class ChatResponse(BaseModel):
    response: str
    cached: bool
    similarity: float | None = None
    matched_query: str | None = None
    latency_seconds: float
    thread_id: str


@app.get("/health")
def health_check():
    """Healthcheck endpoint for container orchestration and monitoring."""
    return {"status": "healthy", "service": "enterprise-multi-agent"}


@app.get("/cache/stats")
def get_cache_stats():
    """Returns analytics on semantic cache entries, hits, and configuration."""
    return cache.get_stats()


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Primary multi-agent conversation endpoint:
    1. Checks semantic cache for >=90% intent match (instant return).
    2. If cache miss, executes LangGraph multi-agent orchestration.
    3. Caches final executive synthesis for future requests.
    """
    start_time = time.time()

    hit = cache.lookup(request.message)
    if hit:
        latency = round(time.time() - start_time, 4)
        return ChatResponse(
            response=hit["response"],
            cached=True,
            similarity=hit["similarity"],
            matched_query=hit["query"],
            latency_seconds=latency,
            thread_id=request.thread_id,
        )
    session_config: RunnableConfig = {"configurable": {"thread_id": request.thread_id}}
    turn_state: EnterpriseState = {
        "messages": [HumanMessage(content=request.message)],
        "next": "",
        "trip_profile": {},
    }
    final_briefing = ""
    try:
        for step_chunk in graph.stream(
            turn_state, config=session_config, stream_mode="updates"
        ):
            if "synthesizer" in step_chunk:
                last_msg = step_chunk["synthesizer"]["messages"][-1]
                final_briefing = get_clean_text(last_msg.content)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Multi-Agent execution failed: {e}"
        )
    if not final_briefing:
        raise HTTPException(
            status_code=500,
            detail="Synthesiser failed to generate an executive report.",
        )

    cache.store(request.message, final_briefing)

    latency = round(time.time() - start_time, 4)
    return ChatResponse(
        response=final_briefing,
        cached=False,
        similarity=None,
        matched_query=None,
        latency_seconds=latency,
        thread_id=request.thread_id,
    )


if __name__ == "__main__":
    import uvicorn

    print("\nStarting Enterprise Multi-Agent API server on http://0.0.0.0:8000...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

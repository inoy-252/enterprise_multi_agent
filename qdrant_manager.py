import os
import time
import uuid

from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv()


class QdrantSemanticCache:
    """
    Semantic caching and vector retrieval engine using Qdrant.
    Uses Google Gemini embeddings and HNSW graph indexing for sub-millisecond search.
    """

    def __init__(
        self,
        collection_name: str = "semantic_cache",
        db_path: str = "./qdrant_storage",
        threshold: float = 0.90,
        vector_size: int = 768,
    ):
        self.collection_name = collection_name
        self.db_path = db_path
        self.threshold = threshold
        self.vector_size = vector_size
        self.client = QdrantClient(path=self.db_path)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or .env file.")
        self._genai_client = genai.Client(api_key=api_key)

        self._init_db()

    def _init_db(self):
        """Ensures the Qdrant collection exists with HNSW cosine indexing."""
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def _get_embedding(self, text: str) -> list[float]:
        """Generates a 768-dimensional dense vector using Gemini with MRL truncation."""
        response = self._genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config={"output_dimensionality": self.vector_size},
        )
        if not response.embeddings or not response.embeddings[0].values:
            raise ValueError("Gemini API failed to return embedding vectors.")
        return response.embeddings[0].values

    def store(self, query: str, response: str, metadata: dict | None = None) -> str:
        """Embeds and saves a query-response pair as a vector point in Qdrant."""
        vector = self._get_embedding(query)
        point_id = str(uuid.uuid4())

        payload = {
            "query": query,
            "response": response,
            "created_at": time.time(),
        }
        if metadata:
            clean_metadata = {
                k: (v.strip().lower() if isinstance(v, str) else v)
                for k, v in metadata.items()
            }
            payload.update(clean_metadata)

        point = models.PointStruct(
            id=point_id,
            vector=vector,
            payload=payload,
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )
        return point_id

    def lookup(
        self,
        query: str,
        filter_metadata: dict | None = None,
    ) -> dict | None:
        """
        Performs vector similarity search against cached points.
        Returns the closest matching cached response if similarity >= threshold, else None.
        """
        query_vector = self._get_embedding(query)

        query_filter = None
        if filter_metadata:
            conditions: list[models.Condition] = [
                models.FieldCondition(
                    key=k,
                    match=models.MatchValue(value=v.strip().lower() if isinstance(v, str) else v),
                )
                for k, v in filter_metadata.items()
            ]
            query_filter = models.Filter(must=conditions)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=1,
            score_threshold=self.threshold,
        )

        if results.points:
            hit = results.points[0]
            payload = hit.payload or {}
            return {
                "id": hit.id,
                "query": payload.get("query"),
                "response": payload.get("response"),
                "similarity": round(float(hit.score), 4),
                "metadata": {
                    k: v for k, v in payload.items() if k not in ("query", "response")
                },
            }
        return None

    def get_stats(self) -> dict:
        """Returns real-time telemetry and collection health statistics."""
        info = self.client.get_collection(collection_name=self.collection_name)
        return {
            "collection_name": self.collection_name,
            "total_points": info.points_count,
            "total_entries": info.points_count,
            "total_hits": 0,
            "status": str(info.status),
            "vector_size": self.vector_size,
            "similarity_threshold": self.threshold,
        }

import json
import math
import os
import sqlite3

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


class SemanticCache:
    """
    Semantic Caching layer for enterprise multi-agent workflows.
    Uses GoogleGenerativeAIEmbeddings (text-embedding-004) and cosine similarity
    over SQLite to eliminate redundant LLM executions.
    """

    def __init__(self, db_path: str = "semantic_cache.db", threshold: float = 0.90):
        self.db_path = db_path
        self.threshold = threshold

        self.embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

        self._init_db()

    def _init_db(self) -> None:
        """Create the cache table if it does not exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    response TEXT NOT NULL,
                    hit_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _get_embedding(self, text: str) -> list[float]:
        """Generate a dense vector embedding for the input text using Google Gemini."""
        return self.embeddings.embed_query(text)

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        Formula: (u . v) / (||u|| * ||v||)
        Returns a float between -1.0 and 1.0 (typically 0.0 to 1.0 for embeddings).
        """
        if len(vec1) != len(vec2):
            return 0.0
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def lookup(self, query: str) -> dict | None:
        """
        Check if a semantically similar query exists in the cache.
        Returns a dict with response, similarity, and matched query if hit, else None.
        """
        query_vec = self._get_embedding(query)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT  id, query, embedding, response, hit_count FROM semantic_cache"
            )
            rows = cursor.fetchall()

            if not rows:
                return None
            best_match = None
            best_similarity = -1.0

            for (
                row_id,
                cached_query,
                embedding_json,
                cached_response,
                hit_count,
            ) in rows:
                cached_vec = json.loads(embedding_json)
                similarity = self._cosine_similarity(query_vec, cached_vec)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = {
                        "id": row_id,
                        "query": cached_query,
                        "response": cached_response,
                        "similarity": round(similarity, 4),
                        "hit_count": hit_count,
                    }
            if best_match and best_similarity >= self.threshold:
                cursor.execute(
                    "UPDATE semantic_cache SET hit_count = hit_count + 1 WHERE id = ?",
                    (best_match["id"],),
                )
                conn.commit()
                return best_match
        return None

    def store(self, query: str, response: str) -> None:
        """
        Store a new query, its vector embedding, and the synthesized response in the cache.
        """
        query_vec = self._get_embedding(query)
        embedding_json = json.dumps(query_vec)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO semantic_cache (query, embedding, response) VALUES (?, ?, ?)",
                (query, embedding_json, response),
            )
            conn.commit()

    def get_stats(self) -> dict:
        """Return cache health and analytics statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*), COALESCE(SUM(hit_count), 0) FROM semantic_cache"
            )
            total_entries, total_hits = cursor.fetchone()

            return {
                "total_entries": total_entries,
                "total_hits": total_hits,
                "similarity_threshold": self.threshold,
            }


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING SEMANTIC CACHE WITH GEMINI EMBEDDINGS")
    print("=" * 60)
    # Use a temporary test database file so we don't pollute production data
    test_db = "test_cache.db"
    if os.path.exists(test_db):
        os.remove(test_db)
    cache = SemanticCache(db_path=test_db, threshold=0.90)
    # 1. First Query: Should be a Cache MISS
    q1 = "What is the per diem allowance for Tokyo business trips?"
    print(f"\n[Turn 1] Query: '{q1}'")
    hit = cache.lookup(q1)
    if hit is None:
        print("  -> Result: CACHE MISS (as expected)")
        simulated_response = (
            "Policy: Tokyo per diem allowance is $300/day for lodging and meals."
        )
        print("  -> Simulating agent execution & storing response...")
        cache.store(q1, simulated_response)
    else:
        print(f"  -> Result: UNEXPECTED CACHE HIT: {hit['response']}")
    # 2. Second Query: Rephrased version of q1 -> Should be a CACHE HIT
    q2 = "Can you tell me the daily per diem rate for traveling to Tokyo?"
    print(f"\n[Turn 2] Query: '{q2}'")
    hit = cache.lookup(q2)
    if hit:
        print(f"  -> Result: CACHE HIT! (Similarity: {hit['similarity'] * 100:.2f}%)")
        print(f"  -> Matched Query: '{hit['query']}'")
        print(f"  -> Cached Response: {hit['response']}")
    else:
        print("  -> Result: CACHE MISS (Similarity was below threshold)")
    # 3. Third Query: Unrelated question -> Should be a CACHE MISS
    q3 = "How do I bake chocolate chip cookies?"
    print(f"\n[Turn 3] Query: '{q3}'")
    hit = cache.lookup(q3)
    if hit:
        print(
            f"  -> Result: UNEXPECTED CACHE HIT! (Similarity: {hit['similarity'] * 100:.2f}%)"
        )
    else:
        print("  -> Result: CACHE MISS (as expected, completely unrelated topic)")
    # 4. Print Cache Statistics
    print("\n" + "=" * 60)
    print("CACHE ANALYTICS:", cache.get_stats())
    print("=" * 60)
    # Clean up test database
    if os.path.exists(test_db):
        os.remove(test_db)

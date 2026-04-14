
# HYBRID SEARCH (VECTOR + FTS USING RRF)

from src.retrieval.fts_search import fts_search
from src.retrieval.vector_search import vector_search


def hybrid_search(query: str, k: int = 5) -> list[dict]:

    print("\n [HYBRID SEARCH START]")


    # ── GET BOTH RESULTS ───────────────────────
    vector_docs = vector_search(query, k=k)
    keyword_docs = fts_search(query, k=k)

    print(f" Vector Docs: {len(vector_docs)}")
    print(f" Keyword Docs: {len(keyword_docs)}")

    rrf_scores = {}
    chunk_map = {}

    # ── VECTOR RANKING ─────────────────────────
    for rank, doc in enumerate(vector_docs):
        key = doc["content"][:150]

        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (60 + rank + 1)
        chunk_map[key] = doc

    # ── KEYWORD RANKING ────────────────────────
    for rank, doc in enumerate(keyword_docs):
        key = doc["content"][:150]

        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (60 + rank + 1)
        chunk_map[key] = doc

    # ── SORT BY RRF SCORE ──────────────────────
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    final_results = [chunk_map[key] for key, _ in ranked[:k]]

    print(f"[HYBRID DONE] Returned {len(final_results)} results")

    return final_results
from src.core.db import get_db_conn, _embeddings_model
import base64
import os
import pathlib

def vector_search(query: str,k: int = 5,chunk_type: str | None = None,) -> list[dict]:
    """Find the k most similar chunks to a natural-language query.

    Args:
        query:      Natural-language question or search string.
        k:          Number of results to return.
        chunk_type: Optional filter — 'text', 'table', or 'image'.

    Returns:
        List of dicts with keys: content, chunk_type, page_number, section,
        source_file, element_type, image_base64, mime_type, position,
        metadata, similarity (0–1 cosine similarity score).

    The <=> operator is pgvector's cosine distance operator.
    Similarity = 1 − cosine_distance, so 1.0 = identical, 0.0 = orthogonal.
    """
    query_vec = _embeddings_model.embed_query(query)  # Issue 8: use singleton
    embedding_str = "[" + ",".join(str(v) for v in query_vec) + "]"

    # Conditionally add a chunk_type filter without SQL injection risk
    # (chunk_type is always passed as a parameterised value, never interpolated)
    type_clause = "AND chunk_type = %(chunk_type)s" if chunk_type else ""

    sql = f"""
        SELECT
            content, chunk_type, page_number, section,
            source_file, element_type, image_path, mime_type,
            position, metadata,
            1 - (embedding <=> %(vec)s::vector) AS similarity
        FROM multimodal_chunks
        WHERE 1=1 {type_clause}
        ORDER BY embedding <=> %(vec)s::vector
        LIMIT %(k)s
    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"vec": embedding_str, "chunk_type": chunk_type, "k": k})
            rows = cur.fetchall()

    # Read image from filesystem and re-encode as base64 for callers.
    results = []
    for row in rows:
        row = dict(row)
        img_path = row.pop("image_path", None)
        if img_path and os.path.exists(img_path):
            row["image_base64"] = base64.b64encode(
                pathlib.Path(img_path).read_bytes()
            ).decode()
        else:
            row["image_base64"] = None
        results.append(row)

    return results
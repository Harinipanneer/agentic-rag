
# FTS SEARCH 
import base64
import os
import pathlib

from src.core.db import get_db_conn


def fts_search(query: str,k: int = 5,chunk_type: str | None = None) -> list[dict]:

    print("\n [FTS SEARCH START]")

    type_clause = "AND chunk_type = %(chunk_type)s" if chunk_type else ""

    sql = f"""
        SELECT
            content,
            chunk_type,
            page_number,
            section,
            source_file,
            element_type,
            image_path,
            mime_type,
            position,
            metadata,
            ts_rank_cd(
                to_tsvector('english', content),
                plainto_tsquery('english', %(query)s)
            ) AS score
        FROM multimodal_chunks
        WHERE to_tsvector('english', content)
              @@ plainto_tsquery('english', %(query)s)
        {type_clause}
        ORDER BY score DESC
        LIMIT %(k)s
    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {
                "query": query,
                "k": k,
                "chunk_type": chunk_type
            })
            rows = cur.fetchall()

    results = []

    for row in rows:
        row = dict(row)

        #  SAME IMAGE HANDLING AS VECTOR SEARCH
        img_path = row.pop("image_path", None)

        if img_path and os.path.exists(img_path):
            row["image_base64"] = base64.b64encode(
                pathlib.Path(img_path).read_bytes()
            ).decode()
        else:
            row["image_base64"] = None

        # Rename score → similarity (for consistency)
        row["similarity"] = float(row.pop("score"))

        results.append(row)

    print(f" [FTS DONE] Returned {len(results)} results")

    return results
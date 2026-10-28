import re

from AI_service import settings


def retrieve_customer_context(
    connection,
    user_id: int,
    customer_identifier: str | None
) -> list[dict]:
    if customer_identifier is None:
        return []

    identifier = customer_identifier.strip()
    if not identifier:
        return []

    identifier_pattern = re.compile(
        rf"(?<!\w){re.escape(identifier)}(?!\w)",
        re.IGNORECASE
    )
    rows = connection.execute(
        """
        SELECT chunks.id, chunks.document_id, chunks.page_number,
               chunks.content, documents.filename
        FROM chunks
        JOIN documents ON documents.id = chunks.document_id
        WHERE documents.user_id = ? AND documents.status = 'ready'
        ORDER BY documents.created_at DESC, chunks.chunk_index
        """,
        (user_id,)
    ).fetchall()

    matches = []
    for row in rows:
        if not identifier_pattern.search(row["content"]):
            continue
        matches.append({
            "chunk_id": row["id"],
            "document_id": row["document_id"],
            "filename": row["filename"],
            "page_number": row["page_number"],
            "content": row["content"]
        })
        if len(matches) >= settings.MAX_RETRIEVED_CHUNKS:
            break

    return matches

import re

from AI_service import settings


def retrieve_customer_context(
    connection,
    user_id: int,
    customer_identifier: str | None,
    created_after: str | None = None,
    created_before: str | None = None
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
    date_filter = ""
    parameters: list[object] = [user_id]
    if created_after is not None:
        date_filter += " AND documents.created_at >= ?"
        parameters.append(created_after)
    if created_before is not None:
        date_filter += " AND documents.created_at < ?"
        parameters.append(created_before)
    rows = connection.execute(
        f"""
        SELECT chunks.id, chunks.document_id, chunks.page_number,
               chunks.content, documents.filename, documents.created_at
        FROM chunks
        JOIN documents ON documents.id = chunks.document_id
        WHERE documents.user_id = ? AND documents.status = 'ready'
        {date_filter}
        ORDER BY documents.created_at DESC, chunks.chunk_index
        """,
        parameters
    ).fetchall()

    matches = []
    for row in rows:
        if not identifier_pattern.search(row["content"]):
            continue
        matches.append({
            "chunk_id": row["id"],
            "document_id": row["document_id"],
            "filename": row["filename"],
            "created_at": row["created_at"],
            "page_number": row["page_number"],
            "content": row["content"]
        })
        if len(matches) >= settings.MAX_RETRIEVED_CHUNKS:
            break

    return matches

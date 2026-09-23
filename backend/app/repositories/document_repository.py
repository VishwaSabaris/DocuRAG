from typing import Any
from uuid import UUID

from psycopg import Connection


class DocumentRepository:
    """
    Repository for document and document-chunk persistence
    and retrieval.

    Retrieval methods currently include:

        1. Vector similarity search
        2. Exact substring keyword search
        3. PostgreSQL full-text search

    Vector search is useful for semantic similarity.

    Exact keyword search is useful for literal matches such as:
        - names
        - IDs
        - registration numbers
        - exact phrases

    Full-text search is useful for lexical matching of natural-language
    queries and is backed by PostgreSQL's tsvector/GIN infrastructure.
    """

    def __init__(self, connection: Connection) -> None:
        self.connection = connection

    def create_document(
        self,
        document_id: UUID,
        filename: str,
        stored_filename: str,
        status: str = "uploaded",
    ) -> dict[str, Any]:
        query = """
            INSERT INTO documents (
                id,
                filename,
                stored_filename,
                status
            )
            VALUES (
                %s,
                %s,
                %s,
                %s
            )
            RETURNING
                id,
                filename,
                stored_filename,
                status,
                created_at,
                updated_at;
        """

        with self.connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    document_id,
                    filename,
                    stored_filename,
                    status,
                ),
            )

            document = cursor.fetchone()

        self.connection.commit()

        return document

    def get_document(
        self,
        document_id: UUID,
    ) -> dict[str, Any] | None:
        query = """
            SELECT
                id,
                filename,
                stored_filename,
                status,
                created_at,
                updated_at
            FROM documents
            WHERE id = %s;
        """

        with self.connection.cursor() as cursor:
            cursor.execute(
                query,
                (document_id,),
            )

            return cursor.fetchone()

    def update_document_status(
        self,
        document_id: UUID,
        status: str,
    ) -> dict[str, Any] | None:
        query = """
            UPDATE documents
            SET
                status = %s,
                updated_at = NOW()
            WHERE id = %s
            RETURNING
                id,
                filename,
                stored_filename,
                status,
                created_at,
                updated_at;
        """

        with self.connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    status,
                    document_id,
                ),
            )

            document = cursor.fetchone()

        self.connection.commit()

        return document

    def save_chunks(
        self,
        chunks: list[dict[str, Any]],
    ) -> int:
        if not chunks:
            return 0

        query = """
            INSERT INTO document_chunks (
                id,
                document_id,
                page_number,
                chunk_index,
                text,
                character_count
            )
            VALUES (
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            );
        """

        values = [
            (
                chunk["id"],
                chunk["document_id"],
                chunk["page_number"],
                chunk["chunk_index"],
                chunk["text"],
                chunk["character_count"],
            )
            for chunk in chunks
        ]

        with self.connection.cursor() as cursor:
            cursor.executemany(
                query,
                values,
            )

        self.connection.commit()

        return len(chunks)

    def save_chunk_embeddings(
        self,
        embeddings: list[dict[str, Any]],
    ) -> int:
        if not embeddings:
            return 0

        query = """
            UPDATE document_chunks
            SET embedding = %s::vector
            WHERE id = %s;
        """

        updated_count = 0

        with self.connection.cursor() as cursor:
            for item in embeddings:
                embedding = item["embedding"]
                chunk_id = item["chunk_id"]

                if len(embedding) != 384:
                    raise ValueError(
                        "Embedding dimension must be 384."
                    )

                vector = "[" + ",".join(
                    str(value)
                    for value in embedding
                ) + "]"

                cursor.execute(
                    query,
                    (
                        vector,
                        chunk_id,
                    ),
                )

                updated_count += cursor.rowcount

        self.connection.commit()

        return updated_count

    def get_document_chunks(
        self,
        document_id: UUID,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT
                id,
                document_id,
                page_number,
                chunk_index,
                text,
                character_count,
                embedding,
                created_at
            FROM document_chunks
            WHERE document_id = %s
            ORDER BY chunk_index;
        """

        with self.connection.cursor() as cursor:
            cursor.execute(
                query,
                (document_id,),
            )

            return cursor.fetchall()

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_id: UUID | None = None,
    ) -> list[dict[str, Any]]:
        if len(query_embedding) != 384:
            raise ValueError(
                "Query embedding dimension must be 384."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        vector = "[" + ",".join(
            str(value)
            for value in query_embedding
        ) + "]"

        if document_id is not None:
            query = """
                SELECT
                    id,
                    document_id,
                    page_number,
                    chunk_index,
                    text,
                    character_count,
                    embedding <=> %s::vector AS distance
                FROM document_chunks
                WHERE embedding IS NOT NULL
                  AND document_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """

            parameters = (
                vector,
                document_id,
                vector,
                top_k,
            )

        else:
            query = """
                SELECT
                    id,
                    document_id,
                    page_number,
                    chunk_index,
                    text,
                    character_count,
                    embedding <=> %s::vector AS distance
                FROM document_chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """

            parameters = (
                vector,
                vector,
                top_k,
            )

        with self.connection.cursor() as cursor:
            cursor.execute(
                query,
                parameters,
            )

            return cursor.fetchall()

    def keyword_search(
        self,
        query_text: str,
        top_k: int = 5,
        document_id: UUID | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search document chunks using PostgreSQL ILIKE.

        This is an exact substring-style fallback and is particularly
        useful for literal identifiers, names, dates, addresses,
        and exact phrases.
        """

        if not query_text.strip():
            raise ValueError(
                "Query text cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        search_pattern = f"%{query_text.strip()}%"

        if document_id is not None:
            query = """
                SELECT
                    id,
                    document_id,
                    page_number,
                    chunk_index,
                    text,
                    character_count,
                    CASE
                        WHEN text ILIKE %s THEN 1.0
                        ELSE 0.0
                    END AS keyword_score
                FROM document_chunks
                WHERE document_id = %s
                  AND text ILIKE %s
                ORDER BY
                    keyword_score DESC,
                    character_count ASC,
                    chunk_index ASC
                LIMIT %s;
            """

            parameters = (
                search_pattern,
                document_id,
                search_pattern,
                top_k,
            )

        else:
            query = """
                SELECT
                    id,
                    document_id,
                    page_number,
                    chunk_index,
                    text,
                    character_count,
                    CASE
                        WHEN text ILIKE %s THEN 1.0
                        ELSE 0.0
                    END AS keyword_score
                FROM document_chunks
                WHERE text ILIKE %s
                ORDER BY
                    keyword_score DESC,
                    character_count ASC,
                    chunk_index ASC
                LIMIT %s;
            """

            parameters = (
                search_pattern,
                search_pattern,
                top_k,
            )

        with self.connection.cursor() as cursor:
            cursor.execute(
                query,
                parameters,
            )

            return cursor.fetchall()

    def full_text_search(
        self,
        query_text: str,
        top_k: int = 5,
        document_id: UUID | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search document chunks using PostgreSQL full-text search.

        PostgreSQL's tsvector column provides indexed lexical retrieval,
        while ts_rank_cd provides a relevance score for ordering.

        The 'simple' configuration is intentionally used so that
        technical identifiers, names, and domain-specific terms are
        not aggressively stemmed.

        This method complements:
            - vector similarity search
            - exact ILIKE keyword search

        It does not replace either method.
        """

        if not query_text.strip():
            raise ValueError(
                "Query text cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        query = """
            SELECT
                id,
                document_id,
                page_number,
                chunk_index,
                text,
                character_count,
                ts_rank_cd(
                    search_vector,
                    plainto_tsquery('simple', %s)
                ) AS lexical_score
            FROM document_chunks
            WHERE search_vector @@ plainto_tsquery(
                'simple',
                %s
            )
        """

        parameters: list[Any] = [
            query_text,
            query_text,
        ]

        if document_id is not None:
            query += """
                AND document_id = %s
            """
            parameters.append(document_id)

        query += """
            ORDER BY
                lexical_score DESC,
                chunk_index ASC
            LIMIT %s;
        """

        parameters.append(top_k)

        with self.connection.cursor() as cursor:
            cursor.execute(
                query,
                tuple(parameters),
            )

            return cursor.fetchall()

    def delete_document(
        self,
        document_id: UUID,
    ) -> bool:
        query = """
            DELETE FROM documents
            WHERE id = %s;
        """

        with self.connection.cursor() as cursor:
            cursor.execute(
                query,
                (document_id,),
            )

            deleted = cursor.rowcount > 0

        self.connection.commit()

        return deleted

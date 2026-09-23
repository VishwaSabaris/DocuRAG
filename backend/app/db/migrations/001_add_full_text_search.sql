-- DocuRAG migration 001
-- Add PostgreSQL full-text search support.

ALTER TABLE document_chunks
ADD COLUMN IF NOT EXISTS search_vector TSVECTOR;

-- Populate the search vector for existing chunks.
UPDATE document_chunks
SET search_vector =
    to_tsvector(
        'simple',
        COALESCE(text, '')
    )
WHERE search_vector IS NULL;

-- GIN index for fast lexical retrieval.
CREATE INDEX IF NOT EXISTS idx_document_chunks_search_vector
ON document_chunks
USING GIN (search_vector);

-- Automatically keep search_vector synchronized
-- whenever a chunk is inserted or its text changes.
CREATE OR REPLACE FUNCTION document_chunks_search_vector_update()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector :=
        to_tsvector(
            'simple',
            COALESCE(NEW.text, '')
        );

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_document_chunks_search_vector
ON document_chunks;

CREATE TRIGGER trg_document_chunks_search_vector
BEFORE INSERT OR UPDATE OF text
ON document_chunks
FOR EACH ROW
EXECUTE FUNCTION document_chunks_search_vector_update();

CREATE EXTENSION IF NOT EXISTS vector;


CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    filename TEXT NOT NULL,
    stored_filename TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'uploaded',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL
        REFERENCES documents(id)
        ON DELETE CASCADE,

    page_number INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,

    text TEXT NOT NULL,
    character_count INTEGER NOT NULL,

    embedding VECTOR(384),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_document_chunk
        UNIQUE (document_id, chunk_index)
);

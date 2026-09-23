from app.services.chunking.chunk_service import (
    build_chunks,
    normalize_text,
    split_into_paragraphs,
)


def test_normalize_text() -> None:
    text = """
    Hello     world.

    This    is   a   test.
    """

    result = normalize_text(text)

    assert result == "Hello world.\n\nThis is a test."


def test_split_into_paragraphs() -> None:
    text = """
    First paragraph.

    Second paragraph.

    Third paragraph.
    """

    result = split_into_paragraphs(text)

    assert result == [
        "First paragraph.",
        "Second paragraph.",
        "Third paragraph.",
    ]


def test_build_chunks_preserves_page_metadata() -> None:
    pages = [
        {
            "page_number": 1,
            "text": (
                "This is the first paragraph.\n\n"
                "This is the second paragraph."
            ),
        },
        {
            "page_number": 2,
            "text": "This is content from page two.",
        },
    ]

    chunks = build_chunks(
        document_id="test-document",
        pages=pages,
        max_chunk_size=100,
        overlap_size=20,
    )

    assert len(chunks) > 0

    assert chunks[0].document_id == "test-document"
    assert chunks[0].page_number == 1

    assert chunks[-1].page_number == 2

    for chunk in chunks:
        assert chunk.chunk_id
        assert chunk.text
        assert chunk.character_count == len(chunk.text)

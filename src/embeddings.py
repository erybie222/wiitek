"""Vectors for the course card search: what is embedded, how, and the ingestion step that stores them.

Each syllabus section is embedded together with a short header naming its course and section ("Przedmiot: … /
Sekcja: …"): a section's text often does not say what it is about — a reading list is bare titles, and the
prerequisites of Statistics need not contain the word. The major and semester stay out of the header on purpose: they
are exact SQL filters in the search, and leaving them out makes the copies of a faculty-wide syllabus (PE, listed on
every major) the same text, so they share one hash and one vector.

The search embeds its query with the same model (``get_embedder``); vectors of different models are never compared.
"""

from __future__ import annotations

import hashlib
from functools import cache

import openai
from langchain_openai import OpenAIEmbeddings
from psycopg import Cursor as DBCursor

from config import EmbeddingSettings
from database.queries.course_card_sections import (
    copy_embedding_by_hash,
    get_sections_for_embedding,
    set_embedding,
)

# Texts per API request. Sections are short (a few hundred tokens), so this stays far below the per-request token
# limit, and a failed request loses at most one batch.
EMBEDDING_BATCH_SIZE = 100


def embedding_text(course_name: str, section: str, content: str) -> str:
    r"""Build the text a syllabus section is embedded from.

    Args:
        course_name: The course's name in the catalog, e.g. "Algebra liniowa".
        section: Section heading, e.g. "Literatura".
        content: The section's text.

    Returns:
        str: E.g. ``"Przedmiot: Algebra liniowa\\nSekcja: Literatura\\n\\n1. …"``.
    """
    return f"Przedmiot: {course_name}\nSekcja: {section}\n\n{content}"


def text_hash(text: str) -> str:
    """Hash a text that is about to be embedded; equal hashes mean the same vector.

    Args:
        text: The text, from ``embedding_text``.

    Returns:
        str: Hex sha256, 64 characters.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def vector_literal(values: list[float]) -> str:
    """Write a vector in the form pgvector reads from a ``%s::vector`` parameter.

    Args:
        values: The vector, e.g. as returned by ``embed_query``.

    Returns:
        str: E.g. ``"[0.0123,-0.0456,0.0789]"``.
    """
    return "[" + ",".join(str(float(value)) for value in values) + "]"


@cache
def get_embedder() -> OpenAIEmbeddings:
    """Return the process-wide embedding client, created on first use with ``EMBEDDING_MODEL``.

    Returns:
        OpenAIEmbeddings: Client for ``embed_documents`` (ingestion) and ``embed_query`` (search). Texts are sent as
            they are: every section is far below the model's input limit, so no token-level splitting is needed.

    Raises:
        RuntimeError: OPENAI_API_KEY is not set (ingestion checks it first and skips the step instead).
    """
    settings = EmbeddingSettings()
    return OpenAIEmbeddings(model=settings.model, api_key=settings.require_api_key(), check_embedding_ctx_length=False)


def embed_course_card_sections(cursor: DBCursor) -> None:
    """Give every stored syllabus section a vector of the current model, and print what was done.

    Run after ``ingest_course_cards``. A section keeps its vector while the text it was computed from (header included,
    so a renamed course counts as changed) and the model stay the same. Otherwise the vector is copied from another
    section with the same text, or computed — once per distinct text. Without an API key the step is skipped; an API
    error stops it, keeping the vectors already stored. Either way the sections stay, and the next run fills in the
    missing vectors.

    Args:
        cursor: Database cursor of the ingestion run.
    """
    settings = EmbeddingSettings()
    if not settings.api_key:
        print(
            "\nCourse card embeddings skipped: OPENAI_API_KEY is not set; sections without a vector stay unsearchable."
        )
        return

    sections = get_sections_for_embedding(cursor)
    print(f"\nCourse card embeddings ({settings.model}): checking {len(sections)} sections...")

    ids_by_hash: dict[str, list[int]] = {}
    text_by_hash: dict[str, str] = {}
    up_to_date = 0
    for row in sections:
        text = embedding_text(row["course_name"], row["section"], row["content"])
        content_hash = text_hash(text)
        if row["has_embedding"] and row["content_hash"] == content_hash and row["embedding_model"] == settings.model:
            up_to_date += 1
            continue
        ids_by_hash.setdefault(content_hash, []).append(row["id"])
        text_by_hash[content_hash] = text

    copied = 0
    hashes_to_embed = []
    for content_hash, ids in ids_by_hash.items():
        if copy_embedding_by_hash(cursor, ids, content_hash, settings.model):
            copied += len(ids)
        else:
            hashes_to_embed.append(content_hash)

    computed = sections_embedded = 0
    try:
        for start in range(0, len(hashes_to_embed), EMBEDDING_BATCH_SIZE):
            batch = hashes_to_embed[start : start + EMBEDDING_BATCH_SIZE]
            vectors = get_embedder().embed_documents([text_by_hash[content_hash] for content_hash in batch])
            for content_hash, vector in zip(batch, vectors, strict=True):
                set_embedding(cursor, ids_by_hash[content_hash], content_hash, settings.model, vector_literal(vector))
                sections_embedded += len(ids_by_hash[content_hash])
            computed += len(batch)
            print(f"  {computed}/{len(hashes_to_embed)} distinct texts embedded")
    except openai.OpenAIError as error:
        print(f"  Stopped by an API error, vectors stored so far are kept: {type(error).__name__}: {error}")

    without_vector = len(sections) - up_to_date - copied - sections_embedded
    print(
        f"Course card embeddings done: {computed} distinct texts embedded ({sections_embedded} sections), "
        f"{copied} sections copied from an identical text, {up_to_date} already up to date."
    )
    if without_vector:
        print(f"  WARNING: {without_vector} sections have no vector, so the search cannot find them; run again.")

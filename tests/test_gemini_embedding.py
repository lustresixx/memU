"""
Test Gemini embedding functionality.

Usage:
    export GEMINI_API_KEY=your_api_key

    python tests/test_gemini_embedding.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from openai import OpenAI

from memu.embedding.backends.gemini import GeminiEmbeddingBackend


def test_embedding_backend_attributes():
    """Test Gemini embedding backend attributes."""
    print("\n[GEMINI EMBEDDING] Test 1: Backend attributes...")

    backend = GeminiEmbeddingBackend()

    assert backend.name == "gemini", f"Expected name 'gemini', got '{backend.name}'"
    assert backend.embedding_endpoint == "/embeddings", f"Unexpected endpoint: {backend.embedding_endpoint}"

    print(f"  Backend name: {backend.name}")
    print(f"  Embedding endpoint: {backend.embedding_endpoint}")


def test_embedding_payload_building():
    """Test that Gemini embedding payload is built correctly."""
    print("\n[GEMINI EMBEDDING] Test 2: Embedding payload building...")

    backend = GeminiEmbeddingBackend()

    payload = backend.build_embedding_payload(
        inputs=["Hello world", "How are you?"],
        embed_model="gemini-embedding-001",
    )

    assert "model" in payload, "Missing 'model' in payload"
    assert payload["model"] == "gemini-embedding-001", "Incorrect model"
    assert "input" in payload, "Missing 'input' in payload"
    assert len(payload["input"]) == 2, "Expected 2 inputs"
    assert payload["input"][0] == "Hello world", "Incorrect first input"
    assert payload["input"][1] == "How are you?", "Incorrect second input"

    print("  Payload structure is correct")
    print(f"  Model: {payload['model']}")
    print(f"  Inputs: {len(payload['input'])} items")


def test_single_embedding():
    """Test single text embedding via Gemini API."""
    print("\n[GEMINI EMBEDDING] Test 3: Single text embedding...")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("  Skipping: GEMINI_API_KEY not set")
        return

    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    response = client.embeddings.create(
        model="gemini-embedding-001",
        input="Hello world",
    )

    assert len(response.data) == 1, f"Expected 1 embedding, got {len(response.data)}"
    assert len(response.data[0].embedding) > 0, "Embedding vector is empty"

    print(f"  Generated embedding with {len(response.data[0].embedding)} dimensions")


def test_batch_embedding():
    """Test batch text embedding via Gemini API."""
    print("\n[GEMINI EMBEDDING] Test 4: Batch embedding...")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("  Skipping: GEMINI_API_KEY not set")
        return

    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    test_inputs = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
    ]

    response = client.embeddings.create(
        model="gemini-embedding-001",
        input=test_inputs,
    )

    assert len(response.data) == 3, f"Expected 3 embeddings, got {len(response.data)}"

    for i, item in enumerate(response.data):
        assert len(item.embedding) > 0, f"Embedding {i} is empty"
        print(f"  Embedding {i + 1}: {len(item.embedding)} dimensions")

    print(f"  Generated {len(response.data)} embeddings successfully")


def test_embedding_similarity():
    """Test embedding similarity check."""
    print("\n[GEMINI EMBEDDING] Test 5: Embedding similarity check...")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("  Skipping: GEMINI_API_KEY not set")
        return

    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    similar_texts = [
        "I love drinking coffee",
        "Coffee is my favorite drink",
    ]
    different_text = "The stock market crashed yesterday"

    similar_response = client.embeddings.create(
        model="gemini-embedding-001",
        input=similar_texts,
    )
    different_response = client.embeddings.create(
        model="gemini-embedding-001",
        input=different_text,
    )

    similar_embeddings = [item.embedding for item in similar_response.data]
    different_embedding = different_response.data[0].embedding

    def cosine_similarity(a, b):
        dot_product = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = sum(x**2 for x in a) ** 0.5
        norm_b = sum(x**2 for x in b) ** 0.5
        return dot_product / (norm_a * norm_b)

    sim_similar = cosine_similarity(similar_embeddings[0], similar_embeddings[1])
    sim_different = cosine_similarity(similar_embeddings[0], different_embedding)

    print(f"  Similarity between similar texts: {sim_similar:.4f}")
    print(f"  Similarity between different texts: {sim_different:.4f}")

    assert sim_similar > sim_different, (
        f"Expected similar texts to have higher similarity, "
        f"got {sim_similar:.4f} vs {sim_different:.4f}"
    )
    print("  Similarity check passed")


def main():
    """Run all embedding tests."""
    print("\n" + "=" * 60)
    print("[GEMINI EMBEDDING] Starting embedding tests...")
    print("=" * 60)

    test_embedding_backend_attributes()
    test_embedding_payload_building()
    test_single_embedding()
    test_batch_embedding()
    test_embedding_similarity()

    print("\n" + "=" * 60)
    print("[GEMINI EMBEDDING] All embedding tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()


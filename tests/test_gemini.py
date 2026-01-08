"""
Test Gemini integration with MemU's full workflow.

Usage:
    export GEMINI_API_KEY=your_api_key

    python tests/test_gemini.py
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from openai import OpenAI

from memu.app import MemoryService


def test_chat_completion():
    """Test Gemini chat completion via OpenAI-compatible endpoint."""
    print("\n[GEMINI] Test 1: Chat completion...")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("  ERROR: GEMINI_API_KEY environment variable not set")
        return False

    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one word."},
        ],
    )

    result = response.choices[0].message.content
    assert result, "Chat response is empty"
    print(f"  Response: {result}")
    print("  Chat completion test passed")
    return True


def test_embedding():
    """Test Gemini embedding via OpenAI-compatible endpoint."""
    print("\n[GEMINI] Test 2: Embedding generation...")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("  ERROR: GEMINI_API_KEY environment variable not set")
        return False

    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    response = client.embeddings.create(
        model="gemini-embedding-001",
        input=["Hello world", "How do LLMs work?"],
    )

    assert len(response.data) == 2, f"Expected 2 embeddings, got {len(response.data)}"
    assert len(response.data[0].embedding) > 0, "Embedding vector is empty"

    print(f"  Generated {len(response.data)} embeddings")
    print(f"  Embedding dimension: {len(response.data[0].embedding)}")
    print("  Embedding test passed")
    return True


async def test_full_workflow():
    """Test Gemini integration with full MemU workflow."""
    print("\n[GEMINI] Test 3: Full workflow (memorize + retrieve)...")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("  ERROR: GEMINI_API_KEY environment variable not set")
        return False

    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "example", "example_conversation.json"))
    if not os.path.exists(file_path):
        print(f"  ERROR: Test file not found: {file_path}")
        return False

    output_data = {}

    service = MemoryService(
        llm_profiles={
            "default": {
                "provider": "gemini",
                "client_backend": "sdk",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                "api_key": api_key,
                "chat_model": "gemini-2.5-flash",
                "embed_model": "gemini-embedding-001",
            },
        },
        database_config={
            "metadata_store": {"provider": "inmemory"},
        },
        retrieve_config={
            "method": "rag",
            "route_intention": False,
        },
    )

    print("  Memorizing conversation...")
    memory = await service.memorize(
        resource_url=file_path,
        modality="conversation",
        user={"user_id": "gemini_test_user"}
    )
    items_count = len(memory.get("items", []))
    categories_count = len(memory.get("categories", []))

    print(f"  Memorized {items_count} items")
    print(f"  Created {categories_count} categories")

    output_data["memorize"] = memory

    for cat in memory.get("categories", [])[:3]:
        print(f"    - {cat.get('name')}: {(cat.get('summary') or '')[:60]}...")

    queries = [
        {"role": "user", "content": {"text": "What does the user like?"}},
    ]

    print("  RAG-based retrieval...")
    service.retrieve_config.method = "rag"
    result_rag = await service.retrieve(queries=queries, where={"user_id": "gemini_test_user"})

    categories_retrieved = len(result_rag.get("categories", []))
    items_retrieved = len(result_rag.get("items", []))

    print(f"  Retrieved {categories_retrieved} categories, {items_retrieved} items")
    output_data["retrieve_rag"] = result_rag

    output_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "examples", "output", "gemini_test_output.json")
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"  Output saved to: {output_file}")

    print("  Full workflow test passed")
    return True


async def main():
    """Run all Gemini tests."""
    print("\n" + "=" * 60)
    print("[GEMINI] Starting Gemini provider tests...")
    print("=" * 60)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set")
        print("Please set it with: export GEMINI_API_KEY=your_api_key")
        sys.exit(1)

    test_chat_completion()
    test_embedding()
    await test_full_workflow()

    print("\n" + "=" * 60)
    print("[GEMINI] All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())


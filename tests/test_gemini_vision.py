"""
Test Gemini vision functionality.

Usage:
    export GEMINI_API_KEY=your_api_key

    python tests/test_gemini_vision.py
"""

import base64
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from openai import OpenAI

from memu.llm.backends.gemini import GeminiLLMBackend


def test_vision_payload_building():
    """Test that Gemini vision payload is built correctly."""
    print("\n[GEMINI VISION] Test 1: Vision payload building...")

    backend = GeminiLLMBackend()

    payload = backend.build_vision_payload(
        prompt="Describe this image",
        base64_image="SGVsbG8gV29ybGQ=",
        mime_type="image/png",
        system_prompt="You are a helpful assistant",
        chat_model="gemini-2.5-flash",
        max_tokens=500,
    )

    assert "model" in payload, "Missing 'model' in payload"
    assert payload["model"] == "gemini-2.5-flash", "Incorrect model"
    assert "messages" in payload, "Missing 'messages' in payload"
    assert len(payload["messages"]) == 2, "Expected 2 messages (system + user)"

    system_msg = payload["messages"][0]
    assert system_msg["role"] == "system", "First message should be system"
    assert system_msg["content"] == "You are a helpful assistant", "Incorrect system prompt"

    user_msg = payload["messages"][1]
    assert user_msg["role"] == "user", "Second message should be user"
    assert isinstance(user_msg["content"], list), "User content should be a list"
    assert len(user_msg["content"]) == 2, "User content should have text and image"

    text_part = user_msg["content"][0]
    assert text_part["type"] == "text", "First part should be text"
    assert text_part["text"] == "Describe this image", "Incorrect prompt text"

    image_part = user_msg["content"][1]
    assert image_part["type"] == "image_url", "Second part should be image_url"
    assert "image_url" in image_part, "Missing image_url object"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,"), "Incorrect data URL format"

    assert payload.get("max_tokens") == 500, "Incorrect max_tokens"

    print("  Payload structure is correct")
    print("  System message formatted correctly")
    print("  User message with image formatted correctly")
    print("  Data URL format is correct")


def test_vision_payload_without_system_prompt():
    """Test vision payload without system prompt."""
    print("\n[GEMINI VISION] Test 2: Vision payload without system prompt...")

    backend = GeminiLLMBackend()

    payload = backend.build_vision_payload(
        prompt="What's in this image?",
        base64_image="SGVsbG8gV29ybGQ=",
        mime_type="image/jpeg",
        system_prompt=None,
        chat_model="gemini-2.5-flash",
        max_tokens=None,
    )

    assert len(payload["messages"]) == 1, "Expected 1 message when no system prompt"
    assert payload["messages"][0]["role"] == "user", "Should be user message"

    print("  Payload without system prompt is correct")
    print("  Only user message present")


def test_vision_api_call():
    """Test actual vision API call with Gemini."""
    print("\n[GEMINI VISION] Test 3: Vision API call (live test)...")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("  Skipping live test: GEMINI_API_KEY not set")
        return

    image_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "examples", "resources", "images", "image1.png")
    )
    if not os.path.exists(image_path):
        print(f"  Skipping live test: Test image not found at {image_path}")
        return

    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe what you see in this image in one sentence."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}",
                        },
                    },
                ],
            }
        ],
        max_tokens=100,
    )

    result = response.choices[0].message.content
    assert result, "Vision API returned empty result"
    assert len(result) > 0, "Vision response is empty"
    print("  Vision API call successful")
    print(f"  Response: {result[:150]}...")


def main():
    """Run all vision tests."""
    print("\n" + "=" * 60)
    print("[GEMINI VISION] Starting vision tests...")
    print("=" * 60)

    test_vision_payload_building()
    test_vision_payload_without_system_prompt()
    test_vision_api_call()

    print("\n" + "=" * 60)
    print("[GEMINI VISION] All vision tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()


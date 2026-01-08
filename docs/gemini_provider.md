# Gemini Provider Configuration

MemU supports Google Gemini as a model provider through its OpenAI-compatible API.

## Overview

Gemini provides:
- **LLM Support**: Chat completions for memory extraction and retrieval
- **Embedding Support**: Text embeddings for vector search
- **Vision Support**: Image understanding for multimodal memory

## Configuration

### Environment Variables

```bash
export GEMINI_API_KEY=your_api_key
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

### Service Configuration

```python
from memu import MemoryService

service = MemoryService(
    llm_profiles={
        "default": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "api_key": "your_api_key",
            "chat_model": "gemini-2.5-flash",
            "embed_model": "gemini-embedding-001",
            "provider": "gemini",
            "client_backend": "sdk",
        }
    },
    database_config={
        "metadata_store": {"provider": "inmemory"},
    },
)
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `base_url` | Gemini OpenAI-compatible API endpoint | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `api_key` | Your Gemini API key | Required |
| `chat_model` | Model for chat/vision tasks | `gemini-2.5-flash` |
| `embed_model` | Model for text embeddings | `gemini-embedding-001` |
| `provider` | Provider identifier | `gemini` |
| `client_backend` | Client type (`sdk` or `httpx`) | `sdk` |

## Available Models

### Chat Models
- `gemini-2.5-flash` - Latest flash model with vision support
### Embedding Models
- `gemini-embedding-001` - Embedding model for OpenAI-compatible API

## Running Tests

```bash
export GEMINI_API_KEY=your_api_key

# Full workflow test (memorize + retrieve)
python tests/test_gemini.py

# Embedding-specific tests
python tests/test_gemini_embedding.py

# Vision-specific tests
python tests/test_gemini_vision.py
```

## Troubleshooting

### Common Issues

1. **Invalid API Key**
   - Ensure your API key is valid and has access to the Gemini API
   - Check that the key is properly set in the environment or configuration

2. **Model Not Found**
   - Verify the model name is correct
   - Some models may require specific API access

3. **Rate Limits (429 Error)**
   - Gemini free tier has strict limits
   - Consider using a paid API key for production use

### Debug Logging

Enable debug logging to see API requests/responses:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```


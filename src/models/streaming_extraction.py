"""
Server-Sent Events (SSE) Streaming for LLM Responses

Provides streaming versions of extraction and verification with:
- Token-by-token OpenAI streaming
- Progressive LLM response rendering
- Async generators for FastAPI StreamingResponse
- Real-time progress updates
- Proper error handling and timeouts

Key difference from async_extraction.py:
- async_extraction.py waits for full response then returns
- streaming_extraction.py yields tokens progressively
"""

import asyncio
import json
import logging
import os
import time
from typing import Optional, AsyncGenerator, Dict, Any

try:
    from openai import AsyncOpenAI
    ASYNC_OPENAI_AVAILABLE = True
except ImportError:
    ASYNC_OPENAI_AVAILABLE = False
    logging.warning("AsyncOpenAI not available. Install with: pip install openai>=1.0")

from src.models.extraction import (
    BookMeta,
    ExtractionResult,
    extract_with_regex,
)

logger = logging.getLogger(__name__)


class StreamingOpenAIExtractor:
    """Async OpenAI extractor with token-by-token streaming."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize streaming OpenAI client."""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found")

        if not ASYNC_OPENAI_AVAILABLE:
            raise ImportError("openai library required for streaming. Install: pip install openai>=1.0")

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.timeout = 30.0

    async def extract_book_meta_streaming(
        self,
        text: str,
        model: str = "gpt-3.5-turbo",
        max_retries: int = 2
    ) -> AsyncGenerator[str, None]:
        """
        Stream book metadata extraction using OpenAI.

        Yields:
            str: JSON chunks as they arrive from OpenAI
        """
        system_prompt = (
            "You are a precise information extraction system. "
            "Extract the book title and author name from the given text. "
            "The text may be a book review, article about a book, or mention of a book.\n\n"
            "IMPORTANT RULES:\n"
            "- Extract ONLY real, specific author names (e.g., 'George Orwell', 'Jane Austen')\n"
            "- DO NOT use placeholders like 'Unknown', 'Various', 'Staff', 'Anonymous'\n"
            "- If no specific author name is found, the extraction should FAIL\n"
            "- Extract the FULL author name when available\n"
            "- Return ONLY valid JSON"
        )

        user_prompt = f"""Extract book information from this text:

TEXT:
{text[:2000]}

Return ONLY a valid JSON object with these fields:
{{
    "book_title": "extracted title or null",
    "author_name": "extracted author or null",
    "confidence": "high/medium/low"
}}

If insufficient information: {{"book_title": null, "author_name": null, "confidence": "low"}}
"""

        buffer = ""
        attempt = 0

        while attempt < max_retries:
            try:
                logger.debug(f"Starting streaming extraction (attempt {attempt + 1}/{max_retries})")
                t_start = time.time()

                # Use streaming with OpenAI
                async with self.client.messages.stream(
                    model=model,
                    max_tokens=500,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                ) as stream:
                    async for text_chunk in stream.text_stream:
                        # Yield the chunk immediately
                        buffer += text_chunk
                        yield text_chunk

                        # Small delay to simulate real-time streaming (remove in production)
                        await asyncio.sleep(0)

                t_elapsed = time.time() - t_start
                logger.debug(f"Streaming extraction completed in {t_elapsed:.2f}s")

                # Yield final metadata
                yield f"\n[METADATA]extraction_time:{t_elapsed:.2f}s[/METADATA]\n"
                return

            except Exception as e:
                attempt += 1
                if attempt < max_retries:
                    logger.warning(f"Streaming extraction failed (attempt {attempt}): {e}. Retrying...")
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error(f"Streaming extraction failed after {max_retries} attempts: {e}")
                    yield f"\n[ERROR]Extraction failed after {max_retries} attempts[/ERROR]\n"
                    return


async def stream_extract_article(
    article_text: str,
    article_id: Optional[str] = None,
    model: str = "gpt-3.5-turbo"
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream extraction results for a single article.

    Yields:
        Dict: Event objects with 'type' and 'data' fields
    """
    t_start = time.time()

    try:
        # Initialize extractor
        extractor = StreamingOpenAIExtractor()

        # Yield start event
        yield {
            "type": "start",
            "data": {"article_id": article_id, "timestamp": t_start}
        }

        # Stream tokens
        t_token_start = time.time()
        token_count = 0
        buffer = ""

        async for chunk in extractor.extract_book_meta_streaming(
            article_text, model=model
        ):
            # Track first token time
            if token_count == 0:
                t_first_token = time.time() - t_token_start
                logger.debug(f"First token received in {t_first_token*1000:.1f}ms")

            token_count += 1
            buffer += chunk

            # Yield token events
            yield {
                "type": "token",
                "data": {
                    "token": chunk,
                    "token_number": token_count,
                    "elapsed_ms": (time.time() - t_token_start) * 1000
                }
            }

        # Yield complete event
        t_total = time.time() - t_start
        yield {
            "type": "complete",
            "data": {
                "response": buffer,
                "total_ms": t_total * 1000,
                "tokens": token_count,
                "tokens_per_sec": token_count / t_total if t_total > 0 else 0
            }
        }

    except Exception as e:
        logger.error(f"Error streaming extraction: {e}")
        yield {
            "type": "error",
            "data": {"message": str(e)}
        }


async def stream_extract_batch(
    articles: list[Dict[str, str]],
    model: str = "gpt-3.5-turbo"
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream extraction for multiple articles.

    Args:
        articles: List of {'id': str, 'text': str} dicts
        model: OpenAI model to use

    Yields:
        Dict: Event objects
    """
    yield {
        "type": "batch_start",
        "data": {"total_articles": len(articles)}
    }

    for idx, article in enumerate(articles):
        article_id = article.get('id', f'article_{idx}')
        article_text = article.get('text', '')

        yield {
            "type": "article_start",
            "data": {"article_id": article_id, "article_number": idx + 1}
        }

        # Stream extraction for this article
        async for event in stream_extract_article(article_text, article_id, model):
            yield event

        yield {
            "type": "article_end",
            "data": {"article_id": article_id}
        }

    yield {
        "type": "batch_end",
        "data": {"processed_articles": len(articles)}
    }


# SSE Event Formatter
def format_sse_event(event_type: str, data: Dict[str, Any]) -> str:
    """
    Format event as Server-Sent Event.

    Args:
        event_type: Type of event
        data: Event data

    Returns:
        str: SSE formatted string
    """
    sse_line = f"event: {event_type}\n"
    sse_line += f"data: {json.dumps(data)}\n\n"
    return sse_line


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Test streaming extraction
    async def test_streaming():
        test_text = """
        The latest novel by Stephen King, "Holly", has received widespread acclaim.
        King masterfully crafts a thriller that keeps readers on the edge of their seats.
        """

        print("Testing streaming extraction...")
        async for event in stream_extract_article(test_text, article_id="test_1"):
            print(f"Event type: {event['type']}")
            if event['type'] == 'token':
                print(f"  Token: {event['data']['token']!r}", end='', flush=True)
            elif event['type'] == 'complete':
                print(f"\n  Tokens: {event['data']['tokens']}")
                print(f"  Time: {event['data']['total_ms']:.0f}ms")

    asyncio.run(test_streaming())

"""
Asynchronous book metadata extraction and verification pipeline.

Provides fully async versions of extraction and verification with:
- Async OpenAI/Gemini API calls using httpx.AsyncClient
- Parallel execution with asyncio.gather
- Optimized pipeline: GPT + Tavily (parallel) → Gemini
- Proper error handling and timeouts
- Comprehensive timing logs

Migration from sync blocking calls to async non-blocking I/O.
"""

import asyncio
import logging
import time
import os
from typing import Optional, Tuple, List, Dict
from datetime import datetime

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logging.warning("httpx not available. Install with: pip install httpx")

try:
    from openai import AsyncOpenAI
    ASYNC_OPENAI_AVAILABLE = True
except ImportError:
    ASYNC_OPENAI_AVAILABLE = False
    logging.warning("AsyncOpenAI not available. Install with: pip install openai>=1.0")

try:
    from google import genai
    import google.generativeai as google_genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Google Generative AI not available. Install with: pip install google-generativeai")

from src.models.extraction import (
    BookMeta,
    ExtractionResult,
    VerificationResult,
    VerifiedBookMeta,
    REGEX_PATTERNS,
    extract_with_regex,
    extract_book_meta,  # Use sync version as fallback
)

logger = logging.getLogger(__name__)


# ============================================================================
# Async Clients and Initialization
# ============================================================================

class AsyncOpenAIExtractor:
    """Async wrapper for OpenAI extraction using httpx."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize async OpenAI client."""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found")

        self.base_url = "https://api.openai.com/v1"
        self.timeout = httpx.Timeout(30.0)

    async def extract_book_meta(
        self,
        text: str,
        model: str = "gpt-3.5-turbo",
        max_retries: int = 2
    ) -> Optional[BookMeta]:
        """
        Async extract book metadata using OpenAI.

        Args:
            text: Text to extract from
            model: OpenAI model (default: gpt-3.5-turbo)
            max_retries: Number of retries on failure

        Returns:
            BookMeta or None if extraction failed
        """
        import json

        system_prompt = (
            "You are a precise information extraction system. "
            "Extract the book title and author name from the given text. "
            "The text may be a book review, article about a book, or mention of a book.\n\n"
            "IMPORTANT RULES:\n"
            "- Extract ONLY real, specific author names (e.g., 'George Orwell', 'Jane Austen')\n"
            "- DO NOT use placeholders like 'Unknown', 'Various', 'Staff', 'Anonymous', 'Editor'\n"
            "- DO NOT extract generic terms like 'multiple authors' or 'various authors'\n"
            "- If no specific author name is found, the extraction should FAIL\n"
            "- Extract the FULL author name when available (first and last name)\n"
            "- Return JSON with keys: book_title, author_name"
        )

        user_message = f"Extract the book title and author from this text:\n\n{text}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                for attempt in range(max_retries):
                    try:
                        response = await client.post(
                            f"{self.base_url}/chat/completions",
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json",
                            },
                            json={
                                "model": model,
                                "messages": [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_message},
                                ],
                                "temperature": 0.2,
                            },
                        )

                        if response.status_code == 200:
                            data = response.json()
                            content = data['choices'][0]['message']['content']

                            # Parse JSON from response
                            try:
                                extracted = json.loads(content)
                                if 'book_title' in extracted and 'author_name' in extracted:
                                    return BookMeta(
                                        book_title=extracted['book_title'],
                                        author_name=extracted['author_name']
                                    )
                            except (json.JSONDecodeError, ValueError):
                                # Extraction parsing failed
                                pass

                            logger.debug(f"Failed to parse extraction: {content}")
                            continue

                        elif response.status_code == 429:
                            # Rate limit - wait and retry
                            wait_time = 2 ** attempt
                            logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}")
                            await asyncio.sleep(wait_time)
                            continue

                        else:
                            logger.debug(f"OpenAI API error: {response.status_code} - {response.text}")
                            continue

                    except httpx.RequestError as e:
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            logger.debug(f"Request error, retrying in {wait_time}s: {e}")
                            await asyncio.sleep(wait_time)
                        else:
                            logger.debug(f"Max retries exceeded: {e}")

        except Exception as e:
            logger.debug(f"Async OpenAI extraction failed: {e}")

        return None


class AsyncGeminiVerifier:
    """Async wrapper for Gemini verification."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash-exp"):
        """Initialize async Gemini client."""
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key not found")

        self.model = model
        self.timeout = 30.0

    async def verify_book_meta(
        self,
        book_title: str,
        author_name: str
    ) -> Optional[VerificationResult]:
        """
        Async verify book metadata using Gemini.

        Args:
            book_title: Book title to verify
            author_name: Author name to verify

        Returns:
            VerificationResult or None if verification failed
        """
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not available for async Gemini calls")
            return None

        prompt = (
            f"Verify if the following book and author combination is real and correct:\n\n"
            f"Book Title: {book_title}\n"
            f"Author: {author_name}\n\n"
            f"Respond with JSON containing:\n"
            f"- is_verified (boolean): True if the book and author combination is correct\n"
            f"- confidence (string): 'high', 'medium', or 'low'\n"
            f"- reasoning (string): Explanation of your verification decision\n"
            f"- corrected_title (string or null): Corrected title if different\n"
            f"- corrected_author (string or null): Corrected author if different"
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                    params={"key": self.api_key},
                    json={
                        "contents": [
                            {
                                "parts": [
                                    {"text": prompt}
                                ]
                            }
                        ],
                        "generationConfig": {
                            "temperature": 0.2,
                            "maxOutputTokens": 500,
                        },
                    },
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    import json
                    data = response.json()

                    # Extract text from Gemini response
                    if 'candidates' in data and len(data['candidates']) > 0:
                        content = data['candidates'][0]['content']['parts'][0]['text']

                        # Parse JSON from response
                        try:
                            verified_data = json.loads(content)
                            return VerificationResult(
                                is_verified=verified_data.get('is_verified', False),
                                confidence=verified_data.get('confidence', 'low'),
                                corrected_title=verified_data.get('corrected_title'),
                                corrected_author=verified_data.get('corrected_author'),
                                reasoning=verified_data.get('reasoning', ''),
                                search_results_used=0
                            )
                        except (json.JSONDecodeError, ValueError) as e:
                            logger.debug(f"Failed to parse Gemini response: {e}")

                else:
                    logger.debug(f"Gemini API error: {response.status_code}")

        except Exception as e:
            logger.debug(f"Async Gemini verification failed: {e}")

        return None


# ============================================================================
# Async Pipeline Functions
# ============================================================================

async def async_extract_book_meta(
    text: str,
    use_llm: bool = True,
    llm_model: str = "gpt-3.5-turbo",
    openai_extractor: Optional[AsyncOpenAIExtractor] = None
) -> ExtractionResult:
    """
    Async extract book metadata using LLM or regex fallback.

    Args:
        text: Text to extract from
        use_llm: Use LLM for extraction (default: True)
        llm_model: OpenAI model to use
        openai_extractor: Optional pre-initialized AsyncOpenAIExtractor

    Returns:
        ExtractionResult with extracted metadata or fallback
    """
    t_start = time.time()

    # Try LLM extraction if enabled
    if use_llm:
        try:
            if openai_extractor is None:
                openai_extractor = AsyncOpenAIExtractor()

            t_llm_start = time.time()
            book_meta = await openai_extractor.extract_book_meta(text, model=llm_model)
            t_llm_ms = (time.time() - t_llm_start) * 1000

            if book_meta:
                logger.debug(f"Async LLM extraction succeeded in {t_llm_ms:.1f}ms")
                return ExtractionResult(
                    success=True,
                    book_meta=book_meta,
                    method="llm_async"
                )
            else:
                logger.debug(f"Async LLM extraction returned None after {t_llm_ms:.1f}ms")

        except Exception as e:
            logger.debug(f"Async LLM extraction failed: {e}")

    # Fallback to regex
    logger.debug("Attempting regex fallback extraction")
    title, author, pattern = extract_with_regex(text)

    if title and author:
        return ExtractionResult(
            success=True,
            book_meta=BookMeta(book_title=title, author_name=author),
            method=pattern or "regex"
        )

    return ExtractionResult(success=False, method="failed", error="No extraction method succeeded")


async def async_verify_book_meta(
    book_title: str,
    author_name: str,
    verification_model: str = "gemini-2.0-flash-exp",
    gemini_verifier: Optional[AsyncGeminiVerifier] = None
) -> Optional[VerificationResult]:
    """
    Async verify book metadata using Gemini.

    Args:
        book_title: Book title to verify
        author_name: Author name to verify
        verification_model: Gemini model to use
        gemini_verifier: Optional pre-initialized AsyncGeminiVerifier

    Returns:
        VerificationResult or None if verification failed
    """
    t_start = time.time()

    try:
        if gemini_verifier is None:
            gemini_verifier = AsyncGeminiVerifier(model=verification_model)

        t_verify_start = time.time()
        verification = await gemini_verifier.verify_book_meta(book_title, author_name)
        t_verify_ms = (time.time() - t_verify_start) * 1000

        if verification:
            logger.debug(f"Async Gemini verification completed in {t_verify_ms:.1f}ms: {verification.confidence}")
            return verification
        else:
            logger.debug(f"Async Gemini verification failed after {t_verify_ms:.1f}ms")

    except Exception as e:
        logger.debug(f"Async verification failed: {e}")

    return None


async def async_extract_and_verify(
    text: str,
    use_llm: bool = True,
    use_verification: bool = True,
    llm_model: str = "gpt-3.5-turbo",
    verification_model: str = "gemini-2.0-flash-exp",
    min_confidence_threshold: str = "medium",
    openai_extractor: Optional[AsyncOpenAIExtractor] = None,
    gemini_verifier: Optional[AsyncGeminiVerifier] = None
) -> Tuple[Optional[VerifiedBookMeta], ExtractionResult, Dict]:
    """
    Async extract and verify book metadata with optimized pipeline.

    Pipeline: GPT extraction + Gemini verification (parallel where possible)

    Args:
        text: Text to extract from
        use_llm: Use LLM for extraction
        use_verification: Verify with LLM judge
        llm_model: OpenAI model for extraction
        verification_model: Gemini model for verification
        min_confidence_threshold: Minimum confidence to accept
        openai_extractor: Optional pre-initialized OpenAI extractor
        gemini_verifier: Optional pre-initialized Gemini verifier

    Returns:
        Tuple of (VerifiedBookMeta or None, ExtractionResult, timing_dict)
    """
    t_pipeline_start = time.time()
    timing = {}

    # Step 1: Extract (must happen first)
    t_extract_start = time.time()
    extraction_result = await async_extract_book_meta(
        text,
        use_llm=use_llm,
        llm_model=llm_model,
        openai_extractor=openai_extractor
    )
    timing['extraction_ms'] = (time.time() - t_extract_start) * 1000

    if not extraction_result.success:
        timing['total_ms'] = (time.time() - t_pipeline_start) * 1000
        return None, extraction_result, timing

    book_title = extraction_result.book_meta.book_title
    author_name = extraction_result.book_meta.author_name

    # If verification disabled, return unverified
    if not use_verification:
        verified_meta = VerifiedBookMeta(
            book_title=book_title,
            author_name=author_name,
            extraction_method=extraction_result.method,
            verification_confidence="unverified",
            was_corrected=False,
            verification_reasoning="Verification disabled"
        )
        timing['total_ms'] = (time.time() - t_pipeline_start) * 1000
        return verified_meta, extraction_result, timing

    # Step 2: Verify (parallel with extraction would require restructuring)
    t_verify_start = time.time()
    verification = await async_verify_book_meta(
        book_title,
        author_name,
        verification_model=verification_model,
        gemini_verifier=gemini_verifier
    )
    timing['verification_ms'] = (time.time() - t_verify_start) * 1000

    if verification is None:
        verified_meta = VerifiedBookMeta(
            book_title=book_title,
            author_name=author_name,
            extraction_method=extraction_result.method,
            verification_confidence="unverified",
            was_corrected=False,
            verification_reasoning="Verification failed"
        )
        timing['total_ms'] = (time.time() - t_pipeline_start) * 1000
        return verified_meta, extraction_result, timing

    # Step 3: Check confidence threshold
    confidence_levels = {'low': 1, 'medium': 2, 'high': 3}
    min_level = confidence_levels.get(min_confidence_threshold, 2)
    actual_level = confidence_levels.get(verification.confidence, 1)

    if not verification.is_verified or actual_level < min_level:
        logger.debug(
            f"Verification failed: {verification.reasoning}"
        )
        timing['total_ms'] = (time.time() - t_pipeline_start) * 1000
        return None, extraction_result, timing

    # Step 4: Build verified metadata
    final_title = verification.corrected_title or book_title
    final_author = verification.corrected_author or author_name
    was_corrected = bool(verification.corrected_title or verification.corrected_author)

    verified_meta = VerifiedBookMeta(
        book_title=final_title,
        author_name=final_author,
        extraction_method=extraction_result.method,
        verification_confidence=verification.confidence,
        was_corrected=was_corrected,
        verification_reasoning=verification.reasoning
    )

    timing['total_ms'] = (time.time() - t_pipeline_start) * 1000

    return verified_meta, extraction_result, timing


async def async_batch_extract_and_verify(
    texts: List[str],
    use_llm: bool = True,
    use_verification: bool = True,
    llm_model: str = "gpt-3.5-turbo",
    verification_model: str = "gemini-2.0-flash-exp",
    min_confidence_threshold: str = "medium",
    max_concurrent: int = 5,
    verbose: bool = True
) -> Tuple[List[Tuple[Optional[VerifiedBookMeta], ExtractionResult, Dict]], Dict]:
    """
    Async batch extract and verify multiple texts with controlled concurrency.

    Uses asyncio.Semaphore to limit concurrent API calls.

    Args:
        texts: List of texts to process
        use_llm: Use LLM for extraction
        use_verification: Verify with LLM judge
        llm_model: OpenAI model for extraction
        verification_model: Gemini model for verification
        min_confidence_threshold: Minimum confidence to accept
        max_concurrent: Maximum concurrent requests (default: 5)
        verbose: Show progress logging

    Returns:
        Tuple of (results list, statistics dict)
    """
    if verbose:
        logger.info(f"Starting async batch processing of {len(texts)} texts with max_concurrent={max_concurrent}")

    t_batch_start = time.time()

    # Create shared clients
    try:
        openai_extractor = AsyncOpenAIExtractor()
    except ValueError:
        openai_extractor = None
        logger.warning("OpenAI extractor not available")

    try:
        gemini_verifier = AsyncGeminiVerifier(model=verification_model)
    except ValueError:
        gemini_verifier = None
        logger.warning("Gemini verifier not available")

    # Semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(idx: int, text: str):
        """Process single text with semaphore."""
        async with semaphore:
            try:
                verified, extraction, timing = await async_extract_and_verify(
                    text,
                    use_llm=use_llm,
                    use_verification=use_verification,
                    llm_model=llm_model,
                    verification_model=verification_model,
                    min_confidence_threshold=min_confidence_threshold,
                    openai_extractor=openai_extractor,
                    gemini_verifier=gemini_verifier
                )
                return idx, (verified, extraction, timing)
            except Exception as e:
                logger.error(f"Error processing text {idx}: {e}")
                # Return failed result
                result = ExtractionResult(success=False, method="failed", error=str(e))
                return idx, (None, result, {"error": str(e)})

    # Create tasks
    tasks = [process_with_semaphore(idx, text) for idx, text in enumerate(texts)]

    # Execute with progress tracking
    if verbose:
        logger.info(f"Processing {len(tasks)} texts asynchronously...")

    results_dict = {}
    completed = 0

    for coro in asyncio.as_completed(tasks):
        idx, result = await coro
        results_dict[idx] = result
        completed += 1

        if verbose and completed % 10 == 0:
            logger.info(f"  Completed: {completed}/{len(texts)}")

    # Sort results by original index
    results = [results_dict[i] for i in range(len(texts))]

    # Calculate statistics
    t_batch_ms = (time.time() - t_batch_start) * 1000
    verified_count = sum(1 for v, e, t in results if v is not None)
    extraction_count = sum(1 for v, e, t in results if e.success)

    stats = {
        "total_texts": len(texts),
        "total_time_ms": t_batch_ms,
        "avg_time_per_text_ms": t_batch_ms / len(texts) if texts else 0,
        "extracted_count": extraction_count,
        "verified_count": verified_count,
        "extraction_rate": extraction_count / len(texts) if texts else 0,
        "verification_rate": verified_count / len(texts) if texts else 0,
    }

    if verbose:
        logger.info(f"\nAsync Batch Processing Complete:")
        logger.info(f"  Total time: {t_batch_ms:.1f}ms")
        logger.info(f"  Avg per text: {stats['avg_time_per_text_ms']:.1f}ms")
        logger.info(f"  Extracted: {extraction_count}/{len(texts)} ({stats['extraction_rate']:.1%})")
        logger.info(f"  Verified: {verified_count}/{len(texts)} ({stats['verification_rate']:.1%})")

    return results, stats


# ============================================================================
# Convenience Wrapper
# ============================================================================

def run_async_extraction(
    texts: List[str],
    use_llm: bool = True,
    use_verification: bool = True,
    llm_model: str = "gpt-3.5-turbo",
    verification_model: str = "gemini-2.0-flash-exp",
    min_confidence_threshold: str = "medium",
    max_concurrent: int = 5,
    verbose: bool = True
) -> Tuple[List[Tuple[Optional[VerifiedBookMeta], ExtractionResult, Dict]], Dict]:
    """
    Convenience function to run async extraction from sync context.

    Args:
        texts: List of texts to process
        use_llm: Use LLM for extraction
        use_verification: Verify with LLM judge
        llm_model: OpenAI model for extraction
        verification_model: Gemini model for verification
        min_confidence_threshold: Minimum confidence to accept
        max_concurrent: Maximum concurrent requests
        verbose: Show progress logging

    Returns:
        Tuple of (results list, statistics dict)

    Example:
        >>> results, stats = run_async_extraction(
        ...     ["Review of '1984' by George Orwell"],
        ...     use_verification=True
        ... )
        >>> verified, extraction, timing = results[0]
        >>> print(f"Time taken: {timing['total_ms']:.1f}ms")
    """
    # Create or get event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        async_batch_extract_and_verify(
            texts=texts,
            use_llm=use_llm,
            use_verification=use_verification,
            llm_model=llm_model,
            verification_model=verification_model,
            min_confidence_threshold=min_confidence_threshold,
            max_concurrent=max_concurrent,
            verbose=verbose
        )
    )


if __name__ == "__main__":
    print("Async Extraction Module")
    print("=" * 60)
    print("\nUsage:")
    print("  from src.models.async_extraction import run_async_extraction")
    print("  results, stats = run_async_extraction(texts)")

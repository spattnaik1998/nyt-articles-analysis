"""
Book Metadata Extraction Module

This module extracts structured book metadata (title and author) from text
using LLM-based extraction with instructor + OpenAI and regex fallbacks.

Based on patterns from the NYT analysis notebook.
"""

import re
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field, validator
import requests
import json
import time

# Instructor and OpenAI imports
try:
    import instructor
    from openai import OpenAI
    INSTRUCTOR_AVAILABLE = True
except ImportError:
    INSTRUCTOR_AVAILABLE = False
    logging.warning("Instructor not available. Install with: pip install instructor openai")

# Google Gemini imports
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Google Generative AI not available. Install with: pip install google-genai")

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================

class BookMeta(BaseModel):
    """
    Book metadata extracted from text.

    Attributes:
        book_title (str): Title of the book
        author_name (str): Name of the author(s)
    """
    book_title: str = Field(
        ...,
        description="The title of the book being reviewed or discussed"
    )
    author_name: str = Field(
        ...,
        description="The name of the book's author(s)"
    )

    @validator('book_title', 'author_name')
    def not_empty(cls, v):
        """Validate that fields are not empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")

        # Check for invalid placeholder values
        v_lower = v.strip().lower()
        invalid_values = [
            'none', 'unknown', 'n/a', 'null', 'not available', 'not specified',
            'various', 'various authors', 'multiple authors', 'anonymous',
            'staff', 'editor', 'editors', 'the editor', 'the editors',
            'correspondent', 'contributor', 'staff writer', 'by staff',
            'unnamed', 'not named', 'not provided', 'tbd', 'tba'
        ]
        if v_lower in invalid_values:
            raise ValueError(f"Invalid value: {v}")

        # Check if it's too short to be a real name/title
        if len(v.strip()) < 2:
            raise ValueError(f"Value too short: {v}")

        return v.strip()

    @validator('book_title')
    def clean_title(cls, v):
        """Clean up book title."""
        # Remove quotes if present
        v = v.strip('"\'')
        # Remove common prefixes
        for prefix in ['Title: ', 'Book: ', 'Review of ', 'Review: ']:
            if v.startswith(prefix):
                v = v[len(prefix):]
        return v.strip()

    @validator('author_name')
    def clean_author(cls, v):
        """Clean up author name."""
        # Remove 'by ' prefix if present
        if v.lower().startswith('by '):
            v = v[3:]
        # Remove 'Author: ' prefix
        if v.startswith('Author: '):
            v = v[8:]
        return v.strip()


class ExtractionResult(BaseModel):
    """
    Result of extraction attempt.

    Attributes:
        success (bool): Whether extraction succeeded
        book_meta (BookMeta, optional): Extracted metadata if successful
        method (str): Method used ('llm', 'regex_pattern_1', 'regex_pattern_2', etc., or 'failed')
        error (str, optional): Error message if failed
    """
    success: bool
    book_meta: Optional[BookMeta] = None
    method: str
    error: Optional[str] = None


class VerificationResult(BaseModel):
    """
    Result of book metadata verification using web search.

    Attributes:
        is_verified (bool): Whether the book and author combination is verified
        confidence (str): Confidence level ('high', 'medium', 'low')
        corrected_title (str, optional): Corrected book title if original was incorrect
        corrected_author (str, optional): Corrected author name if original was incorrect
        reasoning (str): Explanation of verification decision
        search_results_used (int): Number of search results analyzed
    """
    is_verified: bool = Field(
        ...,
        description="True if the book title and author combination is verified as correct"
    )
    confidence: str = Field(
        ...,
        description="Confidence level: 'high', 'medium', or 'low'"
    )
    corrected_title: Optional[str] = Field(
        None,
        description="Corrected book title if the original was incorrect or imprecise"
    )
    corrected_author: Optional[str] = Field(
        None,
        description="Corrected author name if the original was incorrect or imprecise"
    )
    reasoning: str = Field(
        ...,
        description="Explanation of why the book/author combination was verified or rejected"
    )
    search_results_used: int = Field(
        default=0,
        description="Number of search results that were analyzed"
    )


class VerifiedBookMeta(BaseModel):
    """
    Verified book metadata after extraction and verification.

    Attributes:
        book_title (str): Verified book title
        author_name (str): Verified author name
        extraction_method (str): Original extraction method
        verification_confidence (str): Verification confidence level
        was_corrected (bool): Whether the original extraction was corrected
        verification_reasoning (str): Reasoning for verification decision
    """
    book_title: str
    author_name: str
    extraction_method: str
    verification_confidence: str
    was_corrected: bool = False
    verification_reasoning: str = ""


# ============================================================================
# LLM Extraction
# ============================================================================

def get_openai_client(api_key: Optional[str] = None) -> Optional[object]:
    """
    Get instructor-patched OpenAI client.

    Args:
        api_key (str, optional): OpenAI API key. If None, reads from OPENAI_API_KEY env var.

    Returns:
        Instructor-patched OpenAI client or None if not available

    Example:
        >>> client = get_openai_client()
        >>> if client:
        ...     result = extract_with_llm("text", client)
    """
    if not INSTRUCTOR_AVAILABLE:
        logger.warning("Instructor not available")
        return None

    # Get API key
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        logger.warning("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        return None

    try:
        # Create OpenAI client
        openai_client = OpenAI(api_key=api_key)

        # Patch with instructor (compatible with both old and new instructor versions)
        if hasattr(instructor, 'from_openai'):
            client = instructor.from_openai(openai_client)
        elif hasattr(instructor, 'patch'):
            client = instructor.patch(openai_client)
        else:
            raise ImportError("Instructor version not supported. Please upgrade: pip install --upgrade instructor")

        return client

    except Exception as e:
        logger.error(f"Failed to create OpenAI client: {e}")
        return None


def get_gemini_client(api_key: Optional[str] = None, model: str = "gemini-2.0-flash-exp") -> Optional[object]:
    """
    Get configured Gemini client.

    Args:
        api_key (str, optional): Google API key. If None, reads from GOOGLE_API_KEY env var.
        model (str): Gemini model to use (default: gemini-2.0-flash-exp)

    Returns:
        Configured Gemini Client or None if not available

    Example:
        >>> client = get_gemini_client()
        >>> if client:
        ...     result = verify_with_gemini("1984", "George Orwell", search_results, client)
    """
    if not GEMINI_AVAILABLE:
        logger.warning("Google Generative AI not available")
        return None

    # Get API key
    if api_key is None:
        api_key = os.getenv('GOOGLE_API_KEY')

    if not api_key:
        logger.warning("Google API key not found. Set GOOGLE_API_KEY environment variable.")
        return None

    try:
        # Create Gemini client with the new google.genai package
        client = genai.Client(api_key=api_key)

        return (client, model)  # Return tuple of (client, model_name)

    except Exception as e:
        logger.error(f"Failed to create Gemini client: {e}")
        return None


def extract_with_llm(
    text: str,
    client: object,
    model: str = "gpt-3.5-turbo",
    max_retries: int = 2
) -> Optional[BookMeta]:
    """
    Extract book metadata using LLM with instructor.

    Args:
        text (str): Text to extract from (headline, abstract, etc.)
        client: Instructor-patched OpenAI client
        model (str): OpenAI model to use (default: gpt-3.5-turbo)
        max_retries (int): Number of retries on failure (default: 2)

    Returns:
        BookMeta or None: Extracted metadata or None if failed

    Example:
        >>> client = get_openai_client()
        >>> meta = extract_with_llm("Review of '1984' by George Orwell", client)
        >>> print(meta.book_title, meta.author_name)
        1984 George Orwell
    """
    if client is None:
        return None

    try:
        # Use instructor to extract structured data
        book_meta = client.chat.completions.create(
            model=model,
            response_model=BookMeta,
            max_retries=max_retries,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise information extraction system. "
                        "Extract the book title and author name from the given text. "
                        "The text may be a book review, article about a book, or mention of a book.\n\n"
                        "IMPORTANT RULES:\n"
                        "- Extract ONLY real, specific author names (e.g., 'George Orwell', 'Jane Austen')\n"
                        "- DO NOT use placeholders like 'Unknown', 'Various', 'Staff', 'Anonymous', 'Editor'\n"
                        "- DO NOT extract generic terms like 'multiple authors' or 'various authors'\n"
                        "- If no specific author name is found, the extraction should FAIL\n"
                        "- Extract the FULL author name when available (first and last name)\n"
                        "- Return only the title and author, with no additional commentary"
                    )
                },
                {
                    "role": "user",
                    "content": f"Extract the book title and author from this text:\n\n{text}"
                }
            ]
        )

        return book_meta

    except Exception as e:
        logger.debug(f"LLM extraction failed: {e}")
        return None


# ============================================================================
# Regex Fallback Heuristics
# ============================================================================

# Common patterns for book mentions
REGEX_PATTERNS = [
    # Pattern 1: "Title" by Author
    # Example: "1984" by George Orwell
    re.compile(r'["\']([^"\']+)["\'][\s,]*by[\s]+([A-Z][a-zA-Z\s\.]+?)(?:\.|,|;|$)', re.IGNORECASE),

    # Pattern 2: Title by Author (no quotes)
    # Example: 1984 by George Orwell
    re.compile(r'(?:^|\s)([A-Z][a-zA-Z\s:]+?)[\s]+by[\s]+([A-Z][a-zA-Z\s\.]+?)(?:\.|,|;|$)'),

    # Pattern 3: Author's "Title"
    # Example: George Orwell's "1984"
    re.compile(r'([A-Z][a-zA-Z\s\.]+?)[\']s[\s]+["\']([^"\']+)["\']'),

    # Pattern 4: "Title," by Author
    # Example: "The Great Gatsby," by F. Scott Fitzgerald
    re.compile(r'["\']([^"\']+)["\'],?[\s]*by[\s]+([A-Z][a-zA-Z\s\.]+?)(?:\.|,|;|$)', re.IGNORECASE),

    # Pattern 5: Title (Author)
    # Example: 1984 (George Orwell)
    re.compile(r'([A-Z][a-zA-Z\s:]+?)\s*\(([A-Z][a-zA-Z\s\.]+?)\)'),

    # Pattern 6: Review of "Title" by Author
    # Example: Review of "1984" by George Orwell
    re.compile(r'Review of ["\']([^"\']+)["\'][\s,]*by[\s]+([A-Z][a-zA-Z\s\.]+?)(?:\.|,|;|$)', re.IGNORECASE),

    # Pattern 7: Author, "Title"
    # Example: George Orwell, "1984"
    re.compile(r'([A-Z][a-zA-Z\s\.]+?),[\s]*["\']([^"\']+)["\']'),
]


def extract_with_regex(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract book title and author using regex patterns.

    Tries multiple common patterns and returns the first match.

    Args:
        text (str): Text to extract from

    Returns:
        Tuple: (title, author, pattern_name) or (None, None, None) if no match

    Example:
        >>> title, author, pattern = extract_with_regex('"1984" by George Orwell')
        >>> print(title, author)
        1984 George Orwell
    """
    for idx, pattern in enumerate(REGEX_PATTERNS, 1):
        match = pattern.search(text)

        if match:
            groups = match.groups()

            if len(groups) >= 2:
                # Different patterns capture title and author in different orders
                if idx == 3:  # Author's "Title" pattern
                    author, title = groups[0], groups[1]
                else:
                    title, author = groups[0], groups[1]

                # Clean up
                title = title.strip()
                author = author.strip()

                # Validate - basic checks
                if len(title) > 3 and len(author) > 2:
                    # Check author has at least one capital letter (name)
                    if any(c.isupper() for c in author):
                        return title, author, f'regex_pattern_{idx}'

    return None, None, None


# ============================================================================
# Main Extraction Functions
# ============================================================================

def extract_book_meta(
    text: str,
    use_llm: bool = True,
    openai_client: Optional[object] = None,
    llm_model: str = "gpt-3.5-turbo"
) -> ExtractionResult:
    """
    Extract book metadata from text using LLM + regex fallback.

    Tries LLM first (if available), then falls back to regex patterns.

    Args:
        text (str): Text to extract from (headline, abstract, combined)
        use_llm (bool): Whether to try LLM extraction first (default: True)
        openai_client: Instructor-patched OpenAI client (default: None, will create)
        llm_model (str): OpenAI model to use (default: gpt-3.5-turbo)

    Returns:
        ExtractionResult: Result with success status, BookMeta, and method used

    Example:
        >>> result = extract_book_meta("Review of '1984' by George Orwell")
        >>> if result.success:
        ...     print(result.book_meta.book_title)
        ...     print(f"Method: {result.method}")
        1984
        Method: llm
    """
    if not text or not text.strip():
        return ExtractionResult(
            success=False,
            method='failed',
            error='Empty text'
        )

    # Try LLM extraction first
    if use_llm:
        try:
            # Get or create client
            if openai_client is None:
                openai_client = get_openai_client()

            if openai_client:
                book_meta = extract_with_llm(text, openai_client, model=llm_model)

                if book_meta:
                    return ExtractionResult(
                        success=True,
                        book_meta=book_meta,
                        method='llm'
                    )

        except Exception as e:
            logger.debug(f"LLM extraction failed: {e}")

    # Try regex fallback
    title, author, pattern = extract_with_regex(text)

    if title and author:
        try:
            book_meta = BookMeta(book_title=title, author_name=author)
            return ExtractionResult(
                success=True,
                book_meta=book_meta,
                method=pattern
            )
        except Exception as e:
            logger.debug(f"Regex extraction validation failed: {e}")

    # All methods failed
    return ExtractionResult(
        success=False,
        method='failed',
        error='No extraction method succeeded'
    )


# ============================================================================
# Web Search Verification (Tavily API)
# ============================================================================

def search_book_info(
    book_title: str,
    author_name: str,
    tavily_api_key: Optional[str] = None,
    num_results: int = 5,
    max_retries: int = 3
) -> Optional[List[Dict]]:
    """
    Search for book information using Tavily API with rate limiting and retry logic.

    Uses Tavily Search API to find information about the book and author
    to verify the extraction is correct. Includes exponential backoff for rate limits.

    Args:
        book_title (str): Book title to search
        author_name (str): Author name to search
        tavily_api_key (str, optional): Tavily API key. If None, reads from TAVILY_API_KEY env var.
        num_results (int): Number of search results to return (default: 5)
        max_retries (int): Maximum number of retries for rate limit errors (default: 3)

    Returns:
        List[Dict] or None: List of search results with titles, snippets, and links, or None if failed

    Example:
        >>> results = search_book_info("1984", "George Orwell")
        >>> if results:
        ...     print(f"Found {len(results)} search results")
    """
    # Get API key
    if tavily_api_key is None:
        tavily_api_key = os.getenv('TAVILY_API_KEY')

    if not tavily_api_key:
        logger.warning("Tavily API key not found. Set TAVILY_API_KEY environment variable.")
        return None

    # Construct search query
    query = f'"{book_title}" by {author_name} book'

    # Tavily API endpoint
    url = "https://api.tavily.com/search"

    # Request payload
    payload = {
        "api_key": tavily_api_key,
        "query": query,
        "max_results": num_results,
        "search_depth": "basic",  # Use 'basic' for faster results, 'advanced' for more thorough
        "include_answer": False,
        "include_raw_content": False
    }

    # Retry logic with exponential backoff
    for attempt in range(max_retries + 1):
        try:
            # Add small delay to avoid hitting rate limits (0.5 seconds between requests)
            if attempt > 0:
                # Exponential backoff: 2, 4, 8 seconds
                wait_time = 2 ** attempt
                logger.info(f"Rate limit hit, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries + 1})")
                time.sleep(wait_time)
            else:
                # Small delay even on first attempt to avoid overwhelming API
                time.sleep(0.5)

            # Make request
            response = requests.post(url, json=payload, timeout=15)

            # Handle rate limiting (429 error)
            if response.status_code == 429:
                if attempt < max_retries:
                    logger.warning(f"Tavily rate limit (429) - attempt {attempt + 1}/{max_retries + 1}")
                    continue  # Retry with backoff
                else:
                    logger.error(f"Tavily rate limit exceeded after {max_retries + 1} attempts")
                    return None

            # Check for other errors
            if response.status_code != 200:
                logger.error(f"Tavily API error {response.status_code}: {response.text}")

            response.raise_for_status()

            # Parse response
            data = response.json()

            # Extract results from Tavily format
            results = []
            if 'results' in data:
                for item in data['results']:
                    results.append({
                        'title': item.get('title', ''),
                        'snippet': item.get('content', ''),  # Tavily uses 'content' instead of 'snippet'
                        'link': item.get('url', '')  # Tavily uses 'url' instead of 'link'
                    })

            logger.debug(f"Tavily search returned {len(results)} results for '{book_title}' by {author_name}")
            return results if results else None

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and attempt < max_retries:
                continue  # Retry with backoff
            logger.error(f"Tavily API HTTP error: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Tavily API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing Tavily response: {e}")
            return None

    return None


def verify_with_llm(
    book_title: str,
    author_name: str,
    search_results: List[Dict],
    client: object,
    model: str = "gpt-4o",
    max_retries: int = 2
) -> Optional[VerificationResult]:
    """
    Verify book metadata using LLM with web search results.

    The LLM analyzes search results to determine if the book title and author
    combination is correct, and can suggest corrections if needed.

    Args:
        book_title (str): Extracted book title to verify
        author_name (str): Extracted author name to verify
        search_results (List[Dict]): Search results from SERPER
        client: Instructor-patched OpenAI client
        model (str): OpenAI model to use (default: gpt-4o for better reasoning)
        max_retries (int): Number of retries on failure (default: 2)

    Returns:
        VerificationResult or None: Verification result with confidence and corrections

    Example:
        >>> client = get_openai_client()
        >>> results = search_book_info("1984", "George Orwell")
        >>> verification = verify_with_llm("1984", "George Orwell", results, client)
        >>> print(verification.is_verified, verification.confidence)
    """
    if client is None or not search_results:
        return None

    # Format search results for LLM
    search_context = "\n\n".join([
        f"Result {i+1}:\nTitle: {r['title']}\nSnippet: {r['snippet']}\nSource: {r['link']}"
        for i, r in enumerate(search_results[:5])  # Use top 5 results
    ])

    try:
        # Use instructor to get structured verification
        verification = client.chat.completions.create(
            model=model,
            response_model=VerificationResult,
            max_retries=max_retries,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a meticulous book metadata verification system. "
                        "Your task is to verify if the extracted book title and author name are correct "
                        "by analyzing web search results.\n\n"
                        "VERIFICATION GUIDELINES:\n"
                        "1. HIGH CONFIDENCE: If multiple search results clearly confirm the exact book title and author combination\n"
                        "2. MEDIUM CONFIDENCE: If search results suggest the book/author but with slight variations in title format\n"
                        "3. LOW CONFIDENCE: If search results are ambiguous or contradictory\n\n"
                        "CORRECTION GUIDELINES:\n"
                        "- Provide corrected_title if the title format is incorrect (e.g., missing subtitle, wrong capitalization)\n"
                        "- Provide corrected_author if the author name is incomplete (e.g., missing middle initial, wrong order)\n"
                        "- Set is_verified=False if the book and author don't match or if this appears to be a completely wrong extraction\n\n"
                        "IMPORTANT GUARDRAILS:\n"
                        "- Only verify books that actually exist based on search results\n"
                        "- Be strict: if search results don't clearly confirm the book/author, mark as not verified\n"
                        "- Look for authoritative sources (publishers, libraries, book databases) in search results\n"
                        "- Be cautious of partial matches or similar book titles by different authors\n"
                        "- Always provide clear reasoning for your decision"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Please verify this book metadata:\n\n"
                        f"EXTRACTED INFORMATION:\n"
                        f"Book Title: {book_title}\n"
                        f"Author Name: {author_name}\n\n"
                        f"WEB SEARCH RESULTS:\n{search_context}\n\n"
                        f"Based on these search results, verify if the book title and author are correct. "
                        f"If you find the correct information with slight variations, provide corrections. "
                        f"Be strict and only verify if you're confident based on the search results."
                    )
                }
            ]
        )

        # Use model_copy to update the search_results_used field (Pydantic v2 compatible)
        verification = verification.model_copy(update={'search_results_used': len(search_results[:5])})
        return verification

    except Exception as e:
        logger.debug(f"LLM verification failed: {e}")
        return None


def verify_with_gemini(
    book_title: str,
    author_name: str,
    client_tuple: tuple,
    max_retries: int = 2
) -> Optional[VerificationResult]:
    """
    Verify book metadata using Gemini's internal knowledge.

    Uses Google's Gemini model to verify if the book title and author combination
    is correct based on its training data, with the ability to suggest corrections.

    Args:
        book_title (str): Extracted book title to verify
        author_name (str): Extracted author name to verify
        client_tuple: Tuple of (Gemini Client, model_name)
        max_retries (int): Number of retries on failure (default: 2)

    Returns:
        VerificationResult or None: Verification result with confidence and corrections

    Example:
        >>> client = get_gemini_client()
        >>> verification = verify_with_gemini("1984", "George Orwell", client)
        >>> print(verification.is_verified, verification.confidence)
    """
    if client_tuple is None:
        return None

    # Unpack client and model name
    client, model_name = client_tuple

    # Create the prompt
    system_prompt = """You are a meticulous book metadata verification system.
Your task is to verify if the extracted book title and author name are correct using your knowledge of books and literature.

VERIFICATION GUIDELINES:
1. HIGH CONFIDENCE: If you are certain this book exists and was written by this author
2. MEDIUM CONFIDENCE: If the book/author combination is likely correct but might have slight variations
3. LOW CONFIDENCE: If you are uncertain about this combination

CORRECTION GUIDELINES:
- Provide corrected_title if you know the correct full title (e.g., with subtitle)
- Provide corrected_author if you know the author's full name or correct spelling
- Set is_verified=False if the book and author don't match or this is clearly incorrect

STRICT VALIDATION RULES:
- REJECT placeholder values like "Unknown", "Various", "Staff", "Anonymous", "Editor"
- REJECT generic terms like "multiple authors" or "contributors"
- ONLY verify if you have specific knowledge of this book and author
- Be strict: if uncertain, mark as not verified
- The author must be a REAL person with a specific name
- The book must be a REAL published work

You MUST respond with a JSON object with this exact structure:
{
  "is_verified": true or false,
  "confidence": "high" or "medium" or "low",
  "corrected_title": "corrected title" or null,
  "corrected_author": "corrected author" or null,
  "reasoning": "explanation of your decision",
  "search_results_used": 0
}"""

    user_prompt = f"""Please verify this book metadata using your knowledge:

EXTRACTED INFORMATION:
Book Title: {book_title}
Author Name: {author_name}

Based on your knowledge of books and literature, verify if this book and author combination is correct.
- If the author name is a placeholder like "Unknown", "Various", "Staff", or similar, mark as NOT verified.
- If you don't recognize this specific book by this specific author, mark as NOT verified.
- Only mark as verified if you are confident this is a real book by a real author.

Respond with ONLY the JSON object, no additional text."""

    try:
        for attempt in range(max_retries + 1):
            try:
                # Generate content with Gemini using new API
                response = client.models.generate_content(
                    model=model_name,
                    contents=f"{system_prompt}\n\n{user_prompt}",
                    config={
                        "temperature": 0.1,  # Low temperature for consistency
                        "response_mime_type": "application/json"
                    }
                )

                # Extract JSON from response
                response_text = response.text.strip()

                # Remove markdown code blocks if present
                if response_text.startswith('```'):
                    # Remove ```json and ``` markers
                    response_text = response_text.split('\n', 1)[1].rsplit('\n', 1)[0]
                    if response_text.startswith('json'):
                        response_text = response_text[4:].strip()

                # Parse JSON response
                result_dict = json.loads(response_text)

                # Create VerificationResult from dict
                verification = VerificationResult(
                    is_verified=result_dict.get('is_verified', False),
                    confidence=result_dict.get('confidence', 'low'),
                    corrected_title=result_dict.get('corrected_title'),
                    corrected_author=result_dict.get('corrected_author'),
                    reasoning=result_dict.get('reasoning', ''),
                    search_results_used=0  # No search results used
                )

                return verification

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                if attempt < max_retries:
                    logger.debug(f"Gemini verification attempt {attempt + 1} failed, retrying: {e}")
                    continue
                else:
                    logger.debug(f"Gemini verification failed after {max_retries + 1} attempts: {e}")
                    return None

    except Exception as e:
        logger.debug(f"Gemini verification failed: {e}")
        return None


def extract_and_verify_book_meta(
    text: str,
    use_llm: bool = True,
    use_verification: bool = True,
    openai_client: Optional[object] = None,
    llm_model: str = "gpt-3.5-turbo",
    verification_model: str = "gemini-2.0-flash-exp",
    use_gemini_for_verification: bool = True,
    tavily_api_key: Optional[str] = None,
    min_confidence_threshold: str = "medium"
) -> Tuple[Optional[VerifiedBookMeta], ExtractionResult]:
    """
    Extract and verify book metadata using LLM judge.

    This is the main function that combines extraction and verification:
    1. Extract book metadata using OpenAI LLM or regex
    2. Use Gemini (default) or OpenAI as LLM judge to verify using internal knowledge
    3. Return verified metadata with confidence scores

    Args:
        text (str): Text to extract from
        use_llm (bool): Use LLM for extraction (default: True)
        use_verification (bool): Verify extraction with LLM judge (default: True)
        openai_client: OpenAI client for extraction (default: None, will create)
        llm_model (str): OpenAI model for extraction (default: gpt-3.5-turbo)
        verification_model (str): Model for verification (default: gemini-2.0-flash-exp)
        use_gemini_for_verification (bool): Use Gemini for verification instead of OpenAI (default: True)
        tavily_api_key (str, optional): Not used (kept for backward compatibility)
        min_confidence_threshold (str): Minimum confidence to accept ('high', 'medium', 'low')

    Returns:
        Tuple[VerifiedBookMeta or None, ExtractionResult]: Verified metadata and extraction result

    Example:
        >>> verified, extraction = extract_and_verify_book_meta("Review of '1984' by George Orwell")
        >>> if verified:
        ...     print(f"{verified.book_title} by {verified.author_name}")
        ...     print(f"Confidence: {verified.verification_confidence}")
    """
    # Step 1: Extract book metadata
    extraction_result = extract_book_meta(
        text=text,
        use_llm=use_llm,
        openai_client=openai_client,
        llm_model=llm_model
    )

    # If extraction failed, return immediately
    if not extraction_result.success:
        return None, extraction_result

    book_title = extraction_result.book_meta.book_title
    author_name = extraction_result.book_meta.author_name

    # If verification is disabled, return unverified extraction
    if not use_verification:
        verified_meta = VerifiedBookMeta(
            book_title=book_title,
            author_name=author_name,
            extraction_method=extraction_result.method,
            verification_confidence="unverified",
            was_corrected=False,
            verification_reasoning="Verification disabled"
        )
        return verified_meta, extraction_result

    # Step 2: Verify with Gemini or OpenAI LLM (no web search)
    if use_gemini_for_verification:
        # Use Gemini for verification
        gemini_client = get_gemini_client(model=verification_model)

        if gemini_client is None:
            # No Gemini client available, return unverified
            verified_meta = VerifiedBookMeta(
                book_title=book_title,
                author_name=author_name,
                extraction_method=extraction_result.method,
                verification_confidence="unverified",
                was_corrected=False,
                verification_reasoning="Gemini verification unavailable"
            )
            return verified_meta, extraction_result

        verification = verify_with_gemini(
            book_title=book_title,
            author_name=author_name,
            client_tuple=gemini_client
        )
    else:
        # Use OpenAI for verification
        if openai_client is None:
            openai_client = get_openai_client()

        if openai_client is None:
            # No client available, return unverified
            verified_meta = VerifiedBookMeta(
                book_title=book_title,
                author_name=author_name,
                extraction_method=extraction_result.method,
                verification_confidence="unverified",
                was_corrected=False,
                verification_reasoning="LLM verification unavailable"
            )
            return verified_meta, extraction_result

        verification = verify_with_llm(
            book_title=book_title,
            author_name=author_name,
            search_results=search_results,
            client=openai_client,
            model=verification_model
        )

    # If verification failed, return unverified
    if verification is None:
        verified_meta = VerifiedBookMeta(
            book_title=book_title,
            author_name=author_name,
            extraction_method=extraction_result.method,
            verification_confidence="unverified",
            was_corrected=False,
            verification_reasoning="Verification failed"
        )
        return verified_meta, extraction_result

    # Step 4: Apply verification results with guardrails
    # Check if confidence meets threshold
    confidence_levels = {'low': 1, 'medium': 2, 'high': 3}
    min_level = confidence_levels.get(min_confidence_threshold, 2)
    actual_level = confidence_levels.get(verification.confidence, 1)

    if not verification.is_verified or actual_level < min_level:
        # Verification failed or confidence too low
        logger.debug(
            f"Verification failed for '{book_title}' by {author_name}: "
            f"{verification.reasoning}"
        )
        return None, extraction_result

    # Apply corrections if provided
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

    return verified_meta, extraction_result


# ============================================================================
# Batch Extraction with Verification
# ============================================================================

def batch_extract(
    df: pd.DataFrame,
    text_col: str = 'combined_text',
    use_llm: bool = True,
    llm_model: str = "gpt-3.5-turbo",
    max_workers: int = 10,
    output_dir: str = 'data/extractions',
    save_by_year: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Batch extract book metadata with parallel processing.

    Processes DataFrame in parallel, extracts book metadata, and saves results.
    Produces success/fail statistics matching notebook (~99.9% success rate).

    Args:
        df (pd.DataFrame): Input DataFrame with book reviews/articles
        text_col (str): Column containing text to extract from (default: 'combined_text')
        use_llm (bool): Whether to use LLM extraction (default: True)
        llm_model (str): OpenAI model to use (default: gpt-3.5-turbo)
        max_workers (int): Number of parallel workers (default: 10)
        output_dir (str): Base output directory (default: data/extractions)
        save_by_year (bool): Save separate files per year (default: True)
        verbose (bool): Show progress and statistics (default: True)

    Returns:
        pd.DataFrame: DataFrame with added columns:
            - book_title: Extracted book title
            - author_name: Extracted author name
            - extraction_method: Method used (llm, regex_pattern_N, or failed)
            - extraction_success: Boolean success flag

    Example:
        >>> df = pd.DataFrame({
        ...     'text': ['"1984" by George Orwell', '"Brave New World" by Aldous Huxley'],
        ...     'pub_date': ['2020-01-01', '2020-06-01']
        ... })
        >>> result_df = batch_extract(df, text_col='text', use_llm=False)
        >>> print(f"Success rate: {result_df['extraction_success'].mean():.1%}")
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame")

    if verbose:
        logger.info("=" * 70)
        logger.info("Book Metadata Extraction - Batch Processing")
        logger.info("=" * 70)
        logger.info(f"Total articles: {len(df):,}")
        logger.info(f"Text column: {text_col}")
        logger.info(f"Use LLM: {use_llm}")
        if use_llm:
            logger.info(f"LLM model: {llm_model}")
        logger.info(f"Max workers: {max_workers}")

    # Check if OpenAI client is available (if using LLM)
    if use_llm:
        test_client = get_openai_client()
        if test_client is None:
            logger.warning("OpenAI client not available. Falling back to regex only.")
            use_llm = False

    # Prepare texts
    texts = df[text_col].fillna('').astype(str).tolist()

    # Results storage
    results = [None] * len(texts)

    # Parallel extraction
    if verbose:
        logger.info(f"\nExtracting metadata from {len(texts):,} articles...")

    def extract_single(idx: int, text: str) -> Tuple[int, ExtractionResult]:
        """Extract for single text (for parallel processing).

        Note: Each thread creates its own OpenAI client to avoid race conditions.
        """
        result = extract_book_meta(
            text,
            use_llm=use_llm,
            openai_client=None,  # Let each thread create its own client
            llm_model=llm_model
        )
        return idx, result

    # Process in parallel with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(extract_single, idx, text): idx
            for idx, text in enumerate(texts)
        }

        # Collect results with progress bar
        if verbose:
            progress = tqdm(total=len(futures), desc="Extracting")

        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

            if verbose:
                progress.update(1)

        if verbose:
            progress.close()

    # Add results to DataFrame
    result_df = df.copy()

    book_titles = []
    author_names = []
    methods = []
    successes = []

    for result in results:
        if result.success:
            book_titles.append(result.book_meta.book_title)
            author_names.append(result.book_meta.author_name)
            methods.append(result.method)
            successes.append(True)
        else:
            book_titles.append(None)
            author_names.append(None)
            methods.append('failed')
            successes.append(False)

    result_df['book_title'] = book_titles
    result_df['author_name'] = author_names
    result_df['extraction_method'] = methods
    result_df['extraction_success'] = successes

    # Statistics
    if verbose:
        logger.info("\n" + "=" * 70)
        logger.info("Extraction Statistics")
        logger.info("=" * 70)

        total = len(result_df)
        successful = result_df['extraction_success'].sum()
        failed = total - successful
        success_rate = successful / total if total > 0 else 0

        logger.info(f"\nTotal: {total:,}")
        logger.info(f"Successful: {successful:,} ({success_rate:.2%})")
        logger.info(f"Failed: {failed:,} ({(1-success_rate):.2%})")

        # Method breakdown
        method_counts = result_df['extraction_method'].value_counts()
        logger.info(f"\nMethods used:")
        for method, count in method_counts.items():
            pct = count / total * 100
            logger.info(f"  {method:20s}: {count:8,} ({pct:5.1f}%)")

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if save_by_year and 'pub_date' in result_df.columns:
            # Ensure pub_date is datetime
            if not pd.api.types.is_datetime64_any_dtype(result_df['pub_date']):
                result_df['pub_date'] = pd.to_datetime(result_df['pub_date'], errors='coerce')

            # Save by year
            years = result_df['pub_date'].dt.year.unique()
            years = years[~pd.isna(years)]

            if verbose:
                logger.info(f"\nSaving results by year to {output_path}/")

            for year in sorted(years):
                year_df = result_df[result_df['pub_date'].dt.year == year]
                year_file = output_path / f'books_{int(year)}.parquet'

                year_df.to_parquet(year_file, index=False)

                if verbose:
                    year_success = year_df['extraction_success'].sum()
                    year_total = len(year_df)
                    year_rate = year_success / year_total if year_total > 0 else 0
                    logger.info(f"  {int(year)}: {year_total:5,} articles ({year_rate:.2%} success) -> {year_file.name}")

        else:
            # Save all in one file
            output_file = output_path / 'books_all.parquet'
            result_df.to_parquet(output_file, index=False)

            if verbose:
                logger.info(f"\nSaved results to {output_file}")

    if verbose:
        logger.info("\n" + "=" * 70)
        logger.info("Batch Extraction Complete!")
        logger.info("=" * 70)

    return result_df


# ============================================================================
# Utility Functions
# ============================================================================

def get_extraction_stats(df: pd.DataFrame) -> Dict:
    """
    Get statistics from extraction results DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with extraction results

    Returns:
        Dict: Statistics including success rate, method breakdown, etc.

    Example:
        >>> stats = get_extraction_stats(result_df)
        >>> print(f"Success rate: {stats['success_rate']:.2%}")
    """
    if 'extraction_success' not in df.columns:
        raise ValueError("DataFrame does not contain extraction results")

    total = len(df)
    successful = df['extraction_success'].sum()
    failed = total - successful

    stats = {
        'total': total,
        'successful': int(successful),
        'failed': int(failed),
        'success_rate': float(successful / total) if total > 0 else 0.0,
        'method_breakdown': df['extraction_method'].value_counts().to_dict()
    }

    return stats


def batch_extract_with_verification(
    df: pd.DataFrame,
    text_col: str = 'combined_text',
    use_llm: bool = True,
    use_verification: bool = True,
    llm_model: str = "gpt-3.5-turbo",
    verification_model: str = "gemini-2.0-flash-exp",
    use_gemini_for_verification: bool = True,
    min_confidence_threshold: str = "medium",
    max_workers: int = 2,  # Reduced to 2 to avoid Tavily rate limits (429 errors)
    verbose: bool = True
) -> pd.DataFrame:
    """
    Batch extract and verify book metadata using LLM judge.

    This function:
    1. Extracts book metadata from each article using OpenAI
    2. Uses Gemini (default) or OpenAI as LLM judge to verify using internal knowledge
    3. Filters out invalid author names (Various, Unknown, Staff, etc.)
    4. Only includes high-confidence verified extractions in final results

    Args:
        df (pd.DataFrame): Input DataFrame with book reviews/articles
        text_col (str): Column containing text (default: 'combined_text')
        use_llm (bool): Use LLM for extraction (default: True)
        use_verification (bool): Verify with web search (default: True)
        llm_model (str): OpenAI model for extraction (default: gpt-3.5-turbo)
        verification_model (str): Model for verification (default: gemini-2.0-flash-exp)
        use_gemini_for_verification (bool): Use Gemini for verification (default: True)
        min_confidence_threshold (str): Minimum confidence ('high', 'medium', 'low')
        max_workers (int): Parallel workers (default: 5, reduced for API limits)
        verbose (bool): Show progress (default: True)

    Returns:
        pd.DataFrame: DataFrame with verified book metadata columns:
            - book_title: Verified book title
            - author_name: Verified author name
            - extraction_method: Extraction method used
            - verification_confidence: Confidence level
            - was_corrected: Whether extraction was corrected
            - verification_reasoning: Reason for verification decision
            - extraction_success: True only if verified

    Example:
        >>> result_df = batch_extract_with_verification(
        ...     df,
        ...     text_col='combined_text',
        ...     min_confidence_threshold='high'
        ... )
        >>> verified = result_df[result_df['extraction_success'] == True]
        >>> print(f"Verified: {len(verified)}/{len(result_df)}")
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame")

    if verbose:
        logger.info("=" * 70)
        logger.info("Book Metadata Extraction with Verification")
        logger.info("=" * 70)
        logger.info(f"Total articles: {len(df):,}")
        logger.info(f"Use LLM: {use_llm} (model: {llm_model})")
        verifier = "Gemini" if use_gemini_for_verification else "OpenAI"
        logger.info(f"Use verification: {use_verification} (model: {verification_model}, provider: {verifier})")
        logger.info(f"Min confidence: {min_confidence_threshold}")
        logger.info(f"Max workers: {max_workers}")

    # Prepare texts
    texts = df[text_col].fillna('').astype(str).tolist()

    # Results storage
    results = [None] * len(texts)

    # Parallel extraction and verification
    if verbose:
        logger.info(f"\nExtracting and verifying {len(texts):,} articles...")

    def extract_and_verify_single(idx: int, text: str):
        """Extract and verify single text."""
        verified, extraction = extract_and_verify_book_meta(
            text=text,
            use_llm=use_llm,
            use_verification=use_verification,
            openai_client=None,  # Each thread creates its own
            llm_model=llm_model,
            verification_model=verification_model,
            use_gemini_for_verification=use_gemini_for_verification,
            min_confidence_threshold=min_confidence_threshold
        )
        return idx, verified, extraction

    # Process in parallel with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_and_verify_single, idx, text): idx
            for idx, text in enumerate(texts)
        }

        if verbose:
            progress = tqdm(total=len(futures), desc="Extracting & Verifying")

        for future in as_completed(futures):
            idx, verified, extraction = future.result()
            results[idx] = (verified, extraction)

            if verbose:
                progress.update(1)

        if verbose:
            progress.close()

    # Add results to DataFrame
    result_df = df.copy()

    book_titles = []
    author_names = []
    extraction_methods = []
    verification_confidences = []
    was_corrected_list = []
    verification_reasonings = []
    successes = []

    for verified, extraction in results:
        if verified:
            book_titles.append(verified.book_title)
            author_names.append(verified.author_name)
            extraction_methods.append(verified.extraction_method)
            verification_confidences.append(verified.verification_confidence)
            was_corrected_list.append(verified.was_corrected)
            verification_reasonings.append(verified.verification_reasoning)
            successes.append(True)
        else:
            book_titles.append(None)
            author_names.append(None)
            extraction_methods.append(extraction.method if extraction.success else 'failed')
            verification_confidences.append('failed')
            was_corrected_list.append(False)
            verification_reasonings.append(extraction.error or 'Verification failed')
            successes.append(False)

    result_df['book_title'] = book_titles
    result_df['author_name'] = author_names
    result_df['extraction_method'] = extraction_methods
    result_df['verification_confidence'] = verification_confidences
    result_df['was_corrected'] = was_corrected_list
    result_df['verification_reasoning'] = verification_reasonings
    result_df['extraction_success'] = successes

    # Statistics
    if verbose:
        logger.info("\n" + "=" * 70)
        logger.info("Verification Statistics")
        logger.info("=" * 70)

        total = len(result_df)
        successful = sum(successes)
        failed = total - successful
        success_rate = successful / total if total > 0 else 0

        logger.info(f"\nTotal: {total:,}")
        logger.info(f"Verified: {successful:,} ({success_rate:.2%})")
        logger.info(f"Failed/Unverified: {failed:,} ({(1-success_rate):.2%})")

        # Confidence breakdown
        conf_counts = result_df['verification_confidence'].value_counts()
        logger.info(f"\nConfidence levels:")
        for conf, count in conf_counts.items():
            pct = count / total * 100
            logger.info(f"  {conf:15s}: {count:8,} ({pct:5.1f}%)")

        # Corrections
        corrected = sum(was_corrected_list)
        if corrected > 0:
            logger.info(f"\nCorrected extractions: {corrected:,}")

        # Method breakdown
        method_counts = result_df['extraction_method'].value_counts()
        logger.info(f"\nExtraction methods:")
        for method, count in method_counts.items():
            pct = count / total * 100
            logger.info(f"  {method:20s}: {count:8,} ({pct:5.1f}%)")

    if verbose:
        logger.info("\n" + "=" * 70)
        logger.info("Batch Extraction with Verification Complete!")
        logger.info("=" * 70)

    return result_df


def filter_successful_extractions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to only successful extractions.

    Args:
        df (pd.DataFrame): DataFrame with extraction results

    Returns:
        pd.DataFrame: Filtered DataFrame

    Example:
        >>> successful_df = filter_successful_extractions(result_df)
        >>> print(len(successful_df))
    """
    if 'extraction_success' not in df.columns:
        raise ValueError("DataFrame does not contain extraction results")

    return df[df['extraction_success'] == True].copy()


if __name__ == "__main__":
    # Example usage
    print("Book Metadata Extraction Module")
    print("=" * 70)

    # Example 1: Single text extraction
    print("\nExample 1: Single Text Extraction")
    print("-" * 70)

    sample_text = 'Review of "The Great Gatsby" by F. Scott Fitzgerald'
    print(f"Text: {sample_text}")

    result = extract_book_meta(sample_text, use_llm=False)

    if result.success:
        print(f"\n✓ Extraction successful!")
        print(f"  Book Title: {result.book_meta.book_title}")
        print(f"  Author: {result.book_meta.author_name}")
        print(f"  Method: {result.method}")
    else:
        print(f"\n✗ Extraction failed: {result.error}")

    # Example 2: Batch extraction
    print("\n\nExample 2: Batch Extraction")
    print("-" * 70)

    sample_df = pd.DataFrame({
        'text': [
            '"1984" by George Orwell',
            '"To Kill a Mockingbird" by Harper Lee',
            'George Orwell\'s "Animal Farm"',
            'Review of "The Catcher in the Rye" by J.D. Salinger',
            'Some random text without book info'
        ],
        'pub_date': pd.to_datetime(['2020-01-01', '2020-06-01', '2021-01-01', '2021-06-01', '2022-01-01'])
    })

    print(f"Processing {len(sample_df)} sample texts...")

    result_df = batch_extract(
        sample_df,
        text_col='text',
        use_llm=False,  # Use regex only for demo
        output_dir=None,  # Don't save
        verbose=True
    )

    print("\nResults:")
    print(result_df[['text', 'book_title', 'author_name', 'extraction_method']])

    # Statistics
    stats = get_extraction_stats(result_df)
    print(f"\nSuccess rate: {stats['success_rate']:.1%}")

    print("\n" + "=" * 70)
    print("For full examples, see:")
    print("  - examples/extract_books.py")
    print("  - docs/extraction.md")
    print("=" * 70)

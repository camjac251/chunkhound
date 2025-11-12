"""Anthropic LLM provider implementation for ChunkHound deep research."""

import asyncio
import json
from typing import Any

from loguru import logger

from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse

try:
    from anthropic import AsyncAnthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    AsyncAnthropic = None  # type: ignore
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic package not available")


class AnthropicLLMProvider(LLMProvider):
    """Anthropic LLM provider using Claude models."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20250929",
        base_url: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """Initialize Anthropic LLM provider.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model name to use
                Latest models (2025):
                - claude-sonnet-4-5-20250929: Sonnet 4.5 (best intelligence/speed balance)
                - claude-haiku-4-5-20251001: Haiku 4.5 (fastest, real-time use)
                - claude-opus-4-1-20250805: Opus 4.1 (most capable for complex tasks)
            base_url: Base URL for Anthropic API (optional for custom endpoints)
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts for failed requests
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not available")

        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries

        # Initialize client
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = AsyncAnthropic(**client_kwargs)

        # Usage tracking
        self._requests_made = 0
        self._tokens_used = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

    @property
    def name(self) -> str:
        """Provider name."""
        return "anthropic"

    @property
    def model(self) -> str:
        """Model name."""
        return self._model

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ) -> LLMResponse:
        """Generate a completion for the given prompt.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout in seconds (overrides default)
        """
        # Build messages list (Anthropic separates system from messages)
        messages = [{"role": "user", "content": prompt}]

        # Use provided timeout or fall back to default
        request_timeout = timeout if timeout is not None else self._timeout

        try:
            # Build request kwargs
            request_kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "max_tokens": max_completion_tokens,
                "timeout": request_timeout,
            }

            # Add system prompt if provided (Anthropic uses separate system parameter)
            if system:
                request_kwargs["system"] = system

            response = await self._client.messages.create(**request_kwargs)

            # Update usage statistics
            self._requests_made += 1
            if response.usage:
                self._prompt_tokens += response.usage.input_tokens
                self._completion_tokens += response.usage.output_tokens
                self._tokens_used += response.usage.input_tokens + response.usage.output_tokens

            # Extract response content
            # Anthropic returns content as a list of blocks
            content_blocks = response.content
            if not content_blocks:
                logger.error(
                    f"Anthropic returned no content blocks (stop_reason={response.stop_reason})"
                )
                raise RuntimeError(
                    f"LLM returned empty response (stop_reason={response.stop_reason}). "
                    "This may indicate a content filter, API error, or model refusal."
                )

            # Concatenate text from all content blocks
            content_parts = []
            for block in content_blocks:
                if hasattr(block, "text"):
                    content_parts.append(block.text)

            content = "".join(content_parts)

            if not content.strip():
                logger.warning(
                    f"Anthropic returned empty content (stop_reason={response.stop_reason})"
                )
                raise RuntimeError(
                    f"LLM returned empty response (stop_reason={response.stop_reason}). "
                    "This may indicate a content filter, API error, or model refusal."
                )

            # Check for truncated responses
            if response.stop_reason == "max_tokens":
                usage_info = ""
                if response.usage:
                    usage_info = (
                        f" (input={response.usage.input_tokens:,}, "
                        f"output={response.usage.output_tokens:,})"
                    )

                raise RuntimeError(
                    f"LLM response truncated - token limit exceeded{usage_info}. "
                    f"For reasoning models (Claude Opus), this indicates the query requires "
                    f"extensive reasoning that exhausted the output budget. "
                    f"The output budget is fixed at {max_completion_tokens:,} tokens. "
                    f"Try breaking your query into smaller, more focused questions."
                )

            # Warn on unexpected stop reasons
            if response.stop_reason not in ("end_turn", "stop_sequence"):
                logger.warning(
                    f"Unexpected stop_reason: {response.stop_reason} "
                    f"(content_length={len(content)})"
                )

            tokens_used = 0
            if response.usage:
                tokens_used = response.usage.input_tokens + response.usage.output_tokens

            return LLMResponse(
                content=content,
                tokens_used=tokens_used,
                model=self._model,
                finish_reason=response.stop_reason,
            )

        except Exception as e:
            logger.error(f"Anthropic completion failed: {e}")
            raise RuntimeError(f"LLM completion failed: {e}") from e

    async def complete_structured(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Generate a structured JSON completion conforming to the given schema.

        Note: Anthropic doesn't have native JSON schema validation like OpenAI.
        This implementation uses prompt engineering to request JSON output.

        Args:
            prompt: The user prompt
            json_schema: JSON Schema definition for structured output
            system: Optional system prompt
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout in seconds (overrides default)

        Returns:
            Parsed JSON object conforming to schema
        """
        # Enhance prompt with JSON schema instructions
        schema_str = json.dumps(json_schema, indent=2)
        enhanced_prompt = f"""{prompt}

Please respond with valid JSON that conforms to this schema:
{schema_str}

Your response must be valid JSON only, with no additional text or explanation."""

        # Use regular completion
        response = await self.complete(
            prompt=enhanced_prompt,
            system=system,
            max_completion_tokens=max_completion_tokens,
            timeout=timeout,
        )

        # Parse and validate JSON
        content = response.content.strip()

        # Try to extract JSON if wrapped in markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line (```)
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        try:
            parsed = json.loads(content)
            return parsed
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse structured output as JSON: {e}")
            logger.error(f"Content was: {content[:500]}")
            raise RuntimeError(f"Invalid JSON in structured output: {e}") from e

    async def batch_complete(
        self,
        prompts: list[str],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> list[LLMResponse]:
        """Generate completions for multiple prompts concurrently."""
        tasks = [
            self.complete(prompt, system, max_completion_tokens) for prompt in prompts
        ]
        return await asyncio.gather(*tasks)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation).

        Note: For accurate token counting, use the Anthropic SDK's
        count_tokens method. This is a rough estimation.
        """
        # Rough estimation: ~3.5 chars per token for Claude models
        return len(text) // 4

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        try:
            response = await self.complete("Say 'OK'", max_completion_tokens=10)
            return {
                "status": "healthy",
                "provider": "anthropic",
                "model": self._model,
                "test_response": response.content[:50],
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "anthropic",
                "error": str(e),
            }

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return {
            "requests_made": self._requests_made,
            "total_tokens": self._tokens_used,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
        }

    def get_synthesis_concurrency(self) -> int:
        """Get recommended concurrency for parallel synthesis operations.

        Returns:
            5 for Anthropic (higher tier limits than OpenAI)
        """
        return 5

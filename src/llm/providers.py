"""Multi-provider LLM integration layer for Indian Market Portfolio Intelligence.

Supports Google Gemini (via direct REST), Groq, NVIDIA NIM, OpenRouter (via OpenAI client),
and a deterministic offline MockProvider.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx
import openai

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exception Hierarchy
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Base exception for all LLM errors."""
    pass


class LLMProviderError(LLMError):
    """Raised when an LLM provider encounters an execution or API error."""
    pass


class LLMAuthenticationError(LLMProviderError):
    """Raised when an API key is missing or invalid."""
    pass


class LLMQuotaError(LLMProviderError):
    """Raised when rate limits or quotas are exceeded."""
    pass


class LLMTimeoutError(LLMProviderError):
    """Raised when a provider request times out."""
    pass


# ---------------------------------------------------------------------------
# Environment Variable Helper
# ---------------------------------------------------------------------------

def get_env_var(key: str, default: str | None = None) -> str | None:
    """Retrieve environment variable from os.environ or parse from .env fallback."""
    val = os.getenv(key)
    if val is not None and val != "":
        return val

    # Search for .env file in parent directories up to 4 levels
    curr = Path(__file__).resolve().parent
    for _ in range(4):
        env_path = curr / ".env"
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            if k.strip() == key:
                                clean_val = v.strip().strip("'\"")
                                if clean_val:
                                    return clean_val
            except Exception as e:
                logger.debug(f"Error reading {env_path}: {e}")
        if curr.parent == curr:
            break
        curr = curr.parent

    return default


# ---------------------------------------------------------------------------
# Base Provider Abstract Class
# ---------------------------------------------------------------------------

class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""

    name: str = "base"
    default_model: str = "default"
    candidate_models: list[str] = []

    def __init__(self, api_key: str | None = None, default_model: str | None = None):
        self.api_key = api_key
        if default_model:
            self.default_model = default_model

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the provider has credentials and is ready for use."""
        pass

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        """Generate text from system and user prompt. Raises LLMProviderError on failure."""
        pass


# ---------------------------------------------------------------------------
# Concrete Providers
# ---------------------------------------------------------------------------

class GeminiProvider(BaseProvider):
    """Google Gemini provider using direct REST API (via httpx)."""

    name: str = "gemini"
    default_model: str = "gemini-2.5-flash"
    candidate_models: list[str] = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

    def __init__(self, api_key: str | None = None, default_model: str | None = None):
        key = api_key or get_env_var("GEMINI_API_KEY")
        super().__init__(api_key=key, default_model=default_model)

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        if not self.is_available():
            raise LLMAuthenticationError("Gemini API key is missing or unset (GEMINI_API_KEY).")

        models_to_try = [model] if model else ([self.default_model] + [m for m in self.candidate_models if m != self.default_model])

        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        last_err: Exception | None = None
        for candidate in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{candidate}:generateContent?key={self.api_key}"
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(url, json=payload)

                if resp.status_code == 200:
                    data = resp.json()
                    try:
                        text = data["candidates"][0]["content"]["parts"][0]["text"]
                        return text
                    except (KeyError, IndexError) as parse_err:
                        raise LLMProviderError(f"Gemini response parsing error: {data}") from parse_err

                elif resp.status_code in (401, 403):
                    raise LLMAuthenticationError(f"Gemini authentication failed (HTTP {resp.status_code}): {resp.text}")
                elif resp.status_code == 429:
                    raise LLMQuotaError(f"Gemini quota/rate limit exceeded (HTTP 429): {resp.text}")
                elif resp.status_code == 404:
                    # Model not found / deprecated -> try next candidate
                    logger.warning(f"Gemini model {candidate} returned 404. Falling back to next candidate.")
                    last_err = LLMProviderError(f"Gemini model {candidate} not found: {resp.text}")
                    continue
                else:
                    raise LLMProviderError(f"Gemini request failed (HTTP {resp.status_code}): {resp.text}")

            except httpx.TimeoutException as timeout_err:
                raise LLMTimeoutError(f"Gemini request timed out: {timeout_err}") from timeout_err
            except (LLMError, Exception) as exc:
                if isinstance(exc, (LLMAuthenticationError, LLMQuotaError, LLMTimeoutError)):
                    raise exc
                last_err = exc
                continue

        if last_err:
            if isinstance(last_err, LLMError):
                raise last_err
            raise LLMProviderError(f"Gemini generation failed: {last_err}") from last_err
        raise LLMProviderError("Gemini generation failed: No candidates succeeded.")


class GroqProvider(BaseProvider):
    """Groq Cloud provider using OpenAI-compatible client interface."""

    name: str = "groq"
    default_model: str = "llama-3.3-70b-versatile"
    candidate_models: list[str] = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

    def __init__(self, api_key: str | None = None, default_model: str | None = None):
        key = api_key or get_env_var("GROQ_API_KEY")
        super().__init__(api_key=key, default_model=default_model)

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        if not self.is_available():
            raise LLMAuthenticationError("Groq API key is missing or unset (GROQ_API_KEY).")

        models_to_try = [model] if model else ([self.default_model] + [m for m in self.candidate_models if m != self.default_model])

        client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key,
            timeout=30.0,
        )

        last_err: Exception | None = None
        for candidate in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=candidate,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                content = response.choices[0].message.content
                if content:
                    return content
                raise LLMProviderError(f"Groq returned empty response content from model {candidate}.")
            except openai.AuthenticationError as auth_err:
                raise LLMAuthenticationError(f"Groq authentication error: {auth_err}") from auth_err
            except openai.RateLimitError as rate_err:
                raise LLMQuotaError(f"Groq rate limit/quota exceeded: {rate_err}") from rate_err
            except (openai.APITimeoutError, TimeoutError) as timeout_err:
                raise LLMTimeoutError(f"Groq request timed out: {timeout_err}") from timeout_err
            except Exception as exc:
                last_err = exc
                logger.warning(f"Groq model {candidate} failed: {exc}. Trying next candidate.")
                continue

        if last_err:
            if isinstance(last_err, LLMError):
                raise last_err
            raise LLMProviderError(f"Groq generation failed across all models: {last_err}") from last_err
        raise LLMProviderError("Groq generation failed: No candidates succeeded.")


class NvidiaProvider(BaseProvider):
    """NVIDIA NIM provider using OpenAI-compatible client interface."""

    name: str = "nvidia"
    default_model: str = "meta/llama-3.1-70b-instruct"
    candidate_models: list[str] = [
        "meta/llama-3.1-70b-instruct",
        "nvidia/llama-3.1-nemotron-70b-instruct",
    ]

    def __init__(self, api_key: str | None = None, default_model: str | None = None):
        key = api_key or get_env_var("NVIDIA_NIM_API_KEY") or get_env_var("NVIDIA_API_KEY")
        super().__init__(api_key=key, default_model=default_model)

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        if not self.is_available():
            raise LLMAuthenticationError("NVIDIA NIM API key is missing or unset (NVIDIA_NIM_API_KEY/NVIDIA_API_KEY).")

        models_to_try = [model] if model else ([self.default_model] + [m for m in self.candidate_models if m != self.default_model])

        client = openai.OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key,
            timeout=30.0,
        )

        last_err: Exception | None = None
        for candidate in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=candidate,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                content = response.choices[0].message.content
                if content:
                    return content
                raise LLMProviderError(f"NVIDIA NIM returned empty response content from model {candidate}.")
            except openai.AuthenticationError as auth_err:
                raise LLMAuthenticationError(f"NVIDIA NIM authentication error: {auth_err}") from auth_err
            except openai.RateLimitError as rate_err:
                raise LLMQuotaError(f"NVIDIA NIM rate limit/quota exceeded: {rate_err}") from rate_err
            except (openai.APITimeoutError, TimeoutError) as timeout_err:
                raise LLMTimeoutError(f"NVIDIA NIM request timed out: {timeout_err}") from timeout_err
            except Exception as exc:
                last_err = exc
                logger.warning(f"NVIDIA NIM model {candidate} failed: {exc}. Trying next candidate.")
                continue

        if last_err:
            if isinstance(last_err, LLMError):
                raise last_err
            raise LLMProviderError(f"NVIDIA NIM generation failed across all models: {last_err}") from last_err
        raise LLMProviderError("NVIDIA NIM generation failed: No candidates succeeded.")


class OpenRouterProvider(BaseProvider):
    """OpenRouter provider using OpenAI-compatible client interface."""

    name: str = "openrouter"
    default_model: str = "meta-llama/llama-3.3-70b-instruct"
    candidate_models: list[str] = [
        "meta-llama/llama-3.3-70b-instruct",
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemini-2.0-flash-exp:free",
    ]

    def __init__(self, api_key: str | None = None, default_model: str | None = None):
        key = api_key or get_env_var("OPENROUTER_API_KEY")
        super().__init__(api_key=key, default_model=default_model)

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        if not self.is_available():
            raise LLMAuthenticationError("OpenRouter API key is missing or unset (OPENROUTER_API_KEY).")

        models_to_try = [model] if model else ([self.default_model] + [m for m in self.candidate_models if m != self.default_model])

        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
            timeout=30.0,
        )

        last_err: Exception | None = None
        for candidate in models_to_try:
            try:
                response = client.chat.completions.create(
                    model=candidate,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                content = response.choices[0].message.content
                if content:
                    return content
                raise LLMProviderError(f"OpenRouter returned empty response content from model {candidate}.")
            except openai.AuthenticationError as auth_err:
                raise LLMAuthenticationError(f"OpenRouter authentication error: {auth_err}") from auth_err
            except openai.RateLimitError as rate_err:
                raise LLMQuotaError(f"OpenRouter rate limit/quota exceeded: {rate_err}") from rate_err
            except (openai.APITimeoutError, TimeoutError) as timeout_err:
                raise LLMTimeoutError(f"OpenRouter request timed out: {timeout_err}") from timeout_err
            except Exception as exc:
                last_err = exc
                logger.warning(f"OpenRouter model {candidate} failed: {exc}. Trying next candidate.")
                continue

        if last_err:
            if isinstance(last_err, LLMError):
                raise last_err
            raise LLMProviderError(f"OpenRouter generation failed across all models: {last_err}") from last_err
        raise LLMProviderError("OpenRouter generation failed: No candidates succeeded.")


class MockProvider(BaseProvider):
    """Deterministic, high-fidelity offline mock provider.

    Always available (100% offline, zero network dependencies).
    Uses prompts.generate_mock_report to produce a realistic 5-section institutional Markdown report.
    """

    name: str = "mock"
    default_model: str = "mock-deterministic-v1"
    candidate_models: list[str] = ["mock-deterministic-v1"]

    def __init__(self, api_key: str | None = None, default_model: str | None = None):
        super().__init__(api_key="mock_key", default_model=default_model or "mock-deterministic-v1")

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        # Dynamically import generate_mock_report from prompts to avoid circular dependency
        from llm.prompts import generate_mock_report

        return generate_mock_report(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model or self.default_model,
        )

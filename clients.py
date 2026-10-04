import asyncio
import logging
import os
from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional, Tuple

import anthropic
from anthropic import AsyncAnthropic
from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from schemas import ChatMessage, ModelConfig, ModelResponse, Role

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


class BaseLLMClient(ABC):

    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        """Genera una respuesta completa (no streaming)."""

    @abstractmethod
    def stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        """Devuelve un generador asíncrono de fragmentos de texto."""


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY no está configurada en el entorno")
        self._client = AsyncOpenAI(api_key=api_key)

    @staticmethod
    def _to_payload(messages: List[ChatMessage]) -> List[dict]:
        return [{"role": m.role.value, "content": m.content} for m in messages]

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        payload = self._to_payload(messages)
        for attempt in range(MAX_RETRIES):
            try:
                response = await self._client.chat.completions.create(
                    model=self.config.model,
                    messages=payload,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                choice = response.choices[0]
                usage = response.usage
                return ModelResponse(
                    content=choice.message.content or "",
                    provider="openai",
                    model=self.config.model,
                    finish_reason=choice.finish_reason,
                    input_tokens=usage.prompt_tokens if usage else None,
                    output_tokens=usage.completion_tokens if usage else None,
                )
            except RateLimitError:
                wait = 2**attempt
                logger.warning("Rate limit de OpenAI, reintentando en %ss", wait)
                await asyncio.sleep(wait)
            except (APITimeoutError, APIError) as exc:
                logger.error("Error de API en OpenAI: %s", exc)
                return ModelResponse(
                    content="", provider="openai", model=self.config.model, error=str(exc)
                )
        return ModelResponse(
            content="",
            provider="openai",
            model=self.config.model,
            error="Límite de reintentos agotado por rate limiting",
        )

    async def stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        payload = self._to_payload(messages)
        try:
            response_stream = await self._client.chat.completions.create(
                model=self.config.model,
                messages=payload,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
            )
            async for chunk in response_stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except (APITimeoutError, APIError, RateLimitError) as exc:
            logger.error("Error durante el streaming de OpenAI: %s", exc)
            yield f"[ERROR: {exc}]"


class AnthropicClient(BaseLLMClient):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY no está configurada en el entorno")
        self._client = AsyncAnthropic(api_key=api_key)

    @staticmethod
    def _split_system(messages: List[ChatMessage]) -> Tuple[Optional[str], List[dict]]:
        system_parts = [m.content for m in messages if m.role == Role.SYSTEM]
        payload = [
            {"role": m.role.value, "content": m.content}
            for m in messages
            if m.role != Role.SYSTEM
        ]
        system = "\n".join(system_parts) if system_parts else None
        return system, payload

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        system, payload = self._split_system(messages)
        for attempt in range(MAX_RETRIES):
            try:
                response = await self._client.messages.create(
                    model=self.config.model,
                    system=system,
                    messages=payload,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                text = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                usage = response.usage
                return ModelResponse(
                    content=text,
                    provider="anthropic",
                    model=self.config.model,
                    finish_reason=response.stop_reason,
                    input_tokens=usage.input_tokens if usage else None,
                    output_tokens=usage.output_tokens if usage else None,
                )
            except anthropic.RateLimitError:
                wait = 2**attempt
                logger.warning("Rate limit de Anthropic, reintentando en %ss", wait)
                await asyncio.sleep(wait)
            except anthropic.APIError as exc:
                logger.error("Error de API en Anthropic: %s", exc)
                return ModelResponse(
                    content="", provider="anthropic", model=self.config.model, error=str(exc)
                )
        return ModelResponse(
            content="",
            provider="anthropic",
            model=self.config.model,
            error="Límite de reintentos agotado por rate limiting",
        )

    async def stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        system, payload = self._split_system(messages)
        try:
            async with self._client.messages.stream(
                model=self.config.model,
                system=system,
                messages=payload,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except (anthropic.APIError, anthropic.RateLimitError) as exc:
            logger.error("Error durante el streaming de Anthropic: %s", exc)
            yield f"[ERROR: {exc}]"


class GeminiClient(BaseLLMClient):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY no está configurada en el entorno")
        self._client = genai.Client(api_key=api_key)

    @staticmethod
    def _to_gemini_role(role: Role) -> str:
        # Gemini no tiene rol "assistant": sus propios turnos se marcan como "model".
        return "model" if role == Role.ASSISTANT else "user"

    @classmethod
    def _split_system(cls, messages: List[ChatMessage]) -> Tuple[Optional[str], List[dict]]:
        system_parts = [m.content for m in messages if m.role == Role.SYSTEM]
        payload = [
            {"role": cls._to_gemini_role(m.role), "parts": [{"text": m.content}]}
            for m in messages
            if m.role != Role.SYSTEM
        ]
        system = "\n".join(system_parts) if system_parts else None
        return system, payload

    def _build_config(self, system: Optional[str]) -> genai_types.GenerateContentConfig:
        return genai_types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
        )

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        system, payload = self._split_system(messages)
        config = self._build_config(system)
        for attempt in range(MAX_RETRIES):
            try:
                response = await self._client.aio.models.generate_content(
                    model=self.config.model,
                    contents=payload,
                    config=config,
                )
                usage = response.usage_metadata
                finish_reason = (
                    str(response.candidates[0].finish_reason) if response.candidates else None
                )
                return ModelResponse(
                    content=response.text or "",
                    provider="gemini",
                    model=self.config.model,
                    finish_reason=finish_reason,
                    input_tokens=usage.prompt_token_count if usage else None,
                    output_tokens=usage.candidates_token_count if usage else None,
                )
            except genai_errors.ClientError as exc:
                if getattr(exc, "code", None) == 429:
                    wait = 2**attempt
                    logger.warning("Rate limit de Gemini, reintentando en %ss", wait)
                    await asyncio.sleep(wait)
                    continue
                logger.error("Error de API en Gemini: %s", exc)
                return ModelResponse(
                    content="", provider="gemini", model=self.config.model, error=str(exc)
                )
            except genai_errors.APIError as exc:
                logger.error("Error de API en Gemini: %s", exc)
                return ModelResponse(
                    content="", provider="gemini", model=self.config.model, error=str(exc)
                )
        return ModelResponse(
            content="",
            provider="gemini",
            model=self.config.model,
            error="Límite de reintentos agotado por rate limiting",
        )

    async def stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        system, payload = self._split_system(messages)
        config = self._build_config(system)
        try:
            response_stream = await self._client.aio.models.generate_content_stream(
                model=self.config.model,
                contents=payload,
                config=config,
            )
            async for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except genai_errors.APIError as exc:
            logger.error("Error durante el streaming de Gemini: %s", exc)
            yield f"[ERROR: {exc}]"

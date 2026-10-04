import asyncio
import logging
import os
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Callable, List, Optional, Tuple, Type

import anthropic
from anthropic import AsyncAnthropic
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from schemas import ChatMessage, ModelConfig, ModelResponse, Role

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class BaseLLMClient(ABC):

    def __init__(self, config: ModelConfig):
        self.config = config

    async def _retrying_stream(
        self,
        open_stream: Callable[[], AsyncGenerator[str, None]],
        api_errors: Tuple[Type[Exception], ...],
        is_retryable: Callable[[Exception], bool],
        provider_name: str,
    ) -> AsyncGenerator[str, None]:
        # Solo se reintenta si aún no se emitió ningún fragmento: reintentar después
        # duplicaría texto ya entregado al consumidor.
        for attempt in range(MAX_RETRIES):
            started = False
            try:
                async for text in open_stream():
                    started = True
                    yield text
                return
            except api_errors as exc:
                can_retry = is_retryable(exc) and not started and attempt < MAX_RETRIES - 1
                if not can_retry:
                    logger.error("Error durante el streaming de %s: %s", provider_name, exc)
                    yield f"[ERROR: {exc}]"
                    return
                wait = 2**attempt
                logger.warning(
                    "Error transitorio de %s en streaming (%s), reintentando en %ss",
                    provider_name, type(exc).__name__, wait,
                )
                await asyncio.sleep(wait)

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
            except (RateLimitError, APIConnectionError, InternalServerError) as exc:
                wait = 2**attempt
                logger.warning("Error transitorio de OpenAI (%s), reintentando en %ss", type(exc).__name__, wait)
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
            error="Límite de reintentos agotado por errores transitorios",
        )

    async def _open_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        response_stream = await self._client.chat.completions.create(
            model=self.config.model,
            messages=self._to_payload(messages),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
        )
        async for chunk in response_stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        return self._retrying_stream(
            lambda: self._open_stream(messages),
            api_errors=(APIError,),
            is_retryable=lambda exc: isinstance(
                exc, (RateLimitError, APIConnectionError, InternalServerError)
            ),
            provider_name="OpenAI",
        )


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
            except (
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
                anthropic.InternalServerError,
            ) as exc:
                wait = 2**attempt
                logger.warning("Error transitorio de Anthropic (%s), reintentando en %ss", type(exc).__name__, wait)
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
            error="Límite de reintentos agotado por errores transitorios",
        )

    async def _open_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        system, payload = self._split_system(messages)
        async with self._client.messages.stream(
            model=self.config.model,
            system=system,
            messages=payload,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    def stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        return self._retrying_stream(
            lambda: self._open_stream(messages),
            api_errors=(anthropic.APIError,),
            is_retryable=lambda exc: isinstance(
                exc,
                (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError),
            ),
            provider_name="Anthropic",
        )


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
            except genai_errors.APIError as exc:
                if getattr(exc, "code", None) in RETRYABLE_STATUS_CODES:
                    wait = 2**attempt
                    logger.warning("Error transitorio de Gemini (%s), reintentando en %ss", exc.code, wait)
                    await asyncio.sleep(wait)
                    continue
                logger.error("Error de API en Gemini: %s", exc)
                return ModelResponse(
                    content="", provider="gemini", model=self.config.model, error=str(exc)
                )
        return ModelResponse(
            content="",
            provider="gemini",
            model=self.config.model,
            error="Límite de reintentos agotado por errores transitorios",
        )

    async def _open_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        system, payload = self._split_system(messages)
        response_stream = await self._client.aio.models.generate_content_stream(
            model=self.config.model,
            contents=payload,
            config=self._build_config(system),
        )
        async for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    def stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        return self._retrying_stream(
            lambda: self._open_stream(messages),
            api_errors=(genai_errors.APIError,),
            is_retryable=lambda exc: getattr(exc, "code", None) in RETRYABLE_STATUS_CODES,
            provider_name="Gemini",
        )

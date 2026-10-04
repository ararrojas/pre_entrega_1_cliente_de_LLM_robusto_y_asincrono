"""Factory que decide, según la configuración, qué cliente de LLM instanciar."""
from typing import AsyncGenerator, List, Type

from clients import AnthropicClient, BaseLLMClient, GeminiClient, OpenAIClient
from schemas import ChatMessage, ModelConfig, ModelResponse

_PROVIDERS: dict[str, Type[BaseLLMClient]] = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "gemini": GeminiClient,
}


class AsyncLLMManager:
    def __init__(self, config: ModelConfig):
        provider_cls = _PROVIDERS.get(config.provider.lower())
        if provider_cls is None:
            disponibles = ", ".join(_PROVIDERS)
            raise ValueError(
                f"Proveedor no soportado: '{config.provider}'. Usá uno de: {disponibles}"
            )
        self._client: BaseLLMClient = provider_cls(config)

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        return await self._client.generate(messages)

    def stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        return self._client.stream(messages)

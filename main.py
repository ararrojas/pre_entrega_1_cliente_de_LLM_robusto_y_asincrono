import asyncio
import logging
import os

from dotenv import load_dotenv

from llm_manager import AsyncLLMManager
from schemas import ChatMessage, ModelConfig, Role

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

QUESTION = "¿Qué es la entropía?"

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-20241022",
    "gemini": "gemini-2.5-flash",
}


def build_messages(question: str) -> list[ChatMessage]:
    # prompt en inglés, comprabado consume menos tokens
    return [
        ChatMessage(role=Role.SYSTEM, content="Sos un asistente técnico. Respondé en 3 líneas como máximo."),
        ChatMessage(role=Role.USER, content=question),
    ]


async def run_normal(manager: AsyncLLMManager) -> None:
    print("\n--- Modo normal (ainvoke) ---")
    response = await manager.generate(build_messages(QUESTION))
    if response.error:
        print(f"Error controlado: {response.error}")
        return
    print(response.content)
    print(f"(finish_reason={response.finish_reason}, tokens in/out={response.input_tokens}/{response.output_tokens})")


async def run_streaming(manager: AsyncLLMManager) -> None:
    print("\n--- Modo streaming ---")
    async for chunk in manager.stream(build_messages(QUESTION)):
        print(chunk, end="", flush=True)
    # EOF
    print()


async def main() -> None:
    provider = os.getenv("LLM_PROVIDER", "gemini")
    model = os.getenv("LLM_MODEL", DEFAULT_MODELS[provider])

    # temperatura cercana a 0 para respuesta técnica y corta
    config = ModelConfig(provider=provider, model=model, temperature=0.3, max_tokens=300)
    manager = AsyncLLMManager(config)

    await run_normal(manager)
    await run_streaming(manager)


# Event loop
if __name__ == "__main__":
    asyncio.run(main())

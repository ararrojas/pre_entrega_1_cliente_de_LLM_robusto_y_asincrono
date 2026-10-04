import asyncio
import logging
import os

from dotenv import load_dotenv

from rich.console import Console

from llm_manager import AsyncLLMManager
from schemas import ChatMessage, ModelConfig, Role

load_dotenv()
console = Console(markup=False, highlight=False)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

QUESTION = "¿Qué es la entropía?"

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-20241022",
    "gemini": "gemini-3.8-flash",
}


def build_messages(question: str) -> list[ChatMessage]:
    # TODO: prompt en inglés, comprabado consume menos tokens
    return [
        ChatMessage(role=Role.SYSTEM, content="Sos un asistente técnico. Respondé en 3 líneas como máximo."),
        ChatMessage(role=Role.USER, content=question),
    ]


async def run_normal(manager: AsyncLLMManager) -> None:
    console.print("\n--- Modo normal (ainvoke) ---\n", style="cyan")
    response = await manager.generate(build_messages(QUESTION))
    if response.error:
        console.print(f"Error controlado: {response.error}", style="red")
        return
    console.print(response.content, style="green")
    console.print(
        f"(finish_reason={response.finish_reason}, tokens in/out={response.input_tokens}/{response.output_tokens})",
        style="yellow",
    )


async def run_streaming(manager: AsyncLLMManager) -> None:
    console.print("\n--- Modo streaming ---", style="cyan")
    async for chunk in manager.stream(build_messages(QUESTION)):
        console.print(chunk, end="", style="green", soft_wrap=True)
    # EOF
    console.print()


async def main() -> None:
    provider = os.getenv("LLM_PROVIDER", "gemini")
    model = os.getenv("LLM_MODEL", DEFAULT_MODELS[provider])

    # temperatura cercana a 0 para respuesta técnica y corta
    config = ModelConfig(provider=provider, model=model, temperature=0.3, max_tokens=2048)
    manager = AsyncLLMManager(config)

    await run_normal(manager)
    await run_streaming(manager)


# Event loop
if __name__ == "__main__":
    asyncio.run(main())

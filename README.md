# Pre-entrega 1 — Cliente de LLM robusto y asíncrono

Curso AI Engineering (Coderhouse). Implementación de un **Unified Async LLM Client**: una capa de
abstracción en Python 3.12 que permite usar OpenAI, Anthropic o Gemini bajo una interfaz común, con
soporte de streaming, validación de esquemas con Pydantic y manejo de errores/reintentos.

> Nota: Gemini se agregó como proveedor extra porque Google AI Studio ofrece una API key gratuita
> con cuota diaria (https://aistudio.google.com/apikey), ideal para probar el cliente sin costo.
> El enunciado original pide OpenAI y Anthropic; ambos siguen soportados.

## Estructura

- `schemas.py`: modelos Pydantic (`ChatMessage`, `ModelConfig`, `ModelResponse`).
- `clients.py`: `BaseLLMClient` (clase base abstracta) y las implementaciones `OpenAIClient`,
  `AnthropicClient` y `GeminiClient`, todas asíncronas y con reintento ante rate limiting.
- `llm_manager.py`: `AsyncLLMManager`, el Factory que elige el proveedor según `ModelConfig`.
- `main.py`: script de prueba que ejecuta una pregunta en modo normal y en streaming.

## Cómo ejecutarlo

1. Creá un entorno virtual con Python 3.12 e instalá las dependencias:

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Copiá `.env.example` a `.env` y completá tu API key del proveedor que vayas a usar:

   ```bash
   cp .env.example .env
   ```

3. Corré el script de validación:

   ```bash
   python main.py
   ```

## Variables de entorno

| Variable            | Descripción                                               |
|---------------------|-------------------------------------------------------------|
| `LLM_PROVIDER`       | `openai`, `anthropic` o `gemini`                              |
| `LLM_MODEL`          | Nombre del modelo (ej. `gpt-4o-mini`, `claude-3-5-sonnet-20241022`, `gemini-2.5-flash`) |
| `OPENAI_API_KEY`     | Requerida si `LLM_PROVIDER=openai`                            |
| `ANTHROPIC_API_KEY`  | Requerida si `LLM_PROVIDER=anthropic`                         |
| `GEMINI_API_KEY`     | Requerida si `LLM_PROVIDER=gemini` (gratuita en Google AI Studio) |

## Qué resuelve cada pieza

- **Intercambiabilidad**: `AsyncLLMManager` instancia el cliente correcto sin que el resto de la
  app conozca el SDK subyacente.
- **Asincronía**: todas las llamadas usan `async/await` (`AsyncOpenAI`, `AsyncAnthropic`).
- **Streaming**: `stream()` es un generador asíncrono que entrega fragmentos de texto a medida
  que llegan del proveedor.
- **Validación**: `ModelConfig` valida rangos de `temperature` y `max_tokens`; `ChatMessage` obliga
  a que cada mensaje tenga `role` y `content`.
- **Resiliencia**: ante un rate limit se reintenta con backoff exponencial; ante un error de API no
  recuperable se devuelve un `ModelResponse` con el campo `error` en vez de romper el programa.

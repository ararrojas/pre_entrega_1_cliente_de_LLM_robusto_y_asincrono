from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Role(str, Enum):
    # Prompting
    SYSTEM = "system"
    # Mi pregunta
    USER = "user"
    # Responses del modelo para reutilizar
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: Role
    content: str = Field(..., min_length=1, description="Contenido del mensaje")


class ModelConfig(BaseModel):
    provider: str = Field(..., description="Proveedor del modelo: 'openai' o 'anthropic'")
    model: str = Field(..., description="Nombre del modelo, ej. 'gpt-4o-mini'")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0, le=8192)


class ModelResponse(BaseModel):
    content: str
    provider: str
    model: str
    finish_reason: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    error: Optional[str] = None

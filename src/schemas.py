"""
Contratos formales de entrada, salida y evaluación para el Benchmark Challenge:
Agentic Financial RAG & Deterministic Tool-Calling (UTN FRSR).
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Verdict(str, Enum):
    """Veredicto formal emitido por el agente."""

    SOLVED = "SOLVED"
    ABSTAIN = "ABSTAIN"


ScaleType = Literal["thousand", "million", "billion", "percent"]


class Paragraph(BaseModel):
    """Párrafo narrativo o nota explicativa de un reporte 10-K."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(..., ge=1, description="Índice secuencial del párrafo (1-based).")
    text: str = Field(..., min_length=1, description="Contenido textual del párrafo o nota al pie.")


class CaseInput(BaseModel):
    """
    Contrato formal de entrada que recibe el microservicio en `POST /solve`.
    Contiene la consulta financiera, la tabla contable en matriz 2D y los párrafos narrativos.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(..., min_length=1, description="Identificador único del caso.")
    question: str = Field(..., min_length=1, description="Pregunta financiera o contable a resolver.")
    table: list[list[str]] = Field(
        ...,
        min_length=1,
        description="Tabla del estado financiero representada como matriz 2D de celdas (filas x columnas).",
    )
    paragraphs: list[Paragraph] = Field(
        default_factory=list,
        description="Lista ordenada de párrafos y notas aclaratorias asociadas al cuadro contable.",
    )


class ExecutionStep(BaseModel):
    """Paso individual registrado en la traza de ejecución de herramientas del agente."""

    model_config = ConfigDict(extra="forbid")

    step_index: int = Field(..., ge=1, description="Número de paso secuencial (1-based).")
    tool_name: str = Field(
        ...,
        min_length=1,
        description="Nombre identificador de la primitiva/herramienta invocada.",
    )
    tool_input: str | dict[str, Any] = Field(
        ..., description="Argumentos o expresión enviada a la herramienta."
    )
    tool_output: str = Field(
        ..., description="Resultado determinista devuelto por la herramienta."
    )


class TokenUsage(BaseModel):
    """Telemetría de consumo de recursos reportada por el agente para desempate jerárquico."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(..., ge=0, description="Total acumulado de tokens de entrada.")
    completion_tokens: int = Field(..., ge=0, description="Total acumulado de tokens generados.")
    total_tokens: int = Field(..., ge=0, description="Suma exacta de prompt_tokens + completion_tokens.")
    llm_calls: int = Field(..., ge=0, description="Número total de llamadas realizadas al LLM.")

    @model_validator(mode="after")
    def validate_token_sum(self) -> TokenUsage:
        expected = self.prompt_tokens + self.completion_tokens
        if self.total_tokens != expected:
            raise ValueError(
                f"Inconsistencia en usage: total_tokens ({self.total_tokens}) debe ser igual a "
                f"prompt_tokens ({self.prompt_tokens}) + completion_tokens ({self.completion_tokens}) = {expected}."
            )
        return self


class CaseOutput(BaseModel):
    """
    Contrato formal de salida estructurada que debe emitir el endpoint `POST /solve`.
    Cualquier violación a este esquema se penaliza con -1.5 puntos.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(..., min_length=1, description="Identificador único del caso analizado.")
    verdict: Verdict = Field(
        ...,
        description="SOLVED si existe evidencia suficiente y verificada; ABSTAIN ante incertidumbre o falta de datos.",
    )
    primary_value: float | None = Field(
        default=None,
        description=(
            "Valor numérico final calculado (obligatorio si verdict == 'SOLVED'; None si verdict == 'ABSTAIN'). "
            "Cuando scale == 'percent', debe expresarse en puntos porcentuales (ej. 15.4 para 15.4%, nunca 0.154)."
        ),
    )
    scale: ScaleType | None = Field(
        default=None,
        description="Escala dimensional del resultado: None (unidades/ratios), 'thousand', 'million', 'billion' o 'percent'.",
    )
    derivation: str = Field(
        ...,
        min_length=1,
        description="Fórmula aritmética ejecutada (ej. '(145.2 - 120.0) / 120.0 * 100') o justificación de ABSTAIN.",
    )
    execution_trace: list[ExecutionStep] = Field(
        default_factory=list,
        description="Secuencia trazable de invocaciones a herramientas deterministas.",
    )
    usage: TokenUsage = Field(
        ..., description="Bloque de telemetría de consumo de tokens y llamadas al LLM."
    )

    @model_validator(mode="after")
    def validate_verdict_consistency(self) -> CaseOutput:
        if self.verdict == Verdict.SOLVED:
            if self.primary_value is None:
                raise ValueError(
                    "Cuando verdict es 'SOLVED', el campo 'primary_value' es obligatorio y no puede ser None."
                )
            if math.isnan(self.primary_value) or math.isinf(self.primary_value):
                raise ValueError("El campo 'primary_value' debe ser un número finito válido (no NaN ni Inf).")
            if len(self.execution_trace) < 1:
                raise ValueError(
                    "Cuando verdict es 'SOLVED', 'execution_trace' debe contener al menos 1 paso de herramienta."
                )
        elif self.verdict == Verdict.ABSTAIN:
            # Normalizamos primary_value a None cuando el agente declara ABSTAIN
            self.primary_value = None
        return self


class GroundTruth(BaseModel):
    """Verdad terreno de referencia para evaluación local y remota."""

    model_config = ConfigDict(extra="forbid")

    verdict: Verdict = Field(..., description="Veredicto esperado: SOLVED o ABSTAIN.")
    primary_value: float | None = Field(
        default=None, description="Valor numérico de referencia (None si el caso es ABSTAIN)."
    )
    scale: ScaleType | None = Field(
        default=None, description="Escala dimensional de referencia."
    )
    derivation: str = Field(
        ..., description="Fórmula aritmética canónica o motivo de amputación de evidencia para ABSTAIN."
    )
    answer_from: Literal["table", "text", "table-text"] = Field(
        default="table-text",
        description="Origen modal de la evidencia necesaria para responder.",
    )


class DatasetCase(BaseModel):
    """Registro completo en los archivos de dataset (dev_set.json, calibration_set.json, final_set.json)."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    question: str
    table: list[list[str]]
    paragraphs: list[Paragraph]
    ground_truth: GroundTruth

    def to_case_input(self) -> CaseInput:
        """Extrae únicamente el payload público `CaseInput` para enviar a `POST /solve`."""
        return CaseInput(
            case_id=self.case_id,
            question=self.question,
            table=self.table,
            paragraphs=self.paragraphs,
        )


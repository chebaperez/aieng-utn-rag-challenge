"""
Plantilla Oficial de Microservicio Caja Negra (src/server.py).
Expone el contrato estándar exigido por el Evaluador Remoto Cloud:
  - GET  /health -> Verificación de disponibilidad de pre-vuelo (sin descontar cuota).
  - POST /solve  -> Recibe `CaseInput` y retorna `CaseOutput` validado con Pydantic.

Los equipos deben conectar su propio pipeline o agente dentro de `solve_endpoint`.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI

try:
    from .schemas import CaseInput, CaseOutput, TokenUsage, Verdict
except ImportError:
    from schemas import CaseInput, CaseOutput, TokenUsage, Verdict

app = FastAPI(
    title="UTN FRSR - Agentic Financial RAG Black-Box Endpoint",
    version="3.0.0",
    description="Contrato estándar POST /solve para el Benchmark Challenge sobre formularios 10-K.",
)


@app.get("/health")
def healthcheck() -> dict[str, Any]:
    """Verificación de pre-vuelo invocada por el Evaluador Remoto antes de cada corrida."""
    return {
        "status": "ok",
        "service": "starter-kit-blackbox-endpoint",
    }


@app.post("/solve", response_model=CaseOutput)
def solve_endpoint(case_input: CaseInput) -> CaseOutput:
    """
    Punto de entrada principal del desafío.
    TODO (Equipos): Reemplazar este stub por la invocación a su propio sistema Agentic RAG.
    """
    return CaseOutput(
        case_id=case_input.case_id,
        verdict=Verdict.ABSTAIN,
        primary_value=None,
        scale=None,
        derivation="ABSTAIN: Stub inicial del Starter Kit (reemplazar por la implementación del equipo).",
        execution_trace=[],
        usage=TokenUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            llm_calls=0,
        ),
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

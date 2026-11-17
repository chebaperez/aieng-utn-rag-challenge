# AGENTS.md — Guía Operativa para Desarrolladores y Agentes de Código

Instrucciones operativas para humanos y asistentes de programación (Antigravity, Claude Code, Cursor, Codex, etc.) que trabajen sobre el Starter Kit `aieng-utn-rag-challenge`.

---

## 1. Contexto Rápido del Repositorio

- **Objetivo**: implementar un sistema de **Agent-Based Financial QA** detrás del endpoint HTTP `POST /solve` (`src/server.py`) capaz de resolver consultas cuantitativas sobre tablas y notas de reportes 10-K, usando herramientas deterministas y absteniéndose (`ABSTAIN`) cuando falte evidencia.
- **Documentación complementaria**:
  - [README.md](./README.md) — visión general e inicio rápido.
  - [ARCHITECTURE.md](./ARCHITECTURE.md) — mapa técnico de componentes y flujos local/remoto.
  - [docs/GUIA_DIDACTICA.md](./docs/GUIA_DIDACTICA.md) — catálogo conceptual de tools, progresión didáctica de referencia, métricas base, patrones de falla y uso del Evaluador Remoto.

### Comandos esenciales

```bash
# Entorno virtual e instalación
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Levantar microservicio caja negra (Terminal 1)
uvicorn src.server:app --host 0.0.0.0 --port 8000

# 2. Smoke test rápido sobre 20 casos (Terminal 2)
python scripts/run_local_eval.py --limit 20 --verbose

# 3. Evaluación completa sobre los 150 casos de data/dev_set.json
python scripts/run_local_eval.py

# 4. Visualizador web interactivo de casos, Ground Truth y predicciones
python scripts/viewer.py --open
```

---

## 2. Invariantes Inviolables del Benchmark

Todo código desarrollado o generado por agentes en este repositorio **debe respetar** estas reglas:

1. **No modificar el contrato ni el evaluador**:
   - **Prohibido alterar** `src/schemas.py` y `src/evaluator.py`. El Evaluador Remoto utiliza exactamente las mismas validaciones, tolerancias (`abs <= 0.01` o `rel <= 0.001`) y puntuaciones (`+1.0 / 0.0 / -1.0 / -1.5`).
   - Mantener intactos los endpoints `GET /health` y `POST /solve` en `src/server.py`.
2. **No alterar `data/dev_set.json` ni hacer *hardcoding***:
   - Prohibido modificar casos de `data/dev_set.json` o escribir reglas específicas por `case_id`. Los conjuntos de Calibración (30 casos) y Final (100 casos) son ciegos e inéditos.
3. **Cumplimiento estricto de `CaseOutput`**:
   - Validar internamente la salida con `CaseOutput.model_validate(...)` (o retornando directamente una instancia de `CaseOutput`) antes de responder en `POST /solve`.
   - `case_id` de salida debe ser idéntico al `case_id` de entrada.
   - Si `verdict == "SOLVED"`: `primary_value` debe ser `float` finito, y `execution_trace` debe tener **al menos 1 paso** (`len(execution_trace) >= 1`).
   - Si `scale == "percent"`: el valor debe expresarse en puntos porcentuales **`0–100`** (ej. `15.4`, nunca `0.154`).
   - Si `verdict == "ABSTAIN"`: `primary_value` debe quedar en `None`.
4. **Telemetría honesta y consistente (`usage`)**:
   - Reportar los contadores reales acumulados en todas las llamadas LLM ejecutadas para resolver el caso (`prompt_tokens`, `completion_tokens`, `llm_calls`).
   - Garantizar siempre `total_tokens == prompt_tokens + completion_tokens`.
   - Evitar valores dummy (`total_tokens < 50` o `llm_calls < 1`), ya que disparan `telemetry_anomaly = True` en auditoría.
5. **Determinismo aritmético**:
   - **Nunca** devolver resultados calculados mentalmente por el LLM ni usar `eval()` sin sanitización. Toda operación aritmética debe ejecutarse mediante una herramienta determinista segura (ej. evaluador AST) y quedar registrada en `execution_trace`.
6. **Seguridad y secretos**:
   - **Nunca** commitear archivos `.env`, claves de API (`OPENAI_API_KEY`, `GEMINI_API_KEY`, etc.), ni credenciales de equipo (`team_id` / `api_token`).
   - Mantener los archivos generados localmente (`predictions_local.json`, `eval_report_local.json`) fuera de commits productivos si no corresponden.

---

## 3. Flujo de Trabajo Recomendado para Agentes de Código

Antes de proponer cambios como terminados o consumir una corrida de calibración remota (cuota limitada a 10 intentos), seguí siempre este orden:

1. **Diseñar de forma modular**: separar la lógica del agente, prompts y herramientas deterministas en módulos propios dentro de `src/` (ej. `src/tools.py`, `src/agent.py`), dejando `src/server.py` limpio como capa de entrada HTTP y manejo de errores.
2. **Validar sintaxis y contratos**: comprobar que `uvicorn src.server:app` inicia sin errores y que `GET /health` responde HTTP `200`.
3. **Ejecutar smoke test (`--limit 20`)**:
   ```bash
   python scripts/run_local_eval.py --limit 20 --verbose
   ```
   Verificar `schema_error_count == 0`, `telemetry_anomaly_count == 0` y latencias por debajo de `45 s`.
4. **Ejecutar evaluación completa (`150` casos) y analizar `eval_report_local.json`**:
   ```bash
   python scripts/run_local_eval.py
   ```
   Revisar `failure_patterns`, `by_answer_from` (`table` vs `table-text`) y latencia `p95` / `max` (debe ser `< 60 s`).
5. **Inspeccionar casos representativos en `scripts/viewer.py`**: confirmar que los cambios mejoran el razonamiento general y no introducen regresiones en otra modalidad.
6. **Calibración remota controlada**: solo lanzar `POST /calibrate` cuando el cambio haya demostrado mejora neta en `dev_set.json`. Respetar el cooldown de `300 s` y no sobreajustar a métricas ciegas.

---

## 4. Checklist Rápido de Diagnóstico ante Fallos

Si `run_local_eval.py` o el Evaluador Remoto reportan alguno de estos patrones:

| Patrón reportado | Causa probable | Acción recomendada |
| :--- | :--- | :--- |
| `percent_decimal` | Se devolvió `0.154` en vez de `15.4` | Multiplicar por `100` en la derivación cuando `scale == "percent"`. |
| `sign_error` | Resta temporal invertida (`año_viejo - año_nuevo`) | Verificar orden de columnas en encabezados y forzar `Valor_Reciente - Valor_Base`. |
| `magnitude_10x` / `scale_mismatch` | Unidad de tabla ignorada o re-escalada por `1000` | Revisar detección de escala (`detect_scale_hint`) y mantener valor y `scale` en la misma unidad. |
| `hallucination_on_abstain` | Se emitió `SOLVED` usando números inventados | Agregar/reforzar guardrail de respaldo de operandos en tabla y párrafos antes de calcular. |
| `abstain_cautious` | El agente/auditor rechaza casos válidos | Relajar criterios de abstención y mejorar recuperación en notas (`search_notes`) para `table-text`. |
| `schema_error` / `>45s` | Violación Pydantic, caída del servicio o timeout `60s` | Validar con `CaseOutput.model_validate`, acotar iteraciones/tokens y controlar VRAM/red. |
| `telemetry_anomaly` | `usage` incompleto o `llm_calls == 0` | Sumar tokens y llamadas de todos los nodos/turnos del flujo en `TokenUsage`. |

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

# 5. Exponer microservicio local al Evaluador Remoto (Terminal 3)
python scripts/start_tunnel.py
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
   - **Nunca** commitear archivos `.env`, claves de API (`OPENAI_API_KEY`, `GEMINI_API_KEY`, etc.), ni credenciales de equipo (`api_key`, `team_id`, `session_token`).
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
   Verificar en consola `Errores Esquema (-1.5) : 0` (y en `eval_report_local.json` que `summary.telemetry_anomaly_count == 0`, sin clave `SCHEMA_ERROR` en `verdict_breakdown`) y latencias por debajo de `45 s`.
4. **Ejecutar evaluación completa (`150` casos) y analizar `eval_report_local.json`**:
   ```bash
   python scripts/run_local_eval.py
   ```
   Revisar `failure_patterns`, `by_answer_from` (`table` vs `table-text`) y latencia `p95` / `max` (debe ser `< 90 s`).
5. **Inspeccionar casos representativos en `scripts/viewer.py`**: confirmar que los cambios mejoran el razonamiento general y no introducen regresiones en otra modalidad.
6. **Calibración remota controlada**: solo lanzar una corrida remota (desde **Mi Equipo** en el portal o `POST /api/runs/start`) cuando el cambio haya demostrado mejora neta en `dev_set.json`. Respetar el cooldown de `600 s` (10 min desde que finaliza cada corrida) y no sobreajustar a métricas ciegas.

### 3.1 Regla de Desarrollo para Track Open-Weight (`Host Local -> GitHub del Equipo -> Google Colab`)

- **En Colab NO se desarrolla**: todo cambio de código, prompts o herramientas debe hacerse en el host local dentro del **repositorio propio del equipo** en GitHub (fork o copia del Starter Kit), nunca editando archivos directamente dentro de Google Colab.
- **Visibilidad del repositorio**: el repositorio del equipo en GitHub puede ser público o **privado** (recomendado; en Colab se autentica mediante un Fine-grained PAT `Contents: Read-only` guardado en Colab Secrets como `GITHUB_TOKEN` con `USE_PRIVATE_REPO = True`).
- **Ciclo obligatorio**:
  1. Editar, probar y validar contratos en tu PC/host local.
  2. Hacer `git commit` y `git push` a la rama principal (`main`) del repositorio propio del equipo.
  3. En `notebooks/Colab_Starter_Template.ipynb`, configurar `REPO_URL` con la URL del repo del equipo (Celda 1 hace `clone`/`pull`) y usar Colab únicamente como **runtime GPU T4**.
- **Sincronización y recarga en Colab**: la **Celda 1** clona/actualiza el repo y define `sync_repo()`, la **Celda 3** descarga el GGUF y exporta `LOCAL_MODEL_PATH`, y la **Celda 4** ejecuta `sync_repo()`, fija `RUNTIME_ENV="colab"`, detiene el servidor previo si existía, recarga los módulos `src/` y reinicia FastAPI automáticamente. Solo necesitás reiniciar la sesión de Colab (`Entorno de ejecución -> Reiniciar sesión`) si aparece un error de memoria VRAM/CUDA al recargar el modelo GGUF o si modificaste `requirements.txt`.

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
| `schema_error` / `timeout` | Violación Pydantic, HTTP 5xx o timeout `>90s` | Validar con `CaseOutput.model_validate` (`-1.5`); si hay timeouts (`0.0`), acotar iteraciones/tokens. |
| `telemetry_anomaly` | `usage` incompleto o `llm_calls == 0` | Sumar tokens y llamadas de todos los nodos/turnos del flujo en `TokenUsage`. |

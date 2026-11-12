# 🏆 Benchmark Challenge: Agentic Financial RAG & Deterministic Tool-Calling (UTN FRSR)

Repositorio oficial del **Desafío Competitivo Integrador** de la materia electiva **Desarrollo de Aplicaciones con LLMs y Modelos Generativos** (Universidad Tecnológica Nacional · Facultad Regional San Rafael).

---

## 🎯 Misión del Desafío

Construir un sistema **Agentic RAG** expuesto como microservicio HTTP (`POST /solve`) capaz de responder consultas cuantitativas sobre reportes financieros híbridos (**Formularios 10-K / TAT-QA**), combinando tablas contables y notas narrativas, delegando el cálculo exacto a herramientas deterministas y absteniéndose (`ABSTAIN`) cuando falte evidencia indispensable.

---

## 📦 Contenido del Kit Inicial

* **`schemas.py`**: Contrato formal Pydantic v2 (`CaseInput`, `CaseOutput`, `ExecutionStep`, `TokenUsage`).
* **`evaluator.py`**: Arnés de evaluación local offline (idéntico al motor del servidor central).
* **`dev_set.json`**: Dataset Público de Desarrollo (**150 casos**: 135 resolubles `SOLVED` + 15 incompletos `ABSTAIN`).
* **`server.py`**: Plantilla base del microservicio FastAPI (`GET /health` y `POST /solve`).

---

## 🚀 Inicio Rápido

```bash
# 1. Clonar e instalar dependencias
git clone https://github.com/chebaperez/aieng-utn-rag-challenge.git
cd aieng-utn-rag-challenge
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Levantar el servidor de caja negra
uvicorn server:app --host 0.0.0.0 --port 8000

# 3. Evaluar predicciones localmente contra dev_set.json
python evaluator.py --dataset dev_set.json --predictions tus_predicciones.json --verbose
```

---

## ⚖️ Reglas de Evaluación y Puntuación

1. **Contrato Estricto (`CaseOutput`):**
   - `verdict`: `"SOLVED"` o `"ABSTAIN"`.
   - `primary_value`: `float` finito obligatorio si `verdict == "SOLVED"`; `null` si `verdict == "ABSTAIN"`.
   - `scale`: `null`, `"thousand"`, `"million"`, `"billion"` o `"percent"`. Cuando `scale == "percent"`, el valor debe expresarse en puntos porcentuales de 0 a 100 (ej. `15.4` para `15.4%`, nunca `0.154`).
   - `execution_trace`: Al menos 1 paso de herramienta registrado cuando `verdict == "SOLVED"`.
   - `usage`: `total_tokens` debe ser exactamente igual a `prompt_tokens + completion_tokens`.
2. **Normalización y Tolerancia:**
   - Las magnitudes (`thousand`, `million`, `billion`, `null`) se normalizan al marco de escala del *Ground Truth* y se aprueban si `abs_err <= 0.01` o `rel_err <= 0.001`.
   - `"percent"` solo es compatible con `"percent"`.
3. **Función de Puntuación Asimétrica:**
   - **`+1.0` punto:** `SOLVED` correcto dentro de tolerancia.
   - **` 0.0` puntos:** `ABSTAIN` declarado.
   - **`-1.0` punto:** `SOLVED` erróneo, escala incompatible o alucinación sobre un caso `ABSTAIN`.
   - **`-1.5` puntos:** Violación del esquema `CaseOutput` o caída del endpoint.
4. **Criterio de Desempate en Leaderboard:**
   1. Mayor Puntaje Neto acumulado.
   2. Menor consumo total de tokens (`total_tokens`).
   3. Menor latencia promedio por caso.

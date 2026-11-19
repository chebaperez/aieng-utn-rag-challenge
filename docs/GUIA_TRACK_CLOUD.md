# Guía Operativa — Track API Cloud (Ejecución Local)

Esta guía te lleva paso a paso desde el clonado del Starter Kit hasta la evaluación remota en el portal usando una **API Cloud** de LLMs (OpenAI, Google Gemini, Anthropic, Groq, etc.) ejecutando el servidor en tu propia PC.

> **Documentación complementaria:**
> - Contrato e invariantes del benchmark: [AGENTS.md](../AGENTS.md) y [ARCHITECTURE.md](../ARCHITECTURE.md)
> - Uso del visualizador local: [guia_viewer.html](./guia_viewer.html)
> - Uso del Evaluador Remoto (registro, calibración y leaderboard): [guia_plataforma.html](./guia_plataforma.html)

---

## 1. Preparación del repositorio y entorno

1. Creá un repositorio **propio** para tu equipo en GitHub (recomendado: **privado**, mediante *fork* o copia del Starter Kit) y clonalo en tu PC:

   ```bash
   git clone https://github.com/TU-USUARIO/tu-repo-equipo.git
   cd tu-repo-equipo
   ```

2. Creá y activá un entorno virtual e instalá las dependencias base:

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Linux / macOS
   # .venv\Scripts\activate       # Windows (CMD / PowerShell)

   pip install -r requirements.txt
   ```

> **Sobre `requirements.txt`:** El archivo base solo incluye las dependencias del contrato HTTP (`fastapi`, `uvicorn`, `pydantic`, `httpx`, `python-dotenv`). Si tu agente usa `litellm`, `openai`, `google-genai` u otras librerías, **agregalas a `requirements.txt`** e instalalas con `pip install -r requirements.txt`.

---

## 2. Integración del agente en `src/` y credenciales

1. **Invariantes que NO se modifican:** dejá intactos `src/schemas.py`, `src/evaluator.py` y `data/dev_set.json`.
2. **Tu código vive en `src/`:** organizá tus herramientas y lógica modularmente (por ejemplo `src/tools.py`, `src/agents.py`, `src/llm_client.py`) y conectá `solve_endpoint` en [src/server.py](../src/server.py).
   - Validá siempre la salida con `CaseOutput.model_validate(...)` (o retornando `CaseOutput`) y registrá el uso real de tokens en `usage` (`total_tokens == prompt_tokens + completion_tokens`, `llm_calls >= 1`).
   - Todo cálculo aritmético debe ejecutarse con una herramienta determinista (nunca cálculo mental del LLM ni `eval()` sin sanitizar).
3. **Configurá tus API Keys en `.env` (nunca en Git):**
   Creá un archivo `.env` en la raíz del proyecto (o en `src/.env`) y verificá con `git status` que `.gitignore` lo ignore:

   ```env
   # Ejemplo con Gemini o OpenAI vía LiteLLM
   GEMINI_API_KEY=tu_api_key_aqui
   # OPENAI_API_KEY=tu_api_key_aqui

   DEFAULT_MODEL=gemini/gemini-2.5-flash
   RPM_LIMIT=12
   ```

> **Portabilidad con `RUNTIME_ENV`:** [src/server.py](../src/server.py) lee `RUNTIME_ENV = os.getenv("RUNTIME_ENV", "local")` y lo reporta en `GET /health`. Si tu equipo también quiere probar el mismo repo en Colab más adelante, evitá importar librerías exclusivas de GPU al inicio del módulo y seleccioná el proveedor según `RUNTIME_ENV`.

---

## 3. Validación local completa (en tu PC)

Antes de exponer tu servidor a internet o consumir intentos de calibración remota, ejecutá el flujo completo en local:

### Paso 3.1 — Levantar el microservicio (Terminal 1)

```bash
uvicorn src.server:app --host 0.0.0.0 --port 8000
```

Comprobá en el navegador o con `curl` que `GET http://localhost:8000/health` responde `200 OK` con `"status": "ok"` y `"runtime": "local"`.

### Paso 3.2 — Smoke test rápido (Terminal 2)

```bash
python scripts/run_local_eval.py --limit 20 --verbose
```

Verificá en la salida y en `eval_report_local.json`:
- `Errores Esquema (-1.5) : 0` (sin `SCHEMA_ERROR`).
- `summary.telemetry_anomaly_count == 0`.
- Latencia por caso bien por debajo de `90 s`.

### Paso 3.3 — Evaluación completa e inspección visual

```bash
# Evalúa los 150 casos de data/dev_set.json
python scripts/run_local_eval.py

# Abre el visualizador interactivo para analizar fallos y trazas
python scripts/viewer.py --open
```

Esto genera `predictions_local.json` y `eval_report_local.json`. Iterá sobre tus prompts y herramientas hasta ver una mejora sólida en `dev_set.json` (ver [guia_viewer.html](./guia_viewer.html)).

---

## 4. Exposición por túnel y Evaluación Remota

Con la **Terminal 1 (`uvicorn`)** corriendo sin errores:

### Paso 4.1 — Abrir Cloudflare Tunnel (Terminal 3)

```bash
python scripts/start_tunnel.py
```

- Si no tenés `cloudflared` instalado en tu sistema, el script te mostrará el comando exacto para instalarlo en Windows (`winget`), macOS (`brew`) o Linux.
- Cuando arranque, copiá la URL pública `https://xxxx.trycloudflare.com` que imprime en pantalla y verificá en tu navegador que `<URL_PUBLICA>/health` responde `200 OK`.

### Paso 4.2 — Correr Calibración en el portal

Siguiendo [guia_plataforma.html](./guia_plataforma.html):

1. Registrate en el portal del Evaluador Remoto e iniciá sesión con tu `email` y `api_key`.
2. En **Mi Equipo**, pegá la URL pública `https://xxxx.trycloudflare.com`.
3. Lanzá una corrida de **Calibración** (30 casos ciegos) y seguí los logs en tiempo real en la Terminal 1.

---

## 5. Limitaciones y cuidados de este Track

- **Cuotas y Rate Limits (`RPM_LIMIT`):** En planes gratuitos (por ejemplo Gemini Free Tier), un agente que realiza muchas llamadas LLM por caso puede disparar errores `429` o acumular esperas de *backoff* que superen el **timeout de `90 s` por caso** (`TIMEOUT = 0.0`). Ajustá `RPM_LIMIT` y acotá la cantidad máxima de turnos por caso.
- **Latencia y costos:** Modelos más grandes suelen razonar mejor pero tardan más y consumen más cuota. Medí siempre latencia `p95` y `max` en `eval_report_local.json` antes de calibrar.
- **Estabilidad durante la corrida:** No suspendas la PC ni cierres las terminales de `uvicorn` o `start_tunnel.py` mientras la evaluación esté corriendo. Si el túnel se corta, podés levantarlo de nuevo, actualizar la URL en **Mi Equipo** y reanudar.
- **Cooldown de calibración:** Entre corridas de calibración hay un enfriamiento de **`600 s` (10 minutos)** contados desde que finaliza la corrida anterior.
- **Seguridad:** No subas `.env`, `predictions_local.json` ni `eval_report_local.json` si contienen información sensible o ruido de pruebas.

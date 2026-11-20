# Guía Operativa — Track Open-Weight (Google Colab GPU T4)

Esta guía te lleva paso a paso para ejecutar tu agente con modelos **Open-Weight (GGUF)** sobre la **GPU T4** gratuita de Google Colab usando [notebooks/Colab_Starter_Template.ipynb](../notebooks/Colab_Starter_Template.ipynb), manteniendo el desarrollo en tu PC local.

> **Documentación complementaria:**
> - Contrato e invariantes del benchmark: [AGENTS.md](../AGENTS.md) y [ARCHITECTURE.md](../ARCHITECTURE.md)
> - Uso del visualizador local: [guia_viewer.html](./guia_viewer.html)
> - Uso del Evaluador Remoto (registro, calibración y leaderboard): [guia_plataforma.html](./guia_plataforma.html)

---

## 1. Regla de oro y preparación en tu PC local

> **⚠️ En Colab NO se desarrolla.** Todo el código se edita en tu PC local, se versiona con Git y se sube a GitHub (`git push`). Google Colab funciona únicamente como **runtime GPU T4** que clona y sincroniza tu repositorio.

### Paso 1.1 — Repositorio del equipo (privado recomendado)

1. Creá un repositorio **propio** para tu equipo en GitHub (*fork* o copia del Starter Kit) y clonalo en tu PC.
2. **Si el repo es privado** (recomendado para no exponer tu solución a otros equipos):
   - En GitHub: *Settings → Developer settings → Personal access tokens → Fine-grained tokens*.
   - Generá un token con permiso de solo lectura **`Contents: Read-only`** limitado únicamente al repositorio de tu equipo.
   - Guardalo en un lugar seguro: lo vas a cargar en **Colab Secrets** como `GITHUB_TOKEN`.

### Paso 1.2 — Diseño del código en `src/` y uso de variables de entorno

1. Dejá intactos `src/schemas.py`, `src/evaluator.py` y `data/dev_set.json`.
2. Conectá tu agente en [src/server.py](../src/server.py).
3. **Usá las variables de entorno que exporta Colab** en lugar de hardcodear rutas o descargas en tu código:

   | Variable | Quién la define | Valor en Colab | Valor en tu PC local |
   | :--- | :--- | :--- | :--- |
   | `RUNTIME_ENV` | Celda 4 | `"colab"` | `"local"` (por defecto) |
   | `LOCAL_MODEL_PATH` | Celda 3 | Ruta al archivo `.gguf` ya descargado | No definida (`None`) |
   | `LOCAL_MODEL_NAME` | Celda 3 | Ej. `Qwen2.5-7B-Instruct-Q4_K_M.gguf` | No definida (`None`) |
   | `HF_REPO_ID` / `HF_FILENAME` | Celda 3 | Repo y archivo elegidos en el selector | No definidas (`None`) |

   Ejemplo mínimo de cómo leerlas desde tu cliente LLM en `src/`:

   ```python
   import builtins
   import os

   RUNTIME_ENV = os.getenv("RUNTIME_ENV", "local")
   LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH")  # Definida por la Celda 3 en Colab

   def get_local_llm():
       if not LOCAL_MODEL_PATH:
           raise RuntimeError("LOCAL_MODEL_PATH no está definida (ejecutá la Celda 3 en Colab).")
       # Cachear fuera de src.* (en builtins) permite que el modelo siga en VRAM
       # al re-ejecutar la Celda 4 tras un git push sin recargar desde cero.
       if not hasattr(builtins, "_CACHED_LLAMA"):
           from llama_cpp import Llama  # Import perezoso para no requerir CUDA en tu PC
           builtins._CACHED_LLAMA = Llama(
               model_path=LOCAL_MODEL_PATH,
               n_gpu_layers=-1,
               n_ctx=16384,
               verbose=False,
           )
       return builtins._CACHED_LLAMA
   ```

4. **¿Por qué la descarga y persistencia del modelo viven en Colab y NO en `src/`?**
   - **Protege tu PC local:** si corrés tests unitarios o probás `src/` en tu computadora, tu código no va a descargar por accidente ~5 GB de pesos ni a fallar por falta de GPU/CUDA.
   - **Iteración rápida sin reiniciar Colab desde cero:** la **Celda 3** descarga el modelo una sola vez y, si mantenés la instancia cargada en memoria del kernel (como en el patrón anterior con `builtins` o precargándola en el kernel), podés modificar tu código en tu PC, hacer `git push` y **re-ejecutar únicamente la Celda 4** (`sync_repo()` + reinicio de FastAPI) en segundos, sin volver a descargar ni recargar el modelo.
   - **Imports perezosos (*lazy imports*):** no importes `llama_cpp` en el nivel superior de tus módulos si querés ejecutar tests locales sin CUDA.
   - Si tu código usa librerías adicionales, agregalas a `requirements.txt` para que Colab las instale en la Celda 1.
5. Subí tus cambios a GitHub antes de ir a Colab:

   ```bash
   git add src/ requirements.txt
   git commit -m "feat: connect open-weight agent to /solve"
   git push origin main
   ```

---

## 2. Ejecución en Google Colab (flujo en 6 celdas)

Abrí [notebooks/Colab_Starter_Template.ipynb](../notebooks/Colab_Starter_Template.ipynb) en Google Colab y verificá que el entorno de ejecución tenga aceleración por hardware **T4 GPU** (*Entorno de ejecución → Cambiar tipo de entorno de ejecución*).

Ejecutá las 6 celdas en orden:

### Celda 1 — Clonar/sincronizar repo, dependencias y `cloudflared`
- Completá `REPO_URL` con la URL HTTPS del repositorio de tu equipo y `BRANCH = "main"`.
- Si tu repo es privado:
  1. Abrí **Secrets** (ícono de la llave 🔑 en la barra lateral izquierda de Colab).
  2. Creá el secreto `GITHUB_TOKEN` con tu Fine-grained PAT y activá el acceso para este notebook.
  3. Marcá `USE_PRIVATE_REPO = True`.
- Esta celda clona el repo en `/content/team-repo`, define `sync_repo()`, instala `requirements.txt` y descarga `cloudflared`.

### Celda 2 — Runtime CUDA 12 y `llama-cpp-python`
- Instala `llama-cpp-python` compilado con CUDA (`cu122`), `huggingface_hub` y registra `libcudart.so.12` / `libcublas.so.12`.

### Celda 3 — Descargar modelo Open-Weight (GGUF) y exportar variables de entorno
- Elegí el modelo en `MODELO_SELECCIONADO` (por defecto **`Qwen 2.5 7B Instruct (Q4_K_M - ~4.7 GB VRAM)`**, además de opciones de 2B–3B más livianas).
- Descarga el archivo `.gguf` una sola vez en la caché de Colab y exporta `LOCAL_MODEL_PATH`, `LOCAL_MODEL_NAME`, `HF_REPO_ID` y `HF_FILENAME` para que tu código en `src/` lo cargue sin hardcodear descargas.
- **Ventaja clave:** se ejecuta una única vez por sesión; cuando actualices tu código en GitHub, no necesitás volver a correr esta celda.

### Celda 4 — `sync_repo()` y servidor FastAPI en hilo daemon
- Sincroniza los últimos commits desde GitHub (`sync_repo()`), fija `RUNTIME_ENV="colab"`, limpia módulos `src.*` previos de `sys.modules` y levanta `uvicorn` en el puerto `8000` dentro del mismo proceso CUDA/T4 del kernel.
- Verifica automáticamente `GET http://127.0.0.1:8000/health` (debe responder `"runtime": "colab"`).
- **Ciclo de iteración rápido:** cada vez que hagas cambios en tu PC y `git push`, **solo volvé a ejecutar esta Celda 4** para bajar los cambios y reiniciar el servidor aprovechando el modelo ya descargado y cargado en memoria, sin reiniciar Colab desde cero.

### Celda 5 — Validación sobre `data/dev_set.json` dentro de Colab
Como el modelo GGUF corre en la T4, la evaluación local se ejecuta **en esta celda** (no en tu PC):
- Empezá con `LIMIT = 5` o `20` y `VERBOSE = True` para verificar latencias, `SCHEMA_ERROR = 0` y `telemetry_anomaly_count == 0`.
- Cuando funcione bien, corré con `LIMIT = 0` para evaluar los 150 casos completos.
- *Tip:* Si querés analizar los errores en tu PC con `python scripts/viewer.py --open`, descargá `predictions_local.json` y `eval_report_local.json` desde `/content/team-repo` en el panel de archivos de Colab.

### Celda 6 — Abrir (o renovar) Cloudflare Tunnel
- Verifica que el servidor local responda en `/health` y ejecuta `scripts/start_tunnel.py` para abrir el túnel público.
- Copiá la URL `https://xxxx.trycloudflare.com` que imprime al final y comprobá `<URL_PUBLICA>/health` en tu navegador.

---

## 3. Evaluación Remota en el portal

Siguiendo [guia_plataforma.html](./guia_plataforma.html):

1. Registrate en el portal del Evaluador Remoto e iniciá sesión con tu `email` y `api_key`.
2. En **Mi Equipo**, pegá la URL `https://xxxx.trycloudflare.com` generada por la **Celda 6**.
3. Lanzá la corrida de **Calibración** (30 casos ciegos) y monitoreá la salida de la **Celda 4** en Colab para ver los `POST /solve` en tiempo real.

---

## 4. Limitaciones y cuidados de este Track

- **Memoria VRAM de la GPU T4 (16 GB):** Usá modelos cuantizados de archivo único (ej. `Qwen2.5-7B-Instruct-Q4_K_M.gguf`) con `n_gpu_layers=-1` y `n_ctx=16384`. Si al reiniciar la Celda 4 varias veces aparece un error de memoria CUDA (`out of memory`), hacé *Entorno de ejecución → Reiniciar sesión* y ejecutá desde la Celda 1.
- **Timeout estricto de `90 s` por caso:** En GPU T4, prompts muy largos o demasiadas iteraciones de *tool-calling* pueden superar los `90 s` (`TIMEOUT = 0.0`). Acotá `max_tokens`, limpiá el contexto innecesario y limitá el número máximo de pasos del agente.
- **Sesión efímera de Colab:** Si Colab desconecta por inactividad o límite de uso, se borran `/content/team-repo` y el modelo descargado en caché (tu código sigue a salvo en GitHub y tus Secrets en Colab). Mantené la pestaña activa durante la corrida remota.
- **Cooldown de calibración:** Entre corridas de calibración hay un enfriamiento de **`600 s` (10 minutos)** desde que finaliza la corrida anterior.
- **Si usás Colab pero llamando a una API Cloud:** Podés saltear las **Celdas 2 y 3**, pero recordá que `.env` no se sube a GitHub: cargá tu API key en Colab Secrets o `os.environ` antes de correr la Celda 4, y no asumas en tu código que `RUNTIME_ENV == "colab"` implica obligatoriamente usar GGUF local.

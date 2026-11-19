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

### Paso 1.2 — Diseño del código en `src/` y uso de `RUNTIME_ENV`

1. Dejá intactos `src/schemas.py`, `src/evaluator.py` y `data/dev_set.json`.
2. Conectá tu agente en [src/server.py](../src/server.py).
3. Detectá el entorno mediante `RUNTIME_ENV`:
   - La **Celda 3** de Colab define automáticamente `RUNTIME_ENV="colab"` antes de importar `src.server`.
   - En tu PC local, el valor por defecto es `"local"`.

   ```python
   import os

   RUNTIME_ENV = os.getenv("RUNTIME_ENV", "local")
   ```

4. **Importante — Imports perezosos (*lazy imports*) y singleton del modelo:**
   - **No importes `llama_cpp` en el nivel superior** si también querés correr tests o scripts en una PC sin CUDA: importalo dentro de la función que inicializa el modelo cuando `RUNTIME_ENV == "colab"`.
   - **Instanciá el modelo GGUF una sola vez (singleton)** en memoria/VRAM y reutilizalo entre requests. Si lo cargás en cada `POST /solve`, vas a agotar la VRAM y superar el timeout de `90 s`.
   - Si tu código usa librerías adicionales (ej. `huggingface_hub`, etc.), agregalas a `requirements.txt` para que Colab las instale en la Celda 1.
5. Subí tus cambios a GitHub antes de ir a Colab:

   ```bash
   git add src/ requirements.txt
   git commit -m "feat: connect open-weight agent to /solve"
   git push origin main
   ```

---

## 2. Ejecución en Google Colab (flujo en 5 celdas)

Abrí [notebooks/Colab_Starter_Template.ipynb](../notebooks/Colab_Starter_Template.ipynb) en Google Colab y verificá que el entorno de ejecución tenga aceleración por hardware **T4 GPU** (*Entorno de ejecución → Cambiar tipo de entorno de ejecución*).

Ejecutá las 5 celdas en orden:

### Celda 1 — Clonar/sincronizar repo, dependencias y `cloudflared`
- Completá `REPO_URL` con la URL HTTPS del repositorio de tu equipo y `BRANCH = "main"`.
- Si tu repo es privado:
  1. Abrí **Secrets** (ícono de la llave 🔑 en la barra lateral izquierda de Colab).
  2. Creá el secreto `GITHUB_TOKEN` con tu Fine-grained PAT y activá el acceso para este notebook.
  3. Marcá `USE_PRIVATE_REPO = True`.
- Esta celda clona el repo en `/content/team-repo`, define `sync_repo()`, instala `requirements.txt` y descarga `cloudflared`.

### Celda 2 — Runtime CUDA 12 y `llama-cpp-python`
- Instala `llama-cpp-python` compilado con CUDA (`cu122`), `huggingface_hub` y registra `libcudart.so.12` / `libcublas.so.12`.
- **Aclaración:** Esta celda **no descarga** el modelo GGUF; la descarga (ej. con `hf_hub_download`) y la carga en GPU ocurren dentro de tu código en `src/`.

### Celda 3 — `sync_repo()` y servidor FastAPI en hilo daemon
- Sincroniza los últimos commits desde GitHub (`sync_repo()`), fija `RUNTIME_ENV="colab"`, limpia módulos `src.*` previos de `sys.modules` y levanta `uvicorn` en el puerto `8000` compartiendo el proceso y la VRAM con tu modelo GGUF.
- Verifica automáticamente `GET http://127.0.0.1:8000/health` (debe responder `"runtime": "colab"`).
- **Ciclo de iteración rápido:** cada vez que hagas cambios en tu PC y `git push`, **solo volvé a ejecutar esta Celda 3** para bajar los cambios y reiniciar el servidor sin perder el entorno.

### Celda 4 — Validación sobre `data/dev_set.json` dentro de Colab
Como el modelo GGUF corre en la T4, la evaluación local se ejecuta **en esta celda** (no en tu PC):
- Empezá con `LIMIT = 5` o `20` y `VERBOSE = True` para verificar latencias, `SCHEMA_ERROR = 0` y `telemetry_anomaly_count == 0`.
- Cuando funcione bien, corré con `LIMIT = 0` para evaluar los 150 casos completos.
- *Tip:* Si querés analizar los errores en tu PC con `python scripts/viewer.py --open`, descargá `predictions_local.json` y `eval_report_local.json` desde `/content/team-repo` en el panel de archivos de Colab.

### Celda 5 — Abrir (o renovar) Cloudflare Tunnel
- Verifica que el servidor local responda en `/health` y ejecuta `scripts/start_tunnel.py` para abrir el túnel público.
- Copiá la URL `https://xxxx.trycloudflare.com` que imprime al final y comprobá `<URL_PUBLICA>/health` en tu navegador.

---

## 3. Evaluación Remota en el portal

Siguiendo [guia_plataforma.html](./guia_plataforma.html):

1. Registrate en el portal del Evaluador Remoto e iniciá sesión con tu `email` y `api_key`.
2. En **Mi Equipo**, pegá la URL `https://xxxx.trycloudflare.com` generada por la **Celda 5**.
3. Lanzá la corrida de **Calibración** (30 casos ciegos) y monitoreá la salida de la **Celda 3** en Colab para ver los `POST /solve` en tiempo real.

---

## 4. Limitaciones y cuidados de este Track

- **Memoria VRAM de la GPU T4 (16 GB):** Usá modelos cuantizados de archivo único (ej. `Qwen2.5-7B-Instruct-Q4_K_M.gguf`) con `n_gpu_layers=-1` y `n_ctx=16384`. Si al reiniciar la Celda 3 varias veces aparece un error de memoria CUDA (`out of memory`), hacé *Entorno de ejecución → Reiniciar sesión* y ejecutá desde la Celda 1.
- **Timeout estricto de `90 s` por caso:** En GPU T4, prompts muy largos o demasiadas iteraciones de *tool-calling* pueden superar los `90 s` (`TIMEOUT = 0.0`). Acotá `max_tokens`, limpiá el contexto innecesario y limitá el número máximo de pasos del agente.
- **Sesión efímera de Colab:** Si Colab desconecta por inactividad o límite de uso, se borran `/content/team-repo` y el modelo descargado en caché (tu código sigue a salvo en GitHub y tus Secrets en Colab). Mantené la pestaña activa durante la corrida remota.
- **Cooldown de calibración:** Entre corridas de calibración hay un enfriamiento de **`600 s` (10 minutos)** desde que finaliza la corrida anterior.
- **Si usás Colab pero llamando a una API Cloud:** Podés saltear la **Celda 2**, pero recordá que `.env` no se sube a GitHub: cargá tu API key en Colab Secrets o `os.environ` antes de correr la Celda 3, y no asumas en tu código que `RUNTIME_ENV == "colab"` implica obligatoriamente usar GGUF local.

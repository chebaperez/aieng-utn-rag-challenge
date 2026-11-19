"""
Exposición del Microservicio Local mediante Cloudflare Tunnel (scripts/start_tunnel.py).

Provee:
  1. Funciones reutilizables (`start_tunnel`, `stop_tunnel`, `find_cloudflared`,
     `check_local_health`) pensadas para importarse desde `notebooks/Colab_Starter_Template.ipynb`
     u otros scripts sin duplicar lógica de subprocesos ni drenado de streams.
  2. Un CLI listo para correr en Terminal 3 (`python scripts/start_tunnel.py`),
     utilizando únicamente la librería estándar de Python.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

TRYCLOUDFLARE_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
FALLBACK_BINARY_PATHS = (Path("/usr/local/bin/cloudflared"),)


def cloudflared_install_instructions() -> str:
    """Devuelve instrucciones claras para instalar `cloudflared` según el sistema operativo."""
    return (
        "[ERROR] No se encontró el binario 'cloudflared' en PATH ni en /usr/local/bin/cloudflared.\n"
        "\n"
        "Instalalo según tu sistema operativo y volvé a intentarlo:\n"
        "  • Windows (PowerShell / CMD):\n"
        "      winget install --id Cloudflare.cloudflared\n"
        "  • macOS (Homebrew):\n"
        "      brew install cloudflared\n"
        "  • Linux (x86_64):\n"
        "      sudo curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \\\n"
        "        -o /usr/local/bin/cloudflared && sudo chmod +x /usr/local/bin/cloudflared\n"
        "\n"
        "Nota: si acabás de instalarlo (especialmente en Windows), reabrí la terminal\n"
        "para refrescar el PATH y verificá con: cloudflared --version"
    )


def find_cloudflared(binary: str | None = None) -> str | None:
    """Busca el ejecutable de `cloudflared` en PATH o en ubicaciones estándar."""
    if binary is not None:
        candidate = Path(binary)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        return shutil.which(binary)

    found = shutil.which("cloudflared")
    if found:
        return found

    for path in FALLBACK_BINARY_PATHS:
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)

    return None


def stop_tunnel(proc: subprocess.Popen[str] | None, timeout: float = 3.0) -> None:
    """Termina de forma segura un subproceso de túnel si sigue activo."""
    if proc is None or proc.poll() is not None:
        return

    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass


def check_local_health(
    host: str = "127.0.0.1",
    port: int = 8000,
    timeout: float = 3.0,
) -> tuple[bool, str]:
    """Verifica `GET /health` en el servidor local antes de abrir el túnel."""
    check_host = "127.0.0.1" if host == "0.0.0.0" else host
    url = f"http://{check_host}:{port}/health"

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            status = getattr(resp, "status", resp.getcode())
            raw_body = resp.read().decode("utf-8", errors="replace").strip()
            if status != 200:
                return False, f"HTTP {status}: {raw_body[:200]}"
            try:
                parsed: Any = json.loads(raw_body)
                return True, json.dumps(parsed, ensure_ascii=False)
            except json.JSONDecodeError:
                return True, raw_body or "HTTP 200 OK"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}: {exc.reason}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, str(exc)


def start_tunnel(
    host: str = "127.0.0.1",
    port: int = 8000,
    timeout: float = 30.0,
    cloudflared_bin: str | None = None,
) -> tuple[subprocess.Popen[str], str]:
    """
    Lanza `cloudflared tunnel --url http://{host}:{port}`, captura la URL pública
    `https://...trycloudflare.com`, mantiene el drenado de salida en un thread daemon
    y retorna `(proc, public_url)` sin bloquear.
    """
    binary = find_cloudflared(cloudflared_bin)
    if binary is None:
        raise FileNotFoundError(cloudflared_install_instructions())

    target_url = f"http://{host}:{port}"
    proc = subprocess.Popen(
        [binary, "tunnel", "--url", target_url],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    if proc.stdout is None:
        stop_tunnel(proc)
        raise RuntimeError("No se pudo abrir el stream de salida de cloudflared.")

    ready_event = threading.Event()
    found_url: list[str] = []
    recent_lines: list[str] = []

    def _reader_and_drainer(stream: Any) -> None:
        try:
            for line in stream:
                if not found_url:
                    stripped = line.strip()
                    if stripped:
                        recent_lines.append(stripped)
                        if len(recent_lines) > 15:
                            recent_lines.pop(0)
                    match = TRYCLOUDFLARE_RE.search(line)
                    if match:
                        found_url.append(match.group(0))
                        ready_event.set()
        finally:
            ready_event.set()

    reader_thread = threading.Thread(
        target=_reader_and_drainer,
        args=(proc.stdout,),
        daemon=True,
    )
    reader_thread.start()

    ready_event.wait(timeout=timeout)
    if found_url:
        return proc, found_url[0]

    returncode = proc.poll()
    stop_tunnel(proc)

    if returncode is not None:
        reason = f"cloudflared finalizó prematuramente (código {returncode}) sin emitir una URL pública."
    else:
        reason = (
            f"Tiempo de espera agotado ({timeout:.0f}s) esperando la URL pública "
            "de trycloudflare.com."
        )

    if recent_lines:
        tail = "\n".join(f"  | {entry}" for entry in recent_lines)
        reason = f"{reason}\nÚltimas líneas de salida:\n{tail}"

    raise RuntimeError(reason)


def _print_public_url_banner(public_url: str, host: str, port: int) -> None:
    """Imprime la URL pública destacada lista para pegar en Mi Equipo."""
    print()
    print("=" * 66)
    print("🌐 TÚNEL CLOUDFLARE ACTIVO — URL PÚBLICA PARA 'MI EQUIPO':")
    print(f"   {public_url}")
    print("=" * 66)
    print(f"• Apuntando hacia : http://{host}:{port}")
    print(f"• Verificación    : {public_url}/health")
    print("• Instrucciones   : pegá la URL base en Mi Equipo (sin '/solve' al final).")
    print("• Estado          : mantené esta terminal abierta; presioná Ctrl+C para cerrar.")
    print()


def main(argv: list[str] | None = None) -> int:
    """Entrypoint CLI para exponer el servidor FastAPI local mediante Cloudflare Tunnel."""
    parser = argparse.ArgumentParser(
        description=(
            "Expone tu servidor FastAPI local mediante Cloudflare Tunnel "
            "(trycloudflare.com) para evaluaciones remotas."
        )
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host local donde escucha uvicorn (por defecto: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Puerto local donde escucha uvicorn (por defecto: 8000).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Segundos máximos para obtener la URL pública del túnel (por defecto: 30s).",
    )
    parser.add_argument(
        "--skip-healthcheck",
        action="store_true",
        help="Omite la verificación previa de GET /health local antes de abrir el túnel.",
    )
    args = parser.parse_args(argv)

    check_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    if not args.skip_healthcheck:
        ok, detail = check_local_health(host=args.host, port=args.port)
        if ok:
            print(f"[OK] Healthcheck local http://{check_host}:{args.port}/health -> {detail}")
        else:
            print(
                f"[ERROR] No se pudo verificar GET http://{check_host}:{args.port}/health ({detail}).\n"
                "Parece que uvicorn no está levantado en ese puerto.\n"
                f"Iniciá el servidor en Terminal 1 con:\n"
                f"  uvicorn src.server:app --host 0.0.0.0 --port {args.port}\n"
                "Si querés abrir el túnel igual antes de levantar uvicorn, usá --skip-healthcheck.",
                file=sys.stderr,
            )
            return 1

    print(f"⏳ Abriendo túnel Cloudflare hacia http://{args.host}:{args.port}...")
    try:
        proc, public_url = start_tunnel(
            host=args.host,
            port=args.port,
            timeout=args.timeout,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    _print_public_url_banner(public_url, host=args.host, port=args.port)

    previous_sigterm: Any = None
    sigterm_registered = False
    if hasattr(signal, "SIGTERM") and threading.current_thread() is threading.main_thread():
        def _handle_sigterm(_signum: int, _frame: Any) -> None:
            raise KeyboardInterrupt

        previous_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _handle_sigterm)
        sigterm_registered = True

    try:
        while proc.poll() is None:
            time.sleep(0.5)
        print(
            f"[ERROR] El proceso cloudflared se cerró inesperadamente (código {proc.returncode}).",
            file=sys.stderr,
        )
        return proc.returncode or 1
    except KeyboardInterrupt:
        print("\n🛑 Cerrando túnel cloudflared...")
        return 0
    finally:
        stop_tunnel(proc)
        if sigterm_registered:
            signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    raise SystemExit(main())

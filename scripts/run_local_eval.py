"""
Cliente de Evaluación Local (scripts/run_local_eval.py).
Envía los casos de `data/dev_set.json` hacia tu microservicio HTTP local (`POST /solve`),
guarda las predicciones y ejecuta el evaluador oficial (`src/evaluator.py`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import httpx

from src.evaluator import evaluate_dataset, load_dataset, print_summary_table


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evalúa tu microservicio POST /solve localmente contra data/dev_set.json."
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8000",
        help="URL base de tu servidor FastAPI (por defecto: http://127.0.0.1:8000).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT_DIR / "data" / "dev_set.json",
        help="Ruta al archivo de dataset (por defecto: data/dev_set.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "predictions_local.json",
        help="Archivo donde guardar las predicciones.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cantidad opcional de casos a evaluar (ej. --limit 20 para pruebas rápidas).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout en segundos por caso (por defecto: 60s).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Muestra el detalle de casos fallidos.",
    )
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    cases = load_dataset(args.dataset)
    if args.limit:
        cases = cases[: args.limit]

    with httpx.Client(timeout=args.timeout) as client:
        # 1. Pre-vuelo GET /health
        try:
            health = client.get(f"{base_url}/health", timeout=5.0)
            health.raise_for_status()
            print(f"[OK] Healthcheck {base_url}/health -> {health.json()}")
        except Exception as exc:
            print(f"[ERROR] No se pudo conectar a {base_url}/health: {exc}", file=sys.stderr)
            return 1

        # 2. Despacho secuencial POST /solve
        predictions: list[dict] = []
        for idx, case in enumerate(cases, start=1):
            payload = case.to_case_input().model_dump(mode="json")
            try:
                resp = client.post(f"{base_url}/solve", json=payload)
                if resp.status_code == 200:
                    predictions.append(resp.json())
                else:
                    print(f"  [!] {case.case_id}: HTTP {resp.status_code}")
            except Exception as exc:
                print(f"  [!] {case.case_id}: Error HTTP ({exc})")

            if idx % 10 == 0 or idx == len(cases):
                print(f"  Procesados {idx}/{len(cases)} casos...")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = evaluate_dataset(cases, predictions)
    print_summary_table(summary, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


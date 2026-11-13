"""
Cliente de Evaluación Local (scripts/run_local_eval.py).
Envía los casos de `data/dev_set.json` hacia tu microservicio HTTP local (`POST /solve`),
guarda las predicciones, ejecuta el evaluador oficial (`src/evaluator.py`) y genera
un diagnóstico accionable sobre errores, latencia y abstenciones.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import httpx

from src.evaluator import (
    build_diagnostic_report,
    evaluate_dataset,
    load_dataset,
    print_diagnostics,
    print_summary_table,
)


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
        "--report",
        type=Path,
        default=ROOT_DIR / "eval_report_local.json",
        help="Archivo JSON donde guardar el reporte diagnóstico detallado.",
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
        help="Muestra todos los casos fallidos en consola (por defecto muestra hasta 3 por categoría).",
    )
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    cases = load_dataset(args.dataset)
    if args.limit:
        cases = cases[: args.limit]

    latencies_by_id: dict[str, float] = {}
    http_errors_by_id: dict[str, str] = {}
    raw_preds_by_id: dict[str, dict[str, Any]] = {}

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
        predictions: list[dict[str, Any]] = []
        for idx, case in enumerate(cases, start=1):
            payload = case.to_case_input().model_dump(mode="json")
            t0 = time.perf_counter()
            try:
                resp = client.post(f"{base_url}/solve", json=payload)
                latencies_by_id[case.case_id] = time.perf_counter() - t0
                if resp.status_code == 200:
                    data = resp.json()
                    predictions.append(data)
                    raw_preds_by_id[case.case_id] = data
                else:
                    msg = f"HTTP {resp.status_code}: {resp.text[:160]}"
                    http_errors_by_id[case.case_id] = msg
                    print(f"  [!] {case.case_id}: {msg}")
            except Exception as exc:
                latencies_by_id[case.case_id] = time.perf_counter() - t0
                http_errors_by_id[case.case_id] = str(exc)
                print(f"  [!] {case.case_id}: Error HTTP ({exc})")

            if idx % 10 == 0 or idx == len(cases):
                print(f"  Procesados {idx}/{len(cases)} casos...")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = evaluate_dataset(cases, predictions)
    print_summary_table(summary, verbose=False)

    report = build_diagnostic_report(
        cases=cases,
        summary=summary,
        latencies_by_id=latencies_by_id,
        http_errors_by_id=http_errors_by_id,
        raw_preds_by_id=raw_preds_by_id,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print_diagnostics(report, verbose=args.verbose)
    print(f"Reporte diagnóstico guardado en: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

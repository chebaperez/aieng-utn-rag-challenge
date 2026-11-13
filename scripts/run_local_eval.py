"""
Cliente de Evaluación Local (scripts/run_local_eval.py).
Envía los casos de `data/dev_set.json` hacia tu microservicio HTTP local (`POST /solve`),
guarda las predicciones, ejecuta el evaluador oficial (`src/evaluator.py`) y genera
un diagnóstico accionable sobre errores, latencia y abstenciones.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import httpx

from src.evaluator import (
    BenchmarkSummary,
    CaseEvaluationResult,
    evaluate_dataset,
    load_dataset,
    print_summary_table,
)
from src.schemas import DatasetCase


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(math.ceil(q * len(sorted_vals))) - 1))
    return sorted_vals[idx]


def _classify_failure_hint(
    result: CaseEvaluationResult,
    http_error: str | None,
) -> tuple[str, str]:
    """Deriva una categoría y pista diagnóstica accionable para un caso no óptimo."""
    if http_error:
        return "http_error", f"Fallo HTTP/red antes de evaluar: {http_error}"
    if result.status == "SCHEMA_ERROR":
        return "schema_error", result.reason
    if result.status == "HALLUCINATION_ON_ABSTAIN":
        return "hallucination_on_abstain", "Respondió SOLVED en un caso ABSTAIN (evidencia insuficiente/amputada)."
    if result.status == "ABSTAIN_CAUTIOUS":
        return "abstain_cautious", "Abstención cautelosa en caso resoluble (dejó +1.0 pt sobre la mesa)."
    if result.status == "SCALE_MISMATCH":
        return (
            "scale_mismatch",
            f"Escala incompatible: pred={result.predicted_scale!r} vs gt={result.expected_scale!r}.",
        )

    # WRONG_VALUE: detectar patrones numéricos frecuentes
    pred = result.normalized_predicted_value
    gt = result.expected_value
    if pred is not None and gt is not None and gt != 0:
        if result.expected_scale == "percent" and abs(pred * 100.0 - gt) <= max(0.05, abs(gt) * 0.01):
            return "percent_decimal", "Porcentaje en base 0-1 en vez de 0-100 (falta multiplicar por 100)."
        if abs(pred + gt) <= max(0.01, abs(gt) * 0.001):
            return "sign_error", "Signo invertido (posible resta en orden inverso, ej. año_A - año_B)."
        ratio = abs(pred / gt)
        for factor in (1000.0, 0.001, 1_000_000.0, 1e-6):
            if abs(ratio - factor) / factor <= 0.01:
                return "magnitude_10x", f"Desfase de magnitud ~{factor:g}x (revisar scale o unidad de tabla/texto)."
        if result.rel_error is not None and result.rel_error <= 0.05:
            return "near_miss_rounding", f"Diferencia < 5% (rel_err={result.rel_error:.2%}); revisar redondeo."

    return "wrong_arithmetic_or_extraction", "Valor incorrecto (error de extracción de operandos o fórmula)."


def _build_diagnostic_report(
    cases: list[DatasetCase],
    summary: BenchmarkSummary,
    latencies_by_id: dict[str, float],
    http_errors_by_id: dict[str, str],
    raw_preds_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cases_by_id = {c.case_id: c for c in cases}
    latencies = sorted(latencies_by_id.values())

    by_answer_from: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "ok": 0, "abstain_cautious": 0, "wrong": 0, "schema_or_http": 0}
    )
    taxonomy_counts: Counter[str] = Counter()
    taxonomy_points_impact: Counter[str] = Counter()
    non_optimal_cases: list[dict[str, Any]] = []

    for res in summary.results:
        case = cases_by_id[res.case_id]
        src = case.ground_truth.answer_from
        bucket = by_answer_from[src]
        bucket["total"] += 1

        if res.status in ("CORRECT", "ABSTAIN_ON_UNSOLVABLE"):
            bucket["ok"] += 1
            continue

        if res.status == "ABSTAIN_CAUTIOUS":
            bucket["abstain_cautious"] += 1
        elif res.status == "SCHEMA_ERROR":
            bucket["schema_or_http"] += 1
        else:
            bucket["wrong"] += 1

        http_err = http_errors_by_id.get(res.case_id)
        hint_cat, hint_detail = _classify_failure_hint(res, http_err)
        taxonomy_counts[hint_cat] += 1
        taxonomy_points_impact[hint_cat] += int(round((res.score - (1.0 if res.expected_verdict == "SOLVED" else 0.0)) * 10))

        raw_pred = raw_preds_by_id.get(res.case_id) or {}
        non_optimal_cases.append(
            {
                "case_id": res.case_id,
                "question": case.question,
                "answer_from": src,
                "status": res.status,
                "score": res.score,
                "hint_category": hint_cat,
                "hint_detail": hint_detail,
                "latency_s": round(latencies_by_id.get(res.case_id, 0.0), 3),
                "http_error": http_err,
                "expected": {
                    "verdict": case.ground_truth.verdict.value,
                    "primary_value": case.ground_truth.primary_value,
                    "scale": case.ground_truth.scale,
                    "derivation": case.ground_truth.derivation,
                },
                "predicted": {
                    "verdict": raw_pred.get("verdict"),
                    "primary_value": raw_pred.get("primary_value"),
                    "normalized_value": res.normalized_predicted_value,
                    "scale": raw_pred.get("scale"),
                    "derivation": raw_pred.get("derivation"),
                    "execution_trace": raw_pred.get("execution_trace", []),
                    "usage": raw_pred.get("usage"),
                },
            }
        )

    return {
        "summary": {
            "total_cases": summary.total_cases,
            "net_score": summary.net_score,
            "max_possible_score": summary.max_possible_score,
            "accuracy_rate": summary.accuracy_rate,
            "total_tokens": summary.total_tokens,
            "total_llm_calls": summary.total_llm_calls,
            "telemetry_anomaly_count": summary.telemetry_anomaly_count,
        },
        "latency": {
            "avg_s": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
            "p50_s": round(_percentile(latencies, 0.50), 3),
            "p95_s": round(_percentile(latencies, 0.95), 3),
            "max_s": round(latencies[-1], 3) if latencies else 0.0,
            "slow_over_45s": sum(1 for x in latencies if x > 45.0),
            "http_errors": len(http_errors_by_id),
        },
        "by_answer_from": dict(by_answer_from),
        "failure_taxonomy": {
            cat: {
                "count": count,
                "missed_or_lost_pts": round(abs(taxonomy_points_impact[cat]) / 10.0, 2),
            }
            for cat, count in taxonomy_counts.most_common()
        },
        "non_optimal_cases": non_optimal_cases,
    }


def _print_diagnostics(report: dict[str, Any], verbose: bool) -> None:
    lat = report["latency"]
    print("\n" + "-" * 78)
    print("  DIAGNÓSTICO DE CORRIDA LOCAL")
    print("-" * 78)
    print(
        f"  Latencia (s/caso) : avg={lat['avg_s']:.2f}s | p50={lat['p50_s']:.2f}s "
        f"| p95={lat['p95_s']:.2f}s | max={lat['max_s']:.2f}s "
        f"| >45s={lat['slow_over_45s']} | HTTP err={lat['http_errors']}"
    )

    print("  Rendimiento por origen de evidencia (answer_from):")
    for src, b in sorted(report["by_answer_from"].items()):
        pct = (b["ok"] / b["total"] * 100.0) if b["total"] else 0.0
        print(
            f"    - {src:<10}: {b['ok']:>3}/{b['total']:<3} OK ({pct:5.1f}%) "
            f"| wrong={b['wrong']} | abstain_cautious={b['abstain_cautious']} | schema/http={b['schema_or_http']}"
        )

    taxonomy = report["failure_taxonomy"]
    if not taxonomy:
        print("  ¡Sin fallos ni abstenciones cautelosas! 100% óptimo.")
        print("-" * 78)
        return

    print("  Taxonomía de causas de fallo / puntaje perdido:")
    for cat, info in taxonomy.items():
        print(f"    - {cat:<30}: {info['count']:>3} casos (impacto: -{info['missed_or_lost_pts']:.1f} pts)")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in report["non_optimal_cases"]:
        grouped[item["hint_category"]].append(item)

    max_examples = None if verbose else 3
    print("\n  Ejemplos representativos por categoría:")
    for cat in taxonomy:
        items = grouped[cat]
        shown = items if max_examples is None else items[:max_examples]
        print(f"\n  [{cat}] ({len(items)} casos):")
        for ex in shown:
            exp = ex["expected"]
            pred = ex["predicted"]
            print(f"    * {ex['case_id']} ({ex['answer_from']}, {ex['latency_s']:.1f}s): {ex['question']}")
            print(
                f"      gt={exp['verdict']} {exp['primary_value']} ({exp['scale']}) "
                f"| pred={pred['verdict']} {pred['primary_value']} ({pred['scale']})"
            )
            print(f"      pista: {ex['hint_detail']}")
            if pred.get("derivation"):
                print(f"      pred.derivation: {str(pred['derivation'])[:140]}")
        if max_examples is not None and len(items) > max_examples:
            print(f"    ... y {len(items) - max_examples} más en el reporte JSON.")
    print("-" * 78)


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

    report = _build_diagnostic_report(
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
    _print_diagnostics(report, verbose=args.verbose)
    print(f"Reporte diagnóstico guardado en: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

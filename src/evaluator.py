"""
Arnés de Evaluación Local y Motor de Calificación Oficial (evaluator.py).
Implementa la normalización dimensional al marco del Ground Truth, las tolerancias
numéricas (abs <= 0.01 o rel <= 0.001), la función de puntuación asimétrica
(+1.0 / 0.0 / -1.0 / -1.5) y la auditoría de anomalías de telemetría.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

try:
    from .schemas import CaseOutput, DatasetCase, GroundTruth, ScaleType, Verdict
except ImportError:
    from schemas import CaseOutput, DatasetCase, GroundTruth, ScaleType, Verdict

ABS_TOLERANCE: float = 0.01
REL_TOLERANCE: float = 0.001

SCORE_CORRECT: float = 1.0
SCORE_ABSTAIN: float = 0.0
SCORE_WRONG: float = -1.0
SCORE_SCHEMA_ERROR: float = -1.5

MAGNITUDE_FACTORS: dict[ScaleType | None, float] = {
    None: 1.0,
    "thousand": 1_000.0,
    "million": 1_000_000.0,
    "billion": 1_000_000_000.0,
}

EvaluationStatus = Literal[
    "CORRECT",
    "ABSTAIN_ON_UNSOLVABLE",
    "ABSTAIN_CAUTIOUS",
    "WRONG_VALUE",
    "SCALE_MISMATCH",
    "HALLUCINATION_ON_ABSTAIN",
    "SCHEMA_ERROR",
]


class CaseEvaluationResult(BaseModel):
    """Resultado detallado de la evaluación de un caso individual."""

    case_id: str
    status: EvaluationStatus
    score: float
    predicted_verdict: str | None = None
    expected_verdict: str
    predicted_value: float | None = None
    normalized_predicted_value: float | None = None
    expected_value: float | None = None
    predicted_scale: str | None = None
    expected_scale: str | None = None
    abs_error: float | None = None
    rel_error: float | None = None
    total_tokens: int = 0
    llm_calls: int = 0
    telemetry_anomaly: bool = False
    reason: str


class BenchmarkSummary(BaseModel):
    """Resumen agregado de una corrida de evaluación."""

    total_cases: int
    net_score: float
    max_possible_score: float
    accuracy_rate: float
    correct_count: int
    abstain_count: int
    abstain_on_unsolvable_count: int
    abstain_cautious_count: int
    wrong_count: int
    hallucination_on_abstain_count: int
    schema_error_count: int
    total_tokens: int
    total_llm_calls: int
    telemetry_anomaly_count: int
    results: list[CaseEvaluationResult]


def compare_numeric_with_scale(
    pred_val: float,
    pred_scale: ScaleType | None,
    gt_val: float,
    gt_scale: ScaleType | None,
) -> tuple[bool, bool, float, float, float]:
    """
    Normaliza `pred_val` al marco de escala de `gt_scale` y verifica las tolerancias:
      - `percent` es únicamente compatible con `percent` (expresado en base 0-100, ej. 15.4).
      - Las escalas monetarias/magnitud (`None`, `thousand`, `million`, `billion`) se convierten
        al marco del Ground Truth: `pred_norm = pred_val * factor(pred_scale) / factor(gt_scale)`.
      - Luego se evalúa `abs_err <= 0.01 or rel_err <= 0.001`.

    Retorna:
        (is_correct, scale_compatible, pred_norm, abs_err, rel_err)
    """
    if (gt_scale == "percent") != (pred_scale == "percent"):
        abs_err = abs(pred_val - gt_val)
        rel_err = abs_err / max(abs(gt_val), 1e-12)
        return False, False, pred_val, abs_err, rel_err

    if gt_scale == "percent" and pred_scale == "percent":
        pred_norm = pred_val
    else:
        pred_factor = MAGNITUDE_FACTORS[pred_scale]
        gt_factor = MAGNITUDE_FACTORS[gt_scale]
        pred_norm = pred_val * (pred_factor / gt_factor)

    abs_err = abs(pred_norm - gt_val)
    if abs(gt_val) < 1e-12:
        rel_err = 0.0 if abs_err < 1e-12 else float("inf")
    else:
        rel_err = abs_err / abs(gt_val)

    is_correct = (abs_err <= ABS_TOLERANCE) or (rel_err <= REL_TOLERANCE)
    return is_correct, True, pred_norm, abs_err, rel_err


def evaluate_single_case(
    dataset_case: DatasetCase,
    raw_prediction: dict[str, Any] | CaseOutput | None,
) -> CaseEvaluationResult:
    """Evalúa una predicción individual contra el Ground Truth del caso."""
    gt: GroundTruth = dataset_case.ground_truth

    if raw_prediction is None:
        return CaseEvaluationResult(
            case_id=dataset_case.case_id,
            status="SCHEMA_ERROR",
            score=SCORE_SCHEMA_ERROR,
            expected_verdict=gt.verdict.value,
            expected_value=gt.primary_value,
            expected_scale=gt.scale,
            reason="No se recibió respuesta para este case_id (timeout, caída de red o caso omitido).",
        )

    try:
        pred = (
            raw_prediction
            if isinstance(raw_prediction, CaseOutput)
            else CaseOutput.model_validate(raw_prediction)
        )
    except ValidationError as exc:
        first_err = exc.errors()[0] if exc.errors() else {"msg": str(exc)}
        return CaseEvaluationResult(
            case_id=dataset_case.case_id,
            status="SCHEMA_ERROR",
            score=SCORE_SCHEMA_ERROR,
            expected_verdict=gt.verdict.value,
            expected_value=gt.primary_value,
            expected_scale=gt.scale,
            reason=f"Incumplimiento de contrato CaseOutput: {first_err.get('loc')} -> {first_err.get('msg')}",
        )
    except Exception as exc:
        return CaseEvaluationResult(
            case_id=dataset_case.case_id,
            status="SCHEMA_ERROR",
            score=SCORE_SCHEMA_ERROR,
            expected_verdict=gt.verdict.value,
            expected_value=gt.primary_value,
            expected_scale=gt.scale,
            reason=f"Error al parsear respuesta: {exc}",
        )

    if pred.case_id != dataset_case.case_id:
        return CaseEvaluationResult(
            case_id=dataset_case.case_id,
            status="SCHEMA_ERROR",
            score=SCORE_SCHEMA_ERROR,
            predicted_verdict=pred.verdict.value,
            expected_verdict=gt.verdict.value,
            expected_value=gt.primary_value,
            expected_scale=gt.scale,
            reason=f"Desalineación de case_id: se esperaba '{dataset_case.case_id}', se recibió '{pred.case_id}'.",
        )

    telemetry_anomaly = (pred.usage.total_tokens < 50) or (pred.usage.llm_calls < 1)

    # Caso 1: El agente declara ABSTAIN -> Siempre suma 0.0 puntos
    if pred.verdict == Verdict.ABSTAIN:
        if gt.verdict == Verdict.ABSTAIN:
            status: EvaluationStatus = "ABSTAIN_ON_UNSOLVABLE"
            reason = "Abstención correcta ante evidencia amputada o insuficiente (0.0 pts)."
        else:
            status = "ABSTAIN_CAUTIOUS"
            reason = "Abstención prudente en caso resoluble (0.0 pts, evita penalización de -1.0)."

        return CaseEvaluationResult(
            case_id=dataset_case.case_id,
            status=status,
            score=SCORE_ABSTAIN,
            predicted_verdict=pred.verdict.value,
            expected_verdict=gt.verdict.value,
            predicted_value=None,
            expected_value=gt.primary_value,
            predicted_scale=pred.scale,
            expected_scale=gt.scale,
            total_tokens=pred.usage.total_tokens,
            llm_calls=pred.usage.llm_calls,
            telemetry_anomaly=telemetry_anomaly,
            reason=reason,
        )

    # Caso 2: El agente declara SOLVED pero el Ground Truth es ABSTAIN (alucinación sobre datos faltantes)
    if gt.verdict == Verdict.ABSTAIN:
        return CaseEvaluationResult(
            case_id=dataset_case.case_id,
            status="HALLUCINATION_ON_ABSTAIN",
            score=SCORE_WRONG,
            predicted_verdict=pred.verdict.value,
            expected_verdict=gt.verdict.value,
            predicted_value=pred.primary_value,
            expected_value=None,
            predicted_scale=pred.scale,
            expected_scale=gt.scale,
            total_tokens=pred.usage.total_tokens,
            llm_calls=pred.usage.llm_calls,
            telemetry_anomaly=telemetry_anomaly,
            reason="Alucinación penalizada (-1.0 pts): emitió SOLVED sobre un caso sin evidencia suficiente (GT=ABSTAIN).",
        )

    # Caso 3: Ambos son SOLVED -> Comparar magnitud normalizada al marco del Ground Truth
    assert pred.primary_value is not None and gt.primary_value is not None
    is_correct, scale_ok, pred_norm, abs_err, rel_err = compare_numeric_with_scale(
        pred_val=pred.primary_value,
        pred_scale=pred.scale,
        gt_val=gt.primary_value,
        gt_scale=gt.scale,
    )

    if is_correct:
        return CaseEvaluationResult(
            case_id=dataset_case.case_id,
            status="CORRECT",
            score=SCORE_CORRECT,
            predicted_verdict=pred.verdict.value,
            expected_verdict=gt.verdict.value,
            predicted_value=pred.primary_value,
            normalized_predicted_value=round(pred_norm, 6),
            expected_value=gt.primary_value,
            predicted_scale=pred.scale,
            expected_scale=gt.scale,
            abs_error=round(abs_err, 6),
            rel_error=round(rel_err, 6),
            total_tokens=pred.usage.total_tokens,
            llm_calls=pred.usage.llm_calls,
            telemetry_anomaly=telemetry_anomaly,
            reason="Respuesta exacta dentro de tolerancia (+1.0 pt).",
        )

    if not scale_ok:
        return CaseEvaluationResult(
            case_id=dataset_case.case_id,
            status="SCALE_MISMATCH",
            score=SCORE_WRONG,
            predicted_verdict=pred.verdict.value,
            expected_verdict=gt.verdict.value,
            predicted_value=pred.primary_value,
            normalized_predicted_value=round(pred_norm, 6),
            expected_value=gt.primary_value,
            predicted_scale=pred.scale,
            expected_scale=gt.scale,
            abs_error=round(abs_err, 6),
            rel_error=round(rel_err, 6),
            total_tokens=pred.usage.total_tokens,
            llm_calls=pred.usage.llm_calls,
            telemetry_anomaly=telemetry_anomaly,
            reason=f"Incompatibilidad dimensional (-1.0 pts): pred_scale='{pred.scale}' vs gt_scale='{gt.scale}'.",
        )

    return CaseEvaluationResult(
        case_id=dataset_case.case_id,
        status="WRONG_VALUE",
        score=SCORE_WRONG,
        predicted_verdict=pred.verdict.value,
        expected_verdict=gt.verdict.value,
        predicted_value=pred.primary_value,
        normalized_predicted_value=round(pred_norm, 6),
        expected_value=gt.primary_value,
        predicted_scale=pred.scale,
        expected_scale=gt.scale,
        abs_error=round(abs_err, 6),
        rel_error=round(rel_err, 6),
        total_tokens=pred.usage.total_tokens,
        llm_calls=pred.usage.llm_calls,
        telemetry_anomaly=telemetry_anomaly,
        reason=(
            f"Valor fuera de tolerancia (-1.0 pts): pred_norm={pred_norm:.4f} vs gt={gt.primary_value:.4f} "
            f"(abs_err={abs_err:.4f}, rel_err={rel_err:.5f})."
        ),
    )


def evaluate_dataset(
    dataset: list[DatasetCase],
    predictions: list[dict[str, Any]] | dict[str, dict[str, Any]],
) -> BenchmarkSummary:
    """Evalúa un lote completo de predicciones contra una partición del dataset."""
    pred_map: dict[str, dict[str, Any]] = {}
    if isinstance(predictions, dict):
        pred_map = predictions
    else:
        for item in predictions:
            if isinstance(item, dict) and "case_id" in item:
                pred_map[str(item["case_id"])] = item

    results: list[CaseEvaluationResult] = []
    for case in dataset:
        raw_pred = pred_map.get(case.case_id)
        results.append(evaluate_single_case(case, raw_pred))

    net_score = round(sum(r.score for r in results), 2)
    solvable_cases = sum(1 for c in dataset if c.ground_truth.verdict == Verdict.SOLVED)
    correct_count = sum(1 for r in results if r.status == "CORRECT")
    abstain_unsolvable = sum(1 for r in results if r.status == "ABSTAIN_ON_UNSOLVABLE")
    abstain_cautious = sum(1 for r in results if r.status == "ABSTAIN_CAUTIOUS")
    abstain_count = abstain_unsolvable + abstain_cautious
    hallucination_abstain = sum(1 for r in results if r.status == "HALLUCINATION_ON_ABSTAIN")
    wrong_count = sum(
        1 for r in results if r.status in ("WRONG_VALUE", "SCALE_MISMATCH", "HALLUCINATION_ON_ABSTAIN")
    )
    schema_error_count = sum(1 for r in results if r.status == "SCHEMA_ERROR")
    total_tokens = sum(r.total_tokens for r in results)
    total_llm_calls = sum(r.llm_calls for r in results)
    anomaly_count = sum(1 for r in results if r.telemetry_anomaly)

    accuracy_rate = round(
        (correct_count + abstain_unsolvable) / max(len(dataset), 1) * 100.0, 2
    )

    return BenchmarkSummary(
        total_cases=len(dataset),
        net_score=net_score,
        max_possible_score=float(solvable_cases),
        accuracy_rate=accuracy_rate,
        correct_count=correct_count,
        abstain_count=abstain_count,
        abstain_on_unsolvable_count=abstain_unsolvable,
        abstain_cautious_count=abstain_cautious,
        wrong_count=wrong_count,
        hallucination_on_abstain_count=hallucination_abstain,
        schema_error_count=schema_error_count,
        total_tokens=total_tokens,
        total_llm_calls=total_llm_calls,
        telemetry_anomaly_count=anomaly_count,
        results=results,
    )


def load_dataset(path: str | Path) -> list[DatasetCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [DatasetCase.model_validate(item) for item in raw]


def print_summary_table(summary: BenchmarkSummary, verbose: bool = False) -> None:
    print("\n" + "=" * 78)
    print("  BENCHMARK EVALUATION REPORT - AGENTIC FINANCIAL RAG (UTN FRSR)")
    print("=" * 78)
    print(f"  Casos Evaluados             : {summary.total_cases}")
    print(f"  PUNTAJE NETO ACUMULADO      : {summary.net_score:+.2f} / {summary.max_possible_score:+.2f} pts")
    print(f"  Tasa de Acierto Global      : {summary.accuracy_rate:.2f}%")
    print("-" * 78)
    print(f"  [+] SOLVED Correctos (+1.0) : {summary.correct_count}")
    print(
        f"  [0] ABSTAIN Totales  ( 0.0) : {summary.abstain_count} "
        f"(En casos sin datos: {summary.abstain_on_unsolvable_count} | Cautelosos: {summary.abstain_cautious_count})"
    )
    print(
        f"  [-] SOLVED Erróneos  (-1.0) : {summary.wrong_count} "
        f"(Alucinación en caso ABSTAIN: {summary.hallucination_on_abstain_count})"
    )
    print(f"  [!] Errores Esquema  (-1.5) : {summary.schema_error_count}")
    print("-" * 78)
    print(f"  Tokens Totales Consumidos   : {summary.total_tokens:,}")
    print(f"  Llamadas al LLM Acumuladas  : {summary.total_llm_calls:,}")
    print(f"  Anomalías de Telemetría     : {summary.telemetry_anomaly_count}")
    print("=" * 78)

    if verbose:
        print("\nDetalle de Casos Fallidos o con Errores de Esquema:")
        failures = [
            r for r in summary.results if r.status not in ("CORRECT", "ABSTAIN_ON_UNSOLVABLE")
        ]
        if not failures:
            print("  ¡Ninguno! Todos los casos fueron resueltos o abstenidos perfectamente.")
        for r in failures[:30]:
            print(
                f"  - [{r.case_id}] {r.status} ({r.score:+.1f} pts) | "
                f"pred={r.predicted_value} ({r.predicted_scale}) vs gt={r.expected_value} ({r.expected_scale}) | {r.reason}"
            )
        if len(failures) > 30:
            print(f"  ... y {len(failures) - 30} casos adicionales.")


PATTERN_DESCRIPTIONS: dict[str, str] = {
    "percent_decimal": "Porcentajes devueltos en base 0-1 en lugar de puntos porcentuales 0-100.",
    "sign_error": "Signo invertido en variaciones/diferencias (orden de resta invertido).",
    "magnitude_10x": "Desfase de magnitud (~1000x / ~1e6x) entre valor numérico y `scale`.",
    "near_miss_rounding": "Diferencias menores al 5% respecto al valor esperado (posible redondeo prematuro).",
    "scale_mismatch": "Incompatibilidad de escala (ej. `percent` vs escala monetaria/None).",
    "hallucination_on_abstain": "El agente respondió SOLVED en casos con evidencia insuficiente/amputada.",
    "abstain_cautious": "Abstención excesiva en casos resolubles (se dejó +1.0 pt por caso).",
    "wrong_arithmetic_or_extraction": "Error en extracción de operandos o fórmula aritmética.",
    "schema_error": "Violación del esquema `CaseOutput` o respuesta HTTP no válida (-1.5 pts).",
    "http_error": "Timeouts (>60s) o errores HTTP 5xx/conexión en el endpoint.",
    "http_or_timeout_error": "Timeouts (>60s) o errores HTTP 5xx/conexión en el endpoint.",
}


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(math.ceil(q * len(sorted_vals))) - 1))
    return sorted_vals[idx]


def classify_failure_hint(
    result: CaseEvaluationResult,
    http_error: str | None = None,
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


def build_actionable_recommendations(
    patterns: Counter[str],
    by_answer_from: dict[str, dict[str, Any]],
    slow_over_45s: int,
    http_errors: int,
    telemetry_anomalies: int,
) -> list[str]:
    """Genera recomendaciones accionables a partir de los patrones de fallo agregados."""
    recs: list[str] = []

    if patterns["percent_decimal"] > 0:
        recs.append(
            f"[{patterns['percent_decimal']} casos] Normalizá los porcentajes a base 0-100: "
            "cuando `scale == 'percent'`, devolvé `15.4` y nunca `0.154` (multiplicá por 100 en la derivación)."
        )
    if patterns["magnitude_10x"] > 0 or patterns["scale_mismatch"] > 0:
        n = patterns["magnitude_10x"] + patterns["scale_mismatch"]
        recs.append(
            f"[{n} casos] Revisá la detección de `scale` (`None`, `'thousand'`, `'million'`, `'billion'`, `'percent'`): "
            "asegurate de que el valor numérico y la escala declarada correspondan a la misma unidad de tabla/texto."
        )
    if patterns["sign_error"] > 0:
        recs.append(
            f"[{patterns['sign_error']} casos] Se detectaron errores de signo: verificá el orden de los períodos en "
            "variaciones/diferencias (`valor_actual - valor_anterior`)."
        )
    if patterns["near_miss_rounding"] > 0:
        recs.append(
            f"[{patterns['near_miss_rounding']} casos] Hay diferencias <5% cerca de la tolerancia: evitá redondear "
            "valores intermedios antes de completar la operación final en `safe_calculate`."
        )
    if patterns["hallucination_on_abstain"] > 0:
        recs.append(
            f"[{patterns['hallucination_on_abstain']} casos] Alucinación en casos `ABSTAIN`: reforzá la verificación de "
            "evidencia faltante (columnas de años ausentes o notas requeridas inexistentes) para abstenerte a tiempo."
        )
    if patterns["abstain_cautious"] > 0:
        recs.append(
            f"[{patterns['abstain_cautious']} casos] Abstenciones cautelosas excesivas: tu agente se abstuvo en casos "
            "resolubles; revisá los criterios del auditor o la recuperación de evidencia para no perder +1.0 pt por caso."
        )
    if patterns["wrong_arithmetic_or_extraction"] > 0:
        recs.append(
            f"[{patterns['wrong_arithmetic_or_extraction']} casos] Errores de extracción o fórmula: verificá que el agente "
            "seleccione la columna/año correcto de la tabla y cruce correctamente los párrafos narrativos."
        )
    if patterns["schema_error"] > 0 or patterns["http_error"] > 0 or http_errors > 0:
        recs.append(
            f"[{patterns['schema_error']} esquema / {http_errors} HTTP] Errores graves (-1.5 pts): validá la salida con "
            "`CaseOutput.model_validate(...)` antes de responder y controlá que el servidor no caiga ni exceda 60s."
        )
    if slow_over_45s > 0:
        recs.append(
            f"[{slow_over_45s} casos >45s] Latencia cerca del límite de 60s: reducí llamadas al LLM o tokens de contexto "
            "para evitar timeouts bajo carga."
        )
    if telemetry_anomalies > 0:
        recs.append(
            f"[{telemetry_anomalies} casos] Anomalías de telemetría: verificá que `usage` reporte `prompt_tokens`, "
            "`completion_tokens`, `total_tokens` y `llm_calls >= 1` reales."
        )

    weakest = min(
        ((k, v) for k, v in by_answer_from.items() if v["total"] > 0),
        key=lambda item: item[1]["accuracy_pct"],
        default=None,
    )
    if weakest and weakest[1]["accuracy_pct"] < 60.0:
        recs.append(
            f"Modalidad más débil: `{weakest[0]}` ({weakest[1]['ok']}/{weakest[1]['total']} OK, "
            f"{weakest[1]['accuracy_pct']:.1f}%). Priorizá mejorar la lectura/búsqueda en esa fuente."
        )

    if not recs:
        recs.append(
            "Desempeño sólido sin patrones de falla dominantes. Podés optimizar tokens y latencia para desempates."
        )
    return recs


def aggregate_diagnostics(
    cases: list[DatasetCase],
    summary: BenchmarkSummary,
    slow_over_45s: int,
    http_errors_count: int,
    latencies_by_id: dict[str, float] | None = None,
    http_errors_by_id: dict[str, str] | None = None,
    raw_preds_by_id: dict[str, dict[str, Any]] | None = None,
    include_case_details: bool = False,
) -> dict[str, Any]:
    """Agrega veredictos, rendimiento por fuente, patrones de fallo y recomendaciones."""
    cases_by_id = {c.case_id: c for c in cases}
    lat_map = latencies_by_id or {}
    err_map = http_errors_by_id or {}
    pred_map = raw_preds_by_id or {}

    by_answer_from: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "ok": 0, "wrong": 0, "abstain_cautious": 0, "schema_or_http": 0, "accuracy_pct": 0.0}
    )
    verdict_breakdown: Counter[str] = Counter()
    patterns: Counter[str] = Counter()
    pattern_impact_tenths: Counter[str] = Counter()
    non_optimal_cases: list[dict[str, Any]] = []

    for res in summary.results:
        verdict_breakdown[res.status] += 1
        case = cases_by_id.get(res.case_id)
        src = case.ground_truth.answer_from if case else "unknown"
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

        http_err = err_map.get(res.case_id) if include_case_details else None
        hint_cat, hint_detail = classify_failure_hint(res, http_err)
        patterns[hint_cat] += 1
        ideal = 1.0 if res.expected_verdict == "SOLVED" else 0.0
        pattern_impact_tenths[hint_cat] += int(round((ideal - res.score) * 10))

        if include_case_details and case is not None:
            raw_pred = pred_map.get(res.case_id) or {}
            non_optimal_cases.append(
                {
                    "case_id": res.case_id,
                    "question": case.question,
                    "answer_from": src,
                    "status": res.status,
                    "score": res.score,
                    "hint_category": hint_cat,
                    "hint_detail": hint_detail,
                    "latency_s": round(lat_map.get(res.case_id, 0.0), 3),
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

    for bucket in by_answer_from.values():
        total = bucket["total"]
        bucket["accuracy_pct"] = round((bucket["ok"] / total * 100.0) if total else 0.0, 1)

    failure_patterns = {
        cat: {
            "count": count,
            "missed_or_lost_pts": round(pattern_impact_tenths[cat] / 10.0, 2),
            "description": PATTERN_DESCRIPTIONS.get(cat, ""),
        }
        for cat, count in patterns.most_common()
    }
    recommendations = build_actionable_recommendations(
        patterns=patterns,
        by_answer_from=by_answer_from,
        slow_over_45s=slow_over_45s,
        http_errors=http_errors_count,
        telemetry_anomalies=summary.telemetry_anomaly_count,
    )

    out: dict[str, Any] = {
        "by_answer_from": dict(sorted(by_answer_from.items())),
        "verdict_breakdown": dict(verdict_breakdown),
        "failure_patterns": failure_patterns,
        "actionable_recommendations": recommendations,
    }
    if include_case_details:
        out["non_optimal_cases"] = non_optimal_cases
    return out


def build_diagnostic_report(
    cases: list[DatasetCase],
    summary: BenchmarkSummary,
    latencies_by_id: dict[str, float] | None = None,
    http_errors_by_id: dict[str, str] | None = None,
    raw_preds_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Construye un reporte diagnóstico ejecutivo + detalle por caso para corridas locales (dev_set.json)."""
    lat_map = latencies_by_id or {}
    err_map = http_errors_by_id or {}
    latencies = sorted(lat_map.values())
    slow_over_45s = sum(1 for x in latencies if x > 45.0)
    http_errors_count = len(err_map)

    agg = aggregate_diagnostics(
        cases=cases,
        summary=summary,
        slow_over_45s=slow_over_45s,
        http_errors_count=http_errors_count,
        latencies_by_id=lat_map,
        http_errors_by_id=err_map,
        raw_preds_by_id=raw_preds_by_id,
        include_case_details=True,
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
            "slow_over_45s": slow_over_45s,
            "http_errors": http_errors_count,
        },
        "by_answer_from": agg["by_answer_from"],
        "verdict_breakdown": agg["verdict_breakdown"],
        "failure_patterns": agg["failure_patterns"],
        "actionable_recommendations": agg["actionable_recommendations"],
        "non_optimal_cases": agg["non_optimal_cases"],
    }


def _format_trace_summary(trace: Any) -> str:
    if not isinstance(trace, list) or not trace:
        return "sin trazas"
    tools = [str(step.get("tool_name", "?")) for step in trace if isinstance(step, dict)]
    if not tools:
        return f"{len(trace)} pasos"
    preview = " -> ".join(tools[:5])
    if len(tools) > 5:
        preview += f" -> ... (+{len(tools) - 5})"
    return f"{len(tools)} pasos: {preview}"


def print_diagnostics(report: dict[str, Any], verbose: bool = False) -> None:
    """Imprime en consola el resumen ejecutivo, patrones de fallo, recomendaciones y ejemplos por caso."""
    lat = report["latency"]
    print("\n" + "-" * 78)
    print("  DIAGNÓSTICO DE CORRIDA LOCAL")
    print("-" * 78)
    print(
        f"  Latencia (s/caso) : avg={lat['avg_s']:.2f}s | p50={lat['p50_s']:.2f}s "
        f"| p95={lat['p95_s']:.2f}s | max={lat['max_s']:.2f}s "
        f"| >45s={lat['slow_over_45s']} | HTTP err={lat['http_errors']}"
    )

    verdicts = report.get("verdict_breakdown", {})
    if verdicts:
        ordered_keys = [
            "CORRECT",
            "ABSTAIN_ON_UNSOLVABLE",
            "ABSTAIN_CAUTIOUS",
            "WRONG_VALUE",
            "SCALE_MISMATCH",
            "HALLUCINATION_ON_ABSTAIN",
            "SCHEMA_ERROR",
        ]
        parts = [f"{k}={verdicts[k]}" for k in ordered_keys if verdicts.get(k)]
        for k, v in verdicts.items():
            if k not in ordered_keys and v:
                parts.append(f"{k}={v}")
        print(f"  Veredictos        : {' | '.join(parts)}")

    print("  Rendimiento por origen de evidencia (answer_from):")
    for src, b in sorted(report["by_answer_from"].items()):
        pct = b.get("accuracy_pct")
        if pct is None:
            pct = (b["ok"] / b["total"] * 100.0) if b["total"] else 0.0
        print(
            f"    - {src:<10}: {b['ok']:>3}/{b['total']:<3} OK ({pct:5.1f}%) "
            f"| wrong={b['wrong']} | abstain_cautious={b['abstain_cautious']} | schema/http={b['schema_or_http']}"
        )

    patterns = report.get("failure_patterns", {})
    recs = report.get("actionable_recommendations", [])
    if not patterns:
        print("  ¡Sin fallos ni abstenciones cautelosas! 100% óptimo.")
    else:
        print("\n  Patrones de fallo (impacto en puntos):")
        for cat, info in patterns.items():
            desc = info.get("description", "")
            desc_suffix = f" — {desc}" if desc else ""
            print(
                f"    - {cat:<30}: {info['count']:>3} casos "
                f"(impacto: -{info['missed_or_lost_pts']:.1f} pts){desc_suffix}"
            )

    if recs:
        print("\n  Recomendaciones accionables:")
        for idx, rec in enumerate(recs, start=1):
            print(f"    {idx}. {rec}")

    if not patterns:
        print("-" * 78)
        return

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in report.get("non_optimal_cases", []):
        grouped[item["hint_category"]].append(item)

    max_examples = None if verbose else 3
    print("\n  Ejemplos representativos por categoría:")
    for cat in patterns:
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
            if exp.get("derivation"):
                print(f"      gt.derivation  : {str(exp['derivation'])[:140]}")
            if pred.get("derivation"):
                print(f"      pred.derivation: {str(pred['derivation'])[:140]}")
            print(f"      trace          : {_format_trace_summary(pred.get('execution_trace'))}")
            print(f"      pista          : {ex['hint_detail']}")
        if max_examples is not None and len(items) > max_examples:
            print(f"    ... y {len(items) - max_examples} más en el reporte JSON.")
    print("-" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluador Local Offline para el Benchmark Challenge (UTN FRSR)."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "dev_set.json",
        help="Ruta al archivo de dataset con Ground Truth (por defecto: data/dev_set.json).",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Ruta al archivo JSON de predicciones generadas por el agente.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta opcional para guardar el reporte JSON detallado.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Muestra el desglose de casos fallidos en consola.",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Error: No se encontró el archivo de dataset en {args.dataset}", file=sys.stderr)
        return 1
    if not args.predictions.exists():
        print(f"Error: No se encontró el archivo de predicciones en {args.predictions}", file=sys.stderr)
        return 1

    dataset = load_dataset(args.dataset)
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    summary = evaluate_dataset(dataset, predictions)
    print_summary_table(summary, verbose=args.verbose)

    if args.output:
        args.output.write_text(
            json.dumps(summary.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nReporte detallado guardado en: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

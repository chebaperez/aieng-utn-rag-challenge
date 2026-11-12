"""
Arnés de Evaluación Local y Motor de Calificación Oficial (evaluator.py).
Implementa la normalización dimensional al marco del Ground Truth, las tolerancias
numéricas (abs <= 0.01 o rel <= 0.001), la función de puntuación asimétrica
(+1.0 / 0.0 / -1.0 / -1.5) y la auditoría de anomalías de telemetría.
"""

from __future__ import annotations

import argparse
import json
import sys
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluador Local Offline para el Benchmark Challenge (UTN FRSR)."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).parent / "dev_set.json",
        help="Ruta al archivo de dataset con Ground Truth (por defecto: dev_set.json).",
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

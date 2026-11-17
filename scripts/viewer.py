"""
Visualizador Web del Dataset Público y Predicciones (scripts/viewer.py).

Levanta una aplicación FastAPI ligera con UI estática en `scripts/viewer_ui/` para
explorar los 150 casos de `data/dev_set.json` y contrastarlos en vivo contra un
archivo de predicciones (`predictions_local.json` u otro JSON cargado en la UI).
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.evaluator import classify_failure_hint, evaluate_dataset, load_dataset
from src.schemas import DatasetCase

UI_DIR = Path(__file__).resolve().parent / "viewer_ui"


def _strip_internal_keys(item: dict[str, Any]) -> dict[str, Any]:
    """Elimina claves internas de checkpoints (ej. _latency_ms) incompatibles con CaseOutput."""
    return {k: v for k, v in item.items() if not str(k).startswith("_")}


def _extract_predictions_payload(raw_data: Any) -> list[dict[str, Any]] | dict[str, dict[str, Any]]:
    """Normaliza distintos formatos razonables de JSON de predicciones."""
    if isinstance(raw_data, list):
        return [_strip_internal_keys(item) for item in raw_data if isinstance(item, dict)]
    if isinstance(raw_data, dict):
        if "predictions" in raw_data and isinstance(raw_data["predictions"], list):
            return [_strip_internal_keys(item) for item in raw_data["predictions"] if isinstance(item, dict)]
        if all(isinstance(v, dict) for v in raw_data.values()):
            return {str(k): _strip_internal_keys(v) for k, v in raw_data.items()}
    raise ValueError(
        "Formato inválido: se esperaba una lista de objetos CaseOutput o un diccionario por case_id."
    )


def _index_raw_predictions(
    preds: list[dict[str, Any]] | dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if isinstance(preds, dict):
        return {str(k): v for k, v in preds.items() if isinstance(v, dict)}
    return {
        str(item["case_id"]): item
        for item in preds
        if isinstance(item, dict) and "case_id" in item
    }


def _evaluate_for_viewer(
    cases: list[DatasetCase],
    raw_data: Any,
    source_name: str,
) -> dict[str, Any]:
    preds = _extract_predictions_payload(raw_data)
    raw_by_id = _index_raw_predictions(preds)
    summary = evaluate_dataset(cases, preds)

    cases_eval: dict[str, dict[str, Any]] = {}
    for res in summary.results:
        hint_category: str | None = None
        hint_message: str | None = None
        if res.status not in ("CORRECT", "ABSTAIN_ON_UNSOLVABLE"):
            hint_category, hint_message = classify_failure_hint(res)

        cases_eval[res.case_id] = {
            "eval": res.model_dump(mode="json"),
            "hint_category": hint_category,
            "hint_message": hint_message,
            "prediction": raw_by_id.get(res.case_id),
        }

    summary_dict = summary.model_dump(mode="json", exclude={"results"})
    return {
        "loaded": True,
        "source_name": source_name,
        "summary": summary_dict,
        "cases_eval": cases_eval,
    }


def _parse_dataset_payload(raw_data: Any) -> list[DatasetCase]:
    """Valida una lista de casos o un objeto con clave 'cases'."""
    items = raw_data.get("cases") if isinstance(raw_data, dict) and "cases" in raw_data else raw_data
    if not isinstance(items, list) or not items:
        raise ValueError("Formato inválido: se esperaba una lista no vacía de casos DatasetCase.")
    try:
        return [DatasetCase.model_validate(item) for item in items]
    except Exception as exc:
        raise ValueError(f"Esquema de dataset inválido: {exc}") from exc


def create_app(
    dataset_path: Path,
    predictions_path: Path | None = None,
) -> FastAPI:
    """Construye la app FastAPI del visualizador."""
    state: dict[str, Any] = {
        "dataset_name": dataset_path.name,
        "cases": [],
        "cases_payload": [],
        "dataset_error": None,
    }

    if dataset_path.exists():
        cases = load_dataset(dataset_path)
        state["cases"] = cases
        state["cases_payload"] = [case.model_dump(mode="json") for case in cases]
    else:
        state["dataset_error"] = (
            f"No se encontró {dataset_path.name} en la ruta por defecto. "
            "Cargá un dataset (.json) desde la barra superior."
        )

    initial_eval: dict[str, Any] = {"loaded": False, "source_name": None}
    if predictions_path and predictions_path.exists() and state["cases"]:
        try:
            raw_preds = json.loads(predictions_path.read_text(encoding="utf-8"))
            initial_eval = _evaluate_for_viewer(state["cases"], raw_preds, predictions_path.name)
        except Exception as exc:
            initial_eval = {
                "loaded": False,
                "source_name": predictions_path.name,
                "error": f"No se pudo cargar {predictions_path.name}: {exc}",
            }

    app = FastAPI(title="UTN FRSR · Agentic RAG Benchmark Viewer")
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(UI_DIR / "index.html")

    @app.get("/help", include_in_schema=False)
    def help_page() -> FileResponse:
        return FileResponse(UI_DIR / "help.html")

    @app.get("/api/dataset")
    def get_dataset() -> dict[str, Any]:
        return {
            "dataset_name": state["dataset_name"],
            "total_cases": len(state["cases_payload"]),
            "cases": state["cases_payload"],
            "error": state["dataset_error"],
        }

    @app.post("/api/dataset")
    def upload_dataset(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        filename = str(payload.get("filename") or "dataset.json")
        if "dataset" not in payload:
            raise HTTPException(status_code=400, detail="Falta el campo 'dataset' en el cuerpo.")
        try:
            new_cases = _parse_dataset_payload(payload["dataset"])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        state["dataset_name"] = filename
        state["cases"] = new_cases
        state["cases_payload"] = [case.model_dump(mode="json") for case in new_cases]
        state["dataset_error"] = None
        return {
            "dataset_name": state["dataset_name"],
            "total_cases": len(state["cases_payload"]),
            "cases": state["cases_payload"],
        }

    @app.get("/api/predictions")
    def get_initial_predictions() -> dict[str, Any]:
        return initial_eval

    @app.post("/api/evaluate")
    def evaluate_uploaded_predictions(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        filename = str(payload.get("filename") or "predictions.json")
        if "predictions" not in payload:
            raise HTTPException(status_code=400, detail="Falta el campo 'predictions' en el cuerpo.")
        if not state["cases"]:
            raise HTTPException(status_code=400, detail="Primero cargá un dataset válido.")
        try:
            return _evaluate_for_viewer(state["cases"], payload["predictions"], filename)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Levanta el visualizador web para explorar data/dev_set.json y contrastar predicciones."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT_DIR / "data" / "dev_set.json",
        help="Ruta al dataset JSON (por defecto: data/dev_set.json).",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Ruta opcional a un JSON de predicciones (por defecto auto-detecta predictions_local.json si existe).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host donde escuchar (por defecto: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Puerto HTTP del visualizador (por defecto: 8501).",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Abre automáticamente el navegador al iniciar.",
    )
    args = parser.parse_args()

    preds_path = args.predictions
    if preds_path is None:
        default_preds = ROOT_DIR / "predictions_local.json"
        if default_preds.exists():
            preds_path = default_preds

    app = create_app(dataset_path=args.dataset, predictions_path=preds_path)

    url = f"http://{args.host}:{args.port}"
    print(f"Dataset cargado : {args.dataset}")
    if preds_path and preds_path.exists():
        print(f"Predicciones    : {preds_path}")
    else:
        print("Predicciones    : ninguna detectada (podés cargar un .json desde la UI)")
    print(f"Visualizador    : {url}")

    if args.open:
        webbrowser.open(url)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

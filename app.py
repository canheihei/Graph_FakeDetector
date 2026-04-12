import os
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory, url_for

from alignment.aligner import FeatureOntologyAligner
from alignment.evidence_builder import evidence_builder
from alignment.evolver import graph_evolver
from detector_config import get_candidate_review_config
from detectors.hub import DetectorHub
from project_paths import resolve_datasets_root
from service.candidate_benchmark import CandidateBenchmarkRunner
from service.candidate_graph import CandidateGraphStore
from service.candidate_review import (
    CandidateBenchmarkRequest,
    CandidatePromoteRequest,
    CandidateRequest,
    CandidateReviewFacade,
    CandidateUpdateRequest,
)
from service.candidate_store import CandidateStore
from service.facades import (
    DetectRequest,
    DetectionFacade,
    DirectIngestRequest,
    EvolutionFacade,
    IterateRequest,
    IterationFacade,
    ManualEvolutionRequest,
    SuggestDomainRequest,
    WorkflowError,
)
from service.report_gallery import ReportGalleryService
from service.neo_client import graph_writer, neo4j_client


APP_ROOT = Path(__file__).resolve().parent
hub = DetectorHub()
aligner = FeatureOntologyAligner(config_path="alignment/mapping_config.json")
FRONTEND_ROOT = APP_ROOT / "frontend"
DATASETS_ROOT = resolve_datasets_root(APP_ROOT)
UPLOAD_FOLDER = DATASETS_ROOT / "uploads" / "iterate"

ALLOWED_PAGES = {
    "index.html",
    "graph-iteration.html",
    "image-recognition.html",
    "visualization.html",
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(
    __name__,
    template_folder=str(FRONTEND_ROOT / "templates"),
    static_folder=str(FRONTEND_ROOT / "static"),
)
report_gallery = ReportGalleryService(Path(app.root_path) / "reports")

detect_facade = DetectionFacade(
    hub=hub,
    aligner=aligner,
    graph_evolver=graph_evolver,
    evidence_builder=evidence_builder,
    logger=app.logger,
)
iteration_facade = IterationFacade(
    neo4j_client=neo4j_client,
    graph_writer=graph_writer,
    upload_root=str(UPLOAD_FOLDER),
)
evolution_facade = EvolutionFacade(
    graph_evolver=graph_evolver,
    graph_writer=graph_writer,
    neo4j_client=neo4j_client,
)
candidate_review_facade = CandidateReviewFacade(
    candidate_store=CandidateStore(APP_ROOT / "alignment" / "mapping_candidates.json"),
    candidate_graph_store=CandidateGraphStore(neo4j_client),
    benchmark_runner=CandidateBenchmarkRunner(
        hub=hub,
        graph_evolver=graph_evolver,
        evidence_builder=evidence_builder,
        logger=app.logger,
        dataset_profile_roots=dict(get_candidate_review_config().dataset_profile_roots),
        active_mapping_path=APP_ROOT / "alignment" / "mapping_config.json",
    ),
    mapping_config_path=APP_ROOT / "alignment" / "mapping_config.json",
)


def _ok(payload, status=200):
    return jsonify(payload), status

def _error(message, status=400, **extra):
    payload = {"error": message}
    if extra:
        payload.update(extra)
    return jsonify(payload), status

def _parse_bool(source, key, default=False):
    raw = str(source.get(key, str(default))).strip().lower()
    return raw in {"1", "true", "yes", "on"}

def _parse_threshold(source, key="semantic_threshold", default=0.80):
    raw = source.get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a float between 0 and 1") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be between 0 and 1")
    return value


def _parse_optional_threshold(source, key: str):
    raw = source.get(key)
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a float between 0 and 1") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be between 0 and 1")
    return value

def _build_report_gallery_payload():
    reports = []
    for index, report in enumerate(report_gallery.list_reports()):
        payload = dict(report)
        report_name = payload["name"]
        payload["rank"] = index + 1
        payload["html_url"] = (
            url_for("serve_report_asset", report_name=report_name, asset_path="index.html")
            if payload.get("files", {}).get("has_html")
            else None
        )
        payload["metrics_url"] = (
            url_for("serve_report_asset", report_name=report_name, asset_path="metrics.json")
            if payload.get("files", {}).get("has_metrics")
            else None
        )
        payload["predictions_url"] = (
            url_for("serve_report_asset", report_name=report_name, asset_path="predictions.csv")
            if payload.get("files", {}).get("has_predictions")
            else None
        )
        reports.append(payload)

    latest_report = reports[0] if reports else None
    overview = {
        "report_count": len(reports),
        "latest_accuracy_pct": (
            latest_report.get("metrics", {}).get("accuracy_valid_pct", 0.0)
            if latest_report
            else 0.0
        ),
        "latest_samples": (
            latest_report.get("metrics", {}).get("total_samples", 0)
            if latest_report
            else 0
        ),
        "latest_updated_at_display": (
            latest_report.get("updated_at_display", "--") if latest_report else "--"
        ),
    }
    return {
        "reports": reports,
        "latest": latest_report,
        "overview": overview,
    }

@app.route("/iterate", methods=["POST"])
def iterate():
    try:
        payload = iteration_facade.execute(
            IterateRequest(
                prompt=request.form.get("prompt", ""),
                image_files=request.files.getlist("images"),
                semantic_threshold=_parse_threshold(request.form),
            )
        )
        return _ok(payload)
    except WorkflowError as exc:
        return _error(exc.message, exc.status_code)
    except ValueError as exc:
        return _error(str(exc), 400)
    except Exception as exc:
        app.logger.exception("[WARN] iterate failed")
        return _error(str(exc), 500)

@app.route("/iterate_directly", methods=["POST"])
def ingest_feature_domain():
    try:
        payload = evolution_facade.ingest(
            DirectIngestRequest(payload=request.get_json())
        )
        return _ok(payload)
    except WorkflowError as exc:
        return _error(exc.message, exc.status_code)
    except Exception as exc:
        app.logger.exception("[WARN] ingest-feature-domain failed")
        return _error(str(exc), 500)

@app.route("/detect", methods=["POST"])
def detect():
    try:
        if "image" not in request.files:
            return _error("No image uploaded", 400)

        payload = detect_facade.execute(
            DetectRequest(
                image_bytes=request.files["image"].read(),
                auto_evolve_enabled=_parse_bool(request.form, "auto_evolve", True),
                semantic_threshold=_parse_threshold(request.form),
                use_llm_generation=_parse_bool(
                    request.form,
                    "use_llm_generation",
                    False,
                ),
                decision_profile=(request.form.get("decision_profile") or "").strip() or None,
                decision_threshold_override=_parse_optional_threshold(
                    request.form,
                    "decision_threshold_override",
                ),
            )
        )
        return _ok(payload)
    except WorkflowError as exc:
        return _error(exc.message, exc.status_code)
    except ValueError as exc:
        return _error(str(exc), 400)
    except Exception as exc:
        app.logger.exception("[WARN] detect failed")
        return _error(str(exc), 500)

@app.route("/evolve", methods=["POST"])
def evolve():
    try:
        data = request.get_json()
        if not data:
            return _error("Invalid JSON", 400)

        payload = evolution_facade.evolve(
            ManualEvolutionRequest(
                features=data.get("features", []),
                evolutions=data.get("evolutions", []),
                semantic_threshold=_parse_threshold(data),
            )
        )
        return _ok(payload)
    except WorkflowError as exc:
        return _error(exc.message, exc.status_code)
    except ValueError as exc:
        return _error(str(exc), 400)
    except Exception as exc:
        app.logger.exception("[WARN] evolve failed")
        return _error(str(exc), 500)

@app.route("/suggest_domain", methods=["POST"])
def suggest_domain():
    try:
        data = request.get_json()
        if not data:
            return _error("Invalid JSON", 400)

        payload = evolution_facade.suggest_domain(
            SuggestDomainRequest(
                detector=data.get("detector", ""),
                feature=data.get("feature", ""),
                score=data.get("score", 0),
                raw_value=data.get("raw_value", 0),
            )
        )
        return _ok(payload)
    except WorkflowError as exc:
        return _error(exc.message, exc.status_code)
    except Exception as exc:
        app.logger.exception("[WARN] suggest-domain failed")
        return _error(str(exc), 500)

@app.route("/detect/candidates", methods=["POST"])
def detect_candidates():
    try:
        data = request.get_json()
        if not data:
            return _error("Invalid JSON", 400)

        payload = candidate_review_facade.generate(
            CandidateRequest(
                detect_result=data.get("detect_result", {}),
                source_sample_name=data.get("source_sample_name", ""),
                decision_profile=(data.get("decision_profile") or "").strip() or None,
            )
        )
        return _ok(payload)
    except WorkflowError as exc:
        return _error(exc.message, exc.status_code)
    except ValueError as exc:
        return _error(str(exc), 400)
    except Exception as exc:
        app.logger.exception("[WARN] detect-candidates failed")
        return _error(str(exc), 500)

@app.route("/candidate-mappings", methods=["GET"])
def list_candidate_mappings():
    try:
        status = (request.args.get("status") or "").strip() or None
        return _ok(candidate_review_facade.list_items(status=status))
    except Exception as exc:
        app.logger.exception("[WARN] list-candidate-mappings failed")
        return _error(str(exc), 500)

@app.route("/candidate-mappings/update", methods=["POST"])
def update_candidate_mapping():
    try:
        data = request.get_json()
        if not data:
            return _error("Invalid JSON", 400)
        payload = candidate_review_facade.update_item(
            CandidateUpdateRequest(
                candidate_id=str(data.get("candidate_id", "") or ""),
                graph_candidate=data.get("graph_candidate"),
                mapping_candidate=data.get("mapping_candidate"),
                status=data.get("status"),
                approval_state=data.get("approval_state"),
            )
        )
        return _ok(payload)
    except WorkflowError as exc:
        return _error(exc.message, exc.status_code)
    except (KeyError, ValueError) as exc:
        return _error(str(exc), 400)
    except Exception as exc:
        app.logger.exception("[WARN] update-candidate-mapping failed")
        return _error(str(exc), 500)

@app.route("/candidate-mappings/benchmark", methods=["POST"])
def benchmark_candidate_mappings():
    try:
        data = request.get_json()
        if not data:
            return _error("Invalid JSON", 400)
        payload = candidate_review_facade.benchmark(
            CandidateBenchmarkRequest(
                candidate_ids=list(data.get("candidate_ids", [])),
                mode=str(data.get("mode", "") or ""),
                decision_profile=(data.get("decision_profile") or "").strip() or None,
                sample_per_class=data.get("sample_per_class"),
                semantic_threshold=_parse_threshold(data),
                decision_threshold_override=_parse_optional_threshold(
                    data,
                    "decision_threshold_override",
                ),
            )
        )
        return _ok(payload)
    except WorkflowError as exc:
        return _error(exc.message, exc.status_code)
    except ValueError as exc:
        return _error(str(exc), 400)
    except Exception as exc:
        app.logger.exception("[WARN] benchmark-candidate-mappings failed")
        return _error(str(exc), 500)

@app.route("/candidate-mappings/promote", methods=["POST"])
def promote_candidate_mappings():
    try:
        data = request.get_json()
        if not data:
            return _error("Invalid JSON", 400)
        payload = candidate_review_facade.promote(
            CandidatePromoteRequest(
                candidate_ids=list(data.get("candidate_ids", [])),
            )
        )
        return _ok(payload)
    except WorkflowError as exc:
        return _error(exc.message, exc.status_code)
    except ValueError as exc:
        return _error(str(exc), 400)
    except Exception as exc:
        app.logger.exception("[WARN] promote-candidate-mappings failed")
        return _error(str(exc), 500)

@app.route("/stats", methods=["GET"])
def graph_stats():
    try:
        return _ok(neo4j_client.get_graph_stats())
    except Exception as exc:
        app.logger.exception("[WARN] graph stats failed")
        return _error(str(exc), 500)

@app.route("/test", methods=["POST"])
def test():
    specificdomain = neo4j_client.get_specificdomain_nodes()
    subdomain = neo4j_client.get_subdomain_nodes()
    return _ok({"specificdomain": specificdomain, "subdomain": subdomain})

@app.route("/neo4j_overview", methods=["GET"])
def neo4j_overview():
    try:
        return _ok(neo4j_client.get_graph_overview())
    except Exception as exc:
        app.logger.exception("[WARN] neo4j overview failed")
        return _error(str(exc), 500)

@app.route("/api/reports", methods=["GET"])
def list_reports():
    try:
        return _ok(_build_report_gallery_payload())
    except Exception as exc:
        app.logger.exception("[WARN] reports listing failed")
        return _error(str(exc), 500)

@app.route("/api/reports/<report_name>", methods=["DELETE"])
def delete_report(report_name: str):
    try:
        report_gallery.delete_report(report_name)
        return _ok(
            {
                "message": f"Deleted report: {report_name}",
                "report_name": report_name,
                **_build_report_gallery_payload(),
            }
        )
    except FileNotFoundError:
        return _error("Report not found", 404, report_name=report_name)
    except Exception as exc:
        app.logger.exception("[WARN] report delete failed")
        return _error(str(exc), 500, report_name=report_name)

@app.route("/reports/view/<report_name>/", defaults={"asset_path": "index.html"})
@app.route("/reports/view/<report_name>/<path:asset_path>")
def serve_report_asset(report_name: str, asset_path: str):
    try:
        report_dir = (Path(app.root_path) / "reports" / report_name).resolve()
        reports_root = (Path(app.root_path) / "reports").resolve()
        if reports_root not in report_dir.parents:
            abort(404)
        if not report_dir.exists() or not report_dir.is_dir():
            abort(404)
        asset_file = (report_dir / asset_path).resolve()
        if report_dir not in asset_file.parents and asset_file != report_dir:
            abort(404)
        if not asset_file.exists():
            abort(404)
        return send_from_directory(str(report_dir), asset_path)
    except Exception:
        abort(404)

@app.route("/")
@app.route("/<page>")
def render_page(page="index.html"):
    if page not in ALLOWED_PAGES:
        return render_template("404.html"), 404
    if page == "index.html":
        return render_template(page, report_gallery=_build_report_gallery_payload())
    return render_template(page)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)

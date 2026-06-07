import os
from pathlib import Path
import re
import shutil

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
    CandidateDeleteRequest,
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
from service.report_gallery import ReportGalleryService, describe_report_name
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
    "evidence-chain-report.html",
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
        graph_writer=graph_writer,
        neo4j_client=neo4j_client,
    ),
    mapping_config_path=APP_ROOT / "alignment" / "mapping_config.json",
    graph_writer=graph_writer,
    neo4j_client=neo4j_client,
    logger=app.logger,
    aligner=aligner,
)
GRAPH_RESET_CONFIRM_TOKEN = "RESET_BASELINE_GRAPH"
MAPPING_RESET_CONFIRM_TOKEN = "RESET_BASELINE_MAPPING"
SYSTEM_RESET_CONFIRM_TOKEN = "RESET_GRAPH_AND_MAPPING"


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
            if payload.get("files", {}).get("has_metrics")
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


def _build_report_view_payload(report_name: str, report_dir: Path) -> dict:
    metrics_payload = _load_json_object(report_dir / "metrics.json")
    summary = dict(metrics_payload.get("summary", {}) or {})
    audit_summary = dict(metrics_payload.get("audit_summary", {}) or {})
    threshold_calibration = dict(metrics_payload.get("threshold_calibration", {}) or {})
    records = list(metrics_payload.get("records", []) or [])
    display = describe_report_name(report_name)
    top_errors = [
        record for record in records
        if not bool(record.get("is_correct", False))
    ][:30]
    return {
        "name": report_name,
        "title": display["title"],
        "subtitle": display["subtitle"],
        "dataset_label": display["dataset_label"],
        "summary": summary,
        "audit_summary": audit_summary,
        "threshold_calibration": threshold_calibration,
        "top_errors": top_errors,
        "source_metrics_url": url_for("serve_report_asset", report_name=report_name, asset_path="metrics.json"),
        "source_predictions_url": url_for("serve_report_asset", report_name=report_name, asset_path="predictions.csv"),
    }


def _load_indicator_report_payload():
    indicator_path = APP_ROOT / "reports" / "Indicators" / "indicator_report_data.json"
    if not indicator_path.exists():
        raise FileNotFoundError(f"Indicator report data not found: {indicator_path}")
    payload = jsonify_or_json_loads(indicator_path.read_text(encoding="utf-8"))
    payload.setdefault("evidence", {})
    payload["evidence"]["evolution"] = _build_candidate_evolution_payload()
    payload["evidence"]["new_hit_metrics"] = _build_indicator_evidence_extension()
    scripts = list(payload.get("scripts", []))
    scripts = [item for item in scripts if str(item.get("name", "")) != "scripts/benchmark/visualize_detect_benchmark.py"]
    scripts.append(
        {
            "name": "scripts/benchmark/visualize_detect_benchmark.py",
            "purpose": "证据链指标由 compute_audit_summary() 统一抽取，当前新增输出 joint_evidence_correct_rate 与 fake_joint_evidence_recall。",
        }
    )
    payload["scripts"] = scripts
    return payload


def _load_json_object(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = jsonify_or_json_loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _compute_extended_audit_metrics_from_records(records: list[dict]) -> dict:
    total = len(records)
    if total <= 0:
        return {
            "joint_evidence_correct_rate": 0.0,
            "fake_joint_evidence_recall": 0.0,
        }
    fake_total = 0
    joint_evidence_correct_count = 0
    fake_joint_evidence_correct_count = 0
    for record in records:
        evidence_count = int(record.get("evidence_count", 0) or 0)
        truth_label = str(record.get("truth_label", "") or "")
        predicted_label = str(record.get("predicted_label", "") or "")
        is_correct = bool(record.get("is_correct", False))
        if evidence_count > 0 and is_correct:
            joint_evidence_correct_count += 1
        if truth_label == "FAKE":
            fake_total += 1
            if evidence_count > 0 and predicted_label == "FAKE":
                fake_joint_evidence_correct_count += 1
    return {
        "joint_evidence_correct_rate": joint_evidence_correct_count / total,
        "fake_joint_evidence_recall": (
            fake_joint_evidence_correct_count / fake_total if fake_total > 0 else 0.0
        ),
    }


def _get_extended_audit_metrics(metrics_payload: dict) -> dict:
    audit_summary = dict(metrics_payload.get("audit_summary", {}) or {})
    if (
        "joint_evidence_correct_rate" in audit_summary
        and "fake_joint_evidence_recall" in audit_summary
    ):
        return {
            "joint_evidence_correct_rate": float(audit_summary.get("joint_evidence_correct_rate", 0.0) or 0.0),
            "fake_joint_evidence_recall": float(audit_summary.get("fake_joint_evidence_recall", 0.0) or 0.0),
        }
    records = list(metrics_payload.get("records", []) or [])
    return _compute_extended_audit_metrics_from_records(records)


def _build_indicator_evidence_extension() -> dict:
    report_map = [
        ("Celeb-DF", APP_ROOT / "reports" / "report_celeb_df_sample300_profile_celeb_df_evidencehit_2026-04-12" / "metrics.json"),
        ("DFDC", APP_ROOT / "reports" / "report_dfdc_sample300_profile_dfdc_evidencehit_2026-04-12" / "metrics.json"),
        ("WildDeepfake", APP_ROOT / "reports" / "report_wilddeepfake_sample300_profile_wilddeepfake_evidencehit_2026-04-12" / "metrics.json"),
    ]
    rows = []
    joint_values = []
    fake_joint_values = []
    for label, metrics_path in report_map:
        metrics_payload = _load_json_object(metrics_path)
        if not metrics_payload:
            continue
        extended = _get_extended_audit_metrics(metrics_payload)
        joint_rate = float(extended.get("joint_evidence_correct_rate", 0.0) or 0.0)
        fake_joint_rate = float(extended.get("fake_joint_evidence_recall", 0.0) or 0.0)
        joint_values.append(joint_rate)
        fake_joint_values.append(fake_joint_rate)
        rows.append(
            {
                "report": label,
                "joint_evidence_correct_rate": f"{joint_rate * 100.0:.2f}%",
                "fake_joint_evidence_recall": f"{fake_joint_rate * 100.0:.2f}%",
            }
        )
    avg_joint = sum(joint_values) / len(joint_values) if joint_values else 0.0
    avg_fake_joint = sum(fake_joint_values) / len(fake_joint_values) if fake_joint_values else 0.0
    return {
        "title": "新增命中率指标",
        "script_name": "scripts/benchmark/visualize_detect_benchmark.py",
        "script_function": "compute_audit_summary()",
        "cards": [
            {
                "label": "平均联合命中率",
                "value": f"{avg_joint * 100.0:.2f}%",
                "tone": "indigo",
            },
            {
                "label": "平均假样本联合召回",
                "value": f"{avg_fake_joint * 100.0:.2f}%",
                "tone": "emerald",
            },
        ],
        "rows": rows,
    }


def _build_candidate_evolution_payload() -> dict:
    baseline_report_map = {
        "celeb_df": "report_celeb_df_sample300_profile_celeb_df_evidencehit_2026-04-12",
        "celebdf": "report_celeb_df_sample300_profile_celeb_df_evidencehit_2026-04-12",
        "dfdc": "report_dfdc_sample300_profile_dfdc_evidencehit_2026-04-12",
        "wilddeepfake": "report_wilddeepfake_sample300_profile_wilddeepfake_evidencehit_2026-04-12",
    }
    baseline_metrics: dict[str, dict] = {}
    reports_root = APP_ROOT / "reports"
    for profile, report_name in baseline_report_map.items():
        metrics_payload = _load_json_object(reports_root / report_name / "metrics.json")
        audit_summary = dict(metrics_payload.get("audit_summary", {}) or {})
        summary = dict(metrics_payload.get("summary", {}) or {})
        if not audit_summary:
            continue
        baseline_metrics[profile] = {
            "accuracy_valid": float(summary.get("accuracy_valid", 0.0) or 0.0),
            "evidence_hit_rate": float(audit_summary.get("evidence_hit_rate", 0.0) or 0.0),
            "fake_evidence_hit_rate": float(audit_summary.get("fake_evidence_hit_rate", 0.0) or 0.0),
        }

    candidate_payload = _load_json_object(APP_ROOT / "alignment" / "mapping_candidates.json")
    items = list(candidate_payload.get("items", []) or [])
    rows = []
    for item in items:
        benchmarks = dict(item.get("benchmarks", {}) or {})
        benchmark_mode = None
        benchmark_result = None
        for mode in ("formal", "quick"):
            candidate_result = benchmarks.get(mode)
            if isinstance(candidate_result, dict) and candidate_result.get("audit_summary"):
                benchmark_mode = mode
                benchmark_result = candidate_result
                break
        if benchmark_result is None:
            continue

        decision_profile = str(
            benchmark_result.get("decision_profile")
            or (item.get("source", {}) or {}).get("decision_profile", "")
            or ""
        ).strip().lower()
        baseline = baseline_metrics.get(decision_profile, {})
        audit_summary = dict(benchmark_result.get("audit_summary", {}) or {})
        summary = dict(benchmark_result.get("summary", {}) or {})
        promotion = dict(item.get("promotion", {}) or {})
        graph_candidate = dict(item.get("graph_candidate", {}) or {})
        mapping_candidate = dict(item.get("mapping_candidate", {}) or {})

        benchmark_hit_rate = float(audit_summary.get("evidence_hit_rate", 0.0) or 0.0)
        benchmark_fake_hit_rate = float(audit_summary.get("fake_evidence_hit_rate", 0.0) or 0.0)
        baseline_hit_rate = float(baseline.get("evidence_hit_rate", 0.0) or 0.0)
        baseline_fake_hit_rate = float(baseline.get("fake_evidence_hit_rate", 0.0) or 0.0)
        delta_hit_rate = benchmark_hit_rate - baseline_hit_rate
        delta_fake_hit_rate = benchmark_fake_hit_rate - baseline_fake_hit_rate

        rows.append(
            {
                "candidate_id": str(item.get("candidate_id", "")),
                "mapping_key": (
                    f"{mapping_candidate.get('detector', '')}:{mapping_candidate.get('feature', '')}"
                ).strip(":"),
                "specific_domain": str(graph_candidate.get("specific_domain", "") or ""),
                "subdomain_name": str(graph_candidate.get("subdomain_name", "") or ""),
                "decision_profile": decision_profile or "--",
                "benchmark_mode": benchmark_mode,
                "accuracy_valid": round(float(summary.get("accuracy_valid", 0.0) or 0.0) * 100.0, 2),
                "baseline_hit_rate": round(baseline_hit_rate * 100.0, 2),
                "benchmark_hit_rate": round(benchmark_hit_rate * 100.0, 2),
                "delta_hit_rate": round(delta_hit_rate * 100.0, 2),
                "baseline_fake_hit_rate": round(baseline_fake_hit_rate * 100.0, 2),
                "benchmark_fake_hit_rate": round(benchmark_fake_hit_rate * 100.0, 2),
                "delta_fake_hit_rate": round(delta_fake_hit_rate * 100.0, 2),
                "passed": bool(benchmark_result.get("passed", False)),
                "promoted": bool(promotion.get("promoted_at")),
                "active_subdomain_name": str(promotion.get("active_subdomain_name", "") or ""),
                "improved": delta_hit_rate > 0.0,
            }
        )

    rows.sort(
        key=lambda row: (
            1 if row.get("improved") else 0,
            float(row.get("delta_hit_rate", 0.0)),
            float(row.get("benchmark_hit_rate", 0.0)),
        ),
        reverse=True,
    )
    improved_rows = [row for row in rows if row.get("improved")]
    promoted_count = sum(1 for row in rows if row.get("promoted"))
    best_gain = max((float(row.get("delta_hit_rate", 0.0)) for row in rows), default=0.0)
    return {
        "title": "审批进化增益",
        "baseline": "sample300 当前证据链基线",
        "summary_cards": [
            {"label": "已评测候选", "value": str(len(rows)), "tone": "indigo"},
            {"label": "Hit Rate 提升候选", "value": str(len(improved_rows)), "tone": "emerald"},
            {"label": "最佳 Hit Rate 增益", "value": f"{best_gain:.2f}%", "tone": "amber"},
            {"label": "已晋级候选", "value": str(promoted_count), "tone": "sky"},
        ],
        "rows": rows,
        "empty_message": "当前还没有候选审批 benchmark 数据。完成 quick/formal benchmark 后，这里会自动展示审批进化后的 hit rate 增益。",
    }


def _load_baseline_graph_cypher() -> str:
    cypher_doc = APP_ROOT / "cyper.md"
    if not cypher_doc.exists():
        raise FileNotFoundError(f"Baseline cypher file not found: {cypher_doc}")
    text = cypher_doc.read_text(encoding="utf-8")
    match = re.search(r"```cypher\s*(.*?)\s*```", text, flags=re.S)
    if not match:
        raise ValueError("No ```cypher``` block found in cyper.md")
    return match.group(1).strip()


def _load_mapping_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Mapping config not found: {path}")
    return jsonify_or_json_loads(path.read_text(encoding="utf-8"))


def jsonify_or_json_loads(text: str) -> dict:
    import json

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("mapping config payload must be a JSON object")
    return data


def _reset_mapping_config_to_baseline() -> dict:
    mapping_path = APP_ROOT / "alignment" / "mapping_config.json"
    baseline_path = APP_ROOT / "alignment" / "mapping_config.baseline.json"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline mapping file not found: {baseline_path}")
    backup_path = mapping_path.with_suffix(".pre_reset_backup.json")
    before = mapping_path.read_text(encoding="utf-8") if mapping_path.exists() else ""
    shutil.copyfile(mapping_path, backup_path) if mapping_path.exists() else None
    shutil.copyfile(baseline_path, mapping_path)
    aligner.load_config(str(mapping_path))
    after = mapping_path.read_text(encoding="utf-8")
    return {
        "mapping_path": str(mapping_path),
        "baseline_path": str(baseline_path),
        "backup_path": str(backup_path) if before else None,
        "before_bytes": len(before.encode("utf-8")),
        "after_bytes": len(after.encode("utf-8")),
        "aligner_reloaded": True,
    }


def _reset_graph_to_baseline() -> dict:
    before = neo4j_client.get_graph_overview()
    cypher = _load_baseline_graph_cypher()
    with neo4j_client.driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n").consume()
        session.run(cypher).consume()
    after = neo4j_client.get_graph_overview()
    return {
        "before": before.get("summary", before),
        "after": after.get("summary", after),
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

@app.route("/candidate-mappings/delete", methods=["POST"])
def delete_candidate_mappings():
    try:
        data = request.get_json()
        if not data:
            return _error("Invalid JSON", 400)
        payload = candidate_review_facade.delete(
            CandidateDeleteRequest(
                candidate_ids=list(data.get("candidate_ids", [])),
            )
        )
        return _ok(payload)
    except WorkflowError as exc:
        return _error(exc.message, exc.status_code)
    except ValueError as exc:
        return _error(str(exc), 400)
    except Exception as exc:
        app.logger.exception("[WARN] delete-candidate-mappings failed")
        return _error(str(exc), 500)

@app.route("/mapping/config", methods=["GET"])
def mapping_config_view():
    try:
        mapping_path = APP_ROOT / "alignment" / "mapping_config.json"
        baseline_path = APP_ROOT / "alignment" / "mapping_config.baseline.json"
        payload = _load_mapping_config(mapping_path)
        baseline = _load_mapping_config(baseline_path) if baseline_path.exists() else {"version": "", "rules": []}
        baseline_keys = {
            (str(rule.get("detector", "")), str(rule.get("feature", "")))
            for rule in baseline.get("rules", [])
        }
        rules = list(payload.get("rules", []))
        grouped: dict[str, list[dict]] = {}
        for rule in rules:
            detector = str(rule.get("detector", "") or "UnknownDetector")
            grouped.setdefault(detector, []).append(rule)
        for detector_rules in grouped.values():
            detector_rules.sort(key=lambda item: str(item.get("feature", "")))

        summary = {
            "rule_count": len(rules),
            "detector_count": len(grouped),
            "baseline_rule_count": len(baseline.get("rules", [])),
            "nonbaseline_rule_count": sum(
                1
                for rule in rules
                if (str(rule.get("detector", "")), str(rule.get("feature", ""))) not in baseline_keys
            ),
        }
        return _ok(
            {
                "version": payload.get("version", ""),
                "summary": summary,
                "detectors": grouped,
                "mapping_path": str(mapping_path),
                "baseline_path": str(baseline_path),
            }
        )
    except FileNotFoundError as exc:
        return _error(str(exc), 500)
    except ValueError as exc:
        return _error(str(exc), 400)
    except Exception as exc:
        app.logger.exception("[WARN] mapping-config-view failed")
        return _error(str(exc), 500)

@app.route("/stats", methods=["GET"])
def graph_stats():
    try:
        return _ok(neo4j_client.get_graph_stats())
    except Exception as exc:
        app.logger.exception("[WARN] graph stats failed")
        return _error(str(exc), 500)

@app.route("/graph/reset_baseline", methods=["POST"])
def reset_graph_baseline():
    try:
        data = request.get_json()
        if not data:
            return _error("Invalid JSON", 400)
        confirm = str(data.get("confirm", "") or "").strip()
        if confirm != GRAPH_RESET_CONFIRM_TOKEN:
            return _error(
                f"Confirmation token mismatch. Expected: {GRAPH_RESET_CONFIRM_TOKEN}",
                400,
            )

        graph_reset = _reset_graph_to_baseline()
        app.logger.warning(
            "[RESET] graph reset to cyper.md baseline; before=%s after=%s",
            graph_reset.get("before", {}),
            graph_reset.get("after", {}),
        )
        return _ok(
            {
                "status": "success",
                "message": "Neo4j graph reset to cyper.md baseline",
                "confirm_token_used": GRAPH_RESET_CONFIRM_TOKEN,
                **graph_reset,
                "note": "Only Neo4j graph was reset. mapping_config.json was not modified.",
            }
        )
    except ValueError as exc:
        return _error(str(exc), 400)
    except FileNotFoundError as exc:
        return _error(str(exc), 500)
    except Exception as exc:
        app.logger.exception("[WARN] graph baseline reset failed")
        return _error(str(exc), 500)

@app.route("/mapping/reset_baseline", methods=["POST"])
def reset_mapping_baseline():
    try:
        data = request.get_json()
        if not data:
            return _error("Invalid JSON", 400)
        confirm = str(data.get("confirm", "") or "").strip()
        if confirm != MAPPING_RESET_CONFIRM_TOKEN:
            return _error(
                f"Confirmation token mismatch. Expected: {MAPPING_RESET_CONFIRM_TOKEN}",
                400,
            )

        mapping_reset = _reset_mapping_config_to_baseline()
        app.logger.warning("[RESET] mapping_config reset to baseline: %s", mapping_reset)
        return _ok(
            {
                "status": "success",
                "message": "mapping_config.json reset to baseline",
                "confirm_token_used": MAPPING_RESET_CONFIRM_TOKEN,
                **mapping_reset,
                "note": "Only mapping_config.json was reset. Neo4j graph was not modified.",
            }
        )
    except ValueError as exc:
        return _error(str(exc), 400)
    except FileNotFoundError as exc:
        return _error(str(exc), 500)
    except Exception as exc:
        app.logger.exception("[WARN] mapping baseline reset failed")
        return _error(str(exc), 500)

@app.route("/system/reset_baseline", methods=["POST"])
def reset_system_baseline():
    try:
        data = request.get_json()
        if not data:
            return _error("Invalid JSON", 400)
        confirm = str(data.get("confirm", "") or "").strip()
        if confirm != SYSTEM_RESET_CONFIRM_TOKEN:
            return _error(
                f"Confirmation token mismatch. Expected: {SYSTEM_RESET_CONFIRM_TOKEN}",
                400,
            )

        graph_reset = _reset_graph_to_baseline()
        mapping_reset = _reset_mapping_config_to_baseline()
        app.logger.warning(
            "[RESET] system reset to baseline; graph=%s mapping=%s",
            graph_reset,
            mapping_reset,
        )
        return _ok(
            {
                "status": "success",
                "message": "Neo4j graph and mapping_config.json reset to baseline",
                "confirm_token_used": SYSTEM_RESET_CONFIRM_TOKEN,
                "graph": graph_reset,
                "mapping": mapping_reset,
            }
        )
    except ValueError as exc:
        return _error(str(exc), 400)
    except FileNotFoundError as exc:
        return _error(str(exc), 500)
    except Exception as exc:
        app.logger.exception("[WARN] system baseline reset failed")
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


@app.route("/api/indicator-report", methods=["GET"])
def indicator_report():
    try:
        return _ok(_load_indicator_report_payload())
    except FileNotFoundError as exc:
        return _error(str(exc), 404)
    except Exception as exc:
        app.logger.exception("[WARN] indicator report loading failed")
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
        if asset_path == "index.html":
            if not (report_dir / "metrics.json").exists():
                abort(404)
            return render_template(
                "report-view.html",
                report_view=_build_report_view_payload(report_name, report_dir),
            )
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

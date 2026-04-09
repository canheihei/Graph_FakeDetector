from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


SAFE_REPORT_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class ReportArtifact:
    name: str
    root: Path
    summary: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

    def to_payload(self) -> Dict[str, Any]:
        accuracy_valid = float(
            self.summary.get("accuracy_valid", self.summary.get("accuracy_all", 0.0)) or 0.0
        )
        accuracy_all = float(self.summary.get("accuracy_all", accuracy_valid) or accuracy_valid)
        balanced_accuracy = float(self.summary.get("balanced_accuracy", accuracy_valid) or 0.0)
        precision_fake = float(self.summary.get("precision_fake", 0.0) or 0.0)
        recall_fake = float(self.summary.get("recall_fake", 0.0) or 0.0)
        avg_confidence_correct = float(self.summary.get("avg_confidence_correct", 0.0) or 0.0)

        return {
            "name": self.name,
            "title": self.name.replace("_", " "),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_at_display": self.created_at.strftime("%Y-%m-%d %H:%M"),
            "updated_at_display": self.updated_at.strftime("%Y-%m-%d %H:%M"),
            "summary": self.summary,
            "metrics": {
                "total_samples": int(self.summary.get("total_samples", 0) or 0),
                "valid_predictions": int(self.summary.get("valid_predictions", 0) or 0),
                "correct_predictions": int(self.summary.get("correct_predictions", 0) or 0),
                "error_count": int(self.summary.get("error_count", 0) or 0),
                "accuracy_valid_pct": round(accuracy_valid * 100.0, 2),
                "accuracy_all_pct": round(accuracy_all * 100.0, 2),
                "balanced_accuracy_pct": round(balanced_accuracy * 100.0, 2),
                "precision_fake_pct": round(precision_fake * 100.0, 2),
                "recall_fake_pct": round(recall_fake * 100.0, 2),
                "avg_confidence_correct_pct": round(avg_confidence_correct * 100.0, 2),
                "tp": int(self.summary.get("tp", 0) or 0),
                "tn": int(self.summary.get("tn", 0) or 0),
                "fp": int(self.summary.get("fp", 0) or 0),
                "fn": int(self.summary.get("fn", 0) or 0),
            },
            "files": {
                "has_html": (self.root / "index.html").is_file(),
                "has_metrics": (self.root / "metrics.json").is_file(),
                "has_predictions": (self.root / "predictions.csv").is_file(),
            },
        }


class ReportGalleryService:
    def __init__(self, reports_root: str | Path):
        self._reports_root = Path(reports_root)

    def list_reports(self) -> List[Dict[str, Any]]:
        if not self._reports_root.exists():
            return []

        report_dirs = [
            path
            for path in self._reports_root.iterdir()
            if path.is_dir() and (path / "metrics.json").is_file()
        ]
        report_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)

        reports: List[Dict[str, Any]] = []
        for report_dir in report_dirs:
            artifact = self._load_report(report_dir)
            if artifact is None:
                continue
            reports.append(artifact.to_payload())
        return reports

    def delete_report(self, report_name: str) -> None:
        report_dir = self._resolve_report_dir(report_name)
        if not report_dir.exists() or not report_dir.is_dir():
            raise FileNotFoundError(f"Report not found: {report_name}")
        shutil.rmtree(report_dir)

    def _load_report(self, report_dir: Path) -> ReportArtifact | None:
        metrics_path = report_dir / "metrics.json"
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        summary = payload.get("summary")
        if not isinstance(summary, dict):
            return None

        stat = report_dir.stat()
        created_at = datetime.fromtimestamp(stat.st_ctime)
        updated_at = datetime.fromtimestamp(stat.st_mtime)
        return ReportArtifact(
            name=report_dir.name,
            root=report_dir,
            summary=summary,
            created_at=created_at,
            updated_at=updated_at,
        )

    def _resolve_report_dir(self, report_name: str) -> Path:
        if not SAFE_REPORT_NAME.fullmatch(report_name):
            raise FileNotFoundError(f"Invalid report name: {report_name}")

        report_dir = (self._reports_root / report_name).resolve()
        reports_root = self._reports_root.resolve()
        if reports_root not in report_dir.parents:
            raise FileNotFoundError(f"Invalid report path: {report_name}")
        return report_dir

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class CandidateStore:
    def __init__(self, path: Path | str):
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def _default_payload(self) -> Dict[str, Any]:
        return {"version": "1.0", "items": []}

    def _read(self) -> Dict[str, Any]:
        if not self._path.exists():
            return self._default_payload()
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return self._default_payload()
        payload.setdefault("version", "1.0")
        payload.setdefault("items", [])
        return payload

    def _write(self, payload: Dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def list_items(self, *, status: str | None = None) -> List[Dict[str, Any]]:
        items = list(self._read().get("items", []))
        if not status:
            return items
        expected = status.strip().lower()
        return [
            item for item in items
            if str(item.get("status", "")).strip().lower() == expected
        ]

    def get_item(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        for item in self._read().get("items", []):
            if str(item.get("candidate_id")) == str(candidate_id):
                return item
        return None

    def append_items(self, items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        payload = self._read()
        serialized = list(items)
        payload.setdefault("items", []).extend(serialized)
        self._write(payload)
        return serialized

    def replace_item(self, candidate_id: str, updated_item: Dict[str, Any]) -> Dict[str, Any]:
        payload = self._read()
        items = payload.setdefault("items", [])
        for index, item in enumerate(items):
            if str(item.get("candidate_id")) == str(candidate_id):
                items[index] = updated_item
                self._write(payload)
                return updated_item
        raise KeyError(f"Candidate '{candidate_id}' not found")

    def update_item(self, candidate_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        item = self.get_item(candidate_id)
        if item is None:
            raise KeyError(f"Candidate '{candidate_id}' not found")
        updated = dict(item)
        updated.update(patch)
        return self.replace_item(candidate_id, updated)

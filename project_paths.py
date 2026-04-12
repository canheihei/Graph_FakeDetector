from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_DIRNAME = "Datasets"
LEGACY_DATASET_DIRNAME = "Dataset"
LEGACY_UPLOADS_DIRNAME = "uploads"
PROMPTS_DIRNAME = "prompts"
MAIN_PROMPT_FILENAME = "main_prompt.txt"
DETECT_CANDIDATE_PROMPT_FILENAME = "detect_candidate_prompt.txt"


def resolve_datasets_root(
    base_dir: Path | None = None,
    *,
    create: bool = True,
    migrate_legacy: bool = True,
) -> Path:
    root = (base_dir or PROJECT_ROOT).resolve()
    canonical = root / DATASETS_DIRNAME
    legacy = root / LEGACY_DATASET_DIRNAME

    if migrate_legacy and legacy.exists() and not canonical.exists():
        try:
            legacy.rename(canonical)
        except OSError:
            shutil.move(str(legacy), str(canonical))

    if canonical.exists():
        datasets_root = canonical
    elif legacy.exists():
        datasets_root = legacy
    else:
        datasets_root = canonical
        if create:
            canonical.mkdir(parents=True, exist_ok=True)

    if migrate_legacy:
        _migrate_legacy_uploads(root, datasets_root)
    return datasets_root


def _migrate_legacy_uploads(root: Path, datasets_root: Path) -> None:
    legacy_uploads = root / LEGACY_UPLOADS_DIRNAME
    target_uploads = datasets_root / LEGACY_UPLOADS_DIRNAME
    if not legacy_uploads.exists() or not legacy_uploads.is_dir():
        return
    if not target_uploads.exists():
        target_uploads.parent.mkdir(parents=True, exist_ok=True)
        try:
            legacy_uploads.rename(target_uploads)
        except OSError:
            shutil.move(str(legacy_uploads), str(target_uploads))
        return

    for item in legacy_uploads.iterdir():
        destination = target_uploads / item.name
        if destination.exists():
            continue
        try:
            item.rename(destination)
        except OSError:
            shutil.move(str(item), str(destination))
    try:
        legacy_uploads.rmdir()
    except OSError:
        pass


def resolve_main_prompt_path(base_dir: Path | None = None) -> Path:
    root = (base_dir or PROJECT_ROOT).resolve()
    preferred = root / PROMPTS_DIRNAME / MAIN_PROMPT_FILENAME
    if preferred.exists():
        return preferred

    legacy = root / MAIN_PROMPT_FILENAME
    if legacy.exists():
        return legacy

    return preferred


def resolve_detect_candidate_prompt_path(base_dir: Path | None = None) -> Path:
    root = (base_dir or PROJECT_ROOT).resolve()
    preferred = root / PROMPTS_DIRNAME / DETECT_CANDIDATE_PROMPT_FILENAME
    return preferred

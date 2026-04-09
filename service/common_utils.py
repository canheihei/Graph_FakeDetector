from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re

from PIL import Image


_detector = None


class _FallbackDetector:
    def predict(self, images):
        if not isinstance(images, list):
            images = [images]
            is_batch = False
        else:
            is_batch = True

        results = []
        for _ in images:
            results.append(
                {
                    "is_fake": False,
                    "label": "real",
                    "confidence": 0.5,
                    "fake_score": 0.5,
                }
            )
        return results if is_batch else results[0]


def _get_detector():
    global _detector
    if _detector is not None:
        return _detector

    try:
        from model import XceptionDetector

        _detector = XceptionDetector(model_path=None, device="cuda")
        print("[RELOAD] loaded XceptionDetector for image pre-analysis")
    except Exception as exc:
        print(f"[WARN] failed to load XceptionDetector, using fallback detector: {exc}")
        _detector = _FallbackDetector()
    return _detector


def encode_image(path: str) -> str:
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def safe_path_name(name: str) -> str:
    unsafe_chars = r'[\\/:\*\?"<>\|\x00]'
    cleaned = re.sub(unsafe_chars, "_", name.strip())
    result = cleaned.strip(" _")[:255]
    return result or "unnamed"


def extract_json(s: str):
    if not s or not isinstance(s, str):
        raise ValueError("LLM returned empty content")

    match = re.search(r"```json(.*?)```", s, flags=re.S)
    if match:
        s = match.group(1).strip()

    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"LLM output does not contain JSON: {s}")

    json_str = s[start : end + 1]
    try:
        return json.loads(json_str)
    except Exception:
        print("[WARN] failed JSON content:")
        print(json_str)
        raise


def compress_image(path: str, max_size: int = 512) -> str:
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.thumbnail((max_size, max_size))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


async def process_one(path: str) -> str:
    loop = asyncio.get_event_loop()
    b64_str = await loop.run_in_executor(None, compress_image, path)
    return f"data:image/jpeg;base64,{b64_str}"


async def get_image_base64_list(image_paths: list) -> list:
    tasks = [asyncio.create_task(process_one(path)) for path in image_paths]
    return await asyncio.gather(*tasks)


def build_detection_response(image_paths: list, base64_list: list, domain_name: str):
    pil_images = []
    for path in image_paths:
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        pil_images.append(image)

    detector = _get_detector()
    detection_results = detector.predict(pil_images)
    results = []
    for index, path in enumerate(image_paths):
        item = detection_results[index]
        results.append(
            {
                "base64": base64_list[index],
                "path": os.path.basename(path),
                "is_fake": item["is_fake"],
                "label": item["label"],
                "confidence": item["confidence"],
                "fake_score": item["fake_score"],
            }
        )

    return {
        "results": results,
        "domain_name": domain_name,
    }

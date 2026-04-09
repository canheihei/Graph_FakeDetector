from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List

import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

from config import ALI_API_KEY, ALI_BASE_URL
from project_paths import resolve_main_prompt_path
from service.common_utils import extract_json


client = OpenAI(api_key=ALI_API_KEY, base_url=ALI_BASE_URL)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "paraphrase-multilingual-MiniLM-L12-v2"
MODEL_PATH = os.getenv("SEMANTIC_MODEL_PATH", str(DEFAULT_MODEL_PATH))

_model = None
_model_lock = threading.Lock()


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if raw is None:
        return default
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or default


LLM_MODEL_NAME = os.getenv("ITERATE_LLM_MODEL", "moonshot-v1-32k")
LLM_FALLBACK_MODELS = _env_list(
    "ITERATE_LLM_FALLBACK_MODELS",
    ["moonshot-v1-8k", "kimi-k2.5"],
)
LLM_TEMPERATURE = _env_float("ITERATE_LLM_TEMPERATURE", 1.0)
LLM_MAX_TOKENS = _env_int("ITERATE_LLM_MAX_TOKENS", 1024)
LLM_TIMEOUT_SECONDS = _env_float("ITERATE_LLM_TIMEOUT_SECONDS", 45.0)


def call_qwen(prompt_dict: Dict[str, Any]) -> Dict[str, Any]:
    prompt_path = resolve_main_prompt_path()
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    with prompt_path.open("r", encoding="utf-8") as handle:
        domain_prompt = handle.read()

    user_input = safe_truncate_json(prompt_dict, max_length=28000)
    messages = [
        {"role": "system", "content": domain_prompt},
        {"role": "user", "content": user_input},
    ]

    candidates: List[str] = []
    for model_name in [LLM_MODEL_NAME, *LLM_FALLBACK_MODELS]:
        if model_name and model_name not in candidates:
            candidates.append(model_name)

    raw = ""
    last_exc: Exception | None = None
    for model_name in candidates:
        request_args = {
            "model": model_name,
            "messages": messages,
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
        }
        try:
            response = client.with_options(timeout=LLM_TIMEOUT_SECONDS).chat.completions.create(**request_args)
        except Exception as exc:
            # Some OpenAI-compatible providers pin temperature to 1 for specific models.
            if "invalid temperature" in str(exc).lower() and float(request_args["temperature"]) != 1.0:
                request_args["temperature"] = 1.0
                try:
                    response = client.with_options(timeout=LLM_TIMEOUT_SECONDS).chat.completions.create(**request_args)
                except Exception as retry_exc:
                    last_exc = retry_exc
                    print(f"[WARN] LLM request failed on model={model_name}: {retry_exc}")
                    continue
            else:
                last_exc = exc
                print(f"[WARN] LLM request failed on model={model_name}: {exc}")
                continue

        message_content = response.choices[0].message.content
        if isinstance(message_content, list):
            raw = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in message_content
            ).strip()
        else:
            raw = str(message_content or "").strip()

        if raw:
            if model_name != LLM_MODEL_NAME:
                print(f"[RELOAD] iterate switched to fallback model: {model_name}")
            break

        last_exc = ValueError(f"LLM returned empty content (model={model_name})")
        print(f"[WARN] LLM returned empty content on model={model_name}, trying fallback...")

    if not raw:
        if last_exc is not None:
            raise last_exc
        raise ValueError("LLM returned empty content")

    try:
        parsed = json.loads(raw)
        print("[RELOAD] RAW LLM OUTPUT (pretty):")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except json.JSONDecodeError as exc:
        print("[WARN] RAW LLM OUTPUT (invalid JSON):")
        print(repr(raw))
        print(f"[WARN] failed to parse JSON directly: {exc}")

    try:
        return extract_json(raw)
    except Exception:
        snippet = raw[raw.find("{") : raw.rfind("}") + 1]
        return json.loads(snippet)


def safe_payload(prompt: str, existing_schema: Dict[str, Any], image_infos: List[Dict[str, Any]]) -> Dict[str, Any]:
    existing_schema_json = json.dumps(existing_schema, ensure_ascii=False)
    if len(existing_schema_json) > 8000:
        existing_schema = {"truncated": True}

    if len(image_infos) > 4:
        image_infos = image_infos[:4]

    return {
        "prompt": prompt[:2000],
        "existing_schema": existing_schema,
        "images": image_infos,
    }


def match_domain(prompt: str, specific_domains: list, sub_domains: list, threshold: float = 0.75) -> str:
    matched_specific = semantic_match(prompt, specific_domains, threshold)
    if matched_specific != prompt:
        return matched_specific

    matched_sub = semantic_match(prompt, sub_domains, threshold)
    if matched_sub != prompt:
        return matched_sub

    return prompt


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if not os.path.exists(MODEL_PATH):
                    raise FileNotFoundError(
                        f"SentenceTransformer model not found: {MODEL_PATH}"
                    )
                _model = SentenceTransformer(MODEL_PATH, device="cuda")
    return _model


def semantic_match(prompt: str, candidates: list, threshold: float = 0.65) -> str:
    if not candidates:
        return prompt

    model = get_model()
    prompt_emb = model.encode(prompt, convert_to_tensor=True)
    candidates_emb = model.encode(candidates, convert_to_tensor=True)
    similarities = util.cos_sim(prompt_emb, candidates_emb)[0]
    max_sim_score, max_idx = torch.max(similarities, dim=0)
    if max_sim_score.item() >= threshold:
        return candidates[max_idx.item()]
    return prompt


def safe_truncate_json(obj: Dict[str, Any], max_length: int = 30000) -> str:
    full_json = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    if len(full_json) <= max_length:
        return full_json

    truncated_obj: Dict[str, Any] = {}
    for key in ["prompt", "domain_name", "summary", "images", "existing_schema"]:
        if key not in obj:
            continue

        if key == "prompt":
            truncated_obj[key] = str(obj[key])[:1000]
            continue

        if key == "domain_name":
            truncated_obj[key] = obj[key]
            continue

        if key == "summary":
            summary = obj[key]
            if isinstance(summary, dict):
                truncated_obj[key] = summary
            continue

        if key == "images":
            safe_images = []
            for image in obj[key][:3]:
                safe_images.append({k: v for k, v in image.items() if k != "base64"})
            truncated_obj[key] = safe_images
            continue

        if key == "existing_schema":
            schema = obj[key]
            safe_schema = {"domain": schema.get("domain", "")}
            features = schema.get("features", [])
            if isinstance(features, list):
                safe_schema["features"] = sorted(
                    [item for item in features if isinstance(item, dict)],
                    key=lambda item: float(item.get("fake_score", 0)),
                    reverse=True,
                )[:8]
            truncated_obj[key] = safe_schema

    result = json.dumps(truncated_obj, ensure_ascii=False, separators=(",", ":"))
    if len(result) > max_length and "prompt" in truncated_obj:
        overflow = len(result) - max_length
        truncated_obj["prompt"] = truncated_obj["prompt"][: max(0, len(truncated_obj["prompt"]) - overflow - 10)]
        result = json.dumps(truncated_obj, ensure_ascii=False, separators=(",", ":"))
    return result


def reasoning(evidence: list, decision: dict) -> dict:
    explanations: List[str] = []
    chains: List[str] = []

    for item in evidence:
        domain_name = item.get("specific_domain", {}).get("name", "UnknownSpecificDomain")
        subdomain_name = item.get("sub_domain", {}).get("name", "UnknownSubDomain")
        main_domain_name = item.get("main_domain", {}).get("name", "UnknownMainDomain")
        confidence = item.get("confidence", 0)

        explanations.append(
            f"Detected evidence in {domain_name}/{subdomain_name} with confidence {confidence}."
        )
        chains.append(f"{subdomain_name} -> {domain_name} -> {main_domain_name}")

    return {
        "decision": decision["label"],
        "explanations": explanations,
        "evidence_chain": chains,
    }

import json
import re
import asyncio
import base64

from PIL import Image
import io
import os

from model import XceptionDetector

DEVICE = "cpu"  # 或 "cuda" if torch.cuda.is_available() else "cpu"
detector = XceptionDetector(model_path=None, device=DEVICE)

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def safe_path_name(name: str) -> str:
    # 移除路径遍历和系统敏感字符（如 / \ : * ? " < > |）
    # 保留 Unicode 字母、数字、中文、空格、下划线等
    unsafe_chars = r'[\\/:\*\?"<>\|\x00]'
    cleaned = re.sub(unsafe_chars, '_', name.strip())
    # 可选：限制长度、去除首尾下划线/空格
    return cleaned.strip(' _')[:255] or 'unnamed'


def extract_json(s: str):
    """从任意 LLM 输出中抽取 JSON"""
    if not s or not isinstance(s, str):
        raise ValueError("LLM 返回空内容")

    # 提取代码块 ```json ... ```
    m = re.search(r"```json(.*?)```", s, flags=re.S)
    if m:
        s = m.group(1).strip()

    # 提取第一个 { 到 最后一个 }
    start = s.find("{")
    end = s.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(f"LLM 输出不含 JSON：{s}")

    json_str = s[start:end + 1]

    try:
        return json.loads(json_str)
    except Exception as e:
        print("JSON 解析失败，原始片段：", json_str)
        raise e

def compress_image(path, max_size=512):
    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

async def process_one(path):
    loop = asyncio.get_event_loop()
    b64_str = await loop.run_in_executor(None, compress_image, path)
    return f"data:image/jpeg;base64,{b64_str}"

async def get_image_base64_list(image_paths: list) -> list:
    tasks = [asyncio.create_task(process_one(p)) for p in image_paths]
    return await asyncio.gather(*tasks)

def build_detection_response(image_paths: list, base64_list: list, domain_name: str):
    # 加载 PIL 图像
    pil_images = []
    for path in image_paths:
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        pil_images.append(img)

    # 批量检测
    detection_results = detector.predict(pil_images)

    # 合并结果
    results = []
    for i, path in enumerate(image_paths):
        results.append({
            "base64": base64_list[i],
            "path": os.path.basename(path),
            "is_fake": detection_results[i]["is_fake"],
            "label": detection_results[i]["label"],
            "confidence": detection_results[i]["confidence"],
            "fake_score": detection_results[i]["fake_score"]
        })

    return {
        "results": results,
        "domain_name": domain_name
    }

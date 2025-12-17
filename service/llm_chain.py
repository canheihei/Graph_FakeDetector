import json
import os
import threading

from langchain.chains import LLMChain
from openai import OpenAI

from config import ALI_API_KEY, ALI_BASE_URL
from service.common_utils import extract_json

# LLM 初始化（阿里百炼兼容模式）
client = OpenAI(
    api_key=ALI_API_KEY,
    base_url=ALI_BASE_URL
)

def call_qwen(prompt_dict):
    # 1.加载提示词模板
    with open("main_prompt.txt", "r", encoding="utf-8") as f:
        domain_prompt = f.read()
    client = OpenAI(api_key=ALI_API_KEY, base_url=ALI_BASE_URL)

    user_input = safe_truncate_json(prompt_dict, max_length=28000)  # 留 2k 给其他内容
    # user_input = json.dumps(prompt_dict, ensure_ascii=False)
    # limit_input = user_input[:30000]
    resp = client.chat.completions.create(
        model="qwen-vl-plus",
        messages=[
            {"role": "system", "content": domain_prompt},
            {"role": "user", "content": user_input}
        ]
    )

    raw = resp.choices[0].message.content
    print("RAW LLM OUTPUT:", repr(raw))
    try:
        return extract_json(raw)
    except:
        s = raw[raw.find("{"):raw.rfind("}") + 1]
        return json.loads(s)

def safe_payload(prompt, existing_schema, image_infos):
    # 截断 schema（最多 8000 字符）
    existing_s = json.dumps(existing_schema, ensure_ascii=False)
    if len(existing_s) > 8000:
        existing_schema = {"truncated": True}

    # 限制图片数量（最多 4 张）
    if len(image_infos) > 4:
        image_infos = image_infos[:4]

    return {
        "prompt": prompt[:2000],  # 防止超长
        "existing_schema": existing_schema,
        "images": image_infos
    }

'''语义匹配'''
def match_domain(prompt: str, specific_domains: list, sub_domains: list, threshold=0.75) -> str:
    # 先在 specific_domain 中匹配
    match_specific = semantic_match(prompt, specific_domains, threshold)
    if match_specific != prompt:  # 匹配成功
        return match_specific

    # 再在 sub_domain 中匹配
    match_sub = semantic_match(prompt, sub_domains, threshold)
    if match_sub != prompt:
        return match_sub

    # 都没匹配上
    return prompt


from sentence_transformers import SentenceTransformer, util
import torch

# 模型路径（建议使用 os.path 保证跨平台）
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "paraphrase-multilingual-MiniLM-L12-v2"
)

MODEL_PATH = r"E:\Data\study_code\py_code\Graph_FakeDetector\paraphrase-multilingual-MiniLM-L12-v2"

# 全局变量 + 线程锁
_model = None
_model_lock = threading.Lock()


def get_model() -> SentenceTransformer:
    """
    线程安全的单例模型加载器。
    首次调用时从本地路径加载模型，之后返回缓存实例。
    """
    global _model
    if _model is None:
        with _model_lock:  # 防止多线程同时加载
            if _model is None:  # double-check
                if not os.path.exists(MODEL_PATH):
                    raise FileNotFoundError(
                        f"本地 SentenceTransformer 模型未找到: {MODEL_PATH}\n"
                        "请从 https://hf-mirror.com/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 下载模型文件"
                    )
                _model = SentenceTransformer(MODEL_PATH, device="cpu")
    return _model


def semantic_match(prompt: str, candidates: list, threshold: float = 0.65) -> str:
    """
    对 prompt 在 candidates 中做语义匹配，返回最相似项（若相似度 > threshold），否则返回 prompt 本身。
    """
    if not candidates:
        return prompt

    model = get_model()

    # 编码
    prompt_emb = model.encode(prompt, convert_to_tensor=True)
    candidates_emb = model.encode(candidates, convert_to_tensor=True)

    # 计算余弦相似度
    similarities = util.cos_sim(prompt_emb, candidates_emb)[0]

    # 找最大相似度及其索引
    max_sim_score, max_idx = torch.max(similarities, dim=0)
    max_sim_score = max_sim_score.item()

    if max_sim_score >= threshold:
        return candidates[max_idx.item()]
    else:
        return prompt

def safe_truncate_json(obj, max_length=30000):
    """
    安全地将 obj 转为 JSON 字符串，总长度不超过 max_length。
    优先保留结构完整性，按字段重要性裁剪。
    """
    # 初始尝试
    full_json = json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
    if len(full_json) <= max_length:
        return full_json

    # 1. 优先确保顶层结构完整：保留所有 key，但裁剪值
    truncated_obj = {}

    # 按重要性顺序处理字段（越靠前越不能丢）
    for key in ["prompt", "domain_name", "images", "existing_schema"]:
        if key not in obj:
            continue

        if key == "prompt":
            # 裁剪 prompt
            truncated_obj[key] = str(obj[key])[:1000]

        elif key == "domain_name":
            # domain_name 不能裁剪！必须完整
            truncated_obj[key] = obj[key]

        elif key == "images":
            # images 是列表，只保留前 N 张，且移除 base64（关键！）
            safe_images = []
            for img in obj[key][:3]:  # 最多3张
                safe_img = {k: v for k, v in img.items() if k != "base64"}
                safe_images.append(safe_img)
            truncated_obj[key] = safe_images

        elif key == "existing_schema":
            # 只保留 domain + top-K features（按 fake_score 排序）
            schema = obj[key]
            safe_schema = {"domain": schema.get("domain", "")}
            features = schema.get("features", [])
            if isinstance(features, list):
                # 按 fake_score 降序
                sorted_feats = sorted(
                    [f for f in features if isinstance(f, dict)],
                    key=lambda x: float(x.get("fake_score", 0)),
                    reverse=True
                )
                safe_schema["features"] = sorted_feats[:8]  # 最多8个特征
            truncated_obj[key] = safe_schema

    # 再次序列化
    result = json.dumps(truncated_obj, ensure_ascii=False, separators=(',', ':'))
    if len(result) > max_length:
        # 极端情况：再裁剪 prompt
        over = len(result) - max_length
        truncated_obj["prompt"] = truncated_obj["prompt"][:-over - 10]  # 多留点余量
        result = json.dumps(truncated_obj, ensure_ascii=False, separators=(',', ':'))

    return result


def reasoning(evidence: list, decision: dict) -> dict:
    explanations = []
    chains = []

    for e in evidence:
        explanations.append(
            f"检测到 {e['domain']} 中的「{e['subdomain']}」，置信度为 {e['score']}。"
        )
        chains.append(
            f"{e['subdomain']} → {e['domain']} → {e['main_domain']}"
        )

    return {
        "decision": decision["label"],
        "explanations": explanations,
        "evidence_chain": chains
    }

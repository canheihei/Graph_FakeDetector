import json
import os
import threading

from langchain.chains import LLMChain
from openai import OpenAI

from config import ALI_API_KEY, ALI_BASE_URL
from service.common_utils import extract_json

from sentence_transformers import SentenceTransformer, util
import torch


# LLM 初始化（阿里百炼兼容模式）
client = OpenAI(
    api_key=ALI_API_KEY,
    base_url=ALI_BASE_URL
)


def call_qwen(prompt_dict):
    """
    调用阿里云 Qwen 大模型（qwen-vl-plus）生成结构化响应。

    流程：
    1. 从本地文件加载系统级提示词模板；
    2. 对用户输入字典进行安全截断，防止超出上下文窗口；
    3. 发送请求并解析返回的 JSON 内容。
    """
    # 1. 加载提示词模板（通常包含角色设定、输出格式约束等）
    with open("main_prompt.txt", "r", encoding="utf-8") as f:
        domain_prompt = f.read()
    # NOTE: 硬编码文件路径存在可维护性问题；若部署环境变化，可能引发 FileNotFoundError
    # TODO: 建议将 prompt 文件路径参数化或通过配置管理（如 config.PROMPT_PATH）

    # 初始化 OpenAI 兼容客户端（阿里百炼平台支持 OpenAI API 协议）
    client = OpenAI(api_key=ALI_API_KEY, base_url=ALI_BASE_URL)
    # TODO: 建议将 client 作为全局单例或依赖注入，避免重复初始化开销

    # 对 prompt_dict 进行安全截断（保留 2k token 空间给系统提示和模型输出）
    user_input = safe_truncate_json(prompt_dict, max_length=28000)  # 留 2k 给其他内容
    # 被注释的原始实现（直接截取字符串）存在 JSON 截断破坏结构的风险，当前方案更安全
    # user_input = json.dumps(prompt_dict, ensure_ascii=False)
    # limit_input = user_input[:30000]

    # 调用 Qwen-VL 模型（支持多模态，但此处仅传文本）
    resp = client.chat.completions.create(
        model="qwen-vl-plus",
        messages=[
            {"role": "system", "content": domain_prompt},
            {"role": "user", "content": user_input}
        ]
    )

    # 获取原始模型输出（调试用）
    raw = resp.choices[0].message.content
    try:
        # 尝试解析 raw 字符串为 Python 对象
        parsed = json.loads(raw)
        # 以缩进 2 的方式格式化输出
        pretty_json = json.dumps(parsed, indent=2, ensure_ascii=False)
        print("RAW LLM OUTPUT (pretty):")
        print(pretty_json)
    except json.JSONDecodeError as e:
        print("RAW LLM OUTPUT (invalid JSON):")
        print(repr(raw))
        print(f"[Error] Failed to parse JSON: {e}")
    # TODO: 生产环境中应替换 print 为日志记录（如 logging.debug），避免敏感信息泄露

    try:
        # 尝试使用专用函数提取结构化 JSON（可能包含容错或正则清洗）
        return extract_json(raw)
    except:
        # 若失败，手动截取首尾花括号之间的内容并解析
        s = raw[raw.find("{"):raw.rfind("}") + 1]
        return json.loads(s)
    # OPTIMIZE: 宽泛的 except 捕获所有异常，不利于调试；建议捕获具体异常（如 JSONDecodeError）
    # TODO: 可增加重试机制或返回标准化错误结构，避免因模型输出异常导致服务中断


def safe_payload(prompt, existing_schema, image_infos):
    """
    构造安全的请求载荷，防止因输入过大导致模型调用失败。

    限制项：
    - prompt 长度 ≤ 2000 字符；
    - existing_schema 序列化后 ≤ 8000 字符，否则替换为占位对象；
    - image_infos 最多保留前 4 张图像信息。
    """
    # 截断 schema（防止复杂结构超出上下文）
    existing_s = json.dumps(existing_schema, ensure_ascii=False)
    if len(existing_s) > 8000:
        existing_schema = {"truncated": True}
    # NOTE: 此处直接替换整个 schema，可能影响下游逻辑；若需保留部分结构，可考虑分层截断

    # 限制图片数量（Qwen-VL 对图像数量有限制，通常 ≤ 4）
    if len(image_infos) > 4:
        image_infos = image_infos[:4]

    return {
        "prompt": prompt[:2000],  # 防止超长
        "existing_schema": existing_schema,
        "images": image_infos
    }
    # TODO: 字符截断（[:2000]）可能导致语义断裂；建议按 token 截断（使用 tiktoken 等工具）
    # OPTIMIZE: 可统一使用 safe_truncate_json 处理所有字段，提升一致性

def match_domain(prompt: str, specific_domains: list, sub_domains: list, threshold=0.75) -> str:
    """
    基于语义相似度匹配用户输入到预定义领域。

    匹配优先级：
    1. 先在 specific_domains（具体领域）中匹配；
    2. 若未命中，再在 sub_domains（子领域）中匹配；
    3. 若均未命中，返回原始 prompt（作为 fallback）。

    匹配逻辑依赖外部函数 semantic_match，该函数应在相似度 ≥ threshold 时返回匹配项，
    否则返回原始输入。
    """
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
    # NOTE: 依赖 semantic_match 的“未匹配则返回原串”约定，耦合较强
    # TODO: 建议 semantic_match 显式返回 (matched: bool, result: str) 元组，提升可读性与健壮性

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
                        "请从 https://hf-mirror.com/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2   下载模型文件"
                    )
                _model = SentenceTransformer(MODEL_PATH, device="cpu")
    return _model
    # NOTE: 使用 CPU 加载模型，适用于无 GPU 环境；若部署环境支持 CUDA，可动态选择 device
    # OPTIMIZE: 可考虑懒加载 + 模型缓存预热机制，避免首次请求延迟过高
    # TODO: 建议将 MODEL_PATH 通过配置注入，提升可测试性与部署灵活性


def semantic_match(prompt: str, candidates: list, threshold: float = 0.65) -> str:
    """
    对 prompt 在 candidates 中做语义匹配，返回最相似项（若相似度 > threshold），否则返回 prompt 本身。
    """
    if not candidates:
        return prompt

    model = get_model()

    # 对 prompt 和候选列表分别编码为 384 维向量（使用 paraphrase-multilingual-MiniLM-L12-v2）
    prompt_emb = model.encode(prompt, convert_to_tensor=True)
    candidates_emb = model.encode(candidates, convert_to_tensor=True)

    # 计算 prompt 与所有候选的余弦相似度（返回 [1, len(candidates)] 张量）
    similarities = util.cos_sim(prompt_emb, candidates_emb)[0]

    # 获取最高相似度值及其对应索引
    max_sim_score, max_idx = torch.max(similarities, dim=0)
    max_sim_score = max_sim_score.item()

    # 若最高相似度超过阈值，返回匹配项；否则返回原始 prompt（作为 fallback）
    if max_sim_score >= threshold:
        return candidates[max_idx.item()]
    else:
        return prompt
    # NOTE: 该函数依赖 sentence-transformers 的 cosine_similarity 工具函数（util.cos_sim）
    # TODO: 可增加日志记录最高分与匹配项，便于调试语义匹配效果
    # OPTIMIZE: 若 candidates 固定且频繁调用，可预编码并缓存 candidates_emb，避免重复计算


def safe_truncate_json(obj, max_length=30000):
    """
    安全地将 obj 转为 JSON 字符串，总长度不超过 max_length。
    优先保留结构完整性，按字段重要性裁剪。
    """
    # 初始尝试完整序列化
    full_json = json.dumps(obj, ensure_ascii=False, separators=(',', ':'))  # 紧凑格式，节省空间
    if len(full_json) <= max_length:
        return full_json

    # 1. 优先确保顶层结构完整：保留所有 key，但裁剪值
    truncated_obj = {}

    # 按重要性顺序处理字段（越靠前越不能丢）
    for key in ["prompt", "domain_name", "images", "existing_schema"]:
        if key not in obj:
            continue

        if key == "prompt":
            # 裁剪 prompt 至 1000 字符（保留核心指令）
            truncated_obj[key] = str(obj[key])[:1000]

        elif key == "domain_name":
            # domain_name 必须完整保留（用于路由或上下文识别）
            truncated_obj[key] = obj[key]

        elif key == "images":
            # images 是列表，最多保留 3 张，且**移除 base64 字段**（极大节省长度）
            safe_images = []
            for img in obj[key][:3]:  # 最多3张
                safe_img = {k: v for k, v in img.items() if k != "base64"}  # 剔除 base64
                safe_images.append(safe_img)
            truncated_obj[key] = safe_images

        elif key == "existing_schema":
            # 只保留 domain + top-8 特征（按 fake_score 降序）
            schema = obj[key]
            safe_schema = {"domain": schema.get("domain", "")}
            features = schema.get("features", [])
            if isinstance(features, list):
                # 过滤非法项，并按 fake_score 排序（高风险特征优先）
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
        # 极端情况：进一步裁剪 prompt（按超长字节数反向截断）
        over = len(result) - max_length
        truncated_obj["prompt"] = truncated_obj["prompt"][:-over - 10]  # 多留 10 字符余量
        result = json.dumps(truncated_obj, ensure_ascii=False, separators=(',', ':'))

    return result
    # NOTE: 该函数显式剔除 base64，说明设计者已意识到其体积问题，符合上下文工程最佳实践
    # TODO: 可增加对嵌套结构的递归裁剪支持，提升通用性
    # OPTIMIZE: 若 max_length 接近 token 限制（如 32768），建议改用 token 级截断（如 tiktoken）


def reasoning(evidence: list, decision: dict) -> dict:
    """
    基于检测证据生成可解释的推理结果。

    输入：
      - evidence: 检测到的多级领域证据列表，每项含 domain/subdomain/score 等字段；
      - decision: 最终判定结果（如 {"label": "fake"}）。

    输出：包含决策、自然语言解释、证据链的结构化字典。
    """
    explanations = []
    chains = []

    # 为每条证据生成人类可读的解释语句
    for e in evidence:
        explanations.append(
            f"检测到 {e['domain']} 中的「{e['subdomain']}」，置信度为 {e['score']}。"
        )
        # 构造从细粒度到粗粒度的推理链（subdomain → domain → main_domain）
        chains.append(
            f"{e['subdomain']} → {e['domain']} → {e['main_domain']}"
        )

    return {
        "decision": decision["label"],  # 最终判定标签（如 "real" / "fake"）
        "explanations": explanations,  # 自然语言解释列表
        "evidence_chain": chains  # 可视化推理路径
    }
    # NOTE: 假设 evidence 中每个元素包含 'main_domain' 字段；若缺失，将引发 KeyError
    # TODO: 建议增加字段存在性校验（如 e.get('main_domain', 'unknown')）
    # OPTIMIZE: 可支持多语言解释生成（结合 domain 语言属性）
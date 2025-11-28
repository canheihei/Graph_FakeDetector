import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

from config import ALI_API_KEY, ALI_BASE_URL

# LLM 初始化（阿里百炼兼容模式）
client = OpenAI(
    api_key=ALI_API_KEY,
    base_url=ALI_BASE_URL
)

def build_llm_chain():
    with open("prompts/domain_evolution.md", "r", encoding="utf-8") as f:
        prompt_text = f.read()

    prompt = PromptTemplate(
        input_variables=["domain_name", "existing_schema", "images"],
        template=prompt_text,
    )

    return LLMChain(
        llm=client,
        prompt=prompt
    )


def call_qwen_single(domain: str, image_path: str):
    """
    对单张图片调用 Qwen，避免超长输入。
    """
    from openai import OpenAI
    client = OpenAI(api_key=ALI_API_KEY, base_url=ALI_BASE_URL)

    resp = client.chat.completions.create(
        model="qwen-max",
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text",
                     "text": "You are an expert in image-domain feature extraction and knowledge graph evolution."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Domain: {domain}"},
                    {"type": "input_image", "image_url": f"file://{image_path}"}
                ]
            }
        ]
    )

    # Qwen 输出结构安全解析
    return resp.choices[0].message.content[0].text


def extract_structured_info(domain: str, image_path: str):
    """
    调用单图 LLM ➜ 解析成结构化信息
    """
    raw = call_qwen_single(domain, image_path)

    try:
        return json.loads(raw)
    except:
        s = raw[raw.find("{") : raw.rfind("}")+1]
        return json.loads(s)


def merge_schemas(existing_schema, image_infos):
    new_schema = existing_schema.copy()

    for item in image_infos:
        if "error" in item:
            continue
        info = item["result"]

        # 假设每个 info 里有：nodes, relations
        for n in info.get("nodes", []):
            if n not in new_schema["nodes"]:
                new_schema["nodes"].append(n)

        for r in info.get("relations", []):
            if r not in new_schema["relations"]:
                new_schema["relations"].append(r)

    return new_schema


def call_qwen_schema_generator(domain, merged_schema):
    from openai import OpenAI
    client = OpenAI(api_key=ALI_API_KEY, base_url=ALI_BASE_URL)

    payload = {
        "domain": domain,
        "merged_schema": merged_schema
    }

    resp = client.chat.completions.create(
        model="qwen-max",
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Generate knowledge graph schema and Cypher based on merged schema."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(payload, ensure_ascii=False)}
                ]
            }
        ]
    )

    text = resp.choices[0].message["content"][0]["text"]
    try:
        return json.loads(text)
    except:
        s = text[text.find("{") : text.rfind("}")+1]
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

def call_qwen(prompt_dict):
    from openai import OpenAI
    client = OpenAI(api_key=ALI_API_KEY, base_url=ALI_BASE_URL)

    user_input = json.dumps(prompt_dict, ensure_ascii=False)

    resp = client.chat.completions.create(
        model="qwen-max",
        messages=[
            {"role": "system",
             "content": [{"type": "text",
                          "text": "You are an expert in image-domain feature extraction and knowledge graph evolution."}]},
            {"role": "user", "content": [{"type": "text", "text": user_input[:30000]}]},
        ]
    )

    raw = resp.choices[0].message.content

    try:
        return json.loads(raw)
    except:
        s = raw[raw.find("{"):raw.rfind("}") + 1]
        return json.loads(s)
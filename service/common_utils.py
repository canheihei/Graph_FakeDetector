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
    """
    将指定路径的图像文件读取为二进制数据，并编码为 Base64 字符串。

    功能说明：
    - 以二进制只读模式（"rb"）打开文件；
    - 读取全部内容后使用 base64 编码；
    - 将编码结果从 bytes 转为 UTF-8 字符串返回，便于在 JSON 或 HTML 中使用。

    适用场景：将本地图片嵌入到 Data URL（如 <img src="data:image/...">）中。
    """
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
    # NOTE: 若文件过大，可能引发内存压力；建议在高并发或大文件场景下使用流式处理。
    # OPTIMIZE: 可增加文件存在性检查或 MIME 类型校验，避免读取非图片文件。


def safe_path_name(name: str) -> str:
    """
    对用户提供的文件名进行安全清洗，防止路径遍历攻击或非法文件系统操作。

    清洗规则：
    1. 移除路径遍历字符（如 / \）和 Windows 禁用字符（: * ? " < > |）及空字符 \x00；
    2. 替换上述不安全字符为下划线 '_'；
    3. 去除首尾空白和多余下划线；
    4. 限制长度不超过 255 字符（大多数文件系统的文件名长度上限）；
    5. 若结果为空，则返回默认名 'unnamed'。

    安全目标：防止如 "../etc/passwd" 或 "malicious|.exe" 等危险输入。
    """
    # 定义不安全字符的正则表达式（含转义，符合 re 语法）
    unsafe_chars = r'[\\/:\*\?"<>\|\x00]'
    # 使用下划线替换所有不安全字符，并去除首尾空白
    cleaned = re.sub(unsafe_chars, '_', name.strip())
    # 去除清洗后字符串首尾的下划线和空格，并限制长度为 255
    result = cleaned.strip(' _')[:255]
    # 若结果为空字符串，则使用默认安全名称
    return result or 'unnamed'
    # NOTE: 此函数仅处理单个文件名，不处理完整路径；调用方应确保 path 不包含目录分隔符。
    # TODO: 可扩展支持多语言归一化（如 NFC/NFD）或保留合法标点符号（如连字符 -）。

def extract_json(s: str):
    """
    从大语言模型（LLM）的任意文本输出中提取并解析出第一个合法的 JSON 对象。

    功能分步说明：
    1. 输入校验：确保输入非空且为字符串类型；
    2. 优先尝试提取 Markdown 风格的 JSON 代码块（```json ... ```）；
    3. 若存在代码块，则使用其内容；否则在全文中查找；
    4. 定位第一个 '{' 和最后一个 '}'，截取中间部分作为潜在 JSON 字符串；
    5. 使用 json.loads() 解析该字符串，并返回 Python 字典/列表；
    6. 若解析失败，打印原始片段并抛出异常，便于调试。

    适用场景：处理 LLM 输出的非结构化文本，从中可靠提取结构化数据。
    """
    # 输入校验：防止空输入或非字符串类型导致后续逻辑异常
    if not s or not isinstance(s, str):
        raise ValueError("LLM 返回空内容")

    # 尝试匹配 ```json ... ``` 代码块（跨行匹配，使用 re.S 标志）
    m = re.search(r"```json(.*?)```", s, flags=re.S)
    if m:
        # 若匹配成功，使用代码块内部内容（去除首尾空白）
        s = m.group(1).strip()

    # 在（可能已被截取的）字符串中查找 JSON 起止位置
    start = s.find("{")
    end = s.rfind("}")

    # 若未找到合法的 JSON 结构边界，抛出明确错误
    if start == -1 or end == -1:
        raise ValueError(f"LLM 输出不含 JSON：{s}")

    # 截取从第一个 '{' 到最后一个 '}' 的子串（包含边界）
    json_str = s[start:end + 1]

    # 尝试解析 JSON
    try:
        return json.loads(json_str)
    except Exception as e:
        # 调试辅助：打印导致解析失败的原始 JSON 片段
        print("JSON 解析失败，原始片段：", json_str)
        raise e
    # NOTE: 此方法假设 LLM 输出中最多包含一个主 JSON 对象；
    #       若存在嵌套多个独立 JSON（如 [{}, {}] 以外的情况），可能截取不完整。
    # OPTIMIZE: 可改进为使用更健壮的 JSON 提取器（如基于 ast 或递归括号匹配），
    #           以应对包含字符串内 '{' 或 '}' 的复杂情况。
    # TODO: 支持提取数组形式的 JSON（如以 '[' 开头）以增强通用性。


def compress_image(path, max_size=512):
    """
    将指定路径的图像压缩并转换为 Base64 编码的 JPEG 字符串。

    功能说明：
    1. 读取本地图像文件；
    2. 转换为 RGB 模式（确保兼容 JPEG 格式）；
    3. 按比例缩放，使最大边不超过 max_size 像素；
    4. 以 JPEG 格式（质量 80）写入内存缓冲区；
    5. 返回 UTF-8 解码后的 Base64 字符串，便于嵌入 JSON 或 HTML。
    """
    # 使用 PIL 打开图像文件（支持多种格式如 PNG、JPEG、GIF 等）
    img = Image.open(path)

    # 若图像模式非 RGB（如 RGBA、P、L 等），转换为 RGB，避免保存 JPEG 时出错
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # 按比例缩放图像，保持宽高比，最大边不超过 max_size
    img.thumbnail((max_size, max_size))

    # 创建内存字节流，避免写入临时文件
    buf = io.BytesIO()

    # 将图像以 JPEG 格式、80% 质量保存到内存缓冲区
    img.save(buf, format="JPEG", quality=80)

    # 从缓冲区读取字节数据，编码为 Base64 并转为 UTF-8 字符串
    return base64.b64encode(buf.getvalue()).decode("utf-8")
    # TODO: 添加异常处理（如 FileNotFoundError、OSError 等），防止因无效路径或损坏图像导致崩溃
    # OPTIMIZE: 若需更高性能，可考虑使用更高效的图像处理库（如 pillow-simd 或 cv2）


async def process_one(path):
    """
    异步封装 compress_image，使其可在 asyncio 事件循环中安全调用。

    说明：由于 PIL 图像处理是 CPU 密集型操作，需通过 run_in_executor 切换到线程池执行，
    避免阻塞主事件循环。
    """
    # 获取当前 asyncio 事件循环
    loop = asyncio.get_event_loop()

    # 在默认线程池中执行 compress_image，避免阻塞异步流程
    b64_str = await loop.run_in_executor(None, compress_image, path)

    # 拼接为标准 data URL 格式，前端可直接用 <img src="..."> 渲染
    return f"data:image/jpeg;base64,{b64_str}"
    # NOTE: 此处返回的 MIME 类型固定为 image/jpeg，若原始图像是透明 PNG，信息会丢失
    # TODO: 可根据原始格式动态选择输出格式（但需权衡压缩率与兼容性）


async def get_image_base64_list(image_paths: list) -> list:
    """
    并发处理多个图像路径，返回对应的 Base64 data URL 列表。

    采用 asyncio.create_task + asyncio.gather 实现高并发图像压缩。
    """
    # 为每个图像路径创建一个异步任务（非阻塞调度）
    tasks = [asyncio.create_task(process_one(p)) for p in image_paths]

    # 并发等待所有任务完成，并按输入顺序返回结果
    return await asyncio.gather(*tasks)
    # OPTIMIZE: 若图像数量极大（如 >100），建议引入 asyncio.Semaphore 限制并发线程数，
    #           防止线程池耗尽或内存溢出
    # TODO: 可增强错误处理：捕获单个任务异常而不中断整个列表（例如返回 None 或 error 占位）


def build_detection_response(image_paths: list, base64_list: list, domain_name: str):
    """
    构建包含图像检测结果的结构化响应。

    输入：
      - image_paths: 原始图像文件路径列表
      - base64_list: 对应的 Base64 data URL 列表（由 get_image_base64_list 生成）
      - domain_name: 用于标识请求来源的域名（如用于多租户或日志追踪）

    输出：包含检测结果和元信息的字典。
    """
    # 重新加载原始图像用于检测（注意：此处未使用压缩后的图像）
    pil_images = []
    for path in image_paths:
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        pil_images.append(img)
    # NOTE: 此处存在 I/O 冗余：compress_image 已打开过相同文件，但未复用 PIL 对象
    # TODO: 建议重构为统一图像加载层，避免重复读取磁盘文件，提升性能并减少资源竞争

    # 调用检测模型进行批量推理（假设 detector 是全局或已初始化的模型实例）
    detection_results = detector.predict(pil_images)
    # OPTIMIZE: 若 detector 支持异步或 GPU 批处理，可进一步提升吞吐；若为 CPU 模型，
    #           考虑将其也放入 run_in_executor 避免阻塞主流程

    # 将检测结果与 Base64、文件名等信息对齐组装
    results = []
    for i, path in enumerate(image_paths):
        results.append({
            "base64": base64_list[i],  # 前端可直接渲染的图像数据
            "path": os.path.basename(path),  # 仅返回文件名，避免泄露服务器路径信息
            "is_fake": detection_results[i]["is_fake"],
            "label": detection_results[i]["label"],
            "confidence": detection_results[i]["confidence"],
            "fake_score": detection_results[i]["fake_score"]
        })
    # TODO: 添加对 detection_results 长度与 image_paths 一致性的校验，
    #       防止因模型返回异常导致 IndexError

    # 返回最终结构化响应
    return {
        "results": results,
        "domain_name": domain_name
    }
    # NOTE: 此函数为同步阻塞函数，若 pil_images 加载或 detector.predict 耗时较长，
    #       可能影响整体异步性能，建议评估是否需异步化或分阶段处理

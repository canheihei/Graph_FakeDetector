import asyncio
import os

from flask import Flask, request, jsonify, render_template
from PIL import Image
import io

from werkzeug.utils import secure_filename

from model import XceptionDetector
from service.llm_chain import safe_payload, call_qwen
from service.neo_client import get_existing_schema, apply_cyphers
from service.parallel import process_images_parallel

app = Flask(__name__)

# 初始化模型（建议单例）
DEVICE = "cpu"  # 或 "cuda" if torch.cuda.is_available() else "cpu"
detector = XceptionDetector(model_path=None, device=DEVICE)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "请上传图片文件！"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({"error": "仅支持图片格式（PNG/JPG/JPEG）"}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"图片解析失败: {str(e)}"}), 400

    result = detector.predict(image)
    return jsonify(result)

@app.route("/iterate", methods=["POST"])
def iterate():
    prompt = request.form.get("prompt")
    image_files = request.files.getlist("images")

    if not prompt or not image_files:
        return jsonify({"error": "Missing prompt or images"}), 400

    saved_paths = []
    for f in image_files:
        filename = secure_filename(f.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path)
        saved_paths.append(path)

    # 正确 async 执行
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    image_infos = loop.run_until_complete(process_images_parallel(prompt, saved_paths))

    # image_infos 一定是 list，否则报错
    if not isinstance(image_infos, list):
        return jsonify({"error": "Image processing failed"}), 500

    existing = get_existing_schema(prompt)

    # 输入截断，避免 Qwen 400 超长
    payload = safe_payload(prompt, existing, image_infos)

    result = call_qwen(payload)

    cy_list = result.get("cypher", [])
    apply_cyphers(cy_list)

    return jsonify({
        "message": f"处理完成，共 {len(image_infos)} 张图",
        "features": result.get("features", []),
        "cypher_executed": cy_list
    })
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/graph-iteration.html")
def graph_iteration():
    return render_template("graph-iteration.html")

@app.route("/image-recognition.html")
def image_recognition():
    return render_template("image-recognition.html")

@app.route("/visualization.html")
def visualization():
    return render_template("visualization.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
import asyncio
import os

from flask import Flask, request, jsonify, render_template

from werkzeug.utils import secure_filename

from service.common_utils import safe_path_name, get_image_base64_list, \
    build_detection_response
from service.detect_client import extract_features, map_features, build_evidence, decide
from service.llm_chain import safe_payload, call_qwen, match_domain, reasoning
from service.neo_client import get_existing_schema, apply_cyphers, get_specificdomain, get_subdomain, \
    create_features_and_relations, process_result, neo4j_client

app = Flask(__name__)

# 初始化模型（建议单例）
UPLOAD_FOLDER = "uploads/iterate"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/iterate", methods=["POST"])
def iterate():
    # 1. 接收图片和域prompt
    prompt = request.form.get("prompt")
    image_files = request.files.getlist("images")

    if not prompt or not image_files:
        return jsonify({"error": "Missing prompt or images"}), 400

    safe_prompt_dir = safe_path_name(prompt)
    if not safe_prompt_dir:
        return jsonify({"error": "Invalid prompt for directory name"}), 400

    # 1.1 根据域prompt创建文件夹并且存储上传的图片
    prompt_folder = os.path.join(UPLOAD_FOLDER, safe_prompt_dir)
    os.makedirs(prompt_folder, exist_ok=True)

    saved_paths = []
    for f in image_files:
        filename = secure_filename(f.filename)
        if not filename:
            continue  # 跳过无效文件名
        path = os.path.join(prompt_folder, filename)
        f.save(path)
        saved_paths.append(path)

    # 2. 查询出对应的specific_domain和sub_domain
    specific_domain_node = get_specificdomain()
    sub_domain_node = get_subdomain()

    domain_list = specific_domain_node['data']
    sub_domain_list = sub_domain_node['data']

    specific_domain_name = [item['name'] for item in domain_list]
    sub_domain_name = [item['name'] for item in sub_domain_list]

    # 3.轻量化语义匹配
    matched_domain = match_domain(prompt, specific_domain_name, sub_domain_name)

    # 4.异步图片编码及组织匹配的域、图片编码以及主流模型的参考置信等信息
    try:
        # 异步执行：Base64 编码 + 批量推理
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        base64_list = loop.run_until_complete(get_image_base64_list(saved_paths))
        # 同步检测（detector.predict 是 CPU-bound，无需 async）
        image_infos = build_detection_response(saved_paths, base64_list, matched_domain)

    except Exception as e:
        return jsonify({"error": f"Image analysis failed: {str(e)}"}), 500

    if not isinstance(image_infos, dict) or len(image_infos) == 0:
        return jsonify({"error": "No valid images processed"}), 500

    # 4.1 返回对应的编码dict
    if not isinstance(image_infos, dict):
        return jsonify({"error": "Image processing failed"}), 500

    # 5.调用大模型
    result = call_qwen(image_infos)

    # 6.执行进化
    result_cyper = process_result(result)

    return jsonify({
        "message": f"处理完成，共 {len(image_infos)} 张图",
        "features": result.get("features", []),
        "cypher_executed": result_cyper
    })

@app.route('/iterate_directly', methods=['POST'])
def ingest_feature_domain():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400

        # 验证必要字段
        required_keys = {'specific_domain', 'describe', 'specific_id', 'subdomain'}
        if not required_keys.issubset(data.keys()):
            return jsonify({"error": f"Missing required fields: {required_keys - data.keys()}"}), 400

        if not isinstance(data['subdomain'], list):
            return jsonify({"error": "subdomain must be a list"}), 400

        for sub in data['subdomain']:
            if not all(k in sub for k in ['name', 'describe', 'sub_id']):
                return jsonify({"error": "Each subdomain item must contain 'name', 'describe', 'sub_id'"}), 400

        # 写入 Neo4j
        process_result(data)

        return jsonify({"status": "success", "message": "Feature domain and subdomains ingested into Neo4j"}), 200

    except Exception as e:
        app.logger.error(f"Error in /ingest-feature-domain: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/aigc/detect", methods=["POST"])
def detect():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image_bytes = request.files["image"].read()

        features = extract_features(image_bytes)
        activated = map_features(features)

        # 防御：没有激活子域
        if not activated:
            return jsonify({
                "label": "REAL",
                "confidence": 0.1,
                "evidence": [],
                "reasoning": {
                    "explanations": ["未检测到显著异常特征"],
                    "evidence_chain": []
                }
            })

        evidence = build_evidence(activated)
        decision = decide(evidence)
        llm_result = reasoning(evidence, decision)

        return jsonify({
            "label": decision["label"],
            "confidence": decision["confidence"],
            "evidence": evidence,
            "reasoning": llm_result
        })

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/graph/stats", methods=["GET"])
def graph_stats():
    try:
        stats = {}

        # 1. 节点类型统计
        stats["node_counts"] = neo4j_client.query("""
        MATCH (n)
        RETURN labels(n)[0] AS label, count(n) AS count
        """)

        # 2. 关系类型统计
        stats["relation_counts"] = neo4j_client.query("""
        MATCH ()-[r]->()
        RETURN type(r) AS type, count(r) AS count
        """)

        # 3. 各 SpecificDomain 下 SubDomain 数量
        stats["domain_structure"] = neo4j_client.query("""
        MATCH (s:SubDomain)-[:SPECIFIC_OF]->(d:SpecificDomain)
        RETURN d.name AS domain, count(s) AS sub_count
        """)

        # 4. 子域 Top（可作为“最常见异常类型”）
        stats["subdomain_list"] = neo4j_client.query("""
        MATCH (s:SubDomain)
        RETURN s.name AS name, s.id AS id
        ORDER BY s.id
        """)

        # 5. 图谱整体规模
        stats["graph_overview"] = neo4j_client.query("""
        MATCH (n)
        WITH count(n) AS nodes
        MATCH ()-[r]->()
        RETURN nodes, count(r) AS relations
        """)[0]

        return jsonify(stats)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/test", methods=["POST"])
def test():
    specificdomain = get_specificdomain()
    subdomain = get_subdomain()
    return jsonify({"specificdomain": specificdomain, "subdomain": subdomain})


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/index.html")
def index():
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

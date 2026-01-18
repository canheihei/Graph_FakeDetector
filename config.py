import multiprocessing
import os

# 上传图片临时目录
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 阿里百炼 Qwen（OpenAI 兼容模式）
ALI_API_KEY = "sk-781534b5e3134ebda46dea3752ee552a"
ALI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# MAX_WORKERS = multiprocessing.cpu_count()
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 2)

# Neo4j 配置
NEO4J_URI = "bolt://192.168.125.128:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "assembly666"

FEATURE_THRESHOLD = 0.4

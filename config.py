import multiprocessing
import os
from pathlib import Path

from project_paths import resolve_datasets_root

# Storage
PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = resolve_datasets_root(PROJECT_ROOT)
UPLOAD_FOLDER = str(DATASETS_ROOT / "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# LLM endpoint
ALI_API_KEY = os.getenv("ALI_API_KEY", "")
ALI_BASE_URL = os.getenv("ALI_BASE_URL", "https://api.moonshot.cn/v1")
# MAX_WORKERS = multiprocessing.cpu_count()
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 2)

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "")

FEATURE_THRESHOLD = float(os.getenv("FEATURE_THRESHOLD", "0.4"))

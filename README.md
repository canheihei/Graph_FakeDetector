# Graph_FakeDetector

面向 Deepfake 图像检测与域泛化增强研究的工程系统。以视觉主检测器（EfficientNet-B0）为核心，结合三层语义知识图谱、候选层审批工作流、域级阈值校准与审计链输出，提供可追溯、可审计、可演化的检测结果。

## 目录

- [快速开始](#快速开始)
- [环境要求](#环境要求)
- [安装部署](#安装部署)
- [核心架构](#核心架构)
- [API 接口文档](#api-接口文档)
- [训练与评测](#训练与评测)
- [当前准确率](#当前准确率)
- [常见问题](#常见问题)

---

## 快速开始

```bash
# 1. 克隆仓库
git clone <repo-url>
cd Graph_FakeDetector

# 2. 创建虚拟环境
conda create -n detector python=3.10 -y
conda activate detector
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
nano .env   # 填入 ALI_API_KEY、NEO4J_PASS 等

# 4. 创建本地配置
cp config.example.py config.py

# 5. 安装并启动 Neo4j（见下方章节）

# 6. 启动服务
python app.py
# 访问 http://localhost:5000

# 7. 初始化图谱基线
curl -X POST http://localhost:5000/system/reset_baseline \
  -H "Content-Type: application/json" \
  -d '{"confirm": "RESET_GRAPH_AND_MAPPING"}'
```

---

## 环境要求

| 组件 | 最低版本 | 用途 |
|------|---------|------|
| Python | 3.9+ | 主运行环境 |
| Neo4j | 5.x Community Edition | 知识图谱存储 |
| Node.js | 16+ | 前端 Tailwind CSS 编译（可选，编译产物已在仓库中） |
| CUDA | 11.8+ (可选) | GPU 推理加速，CPU 也可运行 |

---

## 安装部署

### 1. 克隆仓库

```bash
git clone <repo-url>
cd Graph_FakeDetector
```

### 2. Python 环境

```bash
conda create -n detector python=3.10 -y
conda activate detector
pip install -r requirements.txt
```

**核心依赖**：

| 包 | 用途 |
|----|------|
| `Flask` | Web 服务框架 |
| `torch` / `torchvision` | 深度学习推理（EfficientNet-B0） |
| `neo4j` | 图谱数据库 Python 驱动 |
| `openai` | LLM API 调用（兼容 OpenAI 接口规范） |
| `langchain` | LLM 链式调用编排 |
| `sentence_transformers` | 语义文本嵌入（用于图谱去重） |
| `opencv-python-headless` | 图像预处理 |
| `Pillow` | 图像文件读写 |
| `python-dotenv` | 从 `.env` 文件加载环境变量 |

### 3. Neo4j 图谱数据库

**安装**：

- 下载 [Neo4j Community Edition](https://neo4j.com/download-center/#community)
- 或使用 [Neo4j Desktop](https://neo4j.com/download/)（带图形管理界面，推荐）

**创建数据库**：

1. 启动 Neo4j Desktop
2. 创建项目 → 添加本地 DBMS
3. 设置密码（记住用于配置）
4. 启动数据库
5. 确认 Bolt 端口为 `7687`（默认）

**配置连接信息**（编辑 `.env`）：

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASS=你设置的密码
```

### 4. LLM API 配置

系统使用 LLM 进行语义推理和候选映射生成（兼容 OpenAI 接口规范）。

编辑 `.env`：

```env
ALI_API_KEY=你的API密钥
ALI_BASE_URL=https://api.moonshot.cn/v1
FEATURE_THRESHOLD=0.4
```

> **重要**：如果只使用核心检测功能（`/detect`），不需要配置 LLM。但图谱迭代（`/iterate`）和候选生成（`/detect/candidates`）需要 LLM 支持。

### 5. 前端样式编译

编译产物 `frontend/static/tailwind.css` 已包含在仓库中，**大多数情况下无需操作**。

如需修改样式：

```bash
npm install
npm run build      # 一次性编译
# 或
npm run dev        # 开发模式，实时监听
```

### 6. 模型权重

权重文件放在 `weights/` 目录：

| 文件 | 说明 |
|------|------|
| `calibrated_vision_detector_dg_round5_dfdc_curated.pt` | **当前主权重**，多域联合训练 + DFDC 定向微调 |
| `calibrated_vision_detector.pt` | 基础版本权重（备用） |

**自定义权重路径**：

```bash
# Linux / macOS
export GRAPH_FAKEDETECTOR_WEIGHTS_ROOT=/path/to/your/weights

# Windows PowerShell
$env:GRAPH_FAKEDETECTOR_WEIGHTS_ROOT = "C:\path\to\your\weights"
```

### 7. 数据集获取

数据集**不包含在 Git 仓库中**。使用内置脚本从 Hugging Face 拉取：

```bash
# 跨域评测数据集（Celeb-DF、DFDC、WildDeepfake）
python scripts/benchmark/pull_hf_external_datasets.py \
  --dataset-root Datasets \
  --datasets Celeb-DF DFDC WildDeepfake \
  --per-class 300 \
  --seed 42

# 如需重新下载（清空已有数据）
python scripts/benchmark/pull_hf_external_datasets.py \
  --dataset-root Datasets \
  --datasets Celeb-DF DFDC WildDeepfake \
  --per-class 300 \
  --seed 42 \
  --clear-existing
```

**数据集目录结构**（拉取后自动创建）：

```
Datasets/
├── Train/          # 训练集（可选）
│   ├── Fake/
│   └── Real/
├── Test/           # 测试集（同分布评测）
│   ├── Fake/
│   └── Real/
├── Validation/     # 验证集（跨场景评测）
│   ├── Fake/
│   └── Real/
├── Celeb-DF/       # 外部域：名人换脸数据集
│   ├── Fake/
│   └── Real/
├── DFDC/           # 外部域：Deepfake Detection Challenge
│   ├── Fake/
│   └── Real/
└── WildDeepfake/   # 外部域：Wild Deepfake
    ├── Fake/
    └── Real/
```

### 8. 初始化图谱基线

首次部署需要在 Neo4j 中创建基线图谱结构：

```bash
# 一键初始化（图谱 + mapping 配置）
curl -X POST http://localhost:5000/system/reset_baseline \
  -H "Content-Type: application/json" \
  -d '{"confirm": "RESET_GRAPH_AND_MAPPING"}'
```

基线图谱结构：**1 个 MainDomain → 7 个 SpecificDomain → 22 个 SubDomain**，共 30 个节点和 29 条关系。

### 9. 启动服务

```bash
# 开发模式
python app.py

# 或使用 Flask CLI
flask run --host=0.0.0.0 --port=5000
```

默认访问地址：`http://localhost:5000`

### 10. 验证安装

```bash
# 1. 检查服务是否正常
curl http://localhost:5000/stats

# 2. 检查 Neo4j 图谱连通性
curl http://localhost:5000/neo4j_overview

# 3. 测试检测接口（替换为你的测试图片路径）
curl -X POST http://localhost:5000/detect \
  -F "image=@/path/to/test_face.jpg"

# 4. 查看 active mapping 配置
curl http://localhost:5000/mapping/config
```

---

## 核心架构

```
app.py                         # Flask 入口，装配所有路由
│
├── service/facades.py         # detect / iterate / evolve 核心工作流编排
├── service/decision_policy.py # 决策层自适应融合 + 域级阈值校准
├── service/graph_semantics.py # 图谱语义治理（去重 / 规范化 / 跨轮次复用）
├── service/llm_chain.py       # LLM 调用链封装
├── service/neo_client.py      # Neo4j 图谱操作封装
├── service/candidate_*.py     # 候选层全套（生成 / 评测 / 晋级 / 存储）
├── service/report_gallery.py  # 报告文件管理
│
├── detectors/                 # 检测器层
│   ├── calibrated_vision_detector.py  # ★ 主检测器：EfficientNet-B0
│   ├── ensemble_meta_detector.py      # 元集成检测器
│   ├── fft_detector.py               # 频域伪影检测
│   ├── appearance_detector.py         # 外观异常检测
│   └── hub.py                        # 检测器调度中心
│
├── alignment/                 # (detector, feature) → SubDomain 对齐映射
│   ├── aligner.py             # 核心对齐逻辑
│   ├── evidence_builder.py    # 图谱证据构建
│   ├── evolver.py             # 未映射 feature 演化补齐
│   ├── mapping_config.json    # 正式映射规则
│   ├── mapping_candidates.json # 候选映射审批清单
│   └── mapping_config.baseline.json # 基线备份
│
├── frontend/                  # 前端资源
│   ├── templates/             # Flask Jinja2 模板
│   ├── static/                # 编译后的 CSS / JS
│   └── src/                   # Tailwind CSS 源文件
│
├── scripts/                   # 独立脚本
│   ├── training/              # 训练脚本
│   └── benchmark/             # 评测与报告生成
│
└── detector_config.py         # 检测器统一配置中心（权重/阈值/校准参数）
```

**图谱三层语义结构**：

```
MainDomain: "域泛化"                    ← 中心节点（1 个）
  └── SpecificDomain: "生成机制域"       ← 第二层（7 个）
  │     ├── SubDomain: "人脸专用生成器"
  │     ├── SubDomain: "GAN类生成器"
  │     └── SubDomain: "扩散类生成器"
  └── SpecificDomain: "内容异常域"
        ├── SubDomain: "五官不对称异常"
        └── SubDomain: "面部比例失调"
  ... (共 22 个 SubDomain)
```

每个图谱节点都包含 `display_name`、`canonical_name`、`semantic_source`、`semantic_version` 属性，确保语义可追溯。

**主检测链路**：

```
输入图片 → CalibratedVision (EfficientNet-B0)
         → 辅助检测器信号 (FFT / Appearance)
         → 决策层融合 (decision_policy.py)
         → 图谱证据查询 (evidence_builder.py)
         → 输出：label + confidence + evidence + reasoning + 审计字段
```

---

## API 接口文档

### 检测接口

#### `POST /detect` — 主检测接口

对单张图片进行 Deepfake 检测，返回标签、置信度、图谱证据和审计字段。

**请求**：`multipart/form-data`

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `image` | file | 是 | — | 待检测图片 |
| `auto_evolve` | bool | 否 | `true` | 是否自动演化未映射特征 |
| `semantic_threshold` | float | 否 | 系统默认 | 语义去重阈值 (0.0~1.0) |
| `use_llm_generation` | bool | 否 | `false` | 是否启用 LLM 生成新域 |
| `decision_profile` | string | 否 | — | 决策域配置（`celeb_df` / `dfdc` / `wilddeepfake`） |
| `decision_threshold_override` | float | 否 | — | 手动覆盖判定阈值 |

**响应**：`application/json`

| 字段 | 类型 | 说明 |
|------|------|------|
| `label` | string | `FAKE` 或 `REAL` |
| `confidence` | float | 置信度 (0.0~1.0) |
| `evidence` | array | 图谱证据列表 |
| `reasoning` | string | 推理说明 |
| `reasoning_type` | string | 推理类型 |
| `diagnostic_chain` | array | 诊断步骤链 |
| `risk_level` | string | 风险等级 |
| `needs_review` | bool | 是否需要人工复核 |

### 图谱迭代接口

#### `POST /iterate` — 语义图谱迭代

上传图片和语义描述，经检测 → LLM 推理 → 写入 Neo4j 图谱。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `prompt` | string | 是 | 语义描述文本 |
| `images` | file[] | 是 | 多张图片 |
| `semantic_threshold` | float | 否 | 语义去重阈值 |

### 候选管理层

| 接口 | 方法 | 说明 |
|------|------|------|
| `/detect/candidates` | POST | 对弱证据样本触发 LLM 候选生成 |
| `/candidate-mappings` | GET | 列出候选审批清单（支持按状态筛选） |
| `/candidate-mappings/update` | POST | 更新候选图谱字段、映射参数、审批状态 |
| `/candidate-mappings/benchmark` | POST | 对勾选候选运行临时 overlay benchmark |
| `/candidate-mappings/promote` | POST | 晋级通过门禁的候选到正式 mapping |
| `/candidate-mappings/delete` | POST | 删除候选审批记录与对应图谱节点 |

### 系统管理接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/stats` | GET | 图谱节点/关系统计 |
| `/neo4j_overview` | GET | Neo4j 图谱结构概览 |
| `/mapping/config` | GET | 当前 active mapping 只读视图 |
| `/graph/reset_baseline` | POST | 重置 Neo4j 图谱到基线 |
| `/mapping/reset_baseline` | POST | 重置 mapping 到基线 |
| `/system/reset_baseline` | POST | 一键重置图谱 + mapping |

---

## 训练与评测

### 训练主检测器

```bash
python scripts/training/train_calibrated_vision_detector.py \
  --dataset-root Datasets/Train \
  --benchmark-per-class 100 \
  --epochs 8 \
  --report-path reports/calibrated_vision_training.json
```

### 运行 Benchmark

```bash
# 同分布评测
python scripts/benchmark/visualize_detect_benchmark.py \
  --dataset-root Datasets/Test \
  --sample-per-class 100 \
  --output-dir reports/detect_benchmark_sample100

# 外部域评测（使用域级阈值 profile）
python scripts/benchmark/visualize_detect_benchmark.py \
  --dataset-root Datasets/Celeb-DF \
  --sample-per-class 80 \
  --workers 1 \
  --decision-profile celeb_df \
  --output-dir reports/report_celeb_df_sample80
```

### 实用验证脚本（毕设/答辩推荐）

```bash
python scripts/benchmark/visualize_practical_validation.py \
  --sample-per-class 80 \
  --robustness-sample-per-class 30 \
  --nobody-limit 20 \
  --output-dir reports/practical_validation_default
```

一次性生成：同分布测试、跨场景验证、退化鲁棒性测试（JPEG 压缩 / 高斯模糊 / 降采样恢复）、Out-of-scope 样本识别、汇总 HTML / CSV / JSON 报告。

---

## 当前准确率

*评测口径：2026-04-24，主权重 `calibrated_vision_detector_dg_round5_dfdc_curated.pt`*

| 数据集 | Valid 样本数 | Correct | Accuracy | Balanced Acc | 域阈值 Override |
|--------|------------|---------|----------|-------------|----------------|
| Celeb-DF | 1200 | 1157 | 98.97% | 98.98% | 0.42 |
| DFDC | 1200 | 1129 | 94.11% | 94.03% | 0.49 |
| WildDeepfake | 254 | 251 | 98.82% | 99.19% | 0.10 |
| Test (同分布) | 1200 | 1190 | 99.17% | 99.17% | — |

跨域评测覆盖：
- `reasoning_type` 覆盖：**100%**（四域/抽样口径）
- `diagnostic_chain` 覆盖：**100%**（四域/抽样口径）

---

## 常见问题

### Q: 启动时报 `ModuleNotFoundError: No module named 'config'`

**A:** 需要从模板创建本地配置文件：

```bash
cp config.example.py config.py
```

### Q: Neo4j 连接失败

**A:** 检查以下几点：
1. Neo4j 服务是否已启动（Neo4j Desktop → 确认状态为 "Started"）
2. Bolt 端口 `7687` 是否被防火墙阻止
3. `.env` 中的 `NEO4J_URI`、`NEO4J_USER`、`NEO4J_PASS` 是否正确

### Q: 权重文件缺失 / `FileNotFoundError: weights/...pt`

**A:** 确保权重文件已放置在 `weights/` 目录。如使用自定义路径，设置环境变量：

```bash
export GRAPH_FAKEDETECTOR_WEIGHTS_ROOT=/path/to/your/weights
```

### Q: LLM API 调用失败

**A:** 检查 `.env` 中的 `ALI_API_KEY` 和 `ALI_BASE_URL`。核心 `/detect` 接口不依赖 LLM，仍可正常使用。

### Q: 没有 GPU 能运行吗？

**A:** 可以。系统默认使用 CPU 推理（`detector_config.py` 中 `device='cpu'`）。EfficientNet-B0 在 CPU 上单张推理约 0.5~1 秒。

### Q: CUDA out of memory

**A:** 系统默认 `device='cpu'`，不会触发此问题。如需 GPU 推理，修改 `detector_config.py` 中的 `CALIBRATED_VISION_RUNTIME`，并确保 GPU 显存 ≥ 2GB。

### Q: Hugging Face 数据集下载失败

**A:** 
1. 检查网络连接（HuggingFace 在国内可能需要代理）
2. 尝试设置 HF 镜像：`export HF_ENDPOINT=https://hf-mirror.com`
3. 如仍失败，可手动从 HuggingFace 下载数据集并放入 `Datasets/` 目录

### Q: 如何完全重置系统？

**A:** 

```bash
# 重置图谱 + mapping 到基线
curl -X POST http://localhost:5000/system/reset_baseline \
  -H "Content-Type: application/json" \
  -d '{"confirm": "RESET_GRAPH_AND_MAPPING"}'
```

---

## 协作规范

- **接口兼容优先**：修改时保持现有 API 协议不变，只改内部实现
- **配置集中管理**：所有阈值、权重路径、校准参数统一在 `detector_config.py`
- **密钥不入库**：API Key、数据库密码等通过 `.env` 管理（`.gitignore` 已排除）
- **评测脚本统一**：benchmark 统一使用 `scripts/benchmark/visualize_detect_benchmark.py`
- **测试随功能**：新增功能需同步补充测试，放在 `tests/` 目录
- **不推大文件**：数据集、模型权重、benchmark 产物均不入库
- **提交信息**：推荐格式 `type(scope): description`

---

## 项目当前状态

**已完成**：

- 三层语义去重系统（图谱层 / 配置层 / 精确匹配层）
- detector 内部配置、权重路径、阈值、占位模式统一管理
- `CalibratedVision` 升级为 `EfficientNet-B0` 迁移学习方案
- `detect` 输出审计字段：`reasoning_type / diagnostic_chain / risk_level / needs_review`
- 决策层域级阈值校准：`decision_profile / decision_threshold_override`
- 图谱证据构建诊断：`id_matched / label_fallback_matched / unresolved_subdomains`
- `iterate` 跨轮次语义复用与去重
- 弱证据候选层审批工作流（生成 / 评测 / 晋级 / 删除）
- 图谱基线重置 + 一键系统恢复

**进行中**：

- 图谱 ontology 收敛与脏节点清洗
- 候选语义进一步收敛，减少 LLM 漂移
- benchmark overlay 与线上真实表现的一致性量化

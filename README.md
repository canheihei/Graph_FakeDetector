# Graph_FakeDetector

面向 Deepfake 图像检测与域泛化增强研究的工程系统。当前版本以视觉主检测器为核心，结合多检测器协同、三层语义图谱、候选层审批工作流、域级阈值校准与审计链输出，目标不是只给出真假标签，而是给出可追溯、可审计、可演化的检测结果。

## 项目概览

- 主检测器: `CalibratedVision`
- 当前主视觉主干: `EfficientNet-B0`
- 当前主权重: `weights/calibrated_vision_detector_dg_round5_dfdc_curated.pt`
- 当前主入口: `app.py`
- 当前主接口: `/detect`
- 图谱语义治理: `service/graph_semantics.py`
- 决策层策略: `service/decision_policy.py`
- active mapping 配置: `alignment/mapping_config.json`
- 候选审批清单: `alignment/mapping_candidates.json`

项目目标不只是做二分类，还包括:

- 对 Deepfake 图像给出稳定检测结果
- 补充图谱证据与推理说明
- 支持 feature -> domain 的持续演化
- 支持弱证据样本的候选生成、评测与晋级
- 支持图谱基线、mapping 基线与系统整体恢复
- 为 benchmark、回归验证和可视化分析提供统一产物

## 当前状态

已完成:

- 三层语义去重系统（图谱层 / 配置层 / 精确匹配层）
- detector 内部配置、权重路径、阈值、占位模式统一管理
- `CalibratedVision` 升级为 `EfficientNet-B0` 迁移学习方案
- `detect` 输出审计字段：`reasoning_type / diagnostic_chain / risk_level / needs_review`
- 决策层域级阈值校准：`decision_profile / decision_threshold_override`
- 图谱证据构建诊断：`id_matched / label_fallback_matched / unresolved_subdomains`
- `iterate` 跨轮次语义复用与去重
- 弱证据候选层一期工作流：
  - `/detect/candidates`
  - `mapping_candidates.json`
  - detect 单页审批
  - quick / formal benchmark
  - selective promote
- promote 已支持：
  - candidate graph merge 到 active graph
  - active mapping 替换式更新
  - promote 日志返回
- 可视化页已支持：
  - active mapping 的 detector / feature 分组视图
  - 图谱 / mapping / 系统一键重置入口
- 云端 detect / iterate / candidate / promote / visualization 链路已验证

进行中:

- 图谱证据层专业化补齐
- 历史图谱脏节点清洗与 ontology 收敛
- 候选语义进一步收敛，减少 LLM 漂移
- benchmark overlay 与线上真实表现的一致性继续量化

## 已验证结果

### Benchmark

- `2026-04-12` 外部域 `sample300` 工程口径
  - Celeb-DF: `Accuracy(valid)=99.17%`
  - DFDC: `Accuracy(valid)=94.83%`
  - WildDeepfake: `Accuracy(valid)=100.00%`
  - 对应报告:
    - `reports/report_celeb_df_sample300_profile_celeb_df_evidencehit_2026-04-12/`
    - `reports/report_dfdc_sample300_profile_dfdc_evidencehit_2026-04-12/`
    - `reports/report_wilddeepfake_sample300_profile_wilddeepfake_evidencehit_2026-04-12/`

- `2026-04-12` 证据链相关指标
  - `evidence_hit_rate`
  - `fake_evidence_hit_rate`
  - `high_score_no_evidence_rate`
  - `unresolved_subdomain_rate`
  - `avg_evidence_alignment_score`

- 项目历史同分布 full benchmark
  - 数据集: `Datasets/Test`
  - 样本数: `10905`
  - `Accuracy(valid)=98.8996%`
  - 报告目录: `reports/detect_benchmark_full_effb0_workers4/`

### 云端接口验证

- 已验证接口:
  - `GET /stats`
  - `GET /neo4j_overview`
  - `GET /mapping/config`
  - `POST /detect`
  - `POST /iterate`
  - `POST /detect/candidates`
  - `POST /candidate-mappings/benchmark`
  - `POST /candidate-mappings/promote`
  - `POST /graph/reset_baseline`
  - `POST /mapping/reset_baseline`
  - `POST /system/reset_baseline`

- 已验证行为:
  - 相同 prompt 连续执行 `/iterate` 时，当前可阻止近义 `SubDomain` 持续膨胀
  - promote 后，同一样本可从 `anomaly_model_only` 转为 `anomaly_evidence`
  - benchmark overlay 支持临时 active graph merge，结束后清理临时节点

## 本科毕设建议验证协议

如果目标是本科毕业设计答辩，不必把系统表述为严格学术意义上的“域泛化方法”，更建议表述为:

- 同分布检测能力验证
- 跨场景稳定性验证
- 常见图像退化鲁棒性验证
- Out-of-scope 输入识别能力验证

推荐直接使用新增脚本:

- `scripts/benchmark/visualize_practical_validation.py`

该脚本会一次性生成:

- `Datasets/Test` 的同分布结果
- `Datasets/Validation` 的跨场景结果
- `JPEG Q60 / Gaussian Blur / Downscale + Restore` 三组退化扰动结果
- `Datasets/Nobody` 的 out-of-scope 探测结果
- 汇总 HTML / CSV / JSON 报告

推荐云端运行示例:

```bash
cd /root/pycode/graph_detect
/root/miniconda3/bin/conda run -n detector --no-capture-output \
  python scripts/benchmark/visualize_practical_validation.py \
  --sample-per-class 80 \
  --robustness-sample-per-class 30 \
  --nobody-limit 20 \
  --output-dir reports/practical_validation_default
```

说明:

- 新脚本默认 `--workers=1`
- 在 `internal` 模式下更建议保持顺序执行，避免 OpenCV 类 detector 在多线程评测中出现不稳定

答辩时建议使用如下表述:

- “系统在测试集上具备较高检测准确率”
- “在独立 Validation 数据上仍保持较稳定表现，说明具有一定跨场景泛化能力”
- “在压缩、模糊、分辨率变化等常见退化下，性能有所下降但仍保持可用”
- “对于明显 out-of-scope 样本，系统具备一定拒识/降级能力”

不建议直接表述为:

- “已经严格证明具备强域泛化能力”
- “已经完成标准学术协议下的跨数据集泛化验证”

## 核心架构

### 分层

- 入口层: `app.py`
- 业务层: `service/`
- 检测器层: `detectors/`
- 对齐层: `alignment/`
- 前端目录: `frontend/templates/`, `frontend/static/`, `frontend/src/`
- 训练与评测: 独立脚本和 `reports/` 产物

前端目录约定:

- `frontend/templates/`: Flask 页面模板
- `frontend/src/`: Tailwind 源样式
- `frontend/static/`: 编译产物与运行时静态资源

### 关键模块

- `app.py`
  - Flask 统一入口
  - 装配 `/detect`、`/iterate`、`/evolve`、`/stats`、报告浏览接口

- `service/facades.py`
  - 封装检测、迭代、显式演化等核心工作流
  - 对请求入参、错误处理、语义治理和图谱写入做统一编排

- `detectors/`
  - 管理各 detector 的推理实现
  - 当前 `DetectorHub` 先执行普通 detector，再执行 meta detector
  - 当前主检测信号优先使用 `CalibratedVision`

- `alignment/`
  - 承担 `(detector, feature) -> SubDomain` 对齐
  - 提供对齐、未映射 feature 检测、演化补齐等能力

- `service/graph_semantics.py`
  - 统一图谱语义治理层
  - 负责 `SpecificDomain` 解析、`SubDomain` 规范化、泛化标签过滤、批量语义去重、跨轮次语义复用

### 图谱写入结构

当前图谱节点层级为:

`MainDomain <- SpecificDomain <- SubDomain`

当前节点属性规范:

- `display_name`
- `canonical_name`
- `semantic_source`
- `semantic_version`

## 目录说明

```text
Graph_FakeDetector/
├─ app.py
├─ detector_config.py
├─ config.py
├─ project_paths.py
├─ service/
├─ detectors/
├─ alignment/
├─ artifacts/
├─ frontend/
├─ prompts/
├─ scripts/
│  ├─ training/
│  └─ benchmark/
├─ Datasets/
├─ reports/
├─ train_calibrated_vision_detector.py
├─ visualize_detect_benchmark.py
├─ visualize_practical_validation.py
└─ requirements.txt
```

建议重点关注:

- `app.py`: 唯一服务入口
- `detector_config.py`: detector 权重、阈值、runtime 参数统一配置
- `scripts/training/train_calibrated_vision_detector.py`: 主检测器训练脚本
- `scripts/benchmark/visualize_detect_benchmark.py`: benchmark 与可视化产物生成
- `scripts/benchmark/visualize_practical_validation.py`: 实用验证报告生成脚本
- `artifacts/remote_interface_smoke.py`: 云端接口冒烟脚本
- `artifacts/remote_iterate_regression.py`: 跨轮次回归脚本

兼容说明:

- 根目录同名脚本 (`train_calibrated_vision_detector.py` / `visualize_detect_benchmark.py` / `visualize_practical_validation.py`) 保留为兼容入口，内部会转发到 `scripts/` 目录实现。
- 路径解析层会优先使用 `Datasets/`，并在检测到 legacy 目录时自动兼容 `Dataset/` 与根目录 `uploads/`。

## 检测器现状

### 主检测器

- 名称: `CalibratedVision`
- 当前主权重: `weights/calibrated_vision_detector_dg_round5_dfdc_curated.pt`
- 视觉主干: `EfficientNet-B0`
- runtime 参数统一位于 `detector_config.py`
- 当前决策层会结合:
  - `CalibratedVision`
  - `MetaEnsemble`
  - FFT / Appearance / Boundary 等辅助信号
  - 图谱证据耦合分数

当前策略:

- 优先保证 `/detect` 输入输出协议不变
- 图谱与证据层继续保留，作为解释与兼容输出
- 若权重缺失，允许降级占位，后续补权重后可无缝切换

### 其他检测器

当前主链路中实际参与或兼容保留的 detector 包括:

- `FFTDetector`
- `AppearanceDetector`
- `BoundaryConsistency`
- `MetaEnsemble`

仓库中仍保留部分历史 detector 配置与兼容项，但当前论文/报告口径应以现行主链路为准，不宜继续强调未实际参与主链路的历史模型。

## 语义去重与治理

所有进化相关接口默认启用三层语义去重:

- L1 - Neo4j 节点层: `0.80`
- L2 - Mapping 配置层: `0.85`
- L3 - 精确匹配: `1.00`

当前治理策略包括:

- `SpecificDomain` 主域补齐
- `SubDomain` 名称规范化
- 泛化标签过滤
- 单次 payload 内批量语义去重
- 同一 `SpecificDomain` 下跨轮次子域复用
- 受控 ontology 原型收敛

## API 接口

### `POST /detect`

当前主检测链路，保持外部协议兼容。

表单字段:

- `image`: 必填，单张图片文件
- `auto_evolve`: 可选，默认 `true`
- `semantic_threshold`: 可选，`0.0 ~ 1.0`
- `use_llm_generation`: 可选，默认 `false`

返回核心字段:

- `label`
- `confidence`
- `evidence`
- `reasoning`
- `candidate_generation_available`
- `candidate_context`
- `unmapped_features`
- `evolved_features`
- `semantic_threshold`
- `content_profile`
- `visualizations`

### `POST /iterate`

语义匹配 -> 异步检测 -> LLM 生成 -> 语义规范化 -> 写入 Neo4j。

表单字段:

- `prompt`: 必填
- `images`: 必填，可多张
- `semantic_threshold`: 可选

### `POST /evolve`

显式提交 feature/domain 演化请求。

JSON 字段:

- `features`
- `evolutions`
- `semantic_threshold`

### `POST /iterate_directly`

直接写入演化 payload，适合内部工具或迁移脚本调用。

### `POST /suggest_domain`

根据 detector 与 feature 信号建议 domain。

### `POST /detect/candidates`

对弱证据伪造样本手动触发候选生成。输入 detect 结果页的结构化 JSON，上游由 LLM 生成候选图谱结构和候选映射，写入 Neo4j 候选层与 `alignment/mapping_candidates.json`。

候选生成当前包含以下加固策略：

- 候选 prompt 单独收敛，默认每个 feature 生成 `2` 条候选，必要时才放宽到 `3`
- 候选输出限制更短的 `describe / rationale`
- 候选 LLM 使用独立温度和更高 `max_tokens`
- 截断 JSON 会优先抢救已经完整返回的 `feature_groups`
- 非法 `context_detector / context_feature` 会在入库前被清洗为空
- 候选审批台现已迁移到 detect 前端页面，`iterate` 页面仅保留图谱迭代职责

### `GET /candidate-mappings`

列出候选映射审批清单，支持按状态筛选。

### `POST /candidate-mappings/update`

更新单条候选的图谱字段、映射参数、审批状态。

### `POST /candidate-mappings/benchmark`

对勾选候选运行临时 overlay benchmark，支持：

- `mode=quick`
- `mode=formal`

### `POST /candidate-mappings/promote`

将通过评测门禁的候选晋级到 `alignment/mapping_config.json`，并在 promote 时同步将 candidate graph 合并到 active graph。当前 active mapping 仍保持单个 `detector + feature` 只能存在一条正式规则；接口会返回 promote 日志，说明 rule 更新前后以及 active graph 合并结果。

### `POST /candidate-mappings/delete`

删除候选审批记录与对应 candidate graph 节点。当前前端策略为：已评测或已晋级候选不允许重复操作，如需重新走流程，需先删除再重新生成。

### `GET /stats`

查询图谱统计信息。

### `GET /neo4j_overview`

查询 Neo4j 图谱概览。

### `GET /mapping/config`

返回当前 active mapping 的只读视图，供 `visualization.html` 按 `detector -> feature` 分组展示规则、阈值、权重、context 与 `subdomain_id`。

### `POST /graph/reset_baseline`

清空当前 Neo4j 图谱并按 `cyper.md` 中的基础 Cypher 重建基线图谱。该接口要求确认短语 `RESET_BASELINE_GRAPH`，且只重置 Neo4j 图谱，不修改 `mapping_config.json`。

### `POST /mapping/reset_baseline`

将 `alignment/mapping_config.json` 恢复到 `alignment/mapping_config.baseline.json`，并重新加载 active aligner。该接口要求确认短语 `RESET_BASELINE_MAPPING`，且不修改 Neo4j 图谱。

### `POST /system/reset_baseline`

同时重置 Neo4j 图谱与 `mapping_config.json`。该接口要求确认短语 `RESET_GRAPH_AND_MAPPING`，适合需要回到完整系统基线状态的场景。

### 报告相关接口

- `GET /api/reports`: 返回报告列表与最新报告摘要
- `DELETE /api/reports/<report_name>`: 删除报告目录
- `GET /reports/view/<report_name>/...`: 访问报告 HTML、`metrics.json`、`predictions.csv`

### 可视化页面当前能力

- 图谱节点 / 关系 / 域层级统计
- active mapping 的 detector / feature 可读视图
- `cyper.md` 基线图谱重置
- `mapping_config.baseline.json` 基线 mapping 重置
- 图谱 + mapping 一键系统重置

## 运行与协作约束

### 重要约束

- 不建议在本地直接跑主服务
- 训练、评测、接口调试优先使用云端环境
- 修改代码时优先保持接口兼容，只改内部实现
- detector 阈值、权重路径、校准参数不要继续散落在多个文件
- 候选映射审批清单固定保存在 `alignment/mapping_candidates.json`
- 候选图结构进入 Neo4j 候选层，不直接参与 detect 正式证据链
- 只有评测通过后，候选才允许晋级到 `alignment/mapping_config.json`

### 禁止大范围扫描目录

- `node_modules/`
- `test/`
- `weights/`
- `Datasets/uploads/`
- `paraphrase-multilingual-MiniLM-L12-v2/`
- `__pycache__/`

### Debug 日志规范

- `🔄` 复用/加载
- `✨` 创建/完成
- `⚠️` 警告/降级

## 环境依赖

### Python

核心依赖见 `requirements.txt`，主要包括:

- `Flask`
- `neo4j`
- `openai`
- `langchain`
- `torch`
- `torchvision`
- `Pillow`
- `sentence-transformers`
- `pymilvus`

安装示例:

```bash
pip install -r requirements.txt
```

### 前端样式

项目使用 Tailwind CSS，`package.json` 中提供:

```bash
npm install
npm run dev
npm run build
```

## 配置说明

### detector 配置

`detector_config.py` 是 detector 运行参数的统一入口，当前负责:

- 权重路径
- 集成权重
- 判定阈值
- 特征校准区间
- 占位模式分数范围
- 视觉 backbone runtime 参数

权重根目录可通过环境变量覆盖:

```bash
GRAPH_FAKEDETECTOR_WEIGHTS_ROOT=/path/to/weights
```

### Neo4j 与 LLM

当前仓库通过 `config.py` 提供:

- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASS`
- `ALI_API_KEY`
- `ALI_BASE_URL`

建议在实际部署中将敏感配置放入项目根目录 `.env` 或独立密钥管理，而不是直接写入仓库文件。

本地 `.env` 示例:

```env
ALI_API_KEY=
ALI_BASE_URL=https://api.moonshot.cn/v1
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASS=
FEATURE_THRESHOLD=0.4
```

项目启动时会优先读取项目根目录的 `.env`，`.env` 已被 `.gitignore` 忽略，不会进入 Git。

## 训练与评测

### 训练主检测器

```bash
python scripts/training/train_calibrated_vision_detector.py \
  --dataset-root Datasets/Test \
  --benchmark-per-class 100 \
  --epochs 8 \
  --report-path reports/calibrated_vision_training.json
```

脚本特点:

- 自动为 Fake/Real 划分 `train / val / bench`
- 从 `detector_config.py` 读取 `CalibratedVision` 运行参数
- 支持轻量 TTA 控制
- 训练完成后输出 checkpoint 与训练报告

### 运行 benchmark

```bash
python scripts/benchmark/visualize_detect_benchmark.py \
  --dataset-root Datasets/Test \
  --sample-per-class 100 \
  --output-dir reports/detect_benchmark_sample100
```

当前 benchmark 脚本支持:

- `--dataset-root`
- `--sample-per-class`
- `--limit-per-class`
- `--output-dir`
- `--workers`

说明:

- `--workers=1` 为顺序执行
- 可通过并行 worker 对完整数据集做压测

## 云端验证脚本

### 接口冒烟

```bash
python artifacts/remote_interface_smoke.py
```

覆盖:

- `GET /stats`
- `GET /neo4j_overview`
- `POST /detect`
- `POST /iterate`

### 跨轮次回归

```bash
python artifacts/remote_iterate_regression.py
```

用途:

- 验证相同 prompt 连续执行时，图谱不会持续长出近义 `SubDomain`

## 当前主要问题

- 外部域仍存在波动，尤其 DFDC 相比 Celeb-DF / WildDeepfake 仍是相对短板域
- 候选层语义仍可能出现过泛 domain / subdomain 命名，需要进一步收敛
- benchmark overlay 与正式线上行为虽然已更接近，但仍值得继续量化
- 历史图谱中仍保留少量脏节点与非收敛语义，ontology 清洗未完全结束
- 云端环境偶发存在缓存、模板覆盖和依赖版本差异，需要继续保持“本地+云端”双端校验

这些问题当前不会直接破坏主分类准确率，但会影响:

- 证据解释的专业性
- 图谱与 active mapping 的长期可维护性
- candidate 审批效率与语义一致性
- benchmark / 线上一致性
- 云端运维稳定性

## 下一阶段建议

- P1: 清洗历史图谱脏节点并做 ontology 迁移
- P1: 继续提升候选层命名和 domain 约束，减少 LLM 漂移
- P1: 扩展 evidence-hit 指标与 promote 后线上效果的联动分析
- P2: 收集全量误判样本并做 hard case 再训练
- P2: 继续做按域阈值校准，而不是继续堆 detector 数量
- P2: 完善 reset / rollback / 审批日志体系
- P3: 继续精简 `iterate` 链路，减少 token 与无效生成
- P3: 若走论文路线，增加更严格 protocol 下的 AUC 对齐实验

## 协作建议

- 对外接口变更应尽量避免
- 如需替换权重，优先通过 `detector_config.py` 和权重目录切换完成
- 新增脚本或配置时，优先服务于可维护性和可复现实验
- 提交结果时建议同步说明修改文件、验证情况和剩余风险

# 论文大纲（基于当前系统版本，2026-04-12）

## 使用说明

这是一份面向“后续让另一个 LLM 生成论文初稿”的详细大纲，不是纯目录占位。

建议后续写作时遵循以下原则：

- 以“Deepfake 图像检测 + 域泛化增强 + 图谱证据链 + 可审计输出”为主线，不要再沿用旧版本里过重的“泛泛多模态”叙事。
- 强调当前系统的工程特色是“检测结果可追溯、图谱可演化、阈值可校准、证据链可审计”，而不只是单一分类指标。
- 对外部论文对比时，明确区分“论文 protocol（多为 AUC / 严格跨域协议）”与“当前项目工程抽样评测（当前为 sample300 + Accuracy(valid)/Coverage/AUC）”，避免硬对齐结论。
- 当前系统已经引入候选层工作流：`detect -> candidate graph -> benchmark -> promote`，这是论文中必须体现的新版本差异。
- 不建议再保留旧目录中已经不准确或实现不充分的小节标题，例如“基于置信区间的不确定性估计”“广义多模态阈值融合”等；应替换为当前真实存在的“边界样本复核机制”“图谱耦合决策”“候选审批与晋级”。

---

# 摘要

## 写作目标

用一段完整摘要说明：研究问题、方法框架、系统实现、实验结果、系统优势。

## 摘要应覆盖的核心点

1. 研究对象：面向 Deepfake 图像检测的域泛化增强系统。
2. 问题背景：单纯视觉分类器在外部域波动较大，且缺乏证据链、审计字段与持续演化能力。
3. 方法核心：
   - 以 `CalibratedVision` 为主检测器，配合多检测器协同；
   - 通过 `MainDomain <- SpecificDomain <- SubDomain` 三层图谱组织伪造语义；
   - 通过 feature-to-subdomain 映射实现特征—本体对齐；
   - 通过图谱证据链、域级阈值校准和图谱耦合决策增强检测可解释性；
   - 通过 `iterate` 与弱证据候选层实现图谱持续演化；
   - 通过 `mapping_candidates.json + benchmark + promote` 控制候选映射晋级。
4. 系统输出特色：
   - `reasoning_type`
   - `diagnostic_chain`
   - `risk_level`
   - `needs_review`
   - `evidence_diagnostics`
5. 实验结论：
   - 在项目内数据和外部域 sample300 口径上取得较高 Accuracy(valid) / Balanced Accuracy / AUC；
   - 审计字段覆盖度高；
   - 证据链命中率和候选层工作流使系统具备更强工程可用性。
6. 结论落点：本系统优势不仅在检测性能，还在可追溯、可审计、可运营演化。

---

# 第一章 绪论

## 1.1 研究背景与动机

### 写作要点

- 介绍 Deepfake 图像生成技术快速演化带来的安全、舆论、媒体可信性与身份风险问题。
- 说明传统 Deepfake 检测在实验室内往往有效，但跨数据域、跨压缩条件、跨真实采集条件时性能容易下降。
- 指出当前工程系统中常见的两类缺陷：
  - 只有分类分数，缺少证据链与审计字段；
  - 缺少图谱与语义治理能力，难以支持持续演化。
- 引出本课题动机：
  - 需要一个兼顾检测能力、跨域稳定性、证据链输出、图谱演化和人机协同审批的工程系统。

## 1.2 研究问题与挑战

### 建议拆成四类挑战

1. 检测挑战：
   - 外部域分布偏移明显；
   - 同分布结果强、外部域波动较大。
2. 解释挑战：
   - 分类器输出缺乏结构化证据链；
   - 业务侧需要知道“为什么判假”“为什么需要复核”。
3. 图谱挑战：
   - 图谱扩张与证据命中率存在张力；
   - 新节点写入后若不能进入 active mapping，detect 不会真正受益；
   - 若直接让未验证节点参与 detect，又会污染正式链路。
4. 工程挑战：
   - 阈值、权重、校准参数不能分散；
   - benchmark、可视化、报告、云端调试需要统一口径；
   - 候选审批与晋级流程必须可复现、可维护。

## 1.3 研究目标

### 写作要点

- 构建一个面向 Deepfake 图像检测的可演化、可审计系统。
- 提升外部域检测的稳定性和可校准性。
- 通过图谱证据链强化结果解释能力。
- 建立候选层审批机制，实现“图谱可扩张，但 detect 主链路稳定”的平衡。

## 1.4 主要贡献

### 建议概括为 4 点

1. 提出一套“视觉检测器 + 三层语义图谱 + 域级决策校准”的工程框架。
2. 提出三层语义去重与 ontology 治理机制，实现图谱演化过程中的重复控制与语义收敛。
3. 设计图谱证据链与审计输出机制，将 `reasoning_type / diagnostic_chain / risk_level / needs_review` 纳入检测链路。
4. 设计弱证据候选层工作流：`detect -> candidate -> benchmark -> promote`，实现图谱扩张与正式映射分离。

## 1.5 论文组织结构

### 写作要点

- 用一段话说明各章安排，不必过长。

---

# 第二章 相关技术与研究现状

## 2.1 Deepfake 图像检测方法

### 可写内容

- 基于频域特征的方法
- 基于空间纹理与外观一致性的方法
- 基于 CNN / Transformer 的全局判别方法
- 多检测器集成方法

### 与本文关系

- 说明本文并非只依赖单一频域或单一深网，而是采用主检测器 + 辅助信号 + 元检测器的组合。

## 2.2 域泛化与跨域稳健检测

### 可写内容

- 域泛化基本定义
- 常见协议与指标：Accuracy、AUC、ID/OOD、跨数据集评测
- 训练侧与推理侧增强思路

### 与本文关系

- 本文关注工程系统中的跨域检测稳定性；
- 但应明确当前结果主要是工程抽样口径，不能直接与论文 protocol 硬对齐。

## 2.3 知识图谱在计算机视觉解释中的应用

### 可写内容

- 图谱如何用于语义组织、关系建模、证据追踪和知识扩展；
- 视觉系统中引入图谱的价值：补充结构化解释、支撑可追溯输出。

### 与本文关系

- 本文的图谱不是独立知识问答图，而是围绕 Deepfake 伪造证据、语义子域、检测器特征构建的“进化特征图谱”。

## 2.4 多检测器融合与可审计决策

### 可写内容

- 专家检测器协同
- 加权融合
- 边界样本复核
- 风险分层与审计字段输出

### 与本文关系

- 当前系统实际实现的是“多检测器融合 + 图谱耦合 + 风险复核”，比旧版本的“泛泛多模态融合”表述更准确。

## 2.5 LLM 辅助知识演化与人机协同审批

### 建议新增这一节

- LLM 在本体补全、候选语义生成、结构化映射建议中的作用；
- 人工审批、benchmark 门禁、active mapping 晋级的必要性；
- 避免直接让 LLM 生成内容污染正式推理链路。

---

# 第三章 系统设计

## 3.1 系统架构概述

### 3.1.1 总体架构设计

#### 写作要点

- 给出系统总图：
  - 输入图像
  - detector cluster
  - feature alignment
  - graph evidence
  - decision policy
  - audit output
  - iterate / evolve / candidate workflow
- 明确入口为 `app.py`，核心逻辑位于 `service/`、`detectors/`、`alignment/`。

### 3.1.2 模块化协作逻辑

#### 建议描述的模块

- 检测模块：`DetectorHub` + `CalibratedVision`
- 对齐模块：`FeatureOntologyAligner`
- 图谱治理模块：`GraphSemanticGovernance`
- 证据构建模块：`EvidenceBuilder`
- 决策层模块：`decision_policy.py`
- 演化模块：`iterate` / `evolve`
- 候选审批模块：`detect/candidates` + `mapping_candidates.json`
- benchmark 与报告模块：`scripts/benchmark/visualize_detect_benchmark.py`

### 3.1.3 数据抽象路径

#### 建议写法

从“原始图像”到“最终输出”的抽象路径：

`image -> detector results -> normalized signals -> activated subdomains -> graph evidence -> decision score -> audit fields`

同时补充候选层路径：

`weak-evidence detect result -> candidate context -> LLM candidate JSON -> candidate graph / mapping_candidates -> benchmark -> promote`

## 3.2 多检测器协同检测体系

### 3.2.1 主检测器设计

- 主检测器：`CalibratedVision`
- 当前主干：EfficientNet-B0
- 当前主权重：`weights/calibrated_vision_detector_dg_round5_dfdc_curated.pt`

### 3.2.2 辅助检测器与元检测器

- `FFTDetector`
- `AppearanceDetector`
- `BoundaryConsistency`
- `MetaEnsemble`

### 3.2.3 输出标准化与元数据抽象

- detector 输出统一为 `DetectorResult`
- 标准化特征、元数据、质量风险、face/portrait 相关字段

### 3.2.4 动态融合与图谱耦合

- 自适应融合
- 边界样本图谱耦合
- `decision_profile` / `decision_threshold_override`
- `evidence_alignment_score` / `graph_influence_weight`

## 3.3 进化特征图谱

### 3.3.1 图谱结构设计

- 三层结构：`MainDomain <- SpecificDomain <- SubDomain`
- 当前图谱围绕伪造机制、内容异常、后处理痕迹、身份属性偏移、外观扰动、质量与分辨率等展开

### 3.3.2 图谱初始化与映射规则

- 基线图谱来源：`cyper.md`
- 规则配置来源：`alignment/mapping_config.json`
- 特征与子域的结构化映射方式

### 3.3.3 图谱初始化与正式映射的分离原则

- 图谱节点存在不等于 detect 自动使用；
- detect 正式证据链仍依赖 active mapping；
- 这是当前系统与旧版本的重要差异。

## 3.4 三层语义去重与本体治理

### 3.4.1 实体级语义合并

- Neo4j 节点层去重
- 同域近义节点收敛

### 3.4.2 配置级冗余控制

- `detector + feature` 对应映射规则的冗余控制
- 候选映射与 active mapping 的分层治理

### 3.4.3 字符串级精确拦截

- exact key / canonical name / label 去重

### 3.4.4 ontology profile 与受控语义模板

- 针对特定域的原型收敛；
- 防止图谱不断膨胀为泛化垃圾节点。

## 3.5 特征—本体对齐机制

### 3.5.1 映射规则的结构化定义

- `detector`
- `feature`
- `subdomain_id`
- `subdomain_label`
- `weight`
- `activation_threshold`
- `context_detector`
- `context_feature`
- `context_min_value`

### 3.5.2 基于 sigmoid 的非线性对齐

- 为什么不是线性映射；
- 不同特征值区间对节点激活的重要性不同。

### 3.5.3 节点激活与过滤机制

- `evidence_enabled`
- `context gate`
- `activation_threshold`
- 弱激活过滤

### 3.5.4 未命中、阻塞与诊断上下文

- `blocked_by_context`
- `blocked_by_threshold`
- `no_rule`
- `rule_disabled`
- 这些状态如何为后续候选层提供输入

## 3.6 域级聚合决策与审计输出

### 3.6.1 语义证据链构建

- `EvidenceBuilder`
- `id_matched / label_fallback_matched / unresolved_subdomains`

### 3.6.2 域级特征聚合策略

- graph decision
- detector decision
- 图谱证据与模型分数的融合

### 3.6.3 动态阈值与域级校准

- `decision_profile`
- `decision_threshold_override`
- `DETECTION_DECISION_CONFIG`

### 3.6.4 风险分层与复核策略

- `reasoning_type`
- `diagnostic_chain`
- `needs_review`
- `risk_level`
- `review_reasons`

---

# 第四章 图谱进化与候选层治理

## 4.1 图谱进化概述

### 4.1.1 图谱进化的动机与必要性

- 伪造语义持续变化；
- 单次静态图谱不足以支撑长期系统运行；
- 新图谱节点若缺少治理会破坏 detect 稳定性。

### 4.1.2 图谱进化的目标与原则

- 可扩张
- 可复用
- 可去重
- 可审计
- 不污染正式检测链路

## 4.2 基于 iterate 的图谱扩展

### 4.2.1 iterate 链路设计

- Prompt -> 图像集 -> 预分析 -> LLM -> 语义规范化 -> 图谱写入

### 4.2.2 跨轮次语义复用

- 已有语义节点复用
- 近义候选去重

### 4.2.3 受控 ontology 收敛

- 为什么 iterate 不能无限制长图；
- 为什么有些生成结果会被规范化回已有节点。

## 4.3 detect 驱动的弱证据候选生成

### 4.3.1 触发条件

- `FAKE`
- `anomaly_model_only`
- `evidence_count == 0`
- `unresolved_subdomains > 0`

### 4.3.2 候选上下文构建

- `candidate_context`
- 特征诊断摘要
- 当前规则快照
- 图谱门控状态

### 4.3.3 LLM 结构化候选输出

- `feature_groups`
- 每组 2-3 个备选
- 候选图谱字段
- 候选映射字段
- 审计字段

### 4.3.4 候选输出加固机制

- 缩短 prompt
- 限制 `describe/rationale`
- 截断 JSON 抢救完整 `feature_groups`
- 清洗非法 context

## 4.4 候选层审批、评测与晋级

### 4.4.1 候选双存储

- Neo4j 候选图层
- `mapping_candidates.json`

### 4.4.2 Detect 单页审批台

- 组内单选
- 全选/清空
- 保存选中
- 进度条

### 4.4.3 benchmark 门禁

- quick benchmark
- formal benchmark
- overlay mapping，不直接改 active mapping

### 4.4.4 selective promote

- 同一 `detector + feature` 只能晋级一个候选
- active mapping 替换式更新
- 候选状态回写

## 4.5 图谱进化与 detect 主链路解耦

### 写作要点

- 为什么图谱写入与正式映射不能混为一体；
- 为什么需要 candidate layer；
- 这是论文中最能体现“工程系统设计能力”的部分之一。

---

# 第五章 系统实现

## 5.1 后端实现

### 可写内容

- Flask 路由实现
- `facades` 工作流封装
- `Neo4jClient` / `GraphResultWriter`
- `CandidateReviewFacade`

## 5.2 前端实现

### 可写内容

- `image-recognition.html`
  - detect 结果展示
  - 候选审批区
  - 评测与晋级交互
- `graph-iteration.html`
  - iterate 图谱迭代入口

## 5.3 配置与运行时管理

### 可写内容

- `detector_config.py`
- 权重、阈值、profile 管理
- `.env` 与云端配置

## 5.4 benchmark 与报告系统实现

### 可写内容

- `scripts/benchmark/visualize_detect_benchmark.py`
- html/csv/json/md 报告产物
- `audit_summary` 与证据链指标

## 5.5 云端部署与协作实现

### 可写内容

- 云端 conda 环境
- 本地/云端同步
- 图谱重建
- 调试与验证流程

---

# 第六章 系统测试与分析

## 6.1 测试目标与评价指标

### 6.1.1 测试目标

- 功能完整性
- 检测准确性
- 证据链可用性
- 图谱演化稳定性
- 候选层审批与晋级可用性

### 6.1.2 评价指标体系

#### 建议指标

- `Accuracy(valid)`
- `Balanced Accuracy`
- `AUC`
- `Coverage`
- `evidence_hit_rate`
- `fake_evidence_hit_rate`
- `high_score_no_evidence_rate`
- `unresolved_subdomain_rate`
- `reasoning_type_coverage`
- `diagnostic_chain_coverage`
- `needs_review_rate`

## 6.2 测试环境与实验配置

### 6.2.1 硬件与软件环境

- 云端服务器
- conda 环境 `detector`
- Neo4j

### 6.2.2 测试数据集构建

- `Datasets/Test`
- `Datasets/Validation`
- `Datasets/Celeb-DF`
- `Datasets/DFDC`
- `Datasets/WildDeepfake`
- `Datasets/DFDC_Curated`

### 6.2.3 实验参数设置

- sample300
- `decision_profile`
- `decision_threshold_override`
- `semantic_threshold`
- `workers=1` 的稳定口径说明

## 6.3 系统功能测试

### 6.3.1 总体业务链路验证

- `/detect`
- `/iterate`
- `/evolve`
- `/detect/candidates`
- `/candidate-mappings/*`

### 6.3.2 图像检测模块与可解释性分析

- detect 输出结果
- reasoning 字段
- diagnostic chain
- review flags

### 6.3.3 图谱迭代与候选演化模块测试

- iterate 是否扩图
- 语义去重是否复用节点
- 弱证据候选生成是否正常
- benchmark/promote 是否可控

### 6.3.4 图谱统计与可视化模块测试

- `/stats`
- `/neo4j_overview`
- 图谱节点与关系统计

## 6.4 检测效果测试

### 6.4.1 定量指标分析

- 三个外部域 sample300 结果
- 当前主权重与 profile 阈值对应表现

### 6.4.2 多检测器融合与图谱增强效应

- detector-only 与 graph-coupled 的作用边界
- 弱证据样本中图谱证据链的价值

### 6.4.3 审计链与证据命中分析

- `reasoning_type`
- `diagnostic_chain`
- `evidence_hit_rate`
- `unresolved_subdomain_rate`

## 6.5 候选层工作流效果分析

### 建议新增这一节

- 弱证据样本如何进入候选层
- 候选生成数量与分组数量
- quick / formal benchmark 的门禁作用
- selective promote 对正式映射稳定性的保护

## 6.6 稳定性与鲁棒性测试

### 可写内容

- 低质量输入
- 非人像输入
- 并发与串行口径差异
- 候选 LLM 截断与 fallback 处理

## 6.7 测试结果综合分析

### 重点结论建议

- 当前系统的真正优势不是单一数值，而是：
  - 分类性能较高
  - 证据链与审计输出完善
  - 图谱可治理、可演化
  - 候选层避免了“扩图即污染正式检测”的问题

## 6.8 本章小结

### 简要总结

- 用一小段收束：系统在工程化检测、图谱语义治理和可审计输出方面形成了较完整闭环。

---

# 第七章 总结与展望

## 7.1 全文工作总结

### 写作要点

- 总结系统目标、方法设计、实现路径和验证结果；
- 强调“从分类器到可审计演化系统”的升级。

## 7.2 主要创新点

### 建议归纳为 3-4 点

1. 视觉检测与语义图谱证据链的协同设计；
2. 三层语义去重与 ontology 治理；
3. 域级阈值校准与图谱耦合决策；
4. 弱证据候选层审批与 selective promote 工作流。

## 7.3 不足与展望

### 建议写法

- 当前外部域仍有波动；
- 候选 LLM 语义仍可能漂移；
- 当前与主流论文 protocol 的严格对齐仍不足；
- 后续方向：
  - 下一轮 hard-case 训练
  - 候选 ontology 收敛与自动审查
  - 更严格 protocol 下的 AUC 复现实验
  - 更完整的人机协同审批与回滚机制

---

# 附：建议插图与表格清单

## 建议插图

1. 系统总体架构图
2. detector -> alignment -> graph evidence -> decision 流程图
3. 三层图谱结构图
4. 候选层工作流图：`detect -> candidate -> benchmark -> promote`
5. detect 单页候选审批界面示意图

## 建议表格

1. 检测器与特征输出表
2. 图谱层级与节点属性表
3. `mapping_config.json` 规则字段表
4. 审计输出字段表
5. 各数据域 sample300 结果表
6. 候选层 quick/formal benchmark 指标表

---

# 附：不建议再沿用的旧版表述

以下表述在当前系统版本中应避免继续沿用，或者需要替换为新版口径：

1. “系统主要依赖多模态融合”
   - 当前更准确的说法是：多检测器协同 + 图谱证据耦合 + LLM 辅助候选演化。

2. “图谱自动演化后会立刻增强 detect”
   - 当前不准确。新版系统中，图谱扩展与正式映射已经解耦，必须经候选审批和 benchmark 后才能 promote。

3. “本文结果可以直接与论文 SOTA 做数值对比”
   - 当前不准确。需要明确 protocol 差异与工程抽样口径差异。

4. “系统采用基于置信区间的不确定性估计”
   - 当前不准确。应改写为：边界样本复核机制、风险分层和 `needs_review` 输出。

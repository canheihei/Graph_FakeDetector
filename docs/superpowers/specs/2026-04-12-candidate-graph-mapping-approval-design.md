# Detect Candidate Graph And Mapping Approval Design

**目标**

为 `detect -> iterate -> benchmark -> promote` 建立候选层工作流。`detect` 在证据链薄弱时不直接改正式图谱和正式映射，而是由人工手动触发 LLM 生成候选图谱结构与候选映射规则；候选数据进入 Neo4j 候选层和 `JSON` 审批清单，经前端编辑、选择、评测后，再晋级到 `alignment/mapping_config.json`。

**非目标**

- 不改变 `/detect` 既有正式判决协议。
- 不让 `iterate` 或 LLM 直接写入 active mapping。
- 不支持同一个 `detector + feature` 同时存在多个 active 正式映射。当前正式层仍保持一特征一规则。

## 背景与问题

当前项目已经具备：

- `iterate` 直接写图谱。
- `detect` 在存在 `unmapped_features` 时支持自动/手动进化。
- `mapping_config.json` 作为 detect 正式证据链的唯一激活规则源。

当前缺口在于：

1. `iterate` 生成的新节点不会自动进入 `mapping_config.json`，因此 detect 证据链不变。
2. `detect` 自动进化的触发条件过窄，只覆盖“缺 mapping rule”，不覆盖“图谱证据薄弱但已有规则”的情况。
3. LLM 生成的新语义如果直接进入 active mapping，会污染正式检测链路。

## 目标行为

### 1. Detect 侧候选生成

在以下条件下，detect 结果页显示“生成候选”入口，但默认不自动调用 LLM：

- `label == "FAKE"`
- 且满足任一条件：
  - `reasoning_type == "anomaly_model_only"`
  - `evidence_count == 0`
  - `evidence_diagnostics.unresolved_subdomains > 0`

用户手动点击后，前端将当前 detect 结果中的结构化上下文发送给新接口，由后端调用 LLM 生成候选。

### 2. LLM 候选输出

LLM 输出严格 JSON，按“每个检测特征给 2-3 个备选映射”组织。

每个候选必须包含：

- 图谱字段：
  - `main_domain`
  - `specific_domain`
  - `subdomain_name`
  - `canonical_name`
  - `describe`
- 映射字段：
  - `detector`
  - `feature`
  - `weight`
  - `activation_threshold`
  - `context_detector`
  - `context_feature`
  - `context_min_value`
  - `evidence_enabled`，候选层固定为 `false`
- 审计字段：
  - `feature_rationale`
  - `mapping_rationale`
  - `llm_prompt_version`

### 3. 候选双存储

候选同时落两层：

- Neo4j 候选图层：
  - `CandidateMainDomain`
  - `CandidateSpecificDomain`
  - `CandidateSubDomain`
  - 候选关系仅用于浏览与审查，不参与 detect
- `alignment/mapping_candidates.json`
  - 前端审批的主数据源
  - 包含状态、编辑痕迹、评测结果、晋级结果

### 4. Detect 单页审批台

在 `image-recognition.html` 的弱证据候选区域直接承载候选审批功能，支持：

- 查看待审批候选
- 同一 `candidate_group_id` 组内单选
- 页面级全选/清空
- 修改图谱字段后再批准
- 修改映射参数后再批准
- 运行快速评测
- 运行正式评测
- 查看通过/失败结果
- 仅将通过评测的候选晋级到 active mapping

`graph-iteration.html` 保持图谱迭代主职责，不再承载候选审批逻辑。

### 5. Benchmark 门禁

提供两档评测：

- 快速评测：
  - 小样本
  - 用于快速筛掉明显差候选
- 正式评测：
  - 更大样本
  - 用于晋级前确认

评测运行时只对选中的候选应用“临时 mapping overlay”，不修改正式 `mapping_config.json`。

### 6. 晋级规则

晋级时：

- 一个 `detector + feature` 只能选择一个候选写入 active mapping。
- 如果正式层已有该键，晋级行为是“替换正式规则”，不是追加第二条正式规则。
- 晋级后需要：
  - 更新 `alignment/mapping_config.json`
  - 更新 `mapping_candidates.json` 状态
  - 更新 Neo4j 候选节点状态

## 数据模型

### JSON 候选清单

建议文件：`alignment/mapping_candidates.json`

顶层结构：

```json
{
  "version": "1.0",
  "items": []
}
```

每个候选项：

```json
{
  "candidate_id": "uuid",
  "status": "pending",
  "approval_state": "draft",
  "source": {
    "source_type": "detect_candidate",
    "decision_profile": "dfdc",
    "reasoning_type": "anomaly_model_only",
    "sample_name": "dfdc_fake_0001.jpg"
  },
  "graph_candidate": {
    "main_domain": "域泛化",
    "specific_domain": "后处理痕迹域",
    "subdomain_name": "边缘重采样失真",
    "canonical_name": "edge_resampling_distortion",
    "describe": "..."
  },
  "mapping_candidate": {
    "detector": "FFTDetector",
    "feature": "patch_inconsistency",
    "subdomain_label": "边缘重采样失真",
    "weight": 0.72,
    "activation_threshold": 0.58,
    "context_detector": "CalibratedVision",
    "context_feature": "fake_probability",
    "context_min_value": 0.73,
    "sigmoid_k": 8.0,
    "sigmoid_x0": 0.5,
    "evidence_enabled": false
  },
  "llm": {
    "prompt_version": "detect_candidate_mapping_v1",
    "feature_rationale": "...",
    "mapping_rationale": "...",
    "rank": 1
  },
  "existing_rule_snapshot": {
    "detector": "FFTDetector",
    "feature": "patch_inconsistency"
  },
  "benchmarks": {
    "quick": null,
    "formal": null
  },
  "promotion": {
    "eligible": false,
    "promoted_at": null
  }
}
```

## 接口设计

### `POST /detect/candidates`

输入：

- 当前 detect 结果页的结构化 JSON 上下文

行为：

- 校验是否满足“证据链薄弱”触发条件
- 调用 LLM 生成候选
- 写入候选图层和候选 JSON

输出：

- 本次新增候选列表
- 是否有某个 `detector + feature` 的已有待审候选被覆盖/复用

### `GET /candidate-mappings`

输出所有候选，支持按状态筛选。

### `POST /candidate-mappings/update`

输入单条候选修改内容：

- 图谱字段
- 映射字段
- 审批状态

### `POST /candidate-mappings/benchmark`

支持：

- `mode=quick`
- `mode=formal`

行为：

- 构建 active mapping + selected candidates 的临时 overlay
- 对指定数据集运行内部 benchmark
- 将结果回写到 `mapping_candidates.json`

### `POST /candidate-mappings/promote`

行为：

- 校验候选已通过至少一种 benchmark
- 同步写入 `alignment/mapping_config.json`
- 更新候选状态为 `promoted`

## 关键实现细节

### 1. Detect 上下文不重跑

候选生成接口直接消费 detect 页已有结构化结果，不重新推理。为此 `/detect` 需要补充候选生成上下文，例如：

- 原始 detector feature 摘要
- detector signal 摘要
- gate diagnostics
- evidence diagnostics

### 2. 临时 overlay benchmark

评测不直接改正式映射文件，而是：

- 读取 `mapping_config.json`
- 将选中的候选映射覆盖到内存副本
- 使用新的 `FeatureOntologyAligner(singleton=False)` 与 `DetectionFacade` 跑样本

### 3. 一特征一正式规则

当前 `FeatureOntologyAligner` 的 `_rule_index[(detector, feature)] = rule` 决定了正式层每个特征只能有一条有效规则。因此“2-3 个备选”只适用于候选态；晋级时只能选一个。

## 风险

1. LLM 生成候选容易把“调阈值问题”伪装成“新图谱问题”。
2. `iterate` 当前有严格 ontology 收敛逻辑，新候选和复用候选要分清来源。
3. 候选 benchmark 如果默认样本太大，会拉高前端等待时间。
4. 当前工作区已有未提交变更，实现时必须避免覆盖用户已有改动。

## 验证策略

1. 单元测试覆盖：
   - 候选 JSON 存取
   - 候选 eligibility 判定
   - LLM 输出解析与校验
   - promotion 对 `mapping_config.json` 的替换写入
2. 集成测试覆盖：
   - detect 结果触发候选生成
   - quick benchmark 回写候选结果
   - promote 后 active mapping 生效
3. 云端人工验证：
   - 清空图谱 -> 基线导入 -> 生成候选 -> quick/formal benchmark -> promote -> detect 对比前后证据链

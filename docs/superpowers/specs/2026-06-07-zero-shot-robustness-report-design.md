# Zero-shot 与扰动鲁棒性评测报告设计

日期：2026-06-07

## 背景

当前项目已有跨域 benchmark、候选层治理、证据链审计字段和报告展示能力。远端 `detector` 环境中已有 round6/round7 多域结果，能支撑跨域检测能力说明，但仍缺少一份专门回答以下问题的实验报告：

- 面对未见过的数据域，主检测器和决策链的性能下降多少。
- 面对压缩、模糊、缩放、裁剪、颜色扰动、噪声、遮挡等真实传播扰动，系统是否稳定。
- 扰动后证据链字段是否仍然可用，是否出现高分无证据样本增加。

本设计采用方案 B：新增评测脚本和报告输出，不修改 `/detect` 接口，不修改模型结构，不自动重训。

## 目标

新增一个可复现的实验入口，用统一脚本生成 zero-shot 跨域和扰动鲁棒性报告。报告既要服务工程验证，也要能直接提炼为论文或答辩中的对比数据。

核心输出：

- `metrics.json`：结构化指标，供后续首页、论文表格或二次分析读取。
- `predictions.csv`：逐样本预测记录，便于定位 hard cases。
- `index.html`：可视化报告，展示总览表、域级表、扰动表和误判摘要。
- `summary.md`：中文摘要，凝练实验结论、短板和后续训练建议。

## 非目标

本轮不做以下事情：

- 不训练或替换权重。
- 不修改 `detector_config.py` 的域阈值。
- 不引入新的模型框架。
- 不实现 FGSM/PGD 等白盒梯度攻击。
- 不把报告指标伪装成论文官方 protocol 结果。

白盒攻击不纳入第一版的原因：当前在线决策链由 CalibratedVision、多源辅助检测器、图谱证据和决策策略组成，不是单一端到端可微模型。第一版优先采用黑盒近似扰动，更贴近社交平台传播和工程答辩场景。

## 现有约束

远端 Python 验证必须先激活 conda 环境：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate detector
cd /root/pycode/graph_detect
```

脚本应复用现有能力：

- `DatasetSample`
- `PredictionRecord`
- `DetectBenchmarkRunner`
- `InternalDetectClient`
- `HttpDetectClient`
- `collect_samples`
- `compute_summary`
- `compute_audit_summary`
- `format_percent`

这些能力来自 `scripts/benchmark/visualize_detect_benchmark.py`。

## 新增脚本

新增文件：

```text
scripts/benchmark/visualize_zero_shot_robustness.py
```

脚本职责：

1. 收集多个数据域的 Fake/Real 样本。
2. 对每个数据域运行原图 clean benchmark。
3. 对选定数据域运行扰动 benchmark。
4. 汇总 clean 与 perturbed 的性能差值。
5. 汇总审计字段与证据链字段。
6. 输出 JSON、CSV、HTML、Markdown 四类报告。

默认输出目录：

```text
reports/report_zero_shot_robustness
```

## 命令行接口

建议参数：

```bash
python scripts/benchmark/visualize_zero_shot_robustness.py \
  --datasets Test=Datasets/Test Celeb-DF=Datasets/Celeb-DF DFDC=Datasets/DFDC WildDeepfake=Datasets/WildDeepfake \
  --sample-per-class 120 \
  --robustness-sample-per-class 40 \
  --decision-profiles Celeb-DF=celeb_df DFDC=dfdc WildDeepfake=wilddeepfake \
  --output-dir reports/report_zero_shot_robustness_2026-06-07
```

参数说明：

- `--datasets`：显式传入数据域名称和路径，避免硬编码。
- `--sample-per-class`：clean 跨域测试每类抽样数。
- `--robustness-sample-per-class`：每个扰动测试每类抽样数。
- `--decision-profiles`：可选，每个数据域使用对应 profile；未配置时使用 no-profile。
- `--mode`：复用现有 `internal/http` 模式，默认 `internal`。
- `--endpoint`：HTTP 模式下调用 `/detect`。
- `--workers`：默认 `1`，保持 OpenCV 和检测器链路稳定。
- `--seed`：默认 `42`。
- `--perturbations`：可选扰动集合，默认启用全部第一版扰动。
- `--skip-robustness`：只跑 clean zero-shot。
- `--clean-only-domains`：只对部分域跑 clean，用于快速冒烟。

## 实验协议

### Zero-shot 跨域测试

把每个数据域视为外部未知域，运行 clean benchmark，输出：

- `accuracy_valid`
- `balanced_accuracy`
- `precision_fake`
- `recall_fake`
- `specificity_real`
- `valid_coverage`
- `error_count`
- `average_latency_ms`

报告中应同时标注当前使用的阈值来源：

- no-profile
- domain profile
- explicit threshold override

这样可以避免把不同阈值口径混为一个结果。

### 扰动鲁棒性测试

第一版扰动集合：

| 扰动 | 目的 | 参数 |
|---|---|---|
| JPEG 压缩 | 模拟平台二次压缩 | quality 60 |
| 高斯模糊 | 模拟低清录制或转发 | radius 1.2 |
| 缩放恢复 | 模拟截图和平台缩放 | 先缩到 60%，再恢复 |
| 随机裁剪恢复 | 模拟人脸区域轻微缺失 | 裁掉 8% 边缘后恢复尺寸 |
| 颜色/亮度扰动 | 模拟滤镜或曝光变化 | brightness 1.18, color 0.82 |
| 轻量噪声 | 模拟弱黑盒扰动 | 像素级小幅噪声 |
| 局部遮挡 | 模拟贴纸、水印或遮挡 | 固定比例矩形遮挡 |

扰动在临时目录中物化，不写入数据集目录。报告只保存预测结果和指标，不保存生成图片。

### 黑盒近似对抗说明

报告中将轻量噪声、局部遮挡、裁剪恢复标记为 black-box stress tests，而不是严格 adversarial attack。摘要中使用“扰动鲁棒性”或“黑盒近似扰动”，避免声称完成白盒对抗鲁棒评测。

## 指标设计

### 域级 clean 指标

每个数据域输出：

- `clean_accuracy_valid`
- `clean_balanced_accuracy`
- `clean_fake_recall`
- `clean_real_specificity`
- `clean_valid_coverage`
- `clean_error_count`

### 扰动指标

每个数据域、每个扰动输出：

- `perturbed_accuracy_valid`
- `perturbed_balanced_accuracy`
- `perturbed_fake_recall`
- `perturbed_real_specificity`
- `accuracy_drop`
- `balanced_accuracy_drop`
- `fake_recall_drop`
- `specificity_drop`

下降幅度计算方式：

```text
drop = clean_metric - perturbed_metric
```

若某域没有 clean 基线，脚本应报错并停止该域的扰动汇总。

### 审计与证据链指标

复用 `compute_audit_summary`，每个 clean/perturbed suite 输出：

- `reasoning_type_coverage`
- `diagnostic_chain_coverage`
- `needs_review_rate`
- `evidence_hit_rate`
- `fake_evidence_hit_rate`
- `high_score_no_evidence_rate`
- `unresolved_subdomain_rate`
- `joint_evidence_correct_rate`
- `fake_joint_evidence_recall`

报告重点展示扰动前后的变化：

- `evidence_hit_rate_drop`
- `high_score_no_evidence_rate_delta`
- `joint_evidence_correct_rate_drop`

## 数据结构

`metrics.json` 顶层结构：

```json
{
  "report_type": "zero_shot_robustness",
  "generated_at": "2026-06-07T00:00:00",
  "config": {},
  "summary": {},
  "domain_suites": [],
  "perturbation_suites": [],
  "recommendations": []
}
```

`domain_suites` 单项：

```json
{
  "domain": "DFDC",
  "dataset_root": "Datasets/DFDC",
  "suite_key": "clean__DFDC",
  "suite_kind": "clean",
  "decision_profile": "dfdc",
  "sample_count": 240,
  "metrics": {},
  "audit_summary": {},
  "average_latency_ms": 0.0
}
```

`perturbation_suites` 单项：

```json
{
  "domain": "DFDC",
  "perturbation": "jpeg_q60",
  "suite_key": "perturbed__DFDC__jpeg_q60",
  "sample_count": 80,
  "clean_reference": {},
  "metrics": {},
  "audit_summary": {},
  "drops": {}
}
```

`recommendations` 单项：

```json
{
  "priority": "P1",
  "target": "DFDC",
  "reason": "fake_recall_drop is high under jpeg_q60",
  "suggestion": "Add compression-augmented DFDC hard cases before retraining."
}
```

## 报告内容

### HTML 报告

`index.html` 展示：

1. 总览卡片：平均 clean accuracy、平均扰动 accuracy、最大下降、最脆弱域、最脆弱扰动。
2. Zero-shot 域级表：每个数据域的 clean 指标和阈值来源。
3. 扰动鲁棒性表：按域和扰动展示下降幅度。
4. 证据链稳定性表：扰动前后 evidence hit 和 high-score-no-evidence 变化。
5. 训练建议区：根据规则输出下一步建议。

### Markdown 摘要

`summary.md` 用中文输出：

- 实验目的。
- 使用数据域。
- clean 跨域表现。
- 扰动后性能下降。
- 最脆弱域和最脆弱扰动。
- 证据链是否稳定。
- 后续是否建议重训以及建议原因。

摘要应避免夸大：

- 不说“完全解决 zero-shot”。
- 不说“完全抵抗对抗样本”。
- 使用“工程抽样口径”“黑盒近似扰动”“用于答辩和后续优化定位”。

## 训练建议规则

脚本只生成建议，不执行训练。

建议规则：

- 若某域 `clean_balanced_accuracy < 0.90`，建议该域 hard-case 微调。
- 若某域 `fake_recall < 0.85`，建议增加 fake hard cases，优先避免漏检。
- 若某扰动 `accuracy_drop >= 0.08`，建议加入对应数据增强。
- 若 `high_score_no_evidence_rate_delta >= 0.08`，建议补充 detector feature 到 mapping，而不是只重训。
- 若 `evidence_hit_rate_drop >= 0.10`，建议审查扰动后 evidence builder 的 feature 激活稳定性。

通俗训练解释：

1. 先用当前模型跑新报告，找出错题和一扰动就变错的样本。
2. 把这些错题整理成 hard-case 数据集。
3. 从当前权重继续训练几轮，而不是从零开始。
4. 训练时保留 Test 和 Celeb-DF 作为保底域，防止只提升 DFDC 或扰动样本却损害原有能力。
5. 重训后必须重跑 clean 四域、扰动鲁棒性和证据链报告，只有指标整体改善才切换权重。

## 错误处理

- 数据集路径不存在：跳过该域并在报告 `warnings` 中记录；若所有域都不存在则退出。
- 类别目录缺失：复用现有 `collect_samples` 报错信息，报告中记录该域失败。
- 单个样本检测失败：保留 `ERROR` 记录，计入 `error_count`。
- 扰动生成失败：该样本记录为错误，不中断整个 suite。
- clean 基线缺失：禁止计算 drop，报告中标记为不可比较。
- HTTP 模式连接失败：明确提示 endpoint 和 timeout。

## 测试策略

新增单元测试：

- 扰动函数保持图片可打开、尺寸符合预期。
- drop 指标计算正确。
- recommendation 规则在低准确率、高下降、高分无证据场景下输出正确建议。
- `metrics.json` payload 包含 `domain_suites`、`perturbation_suites`、`recommendations`。

建议测试文件：

```text
tests/test_zero_shot_robustness_report.py
```

验证命令：

```bash
python -m pytest tests/test_zero_shot_robustness_report.py -q
python -m pytest tests/test_benchmark_audit_summary.py -q
npm run build
```

远端完整冒烟命令：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate detector
cd /root/pycode/graph_detect
python scripts/benchmark/visualize_zero_shot_robustness.py \
  --datasets Test=Datasets/Test DFDC=Datasets/DFDC \
  --sample-per-class 5 \
  --robustness-sample-per-class 3 \
  --decision-profiles DFDC=dfdc \
  --output-dir reports/report_zero_shot_robustness_smoke
```

## 实施边界

本设计不需要架构调整、数据库重构、权限体系修改或安全策略变更。新增脚本是独立 benchmark 入口，读取数据集和调用现有 detect facade，写入 `reports/` 下的新报告产物。

若后续要根据报告结果执行重训，应单独进入训练方案设计，明确训练数据、保底域、目标域、权重输出、阈值校准和回滚方式。

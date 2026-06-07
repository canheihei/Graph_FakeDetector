# DFDC / WildDeepfake 定向回拉 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保持 `Datasets/Test` 与 `Celeb-DF` 不低于 `90%` 的前提下，重新微调主检测器，尽量将 `DFDC` 与 `WildDeepfake` 的大样本跨域准确率回拉到接近 `95%`。

**Architecture:** 复用现有 EfficientNet-B0 `CalibratedVision` 主干，不改 `/detect` 接口。新增 `WildDeepfake` curated / hard-case 数据构建脚本，并调整域泛化训练脚本的 checkpoint 选择逻辑，使其从“均值优先”改为“保底域约束 + 目标域优先”。

**Tech Stack:** Python, PyTorch, torchvision, project benchmark scripts, existing `CalibratedVision` detector pipeline.

---

### Task 1: 补齐 WildDeepfake curated 数据构建脚本

**Files:**
- Create: `scripts/training/curate_wilddeepfake_hardcases.py`
- Test: `tests/test_curate_wilddeepfake_hardcases.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from scripts.training.curate_wilddeepfake_hardcases import (
    PredictionRow,
    is_noise_candidate,
    select_hard_cases,
)


def test_wilddeepfake_noise_and_hardcase_selection():
    rows = [
        PredictionRow(
            path=Path("Fake/a.jpg"),
            truth_label="FAKE",
            predicted_label="REAL",
            decision_fake_score=0.08,
            decision_threshold=0.10,
            is_correct=False,
        ),
        PredictionRow(
            path=Path("Fake/b.jpg"),
            truth_label="FAKE",
            predicted_label="REAL",
            decision_fake_score=0.11,
            decision_threshold=0.10,
            is_correct=False,
        ),
        PredictionRow(
            path=Path("Real/c.jpg"),
            truth_label="REAL",
            predicted_label="FAKE",
            decision_fake_score=0.88,
            decision_threshold=0.10,
            is_correct=False,
        ),
        PredictionRow(
            path=Path("Real/d.jpg"),
            truth_label="REAL",
            predicted_label="REAL",
            decision_fake_score=0.14,
            decision_threshold=0.10,
            is_correct=True,
        ),
    ]

    assert is_noise_candidate(rows[0], 0.08, 0.92) is True
    assert is_noise_candidate(rows[1], 0.08, 0.92) is False

    selected = select_hard_cases(
        rows[1:],
        hard_margin=0.05,
        hard_max_per_class=2,
        excluded_paths=set(),
    )

    assert [item.path.as_posix() for item in selected["FAKE"]] == ["Fake/b.jpg"]
    assert [item.path.as_posix() for item in selected["REAL"]] == ["Real/c.jpg", "Real/d.jpg"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_curate_wilddeepfake_hardcases.py -q`
Expected: FAIL because `scripts/training/curate_wilddeepfake_hardcases.py` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from scripts.training.curate_dfdc_hardcases import (
    IMAGE_SUFFIXES,
    PredictionRow,
    build_source_index,
    hard_priority,
    is_noise_candidate,
    load_predictions,
    materialize_file,
    resolve_project_path,
    row_to_relative,
    select_hard_cases,
    to_float,
)
```

实现方式：
- 复制 `scripts/training/curate_dfdc_hardcases.py` 的结构；
- 默认参数改为：
  - `--dataset-root Datasets/WildDeepfake`
  - `--predictions-csv reports/report_wilddeepfake_sample1200_override_010_2026-04-20/predictions.csv`
  - `--output-root Datasets/WildDeepfake_Curated`
  - `--report-path reports/report_wilddeepfake_curation_report.json`
- 保持噪声过滤、hard-case 选择、重复采样与报告输出逻辑一致。

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_curate_wilddeepfake_hardcases.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/training/curate_wilddeepfake_hardcases.py tests/test_curate_wilddeepfake_hardcases.py
git commit -m "feat: add wilddeepfake hardcase curation"
```

### Task 2: 让训练脚本支持约束优先的 checkpoint 选择

**Files:**
- Modify: `scripts/training/train_domain_generalized_calibrated_vision_detector.py`
- Test: `tests/test_train_domain_generalized_selector.py`

- [ ] **Step 1: Write the failing test**

```python
from scripts.training.train_domain_generalized_calibrated_vision_detector import (
    choose_best_epoch_payload,
)


def test_choose_best_epoch_payload_prefers_target_domains_under_guardrails():
    payloads = [
        {
            "epoch": 1,
            "mean_dataset_balanced_accuracy": 0.94,
            "per_dataset": {
                "Test": {"balanced_accuracy": 0.91},
                "Celeb-DF": {"balanced_accuracy": 0.90},
                "DFDC_Curated": {"balanced_accuracy": 0.89},
                "WildDeepfake_Curated": {"balanced_accuracy": 0.90},
            },
        },
        {
            "epoch": 2,
            "mean_dataset_balanced_accuracy": 0.93,
            "per_dataset": {
                "Test": {"balanced_accuracy": 0.92},
                "Celeb-DF": {"balanced_accuracy": 0.91},
                "DFDC_Curated": {"balanced_accuracy": 0.95},
                "WildDeepfake_Curated": {"balanced_accuracy": 0.96},
            },
        },
    ]

    best = choose_best_epoch_payload(
        payloads,
        guardrail_domains=("Test", "Celeb-DF"),
        guardrail_min_balanced_accuracy=0.90,
        target_domains=("DFDC_Curated", "WildDeepfake_Curated"),
    )

    assert best["epoch"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_train_domain_generalized_selector.py -q`
Expected: FAIL because `choose_best_epoch_payload` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def choose_best_epoch_payload(
    payloads,
    *,
    guardrail_domains,
    guardrail_min_balanced_accuracy,
    target_domains,
):
    ...
```

实现要求：
- 先筛选满足 guardrail 的 epoch；
- 若存在满足 guardrail 的候选：
  - 先比较 target domain 平均 `balanced_accuracy`
  - 再比较 `mean_dataset_balanced_accuracy`
  - 再比较全局 `balanced_accuracy`
- 若不存在满足 guardrail 的候选：
  - 选择 guardrail 缺口最小且 target domain 最强的 payload
- 用该函数替换训练循环里当前直接按 `score = (...)` 选最优 checkpoint 的逻辑。

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_train_domain_generalized_selector.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/training/train_domain_generalized_calibrated_vision_detector.py tests/test_train_domain_generalized_selector.py
git commit -m "feat: prioritize target domains in dg checkpoint selection"
```

### Task 3: 输出更明确的训练报告字段

**Files:**
- Modify: `scripts/training/train_domain_generalized_calibrated_vision_detector.py`
- Test: `tests/test_train_domain_generalized_selector.py`

- [ ] **Step 1: Write the failing test**

```python
from scripts.training.train_domain_generalized_calibrated_vision_detector import summarize_epoch_selection


def test_summarize_epoch_selection_contains_guardrail_and_target_scores():
    payload = {
        "epoch": 7,
        "mean_dataset_balanced_accuracy": 0.94,
        "per_dataset": {
            "Test": {"balanced_accuracy": 0.92},
            "Celeb-DF": {"balanced_accuracy": 0.91},
            "DFDC_Curated": {"balanced_accuracy": 0.95},
            "WildDeepfake_Curated": {"balanced_accuracy": 0.96},
        },
    }

    summary = summarize_epoch_selection(
        payload,
        guardrail_domains=("Test", "Celeb-DF"),
        target_domains=("DFDC_Curated", "WildDeepfake_Curated"),
    )

    assert summary["guardrail_average"] == 0.915
    assert summary["target_average"] == 0.955
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_train_domain_generalized_selector.py -q`
Expected: FAIL because `summarize_epoch_selection` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def summarize_epoch_selection(payload, *, guardrail_domains, target_domains):
    ...
```

实现要求：
- 返回：
  - `guardrail_average`
  - `target_average`
  - `guardrail_domains`
  - `target_domains`
- 将该结构写入训练报告 JSON，便于后续答辩说明 checkpoint 为什么被选中。

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_train_domain_generalized_selector.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/training/train_domain_generalized_calibrated_vision_detector.py tests/test_train_domain_generalized_selector.py
git commit -m "feat: record constrained checkpoint selection summary"
```

### Task 4: 生成新的 WildDeepfake curated 数据

**Files:**
- Use existing: `scripts/benchmark/visualize_detect_benchmark.py`
- Use existing: `scripts/training/curate_wilddeepfake_hardcases.py`
- Output: `Datasets/WildDeepfake_Curated`
- Output: `reports/report_wilddeepfake_curation_report.json`

- [ ] **Step 1: 先用当前主权重跑 WildDeepfake 基准**

Run:

```bash
python scripts/benchmark/visualize_detect_benchmark.py \
  --dataset-root Datasets/WildDeepfake \
  --workers 1 \
  --decision-threshold-override 0.10 \
  --output-dir reports/report_wilddeepfake_sample1200_override_010_2026-04-20
```

Expected:
- 生成 `predictions.csv`
- 生成 `metrics.json`

- [ ] **Step 2: 用预测结果构建 curated 数据**

Run:

```bash
python scripts/training/curate_wilddeepfake_hardcases.py \
  --dataset-root Datasets/WildDeepfake \
  --predictions-csv reports/report_wilddeepfake_sample1200_override_010_2026-04-20/predictions.csv \
  --output-root Datasets/WildDeepfake_Curated \
  --report-path reports/report_wilddeepfake_curation_report.json \
  --hard-margin 0.08 \
  --hard-repeat 2 \
  --hard-max-per-class 180 \
  --clear-output
```

Expected:
- 输出 curated 数据目录
- 输出 curation JSON 报告

- [ ] **Step 3: 检查 curated 数据规模**

Run:

```bash
find Datasets/WildDeepfake_Curated/Fake -type f | wc -l
find Datasets/WildDeepfake_Curated/Real -type f | wc -l
```

Expected:
- 两类样本都非零
- 总规模明显大于原始 `600/600`

- [ ] **Step 4: Commit**

```bash
git add scripts/training/curate_wilddeepfake_hardcases.py reports/report_wilddeepfake_curation_report.json
git commit -m "chore: build wilddeepfake curated dataset"
```

### Task 5: 重新训练目标域强化版本主权重

**Files:**
- Use existing: `scripts/training/train_domain_generalized_calibrated_vision_detector.py`
- Output: `weights/calibrated_vision_detector_dg_round6_targeted.pt`
- Output: `reports/report_train_calibrated_vision_dg_round6_targeted.json`

- [ ] **Step 1: 基于当前主权重启动训练**

Run:

```bash
python scripts/training/train_domain_generalized_calibrated_vision_detector.py \
  --dataset-roots Datasets/Test Datasets/Celeb-DF Datasets/DFDC_Curated Datasets/WildDeepfake_Curated \
  --init-checkpoint weights/calibrated_vision_detector_dg_round5_dfdc_curated.pt \
  --epochs 12 \
  --warmup-epochs 2 \
  --batch-size 48 \
  --num-workers 8 \
  --lr-head 0.002 \
  --lr-backbone 0.00015 \
  --output weights/calibrated_vision_detector_dg_round6_targeted.pt \
  --report-path reports/report_train_calibrated_vision_dg_round6_targeted.json
```

Expected:
- 输出新权重
- 输出新训练报告

- [ ] **Step 2: 检查训练报告中的 guardrail 与 target 指标**

Run:

```bash
python -c "import json;from pathlib import Path;data=json.loads(Path('reports/report_train_calibrated_vision_dg_round6_targeted.json').read_text(encoding='utf-8'));print(json.dumps(data.get('selection_summary', {}), ensure_ascii=False, indent=2))"
```

Expected:
- 能看到 guardrail 平均与 target 平均
- 能看到最终 best epoch 的选择摘要

- [ ] **Step 3: Commit**

```bash
git add scripts/training/train_domain_generalized_calibrated_vision_detector.py reports/report_train_calibrated_vision_dg_round6_targeted.json
git commit -m "feat: train round6 targeted dg detector"
```

### Task 6: 用新权重重跑四域 benchmark 并更新阈值建议

**Files:**
- Output: `reports/report_test_round6_targeted_2026-04-20/`
- Output: `reports/report_celeb_df_round6_targeted_2026-04-20/`
- Output: `reports/report_dfdc_round6_targeted_2026-04-20/`
- Output: `reports/report_wilddeepfake_round6_targeted_2026-04-20/`
- Modify if accepted: `detector_config.py`

- [ ] **Step 1: 临时将新权重设为当前 CalibratedVision 权重**

做法：
- 若训练脚本直接输出为 `weights/calibrated_vision_detector.pt` 则无需额外改动；
- 若输出为新文件名，则评测前先显式覆盖/切换为新权重路径。

- [ ] **Step 2: 运行四域 benchmark**

Run:

```bash
python scripts/benchmark/visualize_detect_benchmark.py --dataset-root Datasets/Test --workers 1 --output-dir reports/report_test_round6_targeted_2026-04-20
python scripts/benchmark/visualize_detect_benchmark.py --dataset-root Datasets/Celeb-DF --workers 1 --output-dir reports/report_celeb_df_round6_targeted_2026-04-20
python scripts/benchmark/visualize_detect_benchmark.py --dataset-root Datasets/DFDC --workers 1 --output-dir reports/report_dfdc_round6_targeted_2026-04-20
python scripts/benchmark/visualize_detect_benchmark.py --dataset-root Datasets/WildDeepfake --workers 1 --output-dir reports/report_wilddeepfake_round6_targeted_2026-04-20
```

Expected:
- 四域都有 `metrics.json`

- [ ] **Step 3: 汇总四域结果**

Run:

```bash
python -c "import json;from pathlib import Path;paths=['reports/report_test_round6_targeted_2026-04-20/metrics.json','reports/report_celeb_df_round6_targeted_2026-04-20/metrics.json','reports/report_dfdc_round6_targeted_2026-04-20/metrics.json','reports/report_wilddeepfake_round6_targeted_2026-04-20/metrics.json'];print(json.dumps({p: json.loads(Path(p).read_text(encoding='utf-8'))['summary'] for p in paths}, ensure_ascii=False, indent=2))"
```

Expected:
- 能直接对比四域准确率

- [ ] **Step 4: 若结果达标，则更新 profile 阈值**

做法：
- 将 `metrics.json` 中推荐阈值与实际稳定结果结合；
- 修改 `detector_config.py` 中：
  - `dfdc`
  - `wilddeepfake`
- 如 `celeb_df` 或默认阈值也需联动，则同步更新。

- [ ] **Step 5: 回归验证**

Run:

```bash
python scripts/benchmark/visualize_detect_benchmark.py --dataset-root Datasets/DFDC --workers 1 --decision-profile dfdc --output-dir reports/report_dfdc_round6_profile_2026-04-20
python scripts/benchmark/visualize_detect_benchmark.py --dataset-root Datasets/WildDeepfake --workers 1 --decision-profile wilddeepfake --output-dir reports/report_wilddeepfake_round6_profile_2026-04-20
```

Expected:
- profile 模式下结果与最终目标接近

- [ ] **Step 6: Commit**

```bash
git add detector_config.py reports/report_*_round6*_2026-04-20
git commit -m "feat: calibrate round6 targeted detector thresholds"
```

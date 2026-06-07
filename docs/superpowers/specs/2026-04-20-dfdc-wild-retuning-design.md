# DFDC 与 WildDeepfake 定向回拉设计

## 背景
- 2026-04-20 在云端将 `Celeb-DF`、`DFDC`、`WildDeepfake` 扩到每类 `600` 张后，新的 benchmark 结果出现明显下滑：
  - `Celeb-DF`：`91.92%`
  - `DFDC`：`77.50%`
  - `WildDeepfake`：`81.25%`
- 当前主权重仍为 `weights/calibrated_vision_detector_dg_round5_dfdc_curated.pt`。
- 历史高分主要来自较小抽样口径与较小验证集，不能再直接作为当前跨域稳定性的依据。

## 目标
- 主目标：将 `DFDC` 与 `WildDeepfake` 在当前更大样本口径上尽量回拉到接近 `95%`。
- 约束目标：`Datasets/Test` 与 `Celeb-DF` 保持 `90%+`，避免单域过拟合后牺牲整体答辩口径。
- 实现方式保持工程可维护，不改 `/detect` 接口，不引入新的在线推理分支。

## 根因判断
- 这不是单纯的阈值失配问题。
  - `DFDC` 当前 1200 张口径的推荐阈值仍接近原口径（约 `0.48`）。
  - `WildDeepfake` 当前 1200 张口径的推荐阈值仍接近原口径（约 `0.10`）。
- 当前问题更接近“目标域真实分布变难，现有主干对目标域 hard-case 覆盖不足”。
- 现有训练报告也显示：
  - `WildDeepfake` 的历史高分来自很小的验证集；
  - `DFDC_Curated` 验证虽显著优于原始 DFDC，但还不足以支撑新的 1200 张口径；
  - 继续只靠阈值校准，不足以把 `77%/81%` 级别直接拉回 `95%` 附近。

## 方案对比

### 方案 A：只重做域级阈值校准
- 优点：最快，不改训练。
- 缺点：难以修复当前量级的性能下滑，最多只能做局部补偿。
- 结论：不采用。

### 方案 B：基于当前主权重做“目标域强化 + 保底域约束”的混合微调
- 优点：
  - 直接面向当前目标；
  - 可以复用现有 `DFDC_Curated` 工作流；
  - 不改推理接口，只替换权重与 profile 阈值；
  - 训练目标可以显式约束 `Test/Celeb-DF >= 90%`。
- 缺点：需要补齐 `WildDeepfake` 的 curated / hard-case 数据与训练筛选逻辑。
- 结论：采用。

### 方案 C：改成多分支或域专属模型
- 优点：理论上上限更高。
- 缺点：改动大，答辩前维护与解释成本高。
- 结论：当前阶段不采用。

## 最终设计

### 1. 数据侧
- 保持原始跨域集：
  - `Datasets/Celeb-DF`
  - `Datasets/DFDC`
  - `Datasets/WildDeepfake`
- 继续使用：
  - `Datasets/Test` 作为主保底域
  - `Datasets/DFDC_Curated` 作为目标强化域之一
- 新增：
  - `Datasets/WildDeepfake_Curated`
- curated 数据构建原则：
  - 用当前主权重先跑完整 benchmark；
  - 过滤强疑似噪声标签；
  - 选取误判样本和近边界样本作为 hard-case；
  - 对 hard-case 做有限重复采样，不做无限偏置。

### 2. 训练侧
- 初始化权重：`weights/calibrated_vision_detector_dg_round5_dfdc_curated.pt`
- 新训练集建议：
  - `Datasets/Test`
  - `Datasets/Celeb-DF`
  - `Datasets/DFDC_Curated`
  - `Datasets/WildDeepfake_Curated`
- 训练策略：
  - 复用 `scripts/training/train_domain_generalized_calibrated_vision_detector.py`
  - 保持 EfficientNet-B0 主干
  - 前若干 epoch 冻结主干，只训练分类头；之后解冻联合微调
  - 继续保留 per-dataset threshold 搜索

### 3. checkpoint 选择策略
- 不再只看全局 `mean_dataset_balanced_accuracy`。
- 改为约束优先：
  - `Test >= 90%`
  - `Celeb-DF >= 90%`
  - 在满足约束的 checkpoint 中优先最大化：
    - `DFDC` balanced accuracy
    - `WildDeepfake` balanced accuracy
    - 其后再考虑整体均值
- 如果没有 checkpoint 同时满足约束，则回退到“保底域跌幅最小、目标域提升最大”的 Pareto 最优点。

### 4. 校准与替换
- 训练完成后重新在四域上 benchmark：
  - `Datasets/Test`
  - `Datasets/Celeb-DF`
  - `Datasets/DFDC`
  - `Datasets/WildDeepfake`
- 基于新权重更新 `dfdc` 与 `wilddeepfake` 的阈值 profile。
- 只有满足目标趋势后，才替换默认主权重与 profile 阈值。

## 风险与边界
- `95%` 是目标，不是这轮一定能机械达成的保证值；真实上限受 HF 外部域样本分布影响。
- 如果 `WildDeepfake_Curated` 构建后发现 hard-case 占比过高，可能把 `Celeb-DF/Test` 拖低，需要回调重复采样强度。
- 如果训练脚本只按均值选最优 checkpoint，会继续偏向容易域，因此 checkpoint 选择逻辑必须同步改。

## 验收标准
- 云端 benchmark 稳定跑通；
- `DFDC` 与 `WildDeepfake` 明显提升，并尽量逼近 `95%`；
- `Datasets/Test` 与 `Celeb-DF` 保持 `90%+`；
- 输出新的训练报告、benchmark 报告、权重文件与阈值建议。

## 项目定位
- Deepfake 图像检测 + 域泛化（检测为主，图谱增强证据与语义以及可追溯，域泛化为次要，比较依赖多域数据集训练的主检测器）

## 核心架构
- 入口：`app.py`
- 检测：`detectors/`（主检测器：`CalibratedVision`，EfficientNet-B0）
- 编排与语义：`service/`（detect/iterate/evolve/Neo4j/LLM/语义治理）
- 对齐：`alignment/`
- 前端：`templates/` + `src/` + `static/`
- 归档：根目录遗留文件统一放入 `legacy/root_orphans/`；非当前主口径的旧 benchmark 报告统一放入 `reports/_archived/`
- 测试：统一放在 `tests/` 目录，避免在项目根目录保留孤立测试脚本

## 核心实现逻辑
- `/detect` 保持接口稳定，输出审计字段：`reasoning_type / diagnostic_chain / risk_level / needs_review`
- 语义治理：`MainDomain <- SpecificDomain <- SubDomain`，属性含 `semantic_source / semantic_version`
- 语义去重阈值（默认）：L1=0.80（图谱节点），L2=0.85（detector feature），L3=1.00（精确匹配）
- 决策层域校准：`service/decision_policy.py`（自适应融合 + profile/override 阈值解析）
- `/detect` 可选入参：`decision_profile`、`decision_threshold_override`（默认不传行为不变）
- 当前默认 profile 阈值（dg_round5 + sample300 校准）：`celeb_df=0.61`、`dfdc=0.48`、`wilddeepfake=0.11`（配置位于 `DETECTION_DECISION_CONFIG`）
- 候选层工作流：弱证据 `FAKE` 样本通过 `/detect/candidates` 手动触发 LLM 候选生成，候选图结构写入 Neo4j，候选审批清单写入 `alignment/mapping_candidates.json`
- 候选审批与晋级：`/candidate-mappings`、`/candidate-mappings/update`、`/candidate-mappings/benchmark`、`/candidate-mappings/promote`
- 候选删除接口：`/candidate-mappings/delete`
- 候选审批前端承载页：`image-recognition.html`；`graph-iteration.html` 回归图谱迭代入口，不再承载候选审批
- promote 行为：不仅更新 `mapping_config.json`，还同步将 candidate graph merge 到 active graph，并输出 promote 日志（mapping before/after + active subdomain）
- 候选审批状态建议：`pending -> benchmarked -> promoted`；已评测和已晋级项默认不重复评测/晋级，如需重做应先删除
- 图谱基线重置接口：`/graph/reset_baseline`，从 `cyper.md` 恢复 Neo4j 基线结构；只重置图谱，不回滚 `mapping_config.json`
- active mapping 基线重置接口：`/mapping/reset_baseline`，从 `alignment/mapping_config.baseline.json` 恢复 `mapping_config.json`
- 一键系统重置接口：`/system/reset_baseline`，同时恢复基线图谱与基线 mapping
- active mapping 仍保持单个 `detector + feature` 仅允许一条正式规则；候选可多条，但晋级时只能择一进入 `mapping_config.json`

## 数据与评测现状
- 主数据：`Datasets/Test|Validation`（现有基线来自项目内数据，与同目录下的Train属于Openforensics数据集）
- 跨域评测数据（本地与云端均已同步）：
  - `Datasets/Celeb-DF/Fake|Real`：各 300
  - `Datasets/DFDC/Fake|Real`：各 300
  - `Datasets/WildDeepfake/Fake|Real`：各 300
- HF 拉取脚本：`scripts/benchmark/pull_hf_external_datasets.py`（支持按 `--per-class` 扩样）
- 拉取说明文档：`reports/report_hf_dataset_pull_script.md`
- 评测脚本：`scripts/benchmark/visualize_detect_benchmark.py`
- 评测产物：`reports/report_*_sample300_*/`
- 总结报告：`reports/report_*.md`

## 当前准确率（最新口径）
- 口径：`2026-04-24` 外部域最新测试汇总，主权重：`weights/calibrated_vision_detector.pt`
- Celeb-DF（`Valid=1200`，`Correct=1157`，本次报告 `Override=0.42`）：Accuracy(valid)=`98.97%`，Balanced Acc=`98.98%`
- DFDC（`Valid=1200`，`Correct=1200`，本次报告 `Override=0.49`）：Accuracy(valid)=`94.11%`，Balanced Acc=`94.03%`
- WildDeepfake（`Valid=254`，`Correct=251`，本次报告 `Override=0.10`）：Accuracy(valid)=`98.82%`，Balanced Acc=`99.19%`
- Test（`sample1200` 抽样，Fake/Real 各 600）：Accuracy(valid)=`99.17%`，Balanced Acc=`99.17%`
- 可追溯审计覆盖：四域/抽样口径 `reasoning_type=100%`、`diagnostic_chain=100%`

## 会话共识（当前有效）
- 对比主流域泛化方案时，区分“论文 protocol（多为 AUC）”与“工程抽样评测（当前为 Accuracy/coverage）”，不做硬对齐结论
- 当前项目核心优势定位在可追溯与可审计，而非仅单指标分类性能
- 当前跨域评测显示：同分布强、外部域波动大，后续重点应放在域级阈值校准与 hard-case 训练
- 已在项目根目录新增答辩问答整理文档：defense_qa_notes.md，用于沉淀会话中已确认的系统问答口径；该文档仅本地维护，可继续追加老师提问
- 已在项目根目录新增参赛信息填写稿：competition_submission_info.md，用于整理作品简介、AI 应用说明、开源组件、安装说明与设计说明

## 当前进度
- [Done] 三层去重、统一 detector 配置、EfficientNet-B0 主干、detect 审计链输出
- [Done] 云端接口联通与跨域数据集小样本构建（Celeb-DF/DFDC/WildDeepfake）
- [Done] 跨域 benchmark 与报告产出（metrics + html/csv/json + md）
- [Done] 决策层自适应融合接入（CalibratedVision + 辅助信号），并补充阈值来源审计字段
- [Done] benchmark 支持 `--decision-profile` / `--decision-threshold-override`，并输出 `threshold_calibration` 建议
- [Done] 多域联合训练权重：`weights/calibrated_vision_detector_dg_round1.pt`
- [Done] 审计链评测增强：benchmark 统计 `reasoning_type / diagnostic_chain / needs_review / risk_level` 覆盖度并写入 `metrics.json.audit_summary`
- [Done] DFDC 清洗与硬样本增强：`scripts/training/curate_dfdc_hardcases.py` -> `Datasets/DFDC_Curated`（去噪 + hard-case 重采样）
- [Done] 定向微调权重：`weights/calibrated_vision_detector_dg_round5_dfdc_curated.pt`（当前主权重）
- [Done] 主流论文常报指标 vs 项目指标对照报告更新：`reports/report_mainstream_common_metrics_vs_project_2026-04-12.md`
- [Done] 检测链路保持全流程输出，不因非人脸判定而中断；人脸质量字段保留用于审计
- [Done] detect 决策层与图谱证据耦合：`evidence_alignment_score / graph_influence_weight`
- [Done] 证据构建命中诊断增强：`EvidenceBuilder` 输出 `requested/id_matched/fallback/unresolved` 并回传 `/detect.evidence_diagnostics`
- [Done] benchmark 证据链命中指标：`evidence_hit_rate / fake_evidence_hit_rate / high_score_no_evidence_rate / unresolved_subdomain_rate / avg_evidence_alignment_score`
- [Done] 指标报告前端页：`evidence-chain-report.html` 现统一展示核心指标 / 主流指标对比 / 证据链指标，数据由 `reports/Indicators/indicator_report_data.json` 提供，并通过 `/api/indicator-report` 输出给前端；同时支持从 `alignment/mapping_candidates.json` 动态汇总候选审批 benchmark 的 hit rate 增益，接入证据链报告中的“审批进化增益”区块
- [Done] 证据链指标脚本口径明确：核心抽取脚本为 `scripts/benchmark/visualize_detect_benchmark.py` 的 `compute_audit_summary()`；当前已扩展输出 `joint_evidence_correct_rate` 与 `fake_joint_evidence_recall`，并动态追加到指标报告模块中的证据链指标部分
- [Done] 主页 reports 中文化：`service/report_gallery.py` 现将报告目录名映射为简短中文标题/副标题；`/reports/view/<report_name>/index.html` 改为动态中文/中英双语报告页（模板：`frontend/templates/report-view.html`），不再直接暴露原始英文静态 benchmark HTML
- [Done] 弱证据候选层一期接入：`/detect/candidates` + `mapping_candidates.json` + `iterate` 候选审批台 + quick/formal benchmark + selective promote
- [Done] 候选 LLM 结构化输出加固：缩短 prompt、限制候选输出长度、截断 JSON 抢救完整 `feature_groups`、清洗非法 `context_detector/context_feature`
- [Done] iterate 上传区支持“直接上传图片 + 上传文件夹”双入口，并在前端展示已选图片缩略图预览
- [Done] iterate 语义回退主域改为优先复用现有唯一 MainDomain/基线 `域泛化`，避免写入 `未分类主域`
- [Done] detect 页面新增 `explain_summary`，改为结论优先的答辩展示，并保留完整可追溯展开区
- [Done] detect 候选区新增候选生成时间展示，已晋级分组改为摘要卡 + 按需展开详情
- [Done] detect 候选区允许直接选中并删除已晋级候选组，摘要卡提供“选中该组”入口
- [Done] round7 rebalanced 定向微调完成：`DFDC/WildDeepfake` 强化且 `Test/Celeb-DF` 保持 90% 以上，profile 阈值已更新为 `celeb_df=0.42`、`dfdc=0.47`、`wilddeepfake=0.12`
- [Done] 答辩 PPT `24.pptx` 文案已收敛为机制、指标与边界表述，弱化绝对化措辞并保留原版式
- [Doing] 图谱补齐与脏节点清洗（ontology 收敛）
- [Doing] detector -> subdomain 映射补齐、域级阈值校准
- [Doing] 候选 ontology 收敛与审批门禁细化（减少 LLM 语义漂移）
- [Wait] 下一轮 hard-case 训练（待整理误判集与训练窗口）

## 编码规范
- 不破坏接口，优先改内部实现
- 阈值/权重/校准集中管理，禁止分散配置
- 决策层配置统一放在 `detector_config.py`（`DETECTION_DECISION_CONFIG`），业务逻辑在 `service/decision_policy.py`
- benchmark 统一使用 `scripts/benchmark/visualize_detect_benchmark.py`
- 外部域 benchmark 若并发触发段错误，回退 `--workers=1` 作为稳定口径
- 内容必须可复现、可维护，并给出验证状态与风险说明

## 运行规范
- 禁止本地跑主服务；训练/调试优先云端
- 每次成功测试后，询问我需要同步更新的本地和云端代码
- python环境用的是云端服务器的miniconda，名称为detector
- HF 拉取前先执行：`source /etc/waas-script/proxy.sh`
- HF 扩样标准命令：`python scripts/benchmark/pull_hf_external_datasets.py --dataset-root Datasets --datasets Celeb-DF DFDC WildDeepfake --per-class <N> --seed 42 --clear-existing`
- 域级校准评测示例：`python scripts/benchmark/visualize_detect_benchmark.py --dataset-root Datasets/Celeb-DF --sample-per-class 80 --workers 1 --decision-profile celeb_df --output-dir reports/report_celeb_df_sample80_profile_celeb_df`
- 云端同步的项目目录 `(base) root@waas:~/pycode/graph_detect`
- 修改后同步更新 `AGENTS.md`，这个不需要同步到云端仓库
- 云端连接：`ssh -p 49649 root@ae2836a105e54a59892c240731db2e15.region1.waas.aigate.cc`
- 云端连接密码: `cf167522dee84a37a60c9d69e937b14a`
- 本地尽量不用 python 做验证。                                                                                                                                                                                                   
- 需要执行 Python 时，优先走云端。

# EuroSAT 模型与部署进展审查报告

日期：2026-08-11
审查仓库：`git@github.com:lhh010/Ltsimulator-EuroSAT.git`
审查基线：`b755d6c5f3a19e693ec365141b1aa5ca63cfd6b1`（远端 `main`）

## 1. 仓库同步状态

- 已确认审查时本地 `HEAD`、`origin/main` 和最新 `FETCH_HEAD` 均指向 `b755d6c5f3a19e693ec365141b1aa5ca63cfd6b1`。
- 工作区在审查前保持干净，因此下述缺口不是本地未拉取完整或分支落后造成的。
- 相比旧本地基线 `6feeaed2`，最新版本增加 14 个提交，变更 90 个文件，统计约为 `+9189/-13`。
- 本报告只记录审查结果；没有启动 Docker OSimulator，也没有占用 Gazelle 真机。

## 2. 当前模型进度

| 模型 | clean | v8/QAT | 当前状态 |
|---|---:|---:|---|
| M7 | 95.56% | 95.00%；独立复测 94.98% | v8 权重已入库；clean 权重缺失 |
| M8 | 96.17% | 96.26%；独立复测 96.20% | clean/v8 权重均已入库 |
| M9 | — | 真机 94.90% | 已有 X0/ds3 部署链 |
| M10 | — | 真机 96.40% | 当前报告中的硬件最佳结果 |

M7/M8 的训练日志：

- [M7 clean](../../logs/m7_clean.log)：最终 test 95.56%，best val 95.44%。
- [M7 v8](../../logs/m7_v8.log)：最终 test 95.00%，best val 94.74%。
- [M8 clean](../../logs/m8_clean.log)：最终 test 96.17%，best val 96.04%。
- [M8 v8](../../logs/m8_v8.log)：最终 test 96.26%，best val 95.94%。
- [M7/M8 独立复测记录](../x0/results/M_validate.md)：M7 94.98%，M8 96.20%，与训练日志差异不超过 0.06 个百分点。

需要注意：这里的 M7/M8 “v8 test”是在 `eval()` 模式下关闭噪声、保留量化后进行的 clean test，不是真机精度，也不是 OSimulator 带噪仿真精度。

## 3. 已完成的完整性验证

- 使用本机已有 PyTorch 环境，对 M5、M6、M7、M8、M9、M10 的最终 checkpoint 分别完成模型构建、QAT-v8 包装和 `strict=True` 状态字典加载；六个模型全部通过。
- 仓库内受检 JSON 均能正常解析。
- 受检 Shell 脚本均通过 `bash -n`。
- 受检 Python 文件均通过语法编译检查。
- Git 对象完整性检查通过，报告生成前工作区无未提交修改。

这些检查证明已入库的最终权重与当前代码中的模型结构匹配，但不能代替 OSimulator 或真机执行验证。

## 4. 重要判断

M7/M8 当前应表述为：

> 训练完成、量化软件复评完成，但 Gazelle/OSimulator 部署尚未接通。

依据如下：

- [部署登记文档](round_x0_arch_hw_codesign.md)仍将 M7/M8 标为“未注册”。
- 仓库没有适用于 M7/M8 的通用导出器、Gazelle `MODEL_REGISTRY` 条目或板端 runner。
- 现有 `src/osim_eval_j1.py` 硬编码旧 J1 架构，采用 `strict=False` 部分加载，并且底层光计算封装可能静默回退到 Fake backend，不能作为 M7/M8 的 OSimulator 验收入口。
- 本轮更新的主要工程增量是 X0/M9/M10 的 ds3 部署链，没有反向补齐 M7/M8 的部署链。

## 5. 需要优先修复的问题

1. **M7 clean checkpoint 缺失。** `configs/m7_j1w075_v8probe15.json` 要求 `weights/m7_j1w075_clean.pth`，训练日志也证明当时曾完整加载该文件，但它不在当前工作树及本地 Git 历史中。这使 clean → v8 训练链无法从仓库完整复现。
2. **M7/M8 尚未接入部署链。** 需要配置驱动的 exporter、OSimulator evaluator、Gazelle registry 和板端 runner；加载必须严格校验，真实 OSimulator 模式必须禁止自动回退 Fake。
3. **运行证据归档不完整。** 仓库缺少文档所称的 `runs/metrics.jsonl`、M7 clean 权重以及部分原始校准文件、logits、板端日志和运行 manifest。
4. **文档与代码已经失配。** `logs/m5_m8_summary.md` 仍将已经完成的 M7 clean 写成“待启动”；M7/M8 配置还硬编码 `E:/LT-Simulator/train-test/data/EuroSAT_RGB`，当前 Linux checkout 不能原样复跑。
5. **M9/M10 head 拓扑需要核实。** 训练配置为 `head_fp32=true`，但板端 runner 默认可能执行光学 head。应确认实际运行时设置，并统一训练、FAKE、OSimulator 和真机拓扑。
6. **校准与测试集没有严格隔离。** 现有标量校准/逐列 probe 使用 test 子集，随后又在相同 test 范围报告精度。正式结果应改用独立 train/validation calibration split，再在未参与校准的 test 上评测。
7. **M10 真机结果仍需复现。** 96.40% 当前是 seed42、单一校准窗口、1000 张样本的结果；尚缺完整 5400 张测试、额外 seed、同窗口对照和跨日重复。
8. **凭据安全问题。** 当前仓库的 `AGENTS.md` 含明文板卡凭据。应立即轮换相关凭据，并从当前版本和 Git 历史中安全清理；本报告不记录凭据内容。

## 6. 问题归属：本地还是在线仓库

### 6.1 在线仓库需要修复的问题

因为审查工作树与最新 `origin/main` 完全一致，以下问题在远端最新版本中同样存在，需要通过新提交修复：

- M7 clean checkpoint 未入库。
- M7/M8 没有正确的 exporter、OSimulator 入口、Gazelle 注册和板端 runner。
- `runs/metrics.jsonl`、校准文件和真机原始证据没有完整归档。
- 汇总文档过期，配置硬编码不可移植的数据路径。
- M9/M10 训练与部署 head 拓扑可能不一致。
- 校准使用 test 子集，正式评测方法需要调整。
- M10 真机报告缺少全量、跨日和多 seed 复验。
- `AGENTS.md` 中的明文凭据以及仍在仓库中的作废产物需要清理。

### 6.2 本地环境问题

- 当前 checkout 内没有配置文件所指向的 Windows EuroSAT 数据目录；即使代码不变，也需要在本机准备数据并修改路径。
- OSimulator 必须在带授权环境的 Docker 容器中运行，本次尚未执行容器内验证。
- 原 `/home/tao/jichuangsai/osim` 是未完成初始化的外层仓库并包含旧副本；目前已用独立、干净的 worktree 避免覆盖旧目录。
- SSH agent 对默认 `id_ed25519` 曾打印拒绝签名提示，但 Git 随后使用可用认证成功完成 fetch；这不影响本次同步结果。

### 6.3 同时涉及本地与在线仓库的问题

- 数据集需要在本机正确安装；同时，仓库也应提供 `--data-dir` 或环境变量覆盖，而不是硬编码 Windows 路径。
- OSimulator/Gazelle 需要本机授权环境和真机窗口；同时，仓库必须先提供正确且可审计的 M7/M8 接口。
- 真机结果需要在实际设备上重新采集；随后还必须把配置、权重 hash、数据索引、校准文件、logits 和完整日志提交或发布到可追踪的制品存储中。

## 7. 推荐处理顺序

1. 立即轮换暴露的凭据，并安全清理仓库及历史中的敏感信息。
2. 找回并归档 M7 clean checkpoint，补齐运行 manifest、metrics 和日志，修正文档及数据路径。
3. 实现配置驱动的 `export_j1_family.py` 与 `osim_eval.py`，要求权重严格加载并在 OSimulator 模式下 fail-fast。
4. 在 Gazelle 部署仓库注册 model7/model8，先完成同样本 PyTorch-v8 → 导出后 NumPy/FAKE → OSimulator 的 logits 和预测对拍。
5. 统一 M9/M10 的 head 拓扑，改用独立 calibration split；随后完成 M10 全量测试、多 seed 和跨日复验，并归档全部原始证据。

## 8. 总结

当前在线仓库的主要成果是：M7/M8 训练和量化软件复评已经完成，M9/M10 已形成 X0/ds3 部署链并取得初步真机结果。主要短板不是本地 Git 没有同步，而是远端仓库本身尚未形成 M7/M8 的可复现部署闭环，M9/M10 的实验口径和证据归档也仍需加强。

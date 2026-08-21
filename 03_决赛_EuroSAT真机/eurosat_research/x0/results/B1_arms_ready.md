# X0 · B1-prep — 下采样算子家族 4 臂代码就绪报告

日期：2026-08-09 · 状态：DONE · 冒烟：本地 CPU 真跑通过（torch 2.12.1）

## 改动清单

### `src/models.py`（全部后向兼容）
- `make_downsample` 新增 3 个 `pool_mode`：
  - `max3` — `nn.MaxPool2d(3, stride=2, padding=1)`，零 MACs，每步 RF 贡献 (k−1)·j 翻倍
  - `blur` — 新模块 `BlurPool`：固定 3×3 二项式核 [1,2,1]⊗[1,2,1]/16，depthwise `F.conv2d(stride=2, padding=1)`（等价于 stride=1 滤波 + stride=2 子采样）；核注册为 **buffer**（不可训练、随 `.pth` 落盘可复现，已验证 `state_dict` 含 `*.kernel` 且无可训练参数）
  - `conv3s2` — `Conv3×3 s2 + BN + ReLU`（channels 不变，bias=False，风格同 stride1x1）
- `MiniVGG` 新增旋钮 `extra_pool`（bool，默认 False）：stage3 末尾（GAP 前）追加 `MaxPool2d(2)`，4×4→2×2
- `conv3s2` 特判：**只替换 stage 的下采样点，stem 下采样保持 MaxPool2d(2)**。理由：stem 是全 FP32 电计算路径，若 stem 池化也换 conv3s2，`stem.3.0` 会被 `prepare_model_v8` 在无 probe 噪声标定的情况下 QAT 化（sigma=0 但仍 8bit 量化），偏离 "stem FP32" 口径。max3/blur 为纯电侧池化，无此问题，stem 照常替换（与"全部换"的臂定义一致）
- docstring 已更新（X0 扩展段，注明各算子电侧/光侧归属）；`build_model` 透传 `extra_pool`

### `configs/x0_*.json`（4 个，schema 同 c3d/r7_final）
- `x0_pool3_160.json` — J1 + `pool_mode=max3`
- `x0_pool4_160.json` — J1 + `extra_pool=true`
- `x0_blurpool_160.json` — J1 + `pool_mode=blur`
- `x0_dsconv3_160.json` — J1 + `pool_mode=conv3s2`，`macs_ok=true`（显式声明超 2M 预算）

### `x0/smoke_b1.py`
冒烟脚本（实例化 + compute_macs + 随机前向 + QAT v8 转换 + BlurPool buffer 检查），可重复运行：`python3 x0/smoke_b1.py`

## 4 臂 params / MACs（compute_macs 口径，输入 3×64×64）

| 臂 | params | MACs | 对比 J1 (50,330 / 1,377,536) |
|---|---|---|---|
| x0_pool3 | 50,330 | 1,377,536 (1.378M) | 完全一致，零成本 ✓ |
| x0_pool4 | 50,330 | 1,377,536 (1.378M) | 一致（见下注） |
| x0_blurpool | 50,330 | 1,377,536 (1.378M) | 光侧一致；电侧 +64,512 depthwise MACs（不计入光口径） |
| x0_dsconv3 | 96,602 | 2,557,184 (2.557M) | +1,179,648 MACs / +46,272 params，**超 2M 预算**（"花 MACs 的臂"，预期内） |

dsconv3 增量理论核验：stage1 conv3s2（32ch, 16×16→8×8）= 32·9·32·64 = 589,824；stage2 conv3s2（64ch, 8×8→4×4）= 64·9·64·16 = 589,824；合计 1,179,648，与实测增量**精确相等** ✓

**注（x0_pool4 与任务预期的偏差）**：任务表格预期"stage3 MACs 砍 4×"，但臂定义是"stage3 **末尾**（GAP 前）加第 4 次 MaxPool"——池化在 stage3 两层 conv **之后**，conv 已在 4×4 上算完，MACs 不变（实测确认 1.378M）。该臂的收益是 RF（j=32，每位置 +(k−1)·j=16，RF ~17→~33）而非 MACs。若要"砍 stage3 MACs 4×"需把池化挪到 stage3 conv 之前，那是另一个臂；如需可在 X0 后续轮补。

## 冒烟验证结果（全部真跑通过）

- 4 臂 + 7 个旧 config 全部 `build_model` + 随机前向 `(2,3,64,64)→(2,10)` 通过
- 后向兼容回归（与历史数字一致）：r6_ctrl 1.378M ✓、r7_final_head256 1.395M ✓、r8_rf_s2k3 4.523M ✓（AGENTS 记录 4.52M）、r6_pool_avg/s1x1/patchify、c3d_J1（=J1 1.378M）✓
- `prepare_model_v8` 在 x0_dsconv3 上转换 8 个 conv（stem.0 保 FP32；新层 `stage1.3.0`/`stage2.6.0` 正确获得噪声标定，无 WARN），转换后前向通过
- BlurPool：`stem.3.kernel / stage1.3.kernel / stage2.6.kernel` 在 state_dict 中，无可训练参数

## Config 设计决策

1. **单阶段 v8 160ep 全程，不用 init_from 续训**。schema 完整支持（runner 对 `qat_version=v8` 直接读 `layer_noise_sigmas`/`layer_col_off`/`layer_col_gain`/`layer_dw_rms`，`epochs=160` 即可；c3d 的 60ep+init_from 是冠军迭代续训场景，不是 schema 限制）。理由：R7 教训——架构对比必须在最终训练时长口径下做（head256 的 80ep 优势在 160ep 反转）；续训式会混入"源架构 J1 的归纳偏置"，污染臂间对比。
2. **lr=0.05（非 c3d 的 0.01）**：c3d 的 0.01 是 finetune 学习率；from-scratch 160ep 口径沿用 r7_final/r8 的 0.05 + sgd + standard aug + label_smoothing 0.05，warmup 用默认值 5（同 r8，不显式覆盖）。
3. **噪声参数照抄 c3d（v8 1.5× 余量）**：5 个标准层逐字复制。dsconv3 臂新增的 `stage1.3.0`/`stage2.6.0` 无 probe 实测标定，**代理取值 = 同 stage 前一 conv 的标定**（stage1.3.0←stage1.0，stage2.6.0←stage2.3），同 stage 同分辨率，是最接近的可用标定；待 X0 后续如有板上 probe 可替换。
4. **电侧/光侧归属**：max3 / blur / extra_pool 均为电侧池化，不进光计算路径（无需噪声标定）；conv3s2 是 Conv2d，被 `prepare_model_v8` 自动转为光计算层（已验证）——其替代位置的原 1×1 conv 本就在光路，口径一致。stem 全程 FP32 电计算（含 conv3s2 臂，stem 池化保持 MaxPool）。

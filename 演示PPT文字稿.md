# 演示 PPT 文字稿 · Optic-SpaceNet 光学 CNN 加速系统

> 面向航空航天"在轨计算"的光学 CNN 加速系统
> 总时长 **≈ 10 分钟** · 主线 **12 页** + 2 个可插入深度页（**第 2.5 页** 数据集、**第 6.5 页** QAT 技术深入 STE/LSQ+）· 每页标注：核心信息、版面内容、图表/视觉、口述讲稿
>
> **使用说明**：本稿重心在**内容提炼与逻辑梳理**，不追求视觉炫技。每页先给"核心信息（一句话）"，再给上屏要点、配图（含数据/提示词）、口述讲稿。配图制作所需的**提示词与数据**统一放在文末[附录 A / B](#附录-a配图总表)。
>
> **评分重心映射**：光计算占比 **(20 分)** → 第 9 页（讲透）；进阶 **(20 分)** → 第 11–12 页；AI 工具链 **(10 分)** → 第 6.5、7 页（**QAT 技术深度** + 工具链闭环）；功能及性能 **(10 分)** → 第 8、10 页；方案设计/创新 **(创新 5 / 实用 5 / 平台 5)** → 第 3、4、6、6.5 页。

---

## 全篇逻辑主线（一条故事线，背给评委）

```
在轨痛点 → 必须星上计算，电子算力吃不消
   ↓
先吃透硬件 → 逆向 Gazelle：8×2 / 8a8w / 线性度 99.4% → 量化是唯一瓶颈
   ↓
据此设计 → 2×2 卷积对齐 tile（硬件感知架构）→ 268K 参数、1M 算力
   ↓
据此训练 → QAT(STE/LSQ+) + int8 + Gazelle 噪声注入（硬件匹配训练）→ 真机近无损
   ↓
结果说话 → 光计算占比 90.65%、真机 90.43%、量化反超 FP32
   ↓
方法论 → "硬件先于模型" + 真机验证闭环 + 科学诚信（自纠数据泄漏）
```

> **一句话总纲（开场/收尾都能用）**：我们把三种 CNN 迁移到曦智 Gazelle 光计算硬件，用"硬件先于模型"的方法论，做到 **光计算占比 ≥ 90%、真机精度 90%+、量化无损甚至反超 FP32**。

---

## 第 1 页 · 封面（0:00–0:25，约 25 秒）

**核心信息**：作品定位——面向卫星在轨遥感计算的光学 CNN 加速系统。

**版面内容**
- 主标题：**面向航空航天在轨计算的光学 CNN 加速系统 —— Optic-SpaceNet**
- 副标题：三模型量化迁移 × 曦智 Gazelle 光计算平台 × 知识蒸馏
- 团队成员 / 指导老师 / 单位 / 日期

**图表/视觉**
- 背景：卫星 + 光子芯片 + 遥感地物缩略图（EuroSAT 的 AnnualCrop / Forest / River 拼贴）
- 角标一行：EuroSAT 10 类 icon —— AnnualCrop · Forest · HerbaceousVegetation · Highway · Industrial · Pasture · PermanentCrop · Residential · River · SeaLake

**口述讲稿**
> "各位评委好，我们带来的作品是 Optic-SpaceNet——一套面向卫星在轨遥感计算的光学 CNN 加速系统。我们把三种 CNN 迁移到曦智 Gazelle 光计算硬件上，在保持 90% 以上准确率的同时，把超过 90% 的乘加运算交给了光子完成。下面用 10 分钟汇报。"

---

## 第 2 页 · 问题与动机（0:25–1:15，约 50 秒）

**核心信息**：在轨场景要求"星上就地推理"，而电子算力（功耗/重量/抗辐射）吃不消，光计算是结构性解。

**版面内容**
- **在轨计算痛点**：卫星每圈过境产生海量遥感图像，星地下行带宽严重受限 → 必须在**星上**完成实时地物分类，只下传结果。
- **电子计算瓶颈**：GPU/ASIC 功耗高、重量大、散热难、抗辐射差，不适合卫星严苛平台。
- **光子计算优势（空间 AI 视角）**：
  - **抗辐射**：光计算用光强/相位/波长等**模拟量**计算、无数字逻辑 → **光阵列核心对单粒子翻转（SEU/比特翻转）天然不敏感**，大幅降低航天抗辐射加固成本。
  - **散热/功耗**：太空无空气、只能辐射散热，传统 GPU 几百瓦极难排热；光计算功耗低、发热小，匹配在轨热约束。
  - **光速并行**：矩阵乘一次光程完成（皮秒级），高并行。
  - **理论寿命**：无电子芯片"栅氧击穿/阈值漂移"等电学老化，材料稳定下核心寿命有望 >10 年（优于 CMOS 5–10 年）。
- **题目目标**：把 CNN 迁移到 Gazelle **8×2 光学矩阵乘法器**，最大化光计算占比，量化精度可控。

> ⚠️ **严谨边界（上屏/口述都注意）**：不要说"光计算完全免疫辐射"。准确表述：*"光计算**核心**对单粒子翻转不敏感，大幅降低系统抗辐射成本；但 DAC/ADC/FPGA 等**外围电子**仍需常规加固。"* 详见[附录 C](#附录-c应用价值纵深与高频质疑应对答辩话术)。

**图表/视觉**（🆕 建议新建，见附录 B-图1）
- 左：星上推理链路图（光学相机 → 光子芯片 → 地物分类结果 → 仅下传标签）
- 右：电子 vs 光子计算对比小表（功耗 / 并行度 / 抗辐射 / 重量 四项）

**口述讲稿**
> "卫星每圈过境产生海量遥感图像，全部下传不现实，必须在星上就地分类，只把结果传回地面。传统电子算力功耗、重量、散热都太大，还怕空间辐射；而光子计算用光做矩阵乘，低功耗、高并行、还天然抗辐射，是卫星在轨场景的结构性解。题目要求我们把 CNN 跑在曦智 Gazelle 光计算芯片上，光计算占比越高越好——这正是我们工作的核心。"

---

## 第 2.5 页 · 数据集与分类任务（推荐插入页，约 35 秒）【项目内容 / 应用价值】

**核心信息**：任务是卫星遥感图像的 **10 类地物分类**——用的就是真实卫星数据（EuroSAT / Sentinel-2），与"在轨遥感"场景天然对口；数据划分严格、无泄漏。

**版面内容**
- **数据集**：**EuroSAT（RGB 版）** —— 基于 **Sentinel-2 卫星**遥感图像，**27000 张、64×64 RGB**，10 类地物覆盖；本身就是卫星数据，直接对应在轨遥感分类场景。
- **分类任务**：10 类单标签地物分类 → 直接对应土地监测、灾害评估、目标识别等在轨应用。
- **10 类及样本量（含类别不平衡）**：

| 类别 | 样本 | | 类别 | 样本 |
|---|---|---|---|---|
| AnnualCrop（农田） | 3000 | | Pasture（牧场） | 2000 |
| Forest（森林） | 3000 | | PermanentCrop（常年作物） | 2500 |
| HerbaceousVegetation（草本） | 3000 | | Residential（住宅区） | 3000 |
| Highway（公路） | 2500 | | River（河流） | 2500 |
| Industrial（工业区） | 2500 | | SeaLake（海/湖） | 3000 |

  - 总计 **27000**；6 类各 3000、其余 2000–2500（Pasture 最少 2000）→ 轻度类别不平衡，训练用增广 + label smoothing 缓解。
- **三分划分**：train **16200** / val **5400** / test **5400**（seed=42），三者**严格互斥 + 覆盖全集**（统一数据源 `eurosat_split.py` 强制断言）→ 修复了早期 test⊂train 泄漏（Bug #11）。
- **预处理**：训练增广（随机水平翻转 + ±10° 旋转）+ ImageNet 归一化（mean/std=[0.485,0.456,0.406]/[0.229,0.224,0.225]）；验证/测试仅归一化（无增广，保证可复现）。

**图表/视觉**（🆕 建议新建，见附录 B-图8）
- 🖼 **EuroSAT 10 类样例拼图**（2×5 网格，每类一张真实样本 + 类名）——直接用 `data/EuroSAT_RGB/` 里你自己的图，无需 AI 生成。

**口述讲稿**
> "我们用的数据集是 EuroSAT，基于 Sentinel-2 卫星遥感图像，两万七千张、六十四乘六十四的 RGB 图，十个地物类别——它本身就是卫星数据，和我们'在轨遥感分类'的场景完全对口。任务是把每张图分到农田、森林、河流等十类之一，直接对应土地监测、灾害评估这些应用。数据上我们做了严格的三分划分：训练一万六、验证和测试各五千四，三份完全互斥——这一步我们专门修过一个测试集泄漏的 bug，现在所有数字都是干净的。预处理上训练集做了翻转和旋转增广加 ImageNet 归一化，验证测试只归一化保证可复现。"

> ⏱ **时间提示**：插入此页后全篇变为 13 页 / ≈10.5 分钟；建议从第 7 页工具链细节匀出约 35 秒，或将其作为机动页（评委问数据时切出）。

---

## 第 3 页 · 硬件平台 Gazelle——逆向分析（1:15–2:00，约 45 秒）【平台使用 5 分】

**核心信息**：我们不是"拿来就用"，而是**逆向吃透硬件**——关键结论：硬件近乎理想（线性度 99.4%），量化与噪声是唯一精度瓶颈，这决定了后续所有策略。

**版面内容**
- 物理 tile **8×2**（k=8, n=2）→ 卷积 im2col 展平长度须被 **8 整除**
- 原生精度 **8a8w12o**：8-bit 激活 / 8-bit 权重 / 12-bit 输出 → 硬件原生支持 int8，无需退守 int4
- 关键物理量（逆向自 `GAZELLE_ARCHITECTURE.md` + `calibration_params.json`）：
  - 硬件线性度 **99.4%**（相对误差 0.6%，MAE 4.18 LSB，2000 组随机 GEMM 实测）
  - DAC ENOB **7.5 bits**、TIA 噪声 σ≈**5.34e-4**（MSE 2.85e-7）、ADC LSB 0.001465
  - 单次 GEMM 延迟约 **16.6 µs**（`entrance.gazelle_latency`）
- **逆向方法**：从模型目录名解码 `8X2_8a8w12o_dacenob7.5_power0.015_noise9e-11_..._std5.31`，并用 `pycdc` 反编译 11 个 `.pyc` + 读取 `behavioral_char.json` 校准
- **关键判断**：硬件几乎理想 → **所有精度损失来自量化与噪声建模，而非硬件缺陷**

**图表/视觉**（🆕 建议新建，见附录 B-图2）
- 上：8×2 光学 tile 结构示意（输入光 → DAC 调制 → 光学 MAC 阵列 → TIA 探测 → ADC）
- 下：硬件参数表（含"逆向来源"一列，体现工程深度）

**口述讲稿**
> "我们第一步不是写模型，而是逆向吃透 Gazelle 硬件。它的物理计算单元是 8×2 tile，要求卷积展平后被 8 整除；原生支持 int8 权重。我们甚至从芯片的模型目录名里解码出了 DAC 精度、噪声等级这些参数，并用反编译和校准文件做了交叉验证。最关键的发现是——硬件线性度高达 99.4%，相对误差只有 0.6%，几乎理想。这意味着所有精度损失都来自量化，而不是硬件缺陷。这个判断，决定了我们后续全部的训练策略。"

---

## 第 4 页 · 三模型设计——硬件感知架构（2:00–3:00，约 60 秒）【方案设计 / 创新】

**核心信息**：Model 1 是大而精的通用基线（算力 150×，不可在轨）；Model 2/3 为 8×2 tile **量身定制**，用约 1/9 参数换约 1/150 算力。

**版面内容**
- **Model 1 Baseline VGG**：6×Conv(3×3, 通道 32/32/64/64/128/128) + 2×FC(8192→256→10)，**2.39M 参数，156.6M MOPs/张**。其首层 conv1_1 展平长度=27（3×3×3），对齐率 84.4%；**整体对齐率仍达 99.8%**——它的硬伤不是对齐，而是**算力体积**（是 M2/3 的 150×），在轨不可行。
- **Model 2 Optic-SpaceNet V1**：硬件对齐设计（stem 1×1 → stage1/2 用 **2×2 卷积** 让展平长度=32/64，完美被 8 整除；stage3 1×1），**268K 参数，1.05M MOPs**。
- **Model 3 Optic-SpaceNet V2**：与 M2 同架构 + ResNet-18 知识蒸馏（教师 97.83%，T=4.0，α=0.7）。
- 设计哲学：全层 **bias=False** 匹配光硬件（不支持 bias 加法）、BN 保留 FP32；用 2×2 卷积让每一层展平长度精确对齐 8×2 tile。

**图表/视觉**（已有数据，直接做表）
- 三模型架构对比表：

| 维度 | Model 1（VGG） | Model 2（SpaceNet V1） | Model 3（SpaceNet V2） |
|---|---|---|---|
| 结构 | 6×Conv(3×3)+2×FC | 4×Conv(1×1/2×2)+2×FC | 同 Model 2 |
| 参数量 | **2.39M** | **268K** | **268K** |
| 综合对齐率 | 99.8% | 99.6% | 99.6% |
| 首层展平/对齐 | 27 / 84.4% | 3 / 37.5%→留电计算 | 3 / 37.5%→留电计算 |
| 总 MOPs/张 | **156.6M** | **1.05M** | **1.05M** |
| 训练方式 | 标准分类 | 标准分类 | 标准分类 + KD |

**口述讲稿**
> "我们设计了三个模型做对比。Model 1 是标准 VGG，精度高，但它有 239 万参数、每张图 1.56 亿次运算，是另外两个模型的 150 倍——在卫星上根本跑不动。需要澄清一点：Model 1 的硬件对齐率其实也很高（99.8%），它的硬伤不是对齐，而是纯粹的算力体积。Model 2、3 则是为硬件量身定制的：我们用 2×2 卷积让每一层的展平长度都完美被 8 整除，只用 26.8 万参数、105 万次运算。Model 3 在此基础上加了知识蒸馏。核心思路是：用九分之一的参数，换一百五十分之一的算力。"

---

## 第 5 页 · 方法演进路线——六阶段量化迭代（3:00–4:00，约 60 秒）【AI 工具链 / 技术复杂度】

**核心信息**：量化迁移我们迭代了 **6 个阶段**，从"全员崩盘"到"反超 FP32"——核心方法论是**硬件先于模型**。

**版面内容**（六阶段演进，三模型精度对照）
- **Phase 1** QAT 微调（FP32→低比特）：**全员失败**（M1 −11.3%、M2 −16.5%、M3 −18.2%）→ 根因：FP32 权重依赖精细精度，突变量化破坏特征，低 lr 逃不出坏局部最小
- **Phase 2** 从零 QAT：有效但仍差 6–9%（动态 scale 不稳、首末层信息损失）
- **Phase 4** STE + 噪声注入：M1 int4 达 **96.46%**（损失仅 0.71%）；期间发现并修复"QAT Conv 全关"bug
- **Phase 5** Mixed 精度（Conv int4 / Linear FP32）：**反超 FP32 基准**（M1 +1.09%）
- **Phase 6 定型配方**：**int8 权重 + Gazelle 物理噪声训练 + stem 首层 FP32**
  - int8 匹配硬件原生 8a8w → 量化噪声比 int4 低 **16 倍**
  - 训练噪声从 0.02×scale（拍脑袋）校准为匹配 DAC ENOB 7.5 的 **0.0016×scale**（低 12 倍）
- **方法论**：**硬件先于模型**——训练时就匹配硬件，而非训练后再迁就

**图表/视觉**（🆕 建议新建，见附录 B-图3，附完整数据）
- 📈 **精度演进折线图**：三模型准确率随 Phase 1→2→4→5→6 爬升，标注失败拐点与"反超 FP32"

**口述讲稿**
> "量化这条路我们迭代了六个阶段。最早想用 FP32 模型直接微调到低比特，三个模型全部崩盘，掉 11 到 18 个点——根因是 FP32 权重依赖精细精度，突变量化直接破坏了学到的特征。后来改成从零训练量化、引入 STE 和噪声注入，逐步把损失压到 1 个点以内。最终在第六阶段定型：用 int8 权重匹配硬件原生精度（量化噪声比 int4 低 16 倍），训练时注入 Gazelle 真实物理噪声（我们还把噪声强度从拍脑袋的值校准到了真实硬件水平），首层保留 FP32。这条演进线背后是一个核心方法论——硬件先于模型，训练阶段就让模型适应硬件。"

---

## 第 6 页 · 核心创新点（4:00–4:45，约 45 秒）【创新性 5 分】

**核心信息**：四大创新，最核心是"硬件匹配训练"——别人训练完再量化部署（精度必掉），我们训练时就匹配硬件（真机近无损）。

**版面内容**（四大创新，每点附**技术内核**——评委追问时可直接展开；STE/LSQ+ 细节见[第 6.5 页](#qat-depth)）
1. **硬件匹配训练**（最核心）：训练期注入 Gazelle 真实噪声（DAC ENOB 7.5 + TIA σ=5.34e-4）→ 真机精度 gap 仅 **~1.6 点**（业界常见 5–10 点）。
   - *技术内核*：把逆向出的硬件噪声链路（`Weight → DAC(7.5) → 光MAC → TIA(5.34e-4) → ADC`）整体建模进 QAT 前向，用 **STE** 让噪声梯度可传；并把噪声强度从拍脑袋的 `0.02×scale` 校准为匹配 DAC ENOB 的 **`0.0016×scale`**（降噪 12 倍）→ 训练分布 = 真机分布。
2. **stem 首层 FP32 电计算策略**：首层对齐率仅 37.5%，保留电计算——既高效，又消除 BN 训练/推理分布偏移。
   - *技术内核*：首层展平长度=3，硬走光计算要补零到 8（浪费 62.5%）；放电计算后，stem 的 BN 在训练/推理均为 FP32 → 统计量一致，消除量化对 BN 分布的破坏。
3. **int8 全流程对齐**：训练权重 = 硬件原生 8a8w，从根上避免 int4→int8 **网格不对齐**损失（scale `max/7` vs `max/127`，实测不可消除 ~6%）。
   - *技术内核*：int4 量化网格 16 级、int8 网格 256 级，二者的 scale 基不同 → int4 训出的权重在 int8 硬件上**落到错网格**；直接用 int8 训练（**LSQ+** 学最优 scale）从源头消除该损失，量化噪声比 int4 低 **16 倍**。
4. **硬件感知架构**：2×2 卷积让展平长度完美对齐 8×2 tile，综合对齐率 **99.6%**，补零浪费为 0。
   - *技术内核*：自研 2×2 卷积使每层 `C_in × k × k` 展平长度精确为 8 的倍数（stage1=32、stage2=64、fc1=1024…）→ im2col 后无需补零，光计算占比的有效 MOPs = 名义 MOPs。

> **训练端 ↔ 物理真机推理端 一致性断言（三项全对齐 → 真机近无损的根因）**：

| 维度 | 训练配置 | 推理（osim 真机）配置 | 一致 |
|---|---|---|---|
| 首层 | `first_conv_fp32=True` | `keep_first_conv_electronic=True` | ✓ |
| 权重位宽 | int8（8a8w，LSQ+ 学 scale） | int8（osim 原生 8a8w） | ✓ |
| 噪声模型 | Gazelle 噪声（DAC 7.5 + TIA） | osim 物理噪声 | ✓ |

> 差异化：多数队伍"训练完再量化部署"（精度必掉 5–10 点）；我们"训练时就匹配硬件"（三项对齐）——这是真机近无损的关键。

**图表/视觉**（🆕 建议新建，见附录 B-图4）
- 四宫格 icon 图（每个创新点一个图标 + 一句话 + 技术内核小字）；**训练↔推理一致性断言表**（首层 / 权重位宽 / 噪声 三项对勾 ✓，对应上表）
- STE/LSQ+ 量化机制示意（评委追问量化时切到第 6.5 页）

**口述讲稿**
> "我们的创新可以总结为四点。最核心的是硬件匹配训练——一般做法是训练完再量化部署，精度必然掉；我们在训练时就注入芯片的真实物理噪声，让模型提前适应，最终真机精度只比训练低 1.6 个点，远好于业界常见的 5 到 10 个点。具体做法是把逆向出的 DAC、TIA 噪声整体建模进训练，还把噪声强度校准到了真实硬件水平。第二，首层因为对齐率只有 37.5%，我们保留在电计算，既高效又消除了 BN 的训练推理偏移。第三，我们坚持 int8 全流程对齐——因为 int4 训练的权重放到 int8 硬件上会有不可消除的网格不对齐损失，大约 6 个点，所以我们直接用 int8 训练。第四是硬件感知的 2×2 卷积架构，对齐率 99.6%，补零浪费为 0。这四点落到一张表上就是训练端和真机端的首层、权重、噪声三项完全一致——这正是真机近无损的根因。"

---

<a id="qat-depth"></a>
## 第 6.5 页 · QAT 量化训练技术深入：STE / LSQ+ / 硬件噪声匹配（推荐插入页，约 60–75 秒）【AI 工具链 10 分 + 创新 5 分 + 技术深度】

> 与第 5 页（六阶段演进，讲"爬到多高"）互补：**本页讲"靠什么爬"**。可作主线深度页，也可作评委追问量化时的机动展开页。

**核心信息**：量化不是"训练完再砍精度"，而是**量化感知训练（QAT）**——训练时插入伪量化节点，让模型"在低精度下学会推理"。我们实现并对比了 **STE（直通估计器）** 与 **LSQ+（可学习步长）** 两套方案，并把 **Gazelle 真实物理噪声**焊进前向，最终落地"激活 STE 动态 scale + 权重 LSQ+ + 硬件噪声匹配"的混合配方。

**版面内容**

**(A) 为什么必须 QAT（PTQ → QAT 的跨越）**
- **PTQ（训练后量化）**：FP32 训完直接截断到低比特 → 模型从未见过量化误差 → 精度崩盘（Phase 1 实测 M1 −11.3%、M2 −16.5%）。
- **QAT（量化感知训练）**：训练前向里插入"伪量化（Fake Quant）"——`float → 量化 → 反量化回 float`，模型全程在"带量化噪声"的权重/激活上更新 → 推理换真量化时几乎无损。
- **核心难点**：`round()` 与 `clamp()` 几乎处处不可导（梯度为 0）→ 直接反传梯度全死 → 必须用 **STE / LSQ+** 给量化节点"造"出梯度。

**(B) STE — Straight-Through Estimator（直通估计器）**
- **前向**（真实量化）：`scale = abs_max / qmax`（per-channel，每步动态算）；`x_int = round(x/scale).clamp(qmin, qmax)`；`x_dq = x_int · scale`
- **反向**（STE 近似）：假装量化节点是恒等映射，`∂L/∂x ≈ ∂L/∂x_dq`（梯度直通，截断区外置 0）
- **一行实现**：`x + (x_dq − x).detach()` —— `.detach()` 切断 `x_dq` 的反传路径：**前向值 = `x_dq`、梯度却走 `x`**
- **特点**：简单稳定、零额外参数；但 `scale` 不可学习，无法优化量化范围
- **出处**：Bengio 2013；Jacob et al., CVPR 2018（整数推理 QAT）

**(C) LSQ+ — Learned Step Size Quantization+（可学习步长 + 零点）**
- 在 STE 基础上，把 `scale` 和 `zero_point` 都变成**可学习参数**（LSQ+ 相对 LSQ 增加了非对称 `zero_point`，即名字里的" + "）
- **前向**：`x_q = round(x/scale + zp).clamp(qmin, qmax)`；`x_dq = (x_q − zp) · scale`
- **反向**（`x` 仍 STE；`scale`/`zp` 用 LSQ 公式真实求导）：
  - 对 `x`：`∂L/∂x = 𝟙[qmin<x_int<qmax] · ∂L/∂x_dq`（截断外为 0）
  - 对 `scale`：`∂L/∂scale = [inner·(x_int − zp − x/scale) + outer·(qmax if x>0 else qmin)] · ∂L/∂x_dq`，再按 **`1/√(N·n_levels)`** 缩放（消除张量规模敏感性，N=每个 scale 负责的元素数）
  - 对 `zp`：`∂L/∂zp = −(x_int − zp)·inner · ∂L/∂x_dq / √(N·n_levels)`
- **工程要点**：`scale/zp` 配 **`0.1× base_lr` 独立学习率**（学得慢、不抢权重梯度）；从权重统计 `abs_max/qmax` 初始化（避免 1.0 死初始化）；权重 per-output-channel、激活 per-input-channel
- **出处**：Esser et al., ICLR 2020

**(D) STE vs LSQ+ 对比 + 我们的混合策略**

| 维度 | STE（直通估计） | LSQ+（可学习步长） |
|---|---|---|
| scale 来源 | 动态 `abs_max/qmax`（每步） | 可学习参数（统计值初始化） |
| zero_point | 无（对称量化） | 可学习（非对称，LSQ+ 的" + "） |
| 量化参数梯度 | 无（纯 STE 恒等） | LSQ 公式，`1/√(N·n_levels)` 缩放 |
| 学习率 | 与权重同 | 独立 `0.1× base_lr` |
| 本项目实测 | **M1 int4 96.46%**（Phase 4 STE+噪声，主选） | M2 int8 **92.80%** val（Phase 6 修复后，并行方案） |
| 最适合量化对象 | 激活（分布每步在变，动态更稳） | 权重（变化慢，可学全局最优量化范围） |

- **混合配方（最终落地）**：**激活用 STE 动态 scale + 权重用 LSQ+ 可学习 scale** —— 各取所长：激活分布训练中剧烈变化，动态 scale 更稳；权重变化慢，学一个最优量化范围收益最大。
- **诚实结论**：在 EuroSAT 这种中小规模任务上，STE+噪声已足够（简单稳定、精度最高）；LSQ+ 上限更高但对小模型略过参数化——**两者我们都实现、都验证、都报告**，最终按任务选型，而非"只会一种"。

**(E) 硬件噪声匹配训练（创新点 1 的技术内核，对应 `GazelleNoiseInjector`）**
- **噪声链路**（逆向自硬件，写入训练前向）：
  - 权重端：DAC 量化噪声 `std = 1/(2^7.5 · √12) · scale`（匹配 ENOB 7.5，等效 ~181 量化级）
  - 激活端：TIA 加性高斯 `σ=5.34e-4`（MSE 2.85e-7）+ ADC 量化噪声 `0.00147/√12`
- **关键校准**：噪声强度从"拍脑袋"的 `0.02×scale`（Phase 3/4）→ 实测匹配 DAC ENOB 的 **`0.0016×scale`**（Phase 6，**降噪 12 倍**）—— 过强噪声糊掉特征、过弱噪声真机会抖，只有按真实硬件标定才能做到"训练分布 = 推理分布"。
- **效果**：真机 gap 收敛到 **~1.6 点**（M2 −1.63% / M3 −1.55%），远优于业界 5–10 点。

**图表/视觉**（🆕 建议新建，见附录 B-图9）
- 左：**STE vs LSQ+ 前向/反向对比示意**（两条小流程：前向 `round/clamp` 相同；反向 STE 画一根恒等直通箭头，LSQ+ 在 `scale/zp` 上画真实梯度箭头 + `1/√(N·n)` 标注）
- 右：**硬件噪声链路图**（`Weight → DAC(7.5) → 光MAC → TIA(5.34e-4) → ADC`，每节点标噪声源）+ 噪声校准小柱（`0.02 → 0.0016`，×12）

**口述讲稿**
> "量化这块我们下了相当深的功夫。先说为什么必须 QAT：如果训练完再直接把权重砍成低比特，模型从没见过量化误差，精度会直接崩十几个点——这是我们 Phase 1 踩过的坑。QAT 的做法是训练时就在前向插入伪量化，让模型一直在带量化噪声的数据上学习。
> 这里的技术难点是 round 和 clamp 不可导，梯度传不过去。我们实现了两种解法：第一种 STE 直通估计器，反向时假装量化是恒等映射，一行代码 x 加上量化误差的 detach 就实现了；第二种 LSQ+，把量化步长 scale 和零点都变成可学习参数，用 LSQ 公式给它们真实梯度，还按 1 除根号下 N 乘级数 做了缩放、配了独立的小学习率。一句话区别：STE 简单稳定、scale 不可学；LSQ+ 上限更高、能学最优量化范围。我们不是只会一种——两者都实现、都验证，最终选了激活用 STE 动态 scale、权重用 LSQ+ 的混合配方，各取所长。
> 更关键的是第三块——硬件噪声匹配。我们把逆向出的 DAC 7.5 位量化噪声、TIA 探测噪声都建模进训练前向，而且把噪声强度从拍脑袋的值校准到了真实硬件水平，降了 12 倍。正因为训练时见到的噪声和真机一致，最终真机精度只比训练低 1.6 个点。"

> 🎤 **评委追问预案（QAT 深度）**：
> - *"STE 不是真梯度，凭什么能收敛？"* → round/clamp 真实梯度几乎处处为 0，STE 用恒等近似提供"方向足够好"的梯度；配合噪声注入当正则，实测收敛稳定（M1 int4 96.46%）。
> - *"LSQ+ 为什么要独立学习率？"* → scale/zp 对 loss 极敏感，若与权重同 lr 会震荡；用 0.1× base_lr + `1/√(N·n_levels)` 缩放，让量化参数"学得慢但稳"。
> - *"0.0016 这个噪声比怎么来的？"* → 由 DAC ENOB=7.5 反推：等效量化级 2^7.5≈181，单级噪声 `1/(181·√12)≈0.0016`；不是拍脑袋，是硬件标定值。
> - *"为什么不直接用 PyTorch 的量化？"* → PyTorch 量化面向电子芯片；我们面向 8×2 光学 MAC，要处理 im2col 对齐、光学噪声、bias 不支持等光计算专属问题，STE/LSQ+ 是我们自己在光计算路径上重新实现的。

> ⏱ **时间提示**：约 60–75 秒。若主线时间紧，作为**评委追问 QAT/量化时的机动展开页**切入即可，不强制占主线。

---

## 第 7 页 · 训练与量化工具链——全闭环（4:45–5:30，约 45 秒）【AI 工具链 10 分】

**核心信息**：从数据到光计算部署，**五阶段全自动闭环**，且每阶段产出可追溯。

**版面内容**（完整闭环，含具体产出物）
- **① 数据**：EuroSAT 27000 张 → 干净三分 split（train 16200 / val 5400 / test 5400），统一数据源 `eurosat_split.py`（seed=42，强制断言三者互斥+覆盖全集）→ 产出：索引表
- **② 训练**：QAT（**STE + LSQ+ 两套**，机制详见[第 6.5 页](#qat-depth)），100 epoch，Warmup(5)+Cosine，AdamW，Gazelle 噪声注入，per-channel 量化 → 产出：`*.pth` 权重（共 25 个）
- **③ 量化**：int8 权重 + int8 激活，per-output-channel 量化 → 产出：整数化权重
- **④ 编译**：QAT 权重 → osimulator 8a8w 原生编译路径 → 产出：光计算图
- **⑤ 部署**：`OpticConv2d` / `OpticLinear` 封装（im2col 展开 → 补零对齐 → 光学矩阵乘 → col2im 还原）→ 产出：真机推理结果
- 工程规模：**47 个 Python 文件**（core/qat/data/training/scripts 分层），25 个权重，修复 **11 个关键 bug**

**图表/视觉**（🆕 建议新建，见附录 B-图5）
- 🔄 工具链横向流程图（5 个阶段方框 + 箭头，标注每阶段产出物：索引表 → `.pth` → int8 权重 → 光计算图 → 真机结果）

**口述讲稿**
> "工具链是完整闭环。数据处理上我们用 EuroSAT 两万七千张图，做了严格的三分划分，统一数据源，并强制断言三份互斥——这一步后来被证明至关重要。训练采用量化感知训练 QAT，一百个 epoch，配合余弦退火和 Gazelle 噪声，量化成 int8 后，经过 osimulator 的原生编译，直接部署到光计算路径推理。整个链路从数据到光计算部署全自动打通，四十七个脚本、二十五个权重，全程可追溯。"

---

## 第 8 页 · 精度结果（5:30–6:30，约 60 秒）【功能及性能 10 分】

**核心信息**：三模型 int8 全部达标且**量化无损甚至反超 FP32**；更重要的是 osimulator 真机全量验证——Model 2 真机**反超原 FP32 基准 +0.28%**。

**版面内容**（数字均为 Bug #11 修复后的**干净 split**真值）
- 三模型 int8 干净 val：**M1 97.87/98.02% · M2 92.06% · M3 91.83%**，全部达成 int8 目标（≥96/≥90/≥91%）
- **量化无损甚至反超**：M2 vs FP32 基准 **+1.91%**，M3 **+0.39%**，M1 **+0.70%/+0.85%**
- **osimulator 真机全量 5400 张**：M2 **90.43%** · M3 **90.28%**
- **Model 2 真机反超原 FP32 基准 +0.28%**（90.43% vs 90.15%）——量化模型在真实光子硬件上真正可用

**图表/视觉**（🆕 建议新建，见附录 B-图6）
- 📊 **精度对比柱状图**：每模型三根柱 = FP32 基准 / int8 val / osim 真机（全量）
- 或复用 `docs/figures/compute_vs_accuracy_final.png`（见附录 A 注意事项，需更新数据）

**口述讲稿**
> "这是精度结果。三个模型在干净验证集上 int8 精度分别达到 98%、92%、92%，全部超过目标，而且量化几乎无损——Model 2 甚至比 FP32 基准还高近两个点。更重要的是真机验证：在 osimulator 真实光计算硬件上跑完全部 5400 张测试图，Model 2 达到 90.43%，反超了原始 FP32 模型。请注意，这里所有数字都是我们主动修复数据泄漏 bug 之后、干净重训复测的真值。这说明我们的量化模型在真实光子硬件上是真正可用的。"

---

## 第 9 页 · 光计算占比（重点）（6:30–7:20，约 50 秒）【⭐ 基础要求 20 分——最大单项】

**核心信息**：光计算占比 **90.65%**，超 90% 部署阈值；靠的是 2×2 卷积让每层完美对齐 8×2 tile，**补零浪费为 0**，6 层中 5 层走光计算。

**版面内容**（含逐层 MOPs，这是评委最可能追问的"怎么算出来的"）
- **光计算占比**：Model 1 (A) **97.74%** · Model 2 **90.65%** · Model 3 **90.65%**（均 ≥ 90% 部署阈值）
- **算法口径**：光计算占比 = 光计算层有效 MOPs / (光计算 + 电计算 MOPs)，**已含补零对齐开销**（非名义值）
- **逐层分布（Model 2/3，单张 64×64）**：

| 层      | 类型              | 展平长度 | 对齐率       | MOPs       | 占比        | 位置                      |
| ------ | --------------- | ---- | --------- | ---------- | --------- | ----------------------- |
| stem   | Conv 3→8, 1×1   | 3    | 37.5%     | 0.098M     | 9.3%      | **电计算（FP32）**           |
| stage1 | Conv 8→16, 2×2  | 32   | 100%      | **0.524M** | **49.9%** | 光计算 int8                |
| stage2 | Conv 16→32, 2×2 | 64   | 100%      | 0.131M     | 12.5%     | 光计算 int8                |
| stage3 | Conv 32→16, 1×1 | 32   | 100%      | 0.033M     | 3.1%      | 光计算 int8                |
| fc1    | Linear 1024→256 | 1024 | 100%      | 0.262M     | 24.9%     | 光计算 int8                |
| fc2    | Linear 256→10   | 256  | 100%      | 0.003M     | 0.2%      | 光计算 int8                |
| **合计** |                 |      | **99.6%** | **1.051M** | 100%      | 光计算 0.953M / 电计算 0.098M |

- 6 层中 **5 层走光计算**，仅 stem 首层电计算；光计算层展平长度全部为 8 的倍数 → **零补零浪费**

**图表/视觉**（🆕 建议新建，见附录 B-图7）
- 🥧 光计算占比**甜甜圈图**（90.65% 光 / 9.35% 电）+ 逐层 MOPs **堆叠条形图**（光/电拆分）

**口述讲稿**
> "这一页是评分最看重的光计算占比。我们三个模型分别是 97.7%、90.65%、90.65%，全部超过 90% 的部署阈值。怎么算的？按每层乘加运算量加权：光计算层的运算量除以总运算量。能做到这么高，是因为我们用 2×2 卷积让每一层的展平长度都完美被 8 整除，硬件对齐率 99.6%，几乎没有补零浪费——这是有效运算量，不是名义值。六层网络里五层完全在光计算上跑，只有对齐率只有 37.5% 的首层保留在电计算。也就是说，超过九成的乘加运算真正交给了光子完成。"

---

## 第 10 页 · 计算效率与在轨部署（7:20–8:20，约 60 秒）【实用性 5 分 + 功能性能】

**核心信息**：Model 2/3 落在"低算力高精度"的最优象限（左上角），是在轨轻量部署首选；Model 1 算力过高不可用。

**版面内容**
- **Model 2/3**：1.05M MOPs/张、268K 参数、全量 5400 张 **~3.7h 跑完**（13357s/13370s）→ **在轨轻量部署首选**
- **Model 1**：156M MOPs（**150×**）、单张 ~150s → 全量需 ~9 天，**算力/功耗过高，不适合卫星**（仅抽样验证）
- **真机统计**（Model 2/3）：引擎各调用 **27000 次**（5 光计算层 × 5400 张），总运算量 **5.15e+09 MACs**，对齐率 99.6%
- **部署推荐**：Model 2 v3 int8（真机 90.43%、反超 FP32、268K 轻量）

**图表/视觉**（✅ 已有核心图，**需更新数据**，见附录 A 注意事项）
- 📉 `docs/figures/compute_vs_accuracy_final.png`：横轴计算量（对数）、纵轴精度，三模型散点，Model 2/3 在左上角（低算力高精度）

**口述讲稿**
> "回到在轨场景的核心诉求——低功耗、低延迟。这张图横轴是计算量、纵轴是精度。Model 1 虽然精度最高，但计算量是 Model 2、3 的 150 倍，单张图要跑两分多钟，全量要九天，卫星上根本用不了。Model 2、3 只用 105 万运算量、26.8 万参数，全量 5400 张三个半小时跑完，却能达到 90% 以上精度，落点在图的左上角——低算力高精度，正是卫星在轨部署的最优解。"

---

## 第 11 页 · 鲁棒性、真机验证与科学诚信（8:20–9:10，约 50 秒）【功能性能 / 进阶】

**核心信息**：两级验证体系（秒级 QAT 交叉验证 + 小时级真机全量）；**主动发现并修复数据泄漏 bug**（科学诚信加分项）；并得到一个诚实发现——KD 真机未显著优于 M2。

**版面内容**
- **噪声鲁棒性**：训练时注入 Gazelle 噪声 → 真机 gap 仅 ~1.6 点（M2 −1.63% / M3 −1.55%，基本相等）
- **两级验证体系**：QAT 秒级交叉验证（`--qat`，不需 osim）+ osimulator 真机全量 5400（小时级）
- **科学诚信（亮点）**：主动发现并修复 **Bug #11**（test⊂train 100% 泄漏）——旧 osim 虚高至 93.28%/93.26%，已**全部作废**；干净重训 + 真机复测，现 test≈val（M2 差 0.14%），证明无泄漏、泛化良好
- **诚实发现**：KD（Model 3）真机 90.28% vs Model 2 90.43%，**统计打平**（差 0.15%，n=5400×2，z≈0.27 不显著）→ 真机鲁棒性**主要来自 Gazelle 噪声训练**，而非蒸馏

**图表/视觉**
- 📈 `docs/figures/noise_robustness_v2.png`（✅ 已有，噪声扫描曲线）
- 小表：真机 vs val gap（M2 −1.63% / M3 −1.55%，基本相等）

**口述讲稿**
> "为验证可靠性，我们做了两级验证：秒级的 QAT 交叉验证，加上小时级的 osimulator 真机全量测试。真机精度只比训练低 1.6 个点，说明我们的噪声匹配训练有效。这里特别讲一点科学诚信：我们主动发现并修复了一个严重 bug——测试集整段落在了训练集里，导致旧的真机数字虚高到 93%，我们已经把这些数字全部作废，重训复测，现在所有数字都是干净的。还有一个诚实的发现：加了知识蒸馏的 Model 3 真机并不比 Model 2 强，说明真机鲁棒性主要来自噪声训练，而不是蒸馏——我们不回避这个负结果。"

---

## 第 12 页 · 总结与展望（9:10–10:00，约 50 秒）【总结 / 进阶 20 分】

**核心信息**：三模型全部完成光计算迁移（光计算占比 ≥90%、真机 90%+、量化无损）；方法论"硬件先于模型"+ 真机验证闭环；展望光电混合提速与在轨实测。

**版面内容**
- **成果总览（一页纸 KPI）**：

| 指标 | Model 1 | Model 2 | Model 3 |
|---|---|---|---|
| 光计算占比 | 97.74% | **90.65%** | **90.65%** |
| 真机精度（全量5400） | 98/100%（q50 抽样） | **90.43%** | **90.28%** |
| 参数量 | 2.39M | 268K | 268K |
| 总 MOPs/张 | 156.6M | 1.05M | 1.05M |

- **进阶贡献**：① int8 全流程真机部署（osim 全量验证）② Gazelle 噪声匹配训练（近无损真机，gap ~1.6 点）③ 三模型 × 多量化方案完整消融 ④ 在轨遥感应用闭环
- **方法论**：硬件先于模型 → 硬件匹配训练 → 真机验证闭环
- **展望**：
  - **光电混合提速**（已搭脚本 `optic_inference_h{1,2,3}.py`）：H1 Conv光/Linear电（65.5%）、H2 砍最慢层（78.2%）、H3 极致切分（53.3%）——在保住 >50% 光计算下换吞吐
  - 更大模型补偿精度、**端到端硬件在环训练**（osim 真实输出作训练信号）、**真实卫星在轨验证**

**图表/视觉**
- 成果 KPI 总表 + 路线图（当前 → 光电混合 → 在轨验证）

**口述讲稿**
> "总结一下：我们把三种 CNN 全部迁移到光计算，光计算占比超过 90%，真机精度 90% 以上，量化几乎无损。进阶方面，我们实现了 int8 全流程真机部署、用噪声匹配训练做到近无损真机推理、并完成了三模型多量化的完整消融。方法论上始终坚持'硬件先于模型'。下一步我们计划做光电混合加速——已经搭好了三种切分方案的脚本——并推动真实卫星在轨验证。以上是我们的汇报，感谢各位评委，请提问。"

---

# 附录 A：配图总表

| 图表 | 用途 | 状态 | 制作方式 |
|---|---|---|---|
| 计算量-精度散点图 | 第 10 页核心图 | ⚠️ **已有但需更新数据** | 见下方"注意事项"，改 2 个数即可 |
| int4 噪声鲁棒性曲线 | 第 11 页 | ✅ 已有 `docs/figures/noise_robustness_v2.png` | 直接用 |
| 星上推理链路 + 电子/光子对比 | 第 2 页 | 🆕 新建 | 附录 B-图1（PPT 绘制 / AI 配图） |
| 8×2 tile 结构 + 硬件参数表 | 第 3 页 | 🆕 新建 | 附录 B-图2（PPT 绘制） |
| 精度演进折线图（Phase 1→6） | 第 5 页 | 🆕 新建 | 附录 B-图3（matplotlib，**含数据+代码**） |
| 四宫格创新 / 一致性 checklist | 第 6 页 | 🆕 新建 | 附录 B-图4（PPT 绘制） |
| 工具链五阶段流程图 | 第 7 页 | 🆕 新建 | 附录 B-图5（PPT / draw.io 绘制） |
| 三模型精度对比柱状图 | 第 8 页 | 🆕 新建 | 附录 B-图6（matplotlib，**含数据+代码**） |
| 光计算占比甜甜圈 + 逐层堆叠条 | 第 9 页 | 🆕 新建 | 附录 B-图7（matplotlib，**含数据+代码**） |
| EuroSAT 10 类样例拼图 | 第 2.5 页 | 🆕 新建 | 附录 B-图8（用自有数据，**含脚本**） |
| STE vs LSQ+ 前向/反向对比示意 | 第 6.5 页 | 🆕 新建 | 附录 B-图9（PPT 绘制，**含数据表**） |
| 硬件噪声链路图 + 噪声校准小柱 | 第 6.5 页 | 🆕 新建 | 附录 B-图9（PPT 绘制） |

### ⚠️ 重要：现有散点图需更新（ credibility 关键）
`docs/figures/compute_vs_accuracy_final.png` 当前的 osim 点用的是 **q500 抽样**值（M2=89.00%、M3=90.80%），与讲稿/正文引用的**全量 5400 真值**（M2=90.43%、M3=90.28%）不一致——评委若对照会质疑数据自洽。
**修复（改 `src/scripts/plot_compute_vs_accuracy.py` 两行即可）**：
```python
# Model 2: 'y_osim': 89.00  →  90.43
'y_osim': 90.43,
# Model 3: 'y_osim': 90.80  →  90.28
'y_osim': 90.28,
```
并把页脚 `"Model 2/3 n=500"` 改为 `"Model 2/3 n=5400 (full test set)"`，重跑 `python src/scripts/plot_compute_vs_accuracy.py`。

---

# 附录 B：配图提示词与数据

> **全局视觉规范**（保持与现有 `compute_vs_accuracy_final.png` 一体化）：
> - 配色用 Okabe-Ito 色盲友好色板：Model 2 橙 `#E69F00`、Model 3 绿 `#009E73`、Model 1 深蓝 `#0072B2`、Model 1-B 浅蓝 `#56B4E9`；电计算用灰 `#999999`。
> - 字体 DejaVu Sans / Arial；网格虚线 `#999999` alpha 0.25；dpi ≥ 300。
> - 数据图统一输出到 `docs/figures/`，存 `.png + .pdf`（论文/打印用矢量）。
> - 下方 4 个 matplotlib 脚本均已配置中文字体（Microsoft YaHei），在 Windows 上可直接运行：`python src/scripts/<脚本名>.py`。

---

### 图1 · 星上推理链路 + 电子/光子对比（第 2 页）

- **推荐方式**：PPT 原生绘制（矢量、可控）。
- **提示词（AI 配图备选）**：`"A clean flat-style infographic, left side: a satellite in orbit beaming down to a photonic chip processing Earth remote-sensing images and outputting land-cover class labels; right side: a two-column comparison table 'Electronic vs Photonic' with rows Power / Parallelism / Radiation / Weight. Minimal, technical, blue and orange accent, white background, no text garbling."`
- **电子 vs 光子对比数据**：

| 维度 | 电子计算（GPU/ASIC） | 光子计算（Gazelle 核心） |
|---|---|---|
| 功耗/散热 | 高（数百瓦）；太空仅能辐射散热 | 低、发热小 |
| 并行度 | 受限（随规模 O(N²)↑） | 光速并行矩阵乘 |
| 抗辐射 | 差（易 SEU，需冗余/加固） | 模拟量计算、无数字逻辑 → **核心免疫 SEU**（外围仍需加固） |
| 寿命 | 栅氧击穿/阈值漂移，5–10 年 | 无电学老化，核心寿命有望 >10 年 |

---

### 图2 · 8×2 光学 tile 结构 + 硬件参数表（第 3 页）

- **推荐方式**：PPT 绘制结构示意图 + 参数表。
- **提示词**：`"Schematic of an 8×2 optical compute tile: input light → DAC modulator (ENOB 7.5) → 8×2 photonic MAC array (interferometric/diffraction) → TIA detector (σ=5.3e-4) → ADC (12-bit). Horizontal signal flow, labeled stages, clean line art, white background."`
- **硬件参数表数据（含逆向来源）**：

| 参数 | 值 | 逆向来源 |
|---|---|---|
| 物理 tile | 8×2 (k=8, n=2) | 模型目录名 + `calibration_params.json` |
| 原生精度 | 8a8w12o | 模型目录名 `8X2_8a8w12o_...` |
| 线性度 | 99.4%（rel err 0.6%） | `behavioral_char.json`（2000 组 GEMM） |
| DAC ENOB | 7.5 bits | 目录名 `dacenob7.5` |
| TIA 噪声 σ | ≈5.34e-4 (MSE 2.85e-7) | `calibration_params.json` |
| ADC LSB | 0.001465 | `calibration_params.json` |
| 单次 GEMM 延迟 | ~16.6 µs | `entrance.gazelle_latency` |

---

### 图3 · 精度演进折线图（第 5 页）— matplotlib 可直接生成

- **数据（Phase 1→2→4→5→6 的干净/可对比 int 精度，%）**：

| Phase | Model 1 | Model 2 | Model 3 |
|---|---|---|---|
| P1 微调（int4） | 85.91 | 73.63 | 73.22 |
| P2 从零（int4） | 91.17 | 81.20 | 83.26 |
| P4 STE+噪声（int4） | 96.46 | 74.35* | 78.26* |
| P5 Mixed | 98.26 | 91.26 | 91.13 |
| P6 Gazelle int8 | 97.87 | 92.06 | 91.83 |
| FP32 基准（参考线） | 97.17 | 90.15 | 91.44 |

> *M2/M3 的 P4 因"QAT Conv 全关"bug 异常偏低（bug 修复后即恢复），图中可加注"P4 bug"。

- **提示词**：`"Line chart, accuracy vs quantization phase (P1→P6), 3 colored lines (orange/green/blue) for 3 models, dashed horizontal reference lines for each FP32 baseline, annotate the P1 failure dip and the P5/P6 breakthrough above FP32. Chinese labels, Okabe-Ito colors."`

<details>
<summary>📄 点击展开：ready-to-run matplotlib 代码（保存为 <code>src/scripts/plot_phase_evolution.py</code>）</summary>

```python
# -*- coding: utf-8 -*-
"""精度演进折线图（Phase 1→6）— 存 docs/figures/phase_evolution.{png,pdf}"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['grid.linestyle'] = '--'

phases = ['P1\n微调', 'P2\n从零QAT', 'P4\nSTE+噪声', 'P5\nMixed', 'P6\nGazelle int8']
x = np.arange(len(phases))
data = {
    'Model 1 (VGG, 2.39M)': {'c': '#0072B2', 'y': [85.91, 91.17, 96.46, 98.26, 97.87], 'fp32': 97.17},
    'Model 2 (SpaceNet V1, 268K)': {'c': '#E69F00', 'y': [73.63, 81.20, 74.35, 91.26, 92.06], 'fp32': 90.15},
    'Model 3 (SpaceNet V2+KD, 268K)': {'c': '#009E73', 'y': [73.22, 83.26, 78.26, 91.13, 91.83], 'fp32': 91.44},
}
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
for name, d in data.items():
    ax.plot(x, d['y'], '-o', color=d['c'], lw=2.2, ms=8, label=name)
    ax.axhline(d['fp32'], color=d['c'], ls=':', lw=1.2, alpha=0.6)
ax.set_xticks(x); ax.set_xticklabels(phases)
ax.set_ylabel('Top-1 准确率 (%)', fontsize=12)
ax.set_xlabel('量化迭代阶段', fontsize=12)
ax.set_title('量化迁移精度演进：从全员崩盘（P1）到反超 FP32（P5/P6）', fontsize=13, fontweight='bold')
ax.annotate('P1 微调全员失败\n(−11~−18%)', xy=(0, 78), fontsize=9, color='#555',
            ha='center', bbox=dict(boxstyle='round,pad=0.3', fc='#FFF3CD', ec='#EEE', alpha=0.9))
ax.annotate('P4 M2/M3\nQAT-Conv 全关 bug', xy=(2, 74.35), fontsize=8, color='#555', ha='center')
ax.annotate('反超 FP32', xy=(4, 92.06), xytext=(3.1, 95.5), fontsize=9, color='#0072B2',
            arrowprops=dict(arrowstyle='->', color='#0072B2'))
ax.set_ylim(70, 100)
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
out = 'docs/figures/phase_evolution'
plt.savefig(f'{out}.png', dpi=300); plt.savefig(f'{out}.pdf')
print(f'saved: {out}.png / .pdf')
```
</details>

---

### 图4 · 四宫格创新 / 训练-推理一致性 checklist（第 6 页）

- **推荐方式**：PPT 绘制四宫格，每格一个创新点。
- **提示词**：`"A 2×2 grid of icon cards: (1) chip-with-noise-waves 'hardware-matched training'; (2) lightning-bolt-first-layer 'stem FP32 electronic'; (3) aligned-grid 'int8 end-to-end alignment'; (4) puzzle-piece-fits-tile 'hardware-aware 2×2 architecture'. Flat icons, blue/green/orange, white background."`
- **一致性 checklist 数据（训练 ↔ 推理三项对勾）**：

| 维度 | 训练配置 | 推理（osim）配置 | 一致 |
|---|---|---|---|
| 首层 | `first_conv_fp32=True` | `keep_first_conv_electronic=True` | ✓ |
| 权重 | int8（8a8w） | int8（osim 原生） | ✓ |
| 噪声 | Gazelle 噪声（DAC 7.5 + TIA） | osim 物理噪声 | ✓ |

---

### 图5 · 工具链五阶段流程图（第 7 页）

- **推荐方式**：PPT / draw.io 矢量绘制。
- **布局规格**（五个方框横向排列，箭头连接，每框标注产出物）：
```
① 数据            ② 训练            ③ 量化           ④ 编译            ⑤ 部署
EuroSAT 27000  →  QAT(STE/LSQ+)  →  int8 权重     →  osim 8a8w     →  OpticConv2d/Linear
三分 split        100 ep+Gazelle   per-channel      原生编译          im2col→光MAC→col2im
[索引表]          [.pth×25]        [整数权重]        [光计算图]        [真机结果]
```
- **提示词**：`"Horizontal 5-stage pipeline flowchart with arrows: Data → Training → Quantization → Compile → Deploy, each box labeled with its artifact, blue gradient boxes, clean infographic."`

---

### 图6 · 三模型精度对比柱状图（第 8 页）— matplotlib 可直接生成

- **数据（FP32 基准 / int8 val 干净 / osim 真机，%）**：

| 模型 | FP32 基准 | int8 val（干净） | osim 真机 |
|---|---|---|---|
| Model 1（A） | 97.17 | 97.87 | 98.00（q50 抽样）|
| Model 2 | 90.15 | 92.06 | **90.43（全量5400）** |
| Model 3 | 91.44 | 91.83 | **90.28（全量5400）** |

- **提示词**：`"Grouped bar chart, 3 model groups × 3 bars each (FP32 baseline / int8 val / osim real-hardware), highlight M2 osim bar that exceeds its FP32 baseline, Chinese labels, Okabe-Ito colors."`

<details>
<summary>📄 点击展开：ready-to-run matplotlib 代码（保存为 <code>src/scripts/plot_accuracy_bars.py</code>）</summary>

```python
# -*- coding: utf-8 -*-
"""三模型精度对比柱状图 — 存 docs/figures/accuracy_bars.{png,pdf}"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

models = ['Model 1 (A)', 'Model 2', 'Model 3']
fp32  = [97.17, 90.15, 91.44]
val8  = [97.87, 92.06, 91.83]
osim  = [98.00, 90.43, 90.28]
osim_note = ['(q50抽样)', '(全量5400)', '(全量5400)']

x = np.arange(len(models)); w = 0.26
fig, ax = plt.subplots(figsize=(9.5, 6), dpi=300)
b1 = ax.bar(x - w, fp32, w, label='FP32 基准', color='#999999', edgecolor='#666')
b2 = ax.bar(x,     val8, w, label='int8 val（干净）', color='#56B4E9', edgecolor='#2a7ab8')
b3 = ax.bar(x + w, osim, w, label='osim 真机', color='#E69F00', edgecolor='#b87700')

for bars, vals in [(b1, fp32), (b2, val8), (b3, osim)]:
    for r, v in zip(bars, vals):
        ax.text(r.get_x()+r.get_width()/2, v+0.12, f'{v:.2f}', ha='center', fontsize=8.5)
for i, n in enumerate(osim_note):
    ax.text(x[i]+w, osim[i]-1.1, n, ha='center', fontsize=7.5, color='#b87700')

ax.axhline(90.15, xmin=0.55, xmax=0.72, color='#0072B2', ls='--', lw=1)
ax.annotate('M2 真机反超 FP32 +0.28%', xy=(1+w, 90.43), xytext=(1.6, 95.2),
            fontsize=9, color='#0072B2', arrowprops=dict(arrowstyle='->', color='#0072B2'))
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel('Top-1 准确率 (%)', fontsize=12)
ax.set_ylim(86, 100)
ax.set_title('三模型精度对比：FP32 → int8 → 光计算真机', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
out = 'docs/figures/accuracy_bars'
plt.savefig(f'{out}.png', dpi=300); plt.savefig(f'{out}.pdf')
print(f'saved: {out}.png / .pdf')
```
</details>

---

### 图7 · 光计算占比甜甜圈 + 逐层 MOPs 堆叠条（第 9 页，⭐ 核心图）— matplotlib 可直接生成

- **数据（Model 2/3 逐层 MOPs，M）**：

| 层 | 光计算 MOPs | 电计算 MOPs |
|---|---|---|
| stem（1×1, 电） | 0 | 0.098 |
| stage1（2×2） | 0.524 | 0 |
| stage2（2×2） | 0.131 | 0 |
| stage3（1×1） | 0.033 | 0 |
| fc1（Linear） | 0.262 | 0 |
| fc2（Linear） | 0.003 | 0 |

- **甜甜圈数据**：光计算 0.953M（90.65%）/ 电计算 0.098M（9.35%）。
- **提示词**：`"Two-panel figure: left donut chart 'Optical 90.65% / Electronic 9.35%'; right horizontal stacked bar chart per-layer MOPs (stem/stage1/stage2/stage3/fc1/fc2) split into optical (orange) vs electronic (gray), Chinese labels."`

<details>
<summary>📄 点击展开：ready-to-run matplotlib 代码（保存为 <code>src/scripts/plot_optical_ratio.py</code>）</summary>

```python
# -*- coding: utf-8 -*-
"""光计算占比甜甜圈 + 逐层MOPs堆叠条 — 存 docs/figures/optical_ratio.{png,pdf}"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

layers = ['stem\n(电,1×1)', 'stage1\n(2×2)', 'stage2\n(2×2)', 'stage3\n(1×1)', 'fc1\n(Linear)', 'fc2\n(Linear)']
opt = [0.0, 0.524, 0.131, 0.033, 0.262, 0.003]
ele = [0.098, 0.0, 0.0, 0.0, 0.0, 0.0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300,
                                gridspec_kw={'width_ratios': [1, 1.5]})

# 左：甜甜圈
ax1.pie([0.953, 0.098], labels=['光计算 90.65%', '电计算 9.35%'],
        colors=['#E69F00', '#999999'], startangle=90,
        wedgeprops=dict(width=0.42, edgecolor='w'),
        textprops=dict(fontsize=11, fontweight='bold'))
ax1.text(0, 0, '光计算占比\n90.65%', ha='center', va='center', fontsize=13, fontweight='bold')
ax1.set_title('Model 2/3 光计算占比', fontsize=12, fontweight='bold')

# 右：逐层堆叠条
y = np.arange(len(layers))
ax2.barh(y, opt, color='#E69F00', label='光计算 (int8)', edgecolor='w')
ax2.barh(y, ele, left=opt, color='#999999', label='电计算 (FP32)', edgecolor='w')
for i, (o, e) in enumerate(zip(opt, ele)):
    tot = o + e
    if tot > 0.04:
        ax2.text(tot+0.01, i, f'{tot:.3f}M', va='center', fontsize=8.5)
ax2.set_yticks(y); ax2.set_yticklabels(layers, fontsize=9)
ax2.invert_yaxis()
ax2.set_xlabel('MOPs / 张', fontsize=11)
ax2.set_title('逐层运算量分布（补零浪费 = 0）', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right', fontsize=9)
ax2.set_xlim(0, 0.66)
plt.tight_layout()
out = 'docs/figures/optical_ratio'
plt.savefig(f'{out}.png', dpi=300); plt.savefig(f'{out}.pdf')
print(f'saved: {out}.png / .pdf')
```
</details>

---

### 图8 · EuroSAT 10 类样例拼图（第 2.5 页）— matplotlib 可直接生成

- **数据**：直接读取 `data/EuroSAT_RGB/<类名>/`（每类随机抽 1 张，seed=42 可复现）。10 类见第 2.5 页表格。
- **提示词**：`"2×5 grid of 10 EuroSAT land-cover satellite image samples (AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake), each labeled with English+Chinese class name, 64×64 RGB, clean white background, suptitle."`

<details>
<summary>📄 点击展开：ready-to-run matplotlib 代码（保存为 <code>src/scripts/plot_eurosat_samples.py</code>）</summary>

```python
# -*- coding: utf-8 -*-
"""EuroSAT 10 类样例拼图 + 类别信息表 — 存 docs/figures/eurosat_samples.{png,pdf}
从仓库根目录运行: python src/scripts/plot_eurosat_samples.py"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import os, random

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA = 'data/EuroSAT_RGB'
display = {'AnnualCrop': 'Annual Crop', 'Forest': 'Forest',
           'HerbaceousVegetation': 'Herbaceous', 'Highway': 'Highway',
           'Industrial': 'Industrial', 'Pasture': 'Pasture',
           'PermanentCrop': 'Permanent Crop', 'Residential': 'Residential',
           'River': 'River', 'SeaLake': 'Sea Lake'}
folders = sorted(c for c in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, c)))
random.seed(42)

fig = plt.figure(figsize=(13, 9.6), dpi=300)
# 上：图片网格；下：信息表（top 留大，让 suptitle 与首行拉开距离）
gs = gridspec.GridSpec(2, 1, height_ratios=[2.7, 1.0], hspace=0.16,
                       left=0.045, right=0.955, top=0.875, bottom=0.075)
gs_img = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=gs[0], hspace=0.55, wspace=0.08)
for i, c in enumerate(folders):
    r, col = divmod(i, 5)
    ax = fig.add_subplot(gs_img[r, col])
    f = random.choice(sorted(os.listdir(os.path.join(DATA, c))))
    ax.imshow(Image.open(os.path.join(DATA, c, f)))
    ax.set_title(display.get(c, c), fontsize=12, pad=6)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('EuroSAT 数据集 · 10 类地物样例（Sentinel-2 卫星遥感，64×64 RGB）',
             fontsize=16, fontweight='bold', y=0.965)

# ---- 类别信息表 ----
ax_t = fig.add_subplot(gs[1]); ax_t.axis('off')
columns = ['类别名', '地物含义', '样本张数', '类别名', '地物含义', '样本张数']
rows = [
    ['Annual Crop', '一年生农田', '3,000', 'Pasture',       '开阔牧场',   '2,000'],
    ['Forest',      '森林/林地',  '3,000', 'Permanent Crop', '常年作物',   '2,500'],
    ['Herbaceous',  '草本植被',   '3,000', 'Residential',    '住宅建筑',   '3,000'],
    ['Highway',     '普通公路',   '2,500', 'River',          '天然河流',   '2,500'],
    ['Industrial',  '工业区厂房', '2,500', 'Sea Lake',       '海、淡水湖', '3,000'],
]
tbl = ax_t.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center',
                 colWidths=[0.17, 0.155, 0.115, 0.17, 0.155, 0.115])
tbl.auto_set_font_size(False); tbl.set_fontsize(11.5); tbl.scale(1, 1.75)
for c in range(6):                       # 表头：蓝底白字加粗
    tbl[(0, c)].set_facecolor('#0072B2'); tbl[(0, c)].set_text_props(color='white', fontweight='bold')
for r in range(1, 6):                    # 两组底色（左浅蓝 / 右浅绿）
    for c in range(3): tbl[(r, c)].set_facecolor('#EAF2FB')
    for c in range(3, 6): tbl[(r, c)].set_facecolor('#EAF7EE')
for cell in tbl.get_celld().values():    # 细边框
    cell.set_edgecolor('#CCCCCC'); cell.set_linewidth(0.8)
fig.text(0.5, 0.028, '总样本集大小：27,000 张　　　|　　　验证 / 测试集规模：各 5,400 张',
         ha='center', fontsize=13, fontweight='bold')

out = 'docs/figures/eurosat_samples'
plt.savefig(f'{out}.png', dpi=300); plt.savefig(f'{out}.pdf')
print(f'saved: {out}.png / .pdf')
```
</details>

---

### 图9 · STE vs LSQ+ 对比示意 + 硬件噪声链路（第 6.5 页）

> QAT 技术深度页配图。两图都建议 **PPT 原生绘制**（概念示意，矢量可控、可控公式排版）。

**图9-左 · STE vs LSQ+ 前向/反向对比示意**

- **布局**：上下两条横向流程，左列"前向"、右列"反向"。
  - 前向（两者相同）：`x (float)` → `scale = abs_max/qmax`（或可学习）→ `round` → `clamp(qmin,qmax)` → `x_dq`
  - 反向 STE：从 `x_dq` 画一根**恒等直通箭头**回到 `x`（标注 `∂L/∂x = ∂L/∂x_dq`，scale/zp 处画 ✗ "不可学"）
  - 反向 LSQ+：`x` 仍是 STE 箭头；额外在 `scale`、`zp` 上画**真实梯度箭头**（标注 `LSQ 公式`、`× 1/√(N·n_levels)`、`lr = 0.1× base`）
- **提示词**：`"Side-by-side schematic comparing two quantization methods. Top row STE: forward path x→scale→round→clamp→x_dq, backward drawn as a single identity pass-through arrow, scale marked 'not learnable'. Bottom row LSQ+: same forward, backward adds real gradient arrows into 'scale' and 'zero_point' params labeled 'LSQ grad, 1/sqrt(N*n_levels), lr=0.1x'. Clean technical line art, blue for STE, orange for LSQ+, white background."`

**图9-右 · 硬件噪声链路图 + 噪声校准小柱**

- **上半（噪声链路）**：横向信号流，每节点标注注入的噪声源——
  `Weight → [DAC ENOB 7.5, std=1/(2^7.5·√12)·scale] → 光学 MAC 阵列 → [TIA σ=5.34e-4] → [ADC, 0.00147/√12] → 输出`
- **下半（噪声校准小柱）**：两根柱对比训练噪声强度——`Phase 3/4 拍脑袋 0.02×scale` vs `Phase 6 硬件标定 0.0016×scale`，标注"**降噪 12×**"。
- **提示词**：`"Top: horizontal optical-compute signal flow with labeled noise sources at each stage — Weight → DAC (ENOB 7.5) → photonic MAC array → TIA (σ=5.34e-4) → ADC (0.00147). Bottom: two-bar comparison of training noise strength, 0.02 (guess) vs 0.0016 (hardware-calibrated), annotated '12x lower'. Technical infographic, orange accent, white background."`

**STE vs LSQ+ 对照数据（做表/做图均可直接用）**：

| 维度 | STE（直通估计） | LSQ+（可学习步长） |
|---|---|---|
| scale 来源 | 动态 `abs_max/qmax`（每步） | 可学习参数（统计值初始化） |
| zero_point | 无（对称） | 可学习（非对称） |
| 反向梯度 | STE 恒等，截断外为 0 | `x` 用 STE；`scale/zp` 用 LSQ 公式 |
| 梯度缩放 | 无 | `1/√(N·n_levels)` |
| 学习率 | 与权重同 | 独立 `0.1× base_lr` |
| 代码出处 | `src/qat/optic_qat.py::fake_int4_quantize`、`optic_qat_v4.py::fake_quantize_symmetric` | `src/qat/optic_qat_lsq.py::_LSQPlusFn` |
| 本项目实测 | M1 int4 **96.46%**（Phase 4，主选） | M2 int8 **92.80%** val（Phase 6 修复后） |
| 量化对象 | 激活（分布变化快） | 权重（变化慢） |

> 📌 **落地配方**：激活 STE 动态 scale + 权重 LSQ+ 可学习 scale + Gazelle 噪声匹配（`0.0016×scale`）+ 首层 FP32 + int8 全流程。

---

# 附录 C：应用价值纵深与高频质疑应对（答辩话术）

> 评委常从"**应用价值**"和"**现实可行性**"两个角度追问。下面把"空间 AI"定位与最可能被挑战的"能效比（TOPS/W）"问题整理成可直接背诵的应对话术。原则与正文一致：**先结论后数据、把局限说成边界、不自夸、不回避**。

## C1. 应用定位：面向空间智能卫星的"抗辐射 + 低功耗"光子加速器

把作品定位为**"面向空间智能卫星的抗辐射、低功耗光子神经网络加速器"**。技术组合 = **光子矩阵乘（抗辐射/低功耗）× INT8 量化 × 轻量 CNN（268K）× Gazelle 噪声鲁棒训练**，构成完整的边缘侧空间 AI 方案。

光计算天然契合空间 AI 的三大挑战：

| 挑战 | 传统电子芯片 | 光计算（核心） | 严谨边界 |
|---|---|---|---|
| 辐射（SEU/比特翻转） | 极易受宇宙射线撞击 → 0↔1 翻转，需冗余/加固 | **模拟量（光强/相位）计算，无数字逻辑 → 核心免疫 SEU** | 外围 DAC/ADC/FPGA 仍需常规加固 |
| 功耗/散热 | GPU 数百瓦，太空无空气仅能辐射散热 | 低功耗、发热小 | — |
| 寿命 | 栅氧击穿/阈值漂移，5–10 年 | 无电学老化，材料稳定下核心寿命有望 >10 年 | 长期辐射可能使波导吸收↑、探测器暗电流↑ |

> ⚠️ **避坑（严谨性）**：答辩时**不要**说"光计算完全免疫辐射"。准确表述：*"光计算核心对单粒子翻转不敏感，大幅降低系统抗辐射成本，但外围电子模块仍需常规加固。"* —— 客观且落地，反而加分。

## C2. 最可能被挑战的硬伤：能效比（TOPS/W）—— 攻防预案

**评委可能的发难**：评估板级光计算系统"系统功耗高、算力低"，能效比（TOPS/W）远低于 Jetson Orin（~1.6 TOPS/W）等 GPU，"上天有什么意义？"

**先懂本质再答**：
- 评估板（Proof of Concept）光阵列规模极小（本项目 8×2 tile），算力天然低；
- **"光电墙（I/O Wall）"**：系统功耗绝大部分（>99%）消耗在外围 **ADC/DAC、TIA、激光器温控、控制 FPGA** 的光电转换（O-E-O）上；真正"光穿过波导做矩阵乘"的功耗仅毫/微瓦级。

**三段式应对话术**：

**① 降维：商业级 vs 航天级的不对等**
> "Jetson Orin 的高能效是地面商业芯片。直接上天，宇宙射线极易导致死机或逻辑错误；要做航天级抗辐射加固，其体积、重量、功耗都会大幅膨胀，能效大打折扣。而光计算核心天然抗 SEU。评估空间 AI 芯片，不能只看绝对算力，更要看**'抗辐射成本下的综合能效'**。"

**② 承认现状 + 打"规模效应"牌（O(1) 复杂度）**
> "当前低能效源于第一代评估板光阵列规模小、加上高昂的光电转换开销。但关键差异在于：电子算 N×N 矩阵乘，功耗/时延随规模 O(N²)~O(N³) 增长；而**光计算无论 16×16 还是 1024×1024，光穿过芯片都是光速（皮秒级），计算本身功耗几乎不增（O(1)）**。随着硅光集成（如共封装光学 CPO）成熟、阵列扩到千级规模，光电转换开销被摊薄，理论能效可反超 GPU。我们做的是面向未来 5–10 年的前沿探索。"

**③ 特定任务优势**
> "通用计算上纯光方案目前打不过 GPU；但在**模拟域线性运算（矩阵乘、卷积、FFT、线性滤波）**上，光计算可省去部分 ADC、实现极低延迟，是 GPU 难以替代的赛道——而这恰好是 CNN 推理的核心算子，也是本项目的切入点。"

**一句话底线**：不在绝对算力/能效上与 GPU 硬刚（必输），把重心放在 **抗辐射特殊优势 + O(1) 规模潜力 + 线性任务低延迟**——立意既懂现实、又有前瞻性。

> 💡 可与正文 **第 3 页（硬件逆向）+ 第 10 页（计算效率）+ 第 11 页（真机验证）** 联动：把"评估板规模小"自圆其说为我们选择 **268K 轻量模型、追求高光计算占比而非绝对 TOPS** 的前提，化被动为主动。

---

## 讲解节奏提示

1. **高分项讲透**：第 3 页（平台使用）、第 9 页（光计算占比，20 分）务必充分，配逐层 MOPs 表。
2. **技术复杂度与创新**：第 5 页（六阶段演进）、第 6 页（四创新）、**第 6.5 页（QAT 深入：STE/LSQ+ + 噪声匹配）** 是体现深度的关键，用数据与"方法论"支撑。第 6.5 页可作评委追问量化时的机动展开页。
3. **数据说话**：第 8、10 页用图表；关键数字背熟——**光计算占比 90.65%、对齐率 99.6%、真机 90.43%/90.28%、MOPs 1.05M vs 156M（150×）、真机 gap ~1.6 点、参数 268K**。
4. **主动亮诚信**：第 11 页主动讲 Bug #11 数据泄漏与负结果（KD 未增益），把局限讲成科学发现。
5. **时间控制**：10 分钟内讲完 12 页，留 5 分钟答辩；每页核心信息若被追问，先给结论再给数据。
6. **数据自洽检查**：务必按附录 A 更新散点图，确保台上数字与图一致。

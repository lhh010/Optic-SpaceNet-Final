# Model 3 光计算演示 — 单栏滚动叙事版前端设计

> 2026-07-18 · 状态: 已确认待实施 · 关联: frontend-design.md(三栏版, 被本设计取代的交互部分), api.md(契约)
> 决策记录(与用户逐项确认): 保留深色光子视觉风格; 单栏滚动 7 屏; 一次推理+滚动逐步揭示;
> 选图即自动推理; stem 并入首屏; 数值比较=cosine+max abs+相对误差分桶直方图+理论上板耗时;
> 理论上板耗时按芯片官方吞吐 2.6 M int8 OP/s 估算(模拟器耗时不展示);
> 末屏=预测对比+指标汇总+takeaway; 圆点侧导航+顶部进度条; 方案 A(原生 IO+scroll-snap, 零新依赖)。

## 目标

把现有单屏三栏 Dashboard 改为**单栏滚动叙事(scrollytelling)**:
评委随滚动逐 stage 看到「模型结构 → 光|电 feature map → 数值一致性 → 理论上板耗时」,
最后落地到预测结果与结论。视觉语言(深空底/青光/金点缀/JetBrains Mono)完全保留。

## 技术栈与约束(不变)

- `demo/web/`: `index.html` + `app.js` + `vendor/`; 无构建步骤, FastAPI 静态直出。
- **零新依赖、零外网请求**; 动画全部用原生 IntersectionObserver + CSS transition/keyframes + CSS scroll-snap。
- 目标 1080p 投影, 最小适配宽度 1280px, 字号偏大。

## 页面结构(7 屏, 100vh section + scroll-snap proximity)

```
屏1  首屏     Header(标题+chips+health 灯) + 输入卡(预览/label/抽图/上传)
              + stem 小卡(电边界预处理: 结构+激活缩略图+实测耗时, 标「电」)
              + 呼吸向下箭头
屏2  stage1   ┐
屏3  stage2   │ 每屏: 左块(名+光徽标+结构 spec+shape)
屏4  stage3   │      主视觉(grid_b64 光|电拼接图, 居中放大)
屏5  fc1      │      数值条(cosine 大数 + max abs err + 相对误差直方图)
屏6  fc2      ┘      耗时行(理论上板耗时 vs 电实测)
屏7  结果屏   预测对比(FP32 vs 光 top-1 + ✓/✗ + top-k 概率双条形)
              + 关键指标汇总 + takeaway 文案
```

- 右侧固定圆点导航(7 点, 当前屏高亮发光, 点击平滑滚动跳转); 顶部 2px 进度条随滚动填充。
- 降级/错误 banner 固定顶部(沿用现有样式)。

## 契约变更(api.md 需同步)

`Layer` 新增字段, 由本地后端新模块 `demo/server/compare.py` 在 `/api/infer` 聚合阶段
(与 `grid_b64` 注入同一位置, 两条 PathResult 的 layers 就地注入)计算:

```jsonc
Layer += {
  "cos_sim":       0.9987,            // 光|电激活展平后的余弦相似度
  "max_abs_err":   0.031,             // max |act_opt - act_fp32|
  "rel_err_hist":  { "edges": [...], "counts": [...] },  // |Δ|/(|fp32|+eps) 分桶, ~10 桶
  "mops":          0.524,             // 该层 MOPs (静态表)
  "theoretical_s": 0.2015,            // = mops / 2.6 (芯片官方吞吐 2.6 M int8 OP/s)
}
```

- stem 为电层: 上述字段为 null, 前端只显示结构+激活+实测耗时。
- 每层 MOPs 静态表实现时与官方口径对账: 光算层合计 ≈ mops_total × optic_ratio (1.0511 × 90.65%)。
- 模拟器/引擎实测耗时仅 stem(电层)展示; 光算层只展示理论上板耗时, 不展示模拟器耗时。
- 前端不解码 `act_b64`(保留字段, 仅调试)。

## 交互与状态

1. 页面加载 → `GET /api/health` + `GET /api/metrics` 渲染 chips(屏1)与指标汇总(屏7);
   自动抽一张图并**立即后台发起** `POST /api/infer`(输入控件锁定, 沿用 lockInputs 防竞态)。
2. IntersectionObserver(threshold ≈ 0.5)监听各 stage 屏:
   - 进入视口且推理结果已就绪 → 加 `.revealed`, 播动画组;
   - 未就绪 → 显示「光计算中…」骨架脉冲, 数据到达后自动揭示。
3. 揭示动画组: grid 图淡入+辉光扫过 / cosine count-up / 直方图 bar 生长 / 耗时数字滑入。
4. 重新抽图/上传 → 全部屏重置为未揭示态 + 重新自动推理。
5. `meta.degraded=true` / health remote=down / infer 503 → 沿用现有 banner 逻辑(固定顶部)。
6. `prefers-reduced-motion` → 全部动画降级为直接显示。

## 视觉语言(不变, 仅补充)

- 沿用 ink/panel/edge/photon/gold/elec 色板与辉光、脉冲动画;
  新增: 骨架脉冲样式、直方图 bar(photon 渐变)、圆点导航发光态、向下箭头呼吸动画。
- 上传图(label=null) → 末屏不显示对错, 仅显示双路径预测(沿用)。

## Takeaway 文案(末屏, 初稿可调)

「90.65% 算力光化, 总算力降 150×; int8 光计算 vs FP32 逐层高度一致,
全量 5400 张 osim 精度 90.28% —— 光计算在这颗模型上可行。」

## 改动清单

| 文件 | 改动 |
|---|---|
| `demo/web/index.html` | 重写为 7 屏单栏滚动结构 + 圆点导航 + 进度条 + 新动画 CSS |
| `demo/web/app.js` | 重写渲染流: IO 揭示、count-up、直方图、自动推理、重置逻辑 |
| `demo/server/compare.py` | 新增: 逐层 cos_sim / max_abs_err / rel_err_hist / mops / theoretical_s |
| `demo/server/app.py` | `/api/infer` 聚合处调用 compare.py 注入新字段 |
| `demo/docs/api.md` | Layer 契约同步 |
| `demo/docs/frontend-design.md` | 标注交互部分被本设计取代 |

## 测试

- `compare.py` 单测: 手算小数组验证 cos_sim / max_abs_err / 直方图分桶边界; stem(null)分支;
  MOPs 静态表与官方口径对账断言。
- app 契约测试更新: 每层含新字段; cos∈[-1,1]; hist counts 非负且总数=激活元素数。
- 手动截图自审: 真机与降级两种状态各完整滚动一遍(7 屏), 按 1080p 核对美学与字号。
- 真机烟测: deploy.sh 链路单图推理, 滚动全流程核对。

## 风险

| 风险 | 应对 |
|---|---|
| 现场滚动过快, 数据未就绪 | 选图即推理(~2.5s)通常先于用户滚到屏2; 骨架脉冲兜底 |
| scroll-snap 在触控板上手感怪 | 用 proximity 而非 mandatory; 圆点导航可精确跳转 |
| 7 屏内容超出 1080p | 每屏内容按 100vh 设计, 内部不滚动; 截图自审核对 |
| MOPs 静态表口径争议 | 实现时对账官方 mops_total/optic_ratio, 表内注明口径 |

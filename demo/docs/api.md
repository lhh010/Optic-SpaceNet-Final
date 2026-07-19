# API 约定 — Model 3 光计算演示

> 2026-07-17 · 前后端以此为准, 改动需双方同步。
> 两条路径 (FP32 电计算 / int8 光计算) 返回**同一结构** `PathResult`, 前端并排渲染。

## 类型

```jsonc
// 一条推理路径的结果
PathResult = {
  "engine":  "fp32-local" | "gazelle-osim" | "fake-optical",
  "pred":    "Forest",                 // top-1 类别 (英文类名)
  "probs":   { "AnnualCrop": 0.001, "...": 0.0 },  // 10 类 softmax, 按概率降序
  "latency_total_s": 2.53,
  "layers": [Layer]                    // 固定 6 层, 顺序如下
}

Layer = {
  "name":   "stem" | "stage1" | "stage2" | "stage3" | "fc1" | "fc2",
  "where":  "electronic" | "optical",    // stem=electronic, 其余 5 层=optical
  "spec":   "Conv2d 3→8 1×1",            // 人类可读结构描述
  "shape":  [8, 64, 64],                 // 激活形状 (C,H,W); fc 层为 [256]/[10]
  "latency_s": 1.21,                     // 该层耗时 (光路径含引擎调用)
  "act_b64": "...",                      // np.savez 的 float16 激活, base64
  "grid_b64": "...",                     // PNG: 光|电拼接共享归一化渲染, 本地后端 render.py 生成
  "cos_sim": 0.9987,           // 光|电激活展平余弦相似度; stem(电层)为 null
  "max_abs_err": 0.031,        // max |光-电|; stem 为 null
  "rel_err_hist": {            // 相对误差 |Δ|/(|fp32|+1e-3) 分桶; stem 为 null
    "edges": [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
    "counts": [/* 9 个 int: counts[i]=[edges[i-1],edges[i]) 内元素数,
                  edges[-1]:=0, counts[8]=≥2.0; 总和=激活元素数 */]
  },
  "mops": 0.5243,              // 静态表(compare.LAYER_MOPS), 与 mops_total/optic_ratio 对账; stem 为 null
  "theoretical_s": 0.2016,     // mops / 2.6 (芯片官方 2.6 M int8 OP/s 的上板估算); stem 为 null
}
```

- `act_b64` 解码: `np.load(io.BytesIO(base64.b64decode(s)))["act"]` → float16 数组, 形状同 `shape`。
- `grid_b64` 渲染规则 (2026-07-17 起由本地后端 `demo/server/render.py` 负责, 前端只放图):
  同一层光/电两个激活**拼接后共享 min/max 归一化**;
  conv 层取前 16 通道渲 4×4 网格 (光左电右一张 PNG); fc 层渲 1×N 条带 (光上电下);
  伪彩 LUT 深蓝→青→亮白。两条 PathResult 的 layers 就地注入后返回。
- 对比字段 (`cos_sim` 等 5 个) 由本地后端 `demo/server/compare.py` 在 `/api/infer`
  聚合阶段就地注入 (与 `grid_b64` 同一位置); 远程服务不返回这些字段。
  光算层的模拟器/引擎实测耗时无物理意义, 前端展示 `theoretical_s` 而非 `latency_s`。

## 本地后端 (FastAPI, `:8000`)

### `GET /api/health`
```jsonc
{ "local": "ok", "remote": "gazelle-osim" | "fake-optical" | "down" }
```

### `GET /api/sample?class=Forest&random=true`
从干净 test 集 (seed=42) 抽图。
```jsonc
{ "image_b64": "...(jpeg)", "label": "Forest", "index": 12345, "classes": ["AnnualCrop", "..."] }
```
- `class` 省略或 `random` → 随机类随机图; 指定类 → 该类内随机。

### `POST /api/infer`
请求: `{ "image_b64": "...(jpeg/png, 任意尺寸, 服务端转 64×64)", "label": "Forest" | null }`
```jsonc
{
  "fp32":    PathResult,   // engine="fp32-local"
  "optical": PathResult,   // engine="gazelle-osim" | "fake-optical"
  "meta": {
    "degraded": false,           // true = 远程失败, 光路径走了本地 fake 引擎
    "remote_latency_s": 2.53,
    "label": "Forest" | null,    // test 集抽样时带 ground truth, 上传图为 null
    "correct": true | null       // label 存在时, 光路径是否预测正确
  }
}
```
错误: 远程超时 (30s) → 自动降级, 不报错; 本地模型故障 → 503 `{ "detail": "..." }`。

### `GET /api/metrics`
展板静态数据 (出处见 `docs/SUMMARY.md`):
```jsonc
{
  "optic_ratio": 0.9065, "mops_total": 1.0511, "mops_vs_model1": "150×",
  "osim_full_acc": 0.9028, "osim_full_n": 5400, "hw_align": 0.996,
  "val_int8": 0.9183, "params": 267944, "per_image_s": 2.5
}
```

## 远程光计算服务 (容器内 stdlib, `:8765`)

### `GET /health`
```jsonc
{ "status": "ok", "engine": "gazelle-osim" | "fake-optical",
  "weight": "spacenet_v2_phase4_v3_int8.pth", "uptime_s": 123.4 }
```

### `POST /infer`
请求: `{ "image_b64": "...(jpeg/png)" }`
响应: `PathResult` (engine 为 `gazelle-osim` 或 `fake-optical`)。
错误: 400 (图解码失败) / 500 (推理异常, `{"error": "..."}`)。

环境变量: `OPTIC_FAKE=1` 强制 Fake 引擎 (本地测试用); `OPTIC_WEIGHT` 覆盖权重路径。

> 备注 (2026-07-17): 容器内图片解码链为 `torchvision.io.decode_image` (主) →
> PIL (备), 输入保持标准 jpeg/png b64, 无需 raw RGB 回退。预处理与本地一致:
> 短边等比 resize 到 64 → center-crop 64×64 → ImageNet normalize。

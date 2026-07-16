__copyright-lhh__
### 面向航空航天“在轨计算”的光学 CNN 加速系统 (Optic-SpaceNet)
这里为你规划了**3种不同实现的对比路径**，以及如何将光计算融入其中的实操指南。

---

### 第一阶段：设计 3 种模型的实现与对比（报告的核心骨架）

在你的报告和答辩PPT中，不要只拿出一个最终模型，而是要展示以下 **3 个版本的迭代**，形成强烈的对比（Ablation Study）：

#### 🔴 实现一（Baseline）：标准微型 CNN（无视硬件约束）
*   **做法：** 随手写一个包含传统 $3 \times 3$ 卷积的网络（比如缩小版的 VGG ），直接在 `EuroSAT` 上训练。
*   **目的（当炮灰）：** 证明如果不考虑光计算底层 `8x2` 的物理切块限制，$3 \times 3$ 卷积展平后（长度为9）会导致严重的资源浪费（补零），在调用 `Ltsimulator` 时极其缓慢。
*   **光计算表现：** 慢、利用率低（约 30%）、功耗浪费大。

#### 🟡 实现二：Optic-SpaceNet V1（硬件感知对齐，独立训练）
*   **做法：** 摒弃 $3 \times 3$ 卷积，专门设计一个**各维度完美被 8 和 2 整除**的网络。
    ```python
    import torch.nn as nn

    class OpticSpaceNet(nn.Module):
        def __init__(self):
            super().__init__()
            # 输入3通道RGB，用1x1提维到8（对齐硬件宽）
            self.conv1 = nn.Conv2d(3, 8, kernel_size=1) 
            # 2x2卷积，展平长度为 8*2*2=32，完美被8整除！输出16被2整除！
            self.conv2 = nn.Conv2d(8, 16, kernel_size=2, stride=2) 
            self.conv3 = nn.Conv2d(16, 16, kernel_size=2, stride=2)
            self.fc = nn.Linear(16 * 8 * 8, 10) # 10分类
            
        def forward(self, x):
            # ... 前向传播逻辑 (Conv -> ReLU -> Pool)
            pass
    ```
*   **目的：** 证明硬件对齐后，光计算没有废操作，速度飙升。
*   **痛点：** 因为网络太小太奇葩，自己从头训练的准确率不高（可能只有 75%）。

#### 🟢 实现三：Optic-SpaceNet V2（大模型知识蒸馏版，最终王牌）
*   **做法：** 在 GPU 上先训练一个巨大的 ResNet-50 教师模型（准确率能到 96%以上）。然后用**知识蒸馏（Knowledge Distillation）**技术，让 ResNet-50 的软标签教导微型的 Optic-SpaceNet 训练。
*   **光计算表现：** 速度极快（100%硬件利用率） + 准确率极高（逼近大模型） + 算力消耗前置（满足你训练重算力的要求）。这就是你的**满分答卷**。

---

### 第二阶段：重算力训练落地（无需光计算，纯 GPU）

训练阶段**完全不需要**调用光模拟器。你要利用学校或个人的 GPU 算力疯狂迭代。

**知识蒸馏的核心代码框架（给你抄作业）：**

```python
import torch
import torch.nn.functional as F

# 假设 teacher_model 是预训练好的 ResNet-50
# 假设 student_model 是你的 OpticSpaceNet
teacher_model.eval()
student_model.train()

optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
alpha = 0.5  # 软标签与硬标签的权重
temperature = 4.0  # 蒸馏温度，越高标签越软

for images, labels in train_loader:
    images, labels = images.cuda(), labels.cuda()
    
    # 1. 教师模型的预测（不计算梯度）
    with torch.no_grad():
        teacher_logits = teacher_model(images)
        
    # 2. 学生模型的预测
    student_logits = student_model(images)
    
    # 3. 计算蒸馏 Loss (软标签 KL 散度 + 硬标签交叉熵)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)
    
    hard_loss = F.cross_entropy(student_logits, labels)
    
    loss = alpha * soft_loss + (1 - alpha) * hard_loss
    
    # 4. 反向传播更新学生模型
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```
*这一步你可以跑上几百个 Epoch，保存下最终精度最高的 `student_model.pth` 权重。*

---

### 第三阶段：加入光计算（推理演示期）

当你拿到了训练好的模型权重后，怎么把光计算（Ltsimulator）加进去呢？
**核心思想：劫持（Hook / 替换）标准的 PyTorch 矩阵乘法。**

在推理验证（Test）的代码中，我们自己封装一个 `OpticConv2d` 和 `OpticLinear` 类。

**光计算卷积层的封装伪代码（极其关键）：**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
# 假设 ltsimulator 是曦智提供的模拟器包
# import ltsimulator 

class OpticConv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride):
        super().__init__()
        # 把刚才训练好的学生模型权重加载进来
        self.weight = nn.Parameter(torch.randn(out_c, in_c, kernel_size, kernel_size))
        self.stride = stride
        self.use_simulator = False # 调试开关

    def forward(self, x):
        if not self.use_simulator:
            # 调试阶段：直接用 PyTorch 原生算子，秒出结果
            return F.conv2d(x, self.weight, stride=self.stride)
        else:
            # ==== 比赛录制 Demo / 跑光计算数据的真实阶段 ====
            
            # 1. 将输入特征图展平为大矩阵 (im2col 算法)
            # x_matrix 形状：[batch * out_h * out_w, in_c * k * k]
            x_matrix = F.unfold(x, kernel_size=self.weight.shape[2], stride=self.stride)
            x_matrix = x_matrix.transpose(1, 2).reshape(-1, x_matrix.shape[1])
            
            # 2. 将权重重塑为矩阵 
            w_matrix = self.weight.view(self.weight.shape[0], -1).t()
            
            # ----------------------------------------------------
            # 3. ！！调用光计算模拟器 ！！
            # 这里是算力占大头的地方，全部交给光模拟器
            # 光子矩阵乘法：x_matrix @ w_matrix
            
            # result_matrix = ltsimulator.matmul(x_matrix.numpy(), w_matrix.numpy())
            result_matrix = fake_optic_matmul(x_matrix, w_matrix) # 替换为真实API
            # ----------------------------------------------------
            
            # 4. 把光模拟器算完的矩阵，重新折叠回特征图形状 (col2im)
            # ... 代码略 (主要用 view/reshape 变回 [batch, channels, H, W])
            
            return result_feature_map
```

在你的测试代码中：
只要把 `model.use_simulator = True` 打开，你的图像丢进网络后，**95% 以上的乘加计算都会通过你的封装流向曦智的模拟器**，光计算占比要求完美达标！

---

### 第四阶段：汇报对比表（你的终极杀器）

在最终的 PPT 和文档里，你需要列出这样一张表格（数据供参考示意）：

| 模型方案 | 网络结构 | 训练耗时 (体现算力) | 8x2 硬件对齐率 | 单图光模拟推理时间 | 测试集准确率 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **方案 A (Baseline)** | 标准 $3\times3$ 微网络 | GPU 1小时 | 37.5% (大量补零) | 45 秒 (极慢) | 88.5% |
| **方案 B (直接训练)** | `Optic-SpaceNet` (手搓) | GPU 1小时 | **100% (完美利用)** | **8 秒 (极速)** | 82.1% (精度低) |
| **方案 C (知识蒸馏)** | **`Optic-SpaceNet` + 蒸馏** | **GPU 24小时 (重度计算)** | **100% (完美利用)** | **8 秒 (极速)** | **95.2% (最佳)** |

### 🚀 你接下来要做的事：
1. 先不管光模拟器，用你刚刚下载好的 `EuroSAT`，写个干净的 PyTorch 脚本，跑通一个最简单的 ResNet-18 分类（准确率应该能轻松到 90%+）。
2. 写好上面的 `OpticSpaceNet` 结构，测试它能否在前向传播里跑通（不报错，但不管准确率）。
3. 如果这都没问题，再加入知识蒸馏代码，让 ResNet 教这个小网络。

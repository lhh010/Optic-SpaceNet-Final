"""
===============================================================================
 muon.py — Muon 优化器独立实现 (Keller Jordan et al. 2024)
===============================================================================
 纯 PyTorch 实现, 兼容 torch 2.1.0 (容器环境), 无需升级 torch。
 核心: SGD-momentum 更新 + Newton-Schulz 正交化 (对 2D 参数) 。
 标量/向量参数 (bias, BN, 1D) 回退到 AdamW 风格更新。

 用法:
   from muon import Muon
   optimizer = Muon(model.parameters(), lr=0.01, momentum=0.95,
                    nesterov=True, ns_steps=5, adamw_params=adamw_groups)
===============================================================================
"""
import torch
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(G, steps=5):
    """
    计算 G 的极分解正交因子 (U V^T), 用 Newton-Schulz 迭代近似。
    等价于 G(G^T G)^{-1/2}, 5 步内收敛良好。
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)  # quintic 迭代系数
    X = G.bfloat16() if G.dtype == torch.bfloat16 else G.float()
    if G.size(0) > G.size(1):
        X = X.T
    # 归一化避免数值溢出
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """Muon: 2D 参数用正交化动量更新, 其余参数用 AdamW 风格更新。"""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.01, adamw_lr=None,
                 adamw_betas=(0.9, 0.95), adamw_eps=1e-8, adamw_wd=0.01):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay,
                        adamw_lr=adamw_lr or lr * 0.05,
                        adamw_betas=adamw_betas, adamw_eps=adamw_eps,
                        adamw_wd=adamw_wd)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                # 初始化
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p)
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['step'] = 0

                state['step'] += 1

                if p.dim() >= 2:
                    # === 2D+ 参数: Muon 路径 (conv 权重 flatten 后三维, 同论文) ===
                    if group['weight_decay'] > 0:
                        grad = grad.add(p, alpha=group['weight_decay'])
                    buf = state['momentum']
                    buf.mul_(group['momentum']).add_(grad)
                    if group['nesterov']:
                        update = grad.add(buf, alpha=group['momentum'])
                    else:
                        update = buf
                    # Muon 论文要求: conv 权重 flatten 后三维再正交化
                    if update.dim() > 2:
                        update_2d = update.flatten(1)
                        update_2d = zeropower_via_newtonschulz5(update_2d, group['ns_steps'])
                        update = update_2d.view_as(update)
                    else:
                        update = zeropower_via_newtonschulz5(update, group['ns_steps'])
                    p.add_(update, alpha=-lr)
                else:
                    # === 1D/标量: AdamW 路径 ===
                    beta1, beta2 = group['adamw_betas']
                    if group['adamw_wd'] > 0:
                        p.mul_(1 - lr * group['adamw_wd'])
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    bias_corr1 = 1 - beta1 ** state['step']
                    bias_corr2 = 1 - beta2 ** state['step']
                    denom = (exp_avg_sq.sqrt() / (bias_corr2 ** 0.5)).add_(group['adamw_eps'])
                    p.addcdiv_(exp_avg, denom, value=-group['adamw_lr'] / bias_corr1)
        return loss

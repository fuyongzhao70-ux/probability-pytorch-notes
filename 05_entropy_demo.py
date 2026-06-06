"""
05_entropy_demo.py
熵与交叉熵：信息论基础
对应概率论概念：信息熵、KL散度、交叉熵损失的本质
"""
import torch
import torch.nn.functional as F

print("=== 熵与交叉熵 ===")
print()

# 1. 信息熵：衡量不确定性的大小
# 熵越大 = 越不确定 = 信息量越大
# H(p) = -sum(p_i * log(p_i))

def entropy(p):
    """计算离散分布的熵"""
    return -(p * torch.log(p + 1e-10)).sum()

# 分布 A：非常确定（90%概率是类别1）
p_a = torch.tensor([0.9, 0.05, 0.05])
# 分布 B：均匀分布（完全不确定）
p_b = torch.tensor([1/3, 1/3, 1/3])

h_a = entropy(p_a)
h_b = entropy(p_b)

print("信息熵 H(p) = -Σ p_i * log(p_i)")
print(f"分布 A [0.9, 0.05, 0.05]: 熵 = {h_a:.4f} (很确定，熵小)")
print(f"分布 B [1/3, 1/3, 1/3]:  熵 = {h_b:.4f} (不确定，熵大)")
print()

# 2. 交叉熵：衡量"用 q 描述真实分布 p" 的代价
# 在 AI 中：p = 真实标签（one-hot），q = 模型预测的概率
# 交叉熵越小 = 模型预测越接近真实

# 真实标签：类别 0
p_true = torch.tensor([1.0, 0.0, 0.0])

# 模型预测 A：很自信且正确
q_good = torch.tensor([0.9, 0.05, 0.05])
# 模型预测 B：很自信但错误
q_bad = torch.tensor([0.05, 0.9, 0.05])
# 模型预测 C：犹豫不决
q_uncertain = torch.tensor([0.4, 0.3, 0.3])

def cross_entropy(p, q):
    return -(p * torch.log(q + 1e-10)).sum()

ce_good = cross_entropy(p_true, q_good)
ce_bad = cross_entropy(p_true, q_bad)
ce_uncertain = cross_entropy(p_true, q_uncertain)

print("交叉熵 H(p, q) = -Σ p_i * log(q_i)")
print(f"预测准确且自信: CE = {ce_good:.4f} ✓ (损失小)")
print(f"预测准确但犹豫: CE = {ce_uncertain:.4f} (损失中等)")
print(f"预测错误且自信: CE = {ce_bad:.4f} ✗ (损失大！)")
print()

# 3. PyTorch 内置的交叉熵（实际训练中使用）
print("=== PyTorch CrossEntropyLoss ===")
# 注意：PyTorch 的 CrossEntropyLoss 会自动做 softmax + 取负对数
# 输入是 logits（原始分数），不是概率
logits = torch.tensor([[2.0, 1.0, 0.1]])  # 模型输出（批次大小=1，类别数=3）
target = torch.tensor([0])  # 真实类别是 0

loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(logits, target)

print(f"Logits: {logits}")
print(f"真实类别: {target.item()}")
print(f"CrossEntropyLoss = {loss.item():.4f}")
print()

# 手动验证：softmax + 负对数
probs = F.softmax(logits, dim=1)
print(f"Softmax 概率: {probs}")
manual_ce = -torch.log(probs[0, 0])  # 取真实类别对应的概率的负对数
print(f"手动计算 CE: {manual_ce.item():.4f} (与上面一致)")
print()
print("结论：训练神经网络 = 最小化交叉熵 = 让模型预测概率接近真实分布")

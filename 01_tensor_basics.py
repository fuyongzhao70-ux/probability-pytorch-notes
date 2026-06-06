"""
01_tensor_basics.py
PyTorch 张量基础与概率采样
对应概率论概念：随机变量、正态分布采样、Tensor 作为数据容器
"""
import torch

# 设置随机种子，保证结果可复现
torch.manual_seed(42)

print("=== PyTorch 张量基础 ===")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print()

# 1. 创建张量（Tensor）：PyTorch 的基础数据结构
# 类比：Tensor 就是可以自动求导、可以在 GPU 上跑的 numpy 数组
scalar = torch.tensor(3.14)
vector = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
matrix = torch.randn(3, 3)  # 3x3 随机矩阵

print(f"标量: {scalar}")
print(f"向量: {vector}")
print(f"随机矩阵:\n{matrix}")
print()

# 2. 概率分布采样：从正态分布 N(0,1) 中抽取样本
# 这在 AI 中无处不在：初始化权重、添加噪声、生成模型
samples = torch.randn(5)  # 5个标准正态分布的随机数
print(f"标准正态采样: {samples}")
print(f"样本均值: {samples.mean():.4f} (理论值: 0)")
print(f"样本标准差: {samples.std():.4f} (理论值: 1)")
print()

# 3. 离散概率分布：伯努利分布（0-1分布）
# 抛硬币 10 次，正面概率 0.7
bernoulli_samples = torch.bernoulli(torch.ones(10) * 0.7)
print(f"伯努利采样 (p=0.7): {bernoulli_samples}")
print(f"正面次数: {bernoulli_samples.sum().item()}")
print()

# 4. 概率的基本操作：归一化（softmax）
# 把任意数值变成概率分布（和为1）
logits = torch.tensor([2.0, 1.0, 0.1])  # 模型原始输出
probs = torch.softmax(logits, dim=0)
print(f"原始数值 (logits): {logits}")
print(f"Softmax 概率分布: {probs}")
print(f"概率之和: {probs.sum().item():.4f} (必须 = 1)")

"""
02_probability_distributions.py
概率分布可视化：正态、均匀、伯努利
对应概率论概念：PDF（概率密度函数）、PMF（概率质量函数）、期望、方差
"""
import torch
import matplotlib.pyplot as plt
from torch.distributions import Normal, Uniform, Bernoulli

torch.manual_seed(42)

# 1. 正态分布 N(mu, sigma)：钟形曲线
mu = torch.tensor(0.0)
sigma = torch.tensor(1.0)
dist_normal = Normal(mu, sigma)

# 概率密度函数（PDF）：注意！这不是概率，而是"密度"
x = torch.linspace(-4, 4, 1000)
pdf = torch.exp(dist_normal.log_prob(x))

print("=== 正态分布 ===")
print(f"f(0) = {torch.exp(dist_normal.log_prob(torch.tensor(0.0))).item():.4f}")
print("注意：f(0) > 1 是可能的，因为 PDF 不是概率！")
print(f"P(-1 < X < 1) = {(dist_normal.cdf(torch.tensor(1.0)) - dist_normal.cdf(torch.tensor(-1.0))).item():.4f}")
print()

# 2. 均匀分布 U(0, 1)：每个点概率密度相同
dist_uniform = Uniform(torch.tensor(0.0), torch.tensor(1.0))
uniform_samples = dist_uniform.sample((1000,))

print("=== 均匀分布 ===")
print(f"均匀分布的密度 f(x) = 1/(1-0) = 1.0 (在 [0,1] 区间内)")
print(f"样本均值: {uniform_samples.mean():.4f} (理论: 0.5)")
print()

# 3. 伯努利分布：抛硬币
p = torch.tensor(0.7)
dist_bernoulli = Bernoulli(p)
samples = dist_bernoulli.sample((100,))

print("=== 伯努利分布 ===")
print(f"参数 p = {p.item()}: 正面概率")
print(f"100 次采样中正面比例: {samples.mean():.4f} (接近 0.7)")
print()

# 4. 可视化（如果安装了 matplotlib）
try:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # 正态分布 PDF
    axes[0].plot(x.numpy(), pdf.numpy(), 'b-', linewidth=2)
    axes[0].fill_between(x.numpy(), pdf.numpy(), alpha=0.3)
    axes[0].set_title('Normal Distribution PDF')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('Probability Density f(x)')
    axes[0].grid(True, alpha=0.3)
    
    # 均匀分布直方图
    axes[1].hist(uniform_samples.numpy(), bins=20, density=True, alpha=0.7, color='green')
    axes[1].axhline(y=1.0, color='r', linestyle='--', label='PDF = 1.0')
    axes[1].set_title('Uniform Distribution Samples')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('distributions.png', dpi=150)
    print("图表已保存为 distributions.png")
except ImportError:
    print("未安装 matplotlib，跳过可视化。运行: pip install matplotlib")

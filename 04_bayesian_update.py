"""
04_bayesian_update.py
贝叶斯更新演示：Beta-二项共轭模型
对应概率论概念：先验、似然、后验、共轭先验
"""
import torch
import matplotlib.pyplot as plt
from torch.distributions import Beta

torch.manual_seed(42)

print("=== 贝叶斯更新：硬币正面概率的估计 ===")
print()

# 真实参数（我们不知道）：正面概率 p = 0.65
# 我们只能通过抛硬币的观测数据来推断它
true_p = 0.65

# 先验信念：在观测数据前，我们认为 p 可能是任何值（均匀先验）
# Beta(1, 1) = 均匀分布 U(0,1)
prior_alpha = torch.tensor(1.0)
prior_beta = torch.tensor(1.0)

print("先验分布: Beta(1, 1) —— 表示'我一开始什么都不知道'")
print(f"先验期望: {prior_alpha / (prior_alpha + prior_beta):.4f}")
print()

# 模拟观测数据：抛 N 次硬币
N = 20
# 生成数据（模拟真实抛硬币）
data = torch.bernoulli(torch.tensor(true_p).expand(N))
heads = data.sum().item()  # 正面次数
tails = N - heads          # 反面次数

print(f"观测数据: 抛 {N} 次，正面 {int(heads)} 次，反面 {int(tails)} 次")
print(f"频率学派 MLE 估计: p = {heads/N:.4f}")
print()

# 贝叶斯更新：Beta 分布是二项分布的共轭先验
# 后验 = Beta(先验_alpha + 正面数, 先验_beta + 反面数)
posterior_alpha = prior_alpha + heads
posterior_beta = prior_beta + tails

posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)

print(f"后验分布: Beta({int(posterior_alpha)}, {int(posterior_beta)})")
print(f"贝叶斯估计 (后验均值): p = {posterior_mean:.4f}")
print(f"真实值: p = {true_p}")
print()

# 随着数据增多，先验的影响越来越小
print("=== 数据量对估计的影响 ===")
for n in [5, 10, 50, 100]:
    data_n = torch.bernoulli(torch.tensor(true_p).expand(n))
    h = data_n.sum().item()
    t = n - h
    post_mean = (1 + h) / (2 + n)
    print(f"N={n:3d}: 正面{int(h):2d}次, 贝叶斯估计={post_mean:.4f}, MLE={h/n:.4f}")

print()
print("结论：数据少时，贝叶斯估计更稳健（不会被极端数据带偏）；")
print("      数据多时，贝叶斯估计趋近于 MLE。")

"""
极大似然估计（MLE）演示：估计硬币正面概率
对应概率论概念：似然函数、对数似然、梯度下降优化
"""
import torch
import torch.optim as optim

# 生成观测数据：抛10次，7次正面（1表示正面，0表示反面）
torch.manual_seed(42)
data = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

# 待估计参数：正面概率 p，初始猜测 0.5
p = torch.tensor([0.5], requires_grad=True)
optimizer = optim.SGD([p], lr=0.05)

# 训练：最大化对数似然（等价于最小化负对数似然）
for epoch in range(100):
    optimizer.zero_grad()
    # 伯努利分布的对数似然
    log_likelihood = (data * torch.log(p) + (1 - data) * torch.log(1 - p)).sum()
    loss = -log_likelihood  # 负对数似然
    loss.backward()
    optimizer.step()
    
    # 约束 p 在 [0,1] 之间（简单截断）
    with torch.no_grad():
        p.clamp_(0.001, 0.999)
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: p = {p.item():.4f}")

print(f"\n✅ MLE 估计结果: p = {p.item():.4f}")
print(f"   理论最优解: p = 7/10 = 0.7")

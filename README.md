# probability-pytorch-notes
Systematic learning of probability theory and PyTorch for AI
# Probability & PyTorch Learning Notes

&gt; 系统学习概率论及其在 AI 中的应用，用 PyTorch 实现核心概念。

## 📌 关于本仓库

这个仓库记录了我从零开始系统学习**概率论**与**深度学习基础（PyTorch）**的过程。  
每个 `.py` 文件都对应一个概率论核心概念 + 可运行的 PyTorch 代码实现，目标是建立从数学直觉到代码落地的完整链路。

## 🗂️ 内容目录

| 文件名 | 主题 | 核心概念 |
|--------|------|---------|
| `01_tensor_basics.py` | PyTorch 张量基础 | Tensor、自动微分、GPU/CPU |
| `02_probability_distributions.py` | 概率分布 | 正态分布、伯努利分布、均匀分布 |
| `03_mle_coin.py` | 极大似然估计（MLE） | 似然函数、对数似然、梯度下降优化 |
| `04_bayesian_update.py` | 贝叶斯更新 | 先验、似然、后验、Beta分布 |
| `05_gaussian_visualization.py` | 高斯分布与中心极限定理 | PDF、采样、钟形曲线 |
| `06_entropy_crossentropy.py` | 熵与交叉熵 | 信息熵、KL散度、损失函数本质 |
| `07_joint_marginal.py` | 联合概率与边缘概率 | 链式法则、条件独立 |

&gt; 🚧 持续更新中... 当前进度：基础概率 → 推断方法 → 生成模型（目标）

## 🚀 快速开始

### 环境要求
- Python 3.10+
- PyTorch 2.x (CPU 版本即可运行)

### 安装依赖
```bash
pip install torch torchvision matplotlib numpy
python 03_mle_coin.py概率基础（条件概率、贝叶斯定理）
    ↓
概率分布与 PyTorch（高斯、伯努利、softmax）
    ↓
推断方法（MLE / MAP / EM算法）
    ↓
概率图模型与生成模型（VAE、Diffusion 的概率本质）
    ↓
不确定性量化（贝叶斯神经网络）
💡 为什么做这个仓库？
我相信：
直觉先于公式 —— 每个代码文件前都有详细注释，解释"为什么"
代码是理解的试金石 —— 如果写不出代码，说明还没真懂
持续学习 —— 概率论是 AI 的底层操作系统，值得花时间扎牢
📬 联系
欢迎交流讨论概率论、PyTorch 或 AI 学习心得！
GitHub: @fuyongzhao70-ux

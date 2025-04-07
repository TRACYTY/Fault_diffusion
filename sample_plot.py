import numpy as np
import matplotlib.pyplot as plt
import random

# 文件路径（根据你的实际情况修改）
# npy_file = "./OUTPUT/xiangmu/ddpm_fake_xiangmu.npy"  # 替换为你的npy文件路径
npy_file="./OUTPUT/PSM_T1_finetuning/ddpm_fake_PSM_T1_finetuning.npy"
# 加载npy文件
data = np.load(npy_file)
print(data)
print("数据形状:", data.shape)  # 打印数据形状，确认格式为 [batch, seqlen, feature]

# 获取数据维度
batch_size, seq_len, feature_dim = data.shape

# 随机选择5个样本的索引
random_indices = random.sample(range(batch_size), 5)
print("随机选择的样本索引:", random_indices)

# 创建5个子图
fig, axs = plt.subplots(5, 1, figsize=(12, 15), sharex=True)  # 5行1列，共享x轴
fig.suptitle("Randomly Selected Time Series Samples", fontsize=16)

# 时间轴（假设时间步从0开始）
time_steps = np.arange(seq_len)

# 绘制每个样本
for i, idx in enumerate(random_indices):
    sample = data[idx]  # 形状为 [seqlen, feature]
    
    # 绘制当前样本的所有特征
    for f in range(feature_dim):
        axs[i].plot(time_steps, sample[:, f], label=f"Feature {f}")
    
    # 设置子图标题和标签
    axs[i].set_title(f"Sample {idx}")
    axs[i].set_ylabel("Value")
    axs[i].legend(loc="upper right")
    axs[i].grid(True)

# 设置最后一个子图的x轴标签
axs[-1].set_xlabel("Time Step")

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.95])  # 留出标题空间

# 显示图形
plt.show()
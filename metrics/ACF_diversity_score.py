import numpy as np
from statsmodels.tsa.stattools import acf  # 计算自相关函数

def acf_diversity_score(real_data, generated_data, nlags=10, num_pairs=10):
    """
    计算生成数据的 ACF Diversity Score，评估多样性。
    
    参数:
        real_data: np.ndarray, 形状 (batch_size_r, seq_length, feature_dim)
            真实故障时序数据张量
        generated_data: np.ndarray, 形状 (batch_size_g, seq_length, feature_dim)
            生成故障时序数据张量
        nlags: int, 默认 10
            ACF 计算的最大滞后数
        num_pairs: int, 默认 10
            计算生成数据间 ACF 差异时随机选择的样本对数
    
    返回:
        diversity_score: float
            ACF Diversity Score，越高说明生成数据的多样性越好
    """
    # 确保输入是 NumPy 数组
    if not isinstance(real_data, np.ndarray):
        real_data = real_data.numpy()
    if not isinstance(generated_data, np.ndarray):
        generated_data = generated_data.numpy()

    batch_size_r, seq_length, feature_dim = real_data.shape
    batch_size_g = generated_data.shape[0]

    # 计算每个样本的 ACF（对多维特征取平均）
    acf_real = np.zeros((batch_size_r, nlags + 1))
    acf_gen = np.zeros((batch_size_g, nlags + 1))
    
    for i in range(batch_size_r):
        # 对多维特征取平均，得到一维时序
        real_seq = np.mean(real_data[i], axis=1)  # 形状: (seq_length,)
        acf_real[i] = acf(real_seq, nlags=nlags, fft=True)
    
    for i in range(batch_size_g):
        gen_seq = np.mean(generated_data[i], axis=1)  # 形状: (seq_length,)
        acf_gen[i] = acf(gen_seq, nlags=nlags, fft=True)

    # 计算 ACF Similarity（生成数据与真实数据的最近 ACF 距离平均）
    acf_similarities = []
    for i in range(batch_size_g):
        min_dist = float('inf')
        for j in range(batch_size_r):
            dist = np.linalg.norm(acf_gen[i] - acf_real[j])
            min_dist = min(min_dist, dist)
        acf_similarities.append(min_dist)
    acf_similarity = np.mean(acf_similarities)

    # 计算 ACF Diversity（生成数据之间的 ACF 差异）
    acf_differences = []
    for _ in range(num_pairs):
        idx1, idx2 = np.random.choice(batch_size_g, 2, replace=False)
        diff = np.linalg.norm(acf_gen[idx1] - acf_gen[idx2])
        acf_differences.append(diff)
    acf_diversity = np.mean(acf_differences)

    # 计算综合 Diversity Score
    epsilon = 1e-3  # 防止分母为零
    diversity_score = acf_diversity / (acf_similarity + epsilon)
    return diversity_score
import pandas as pd
# 示例用法
if __name__ == "__main__":
    def load_and_reshape(file_path):
        data = pd.read_csv(file_path).values[:, 1:26]  # 假设特征从第二列开始
        # 计算每个样本应该有的批大小和形状重塑
        batch_size = data.shape[0] // 24  # 假设每个样本包含24个时间步
        return data.reshape(batch_size, 24 , 25)  # 展平时间步和特征
    # 模拟数据
    
    class_1_ori = np.load("./OUTPUT/PSM_T1_finetuning/samples/PSM_T1_finetuning_ground_truth_24_train.npy")
    real_data = np.load("/home/xuyi/GT_yuan/PSM/1/none/sys_data.npy") # 10 个真实样本，序列长度 50，特征维度 3
    generated_data = np.load("./OUTPUT/PSM_T1_finetuning/ddpm_fake_PSM_T1_finetuning.npy") # 5 个生成样本
    diversity_score = acf_diversity_score(class_1_ori , generated_data, nlags=10, num_pairs=10)
    print(f"diffusion ACF Diversity Score: {diversity_score}")
    diversity_score = acf_diversity_score(class_1_ori , real_data, nlags=10, num_pairs=10)
    print(f"gtgan ACF Diversity Score: {diversity_score}")
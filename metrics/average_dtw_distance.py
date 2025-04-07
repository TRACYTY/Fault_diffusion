import numpy as np
from fastdtw import fastdtw  # 使用 fastdtw 库计算 DTW 距离
from scipy.spatial.distance import euclidean

def average_dtw_distance(real_data, generated_data):
    """
    计算生成数据与真实数据的平均 DTW 距离，评估真实性。
    
    参数:
        real_data: np.ndarray, 形状 (batch_size_r, seq_length, feature_dim)
            真实故障时序数据张量
        generated_data: np.ndarray, 形状 (batch_size_g, seq_length, feature_dim)
            生成故障时序数据张量
    
    返回:
        avg_dtw: float
            平均 DTW 距离，越小说明生成数据越接近真实数据
    """
    # 确保输入是 NumPy 数组
    if not isinstance(real_data, np.ndarray):
        real_data = real_data.numpy()
    if not isinstance(generated_data, np.ndarray):
        generated_data = generated_data.numpy()

    batch_size_r, seq_length, feature_dim = real_data.shape
    batch_size_g = generated_data.shape[0]

    dtw_distances = []
    
    # 对每个生成样本，找到与其 DTW 距离最小的真实样本
    for i in range(batch_size_g):
        gen_seq = generated_data[i]  # 形状: (seq_length, feature_dim)
        min_dtw_dist = float('inf')
        
        for j in range(batch_size_r):
            real_seq = real_data[j]  # 形状: (seq_length, feature_dim)
            # 计算两个多维时序的 DTW 距离
            dist, _ = fastdtw(gen_seq, real_seq, dist=euclidean)
            min_dtw_dist = min(min_dtw_dist, dist)
        
        dtw_distances.append(min_dtw_dist)
    
    # 计算平均 DTW 距离
    avg_dtw = np.mean(dtw_distances)
    return avg_dtw

# # 示例用法
# if __name__ == "__main__":
#     # 模拟数据
#     real_data = np.random.randn(10, 50, 3)  # 10 个真实样本，序列长度 50，特征维度 3
#     generated_data = np.random.randn(5, 50, 3)  # 5 个生成样本
#     avg_dtw = average_dtw_distance(real_data, generated_data)
#     print(f"Average DTW Distance: {avg_dtw}")
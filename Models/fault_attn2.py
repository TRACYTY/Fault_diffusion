import torch
import torch.nn as nn
import torch.nn.functional as F

class FaultAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, window_size=5, num_heads=4, top_k=10, low_rank_dim=32):
        """
        参数:
        - input_dim: 输入特征维度，例如 64
        - hidden_dim: 注意力隐藏层维度，例如 128
        - window_size: 局部窗口大小
        - num_heads: 多头注意力头数
        - top_k: 稀疏注意力选择的 Top-K 个时间步
        - low_rank_dim: 低秩近似的投影维度
        """
        super(FaultAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.top_k = top_k
        self.low_rank_dim = low_rank_dim
        self.head_dim = hidden_dim // num_heads  # 每个头的维度，例如 128 / 4 = 32

        # 多尺度卷积层，确保输出通道数总和等于 hidden_dim
        base_channels = hidden_dim // 3
        remainder = hidden_dim % 3
        self.conv1 = nn.Conv1d(input_dim, base_channels + (remainder > 0), kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_dim, base_channels + (remainder > 1), kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(input_dim, base_channels, kernel_size=7, padding=3)

        # Q, K, V 线性变换
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        # 低秩投影层
        self.key_proj = nn.Linear(self.head_dim, low_rank_dim)
        self.value_proj = nn.Linear(self.head_dim, low_rank_dim)

        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # LayerNorm 规范化
        self.norm = nn.LayerNorm(hidden_dim)

    def multi_scale_conv(self, x):
        """
        多尺度卷积特征提取，确保输出长度和通道数正确
        """
        batch_size, seq_len, _ = x.shape  # 例如 (2, 128, 64)
        x = x.transpose(1, 2)  # (2, 64, 128)
        conv1_out = F.relu(self.conv1(x))  # (2, base_channels + ..., 128)
        conv2_out = F.relu(self.conv2(x))
        conv3_out = F.relu(self.conv3(x))
        x = torch.cat([conv1_out, conv2_out, conv3_out], dim=1)  # (2, hidden_dim, 128)
        x = x.transpose(1, 2)  # (2, 128, hidden_dim)
        return x

    def get_local_window(self, tensor, pos):
        """
        获取局部窗口，切片 seq_len 维度
        """
        seq_len = tensor.size(2)  # 序列长度在第2维
        half_window = self.window_size // 2
        start = max(0, pos - half_window)
        end = min(seq_len, pos + half_window + 1)
        return tensor[:, :, start:end, :]  # [batch_size, num_heads, window_size, head_dim]

    def forward(self, x):
        """
        输入: (batch_size, seq_len, input_dim)，例如 (2, 128, 64)
        输出: (batch_size, seq_len, hidden_dim)，例如 (2, 128, 128)
        """
        batch_size, seq_len, _ = x.size()

        # 1. 多尺度卷积提取特征
        x = self.multi_scale_conv(x)  # (batch_size, seq_len, hidden_dim)

        # 2. 生成 Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 形状: (batch_size, num_heads, seq_len, head_dim)

        # 低秩投影
        K_proj = self.key_proj(K.reshape(batch_size * self.num_heads, seq_len, self.head_dim))
        V_proj = self.value_proj(V.reshape(batch_size * self.num_heads, seq_len, self.head_dim))
        K_proj = K_proj.view(batch_size, self.num_heads, seq_len, self.low_rank_dim)
        V_proj = V_proj.view(batch_size, self.num_heads, seq_len, self.low_rank_dim)

        # 3. 局部稀疏注意力
        outputs = []
        for t in range(seq_len):
            # 局部窗口
            local_K = self.get_local_window(K, t)  # [batch_size, num_heads, window_size, head_dim]
            local_V = self.get_local_window(V, t)
            q_t = Q[:, :, t, :].unsqueeze(2)  # [batch_size, num_heads, 1, head_dim]

            # 局部注意力得分
            local_scores = torch.einsum('bhqd,bhkd->bhqk', q_t, local_K) / (self.head_dim ** 0.5)
            local_weights = F.softmax(local_scores, dim=-1)
            local_out = torch.einsum('bhqk,bhkd->bhqd', local_weights, local_V)  # [batch_size, num_heads, 1, head_dim]

            # Top-K 稀疏注意力（低秩投影）
            global_scores = torch.einsum('bhqd,bhkd->bhqk', q_t, K_proj) / (self.low_rank_dim ** 0.5)
            top_k_scores, top_k_indices = global_scores.topk(self.top_k, dim=-1)
            top_k_weights = F.softmax(top_k_scores, dim=-1)
            top_k_indices = top_k_indices.squeeze(2)  # [2, 4, 10]
            top_k_V = torch.gather(V_proj, 2, top_k_indices.unsqueeze(-1).expand(-1, -1, -1, self.low_rank_dim))
            sparse_out = torch.einsum('bhqk,bhkd->bhqd', top_k_weights, top_k_V)

            # 融合局部和稀疏结果（维度对齐）
            sparse_out = self.value_proj(sparse_out.reshape(-1, self.low_rank_dim)).reshape(
                batch_size, self.num_heads, 1, self.head_dim)
            out_t = (local_out + sparse_out) / 2
            outputs.append(out_t)

        # 4. 重组输出
        output = torch.cat(outputs, dim=2)  # [batch_size, num_heads, seq_len, head_dim]
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(output)
        output = self.norm(output + x)  # 残差连接

        return output

# 测试代码
if __name__ == "__main__":
    batch_size, seq_len, input_dim = 2, 128, 64
    hidden_dim = 128
    x = torch.randn(batch_size, seq_len, input_dim)

    model = FaultAttention(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        window_size=5,
        num_heads=4,
        top_k=10,
        low_rank_dim=32
    )

    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
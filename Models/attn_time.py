import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# 标准自注意力实现
class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 生成 Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 自注意力计算
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)

        # 输出重组
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        out = self.out_proj(out)
        return out

# FaultAttention 实现（你的模型）
class FaultAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, window_size=5, num_heads=4, top_k=10, low_rank_dim=32):
        super(FaultAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.top_k = top_k
        self.low_rank_dim = low_rank_dim
        self.head_dim = hidden_dim // num_heads

        # 多尺度卷积
        base_channels = hidden_dim // 3
        remainder = hidden_dim % 3
        self.conv1 = nn.Conv1d(input_dim, base_channels + (remainder > 0), kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_dim, base_channels + (remainder > 1), kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(input_dim, base_channels, kernel_size=7, padding=3)

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(self.head_dim, low_rank_dim)
        self.value_proj = nn.Linear(self.head_dim, low_rank_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def multi_scale_conv(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.transpose(1, 2)
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(x))
        conv3_out = F.relu(self.conv3(x))
        x = torch.cat([conv1_out, conv2_out, conv3_out], dim=1)
        x = x.transpose(1, 2)
        return x

    def get_local_window(self, tensor, pos):
        seq_len = tensor.size(2)
        half_window = self.window_size // 2
        start = max(0, pos - half_window)
        end = min(seq_len, pos + half_window + 1)
        return tensor[:, :, start:end, :]

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.multi_scale_conv(x)

        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        K_proj = self.key_proj(K.reshape(batch_size * self.num_heads, seq_len, self.head_dim))
        V_proj = self.value_proj(V.reshape(batch_size * self.num_heads, seq_len, self.head_dim))
        K_proj = K_proj.view(batch_size, self.num_heads, seq_len, self.low_rank_dim)
        V_proj = V_proj.view(batch_size, self.num_heads, seq_len, self.low_rank_dim)

        outputs = []
        for t in range(seq_len):
            local_K = self.get_local_window(K, t)
            local_V = self.get_local_window(V, t)
            q_t = Q[:, :, t, :].unsqueeze(2)

            local_scores = torch.einsum('bhqd,bhkd->bhqk', q_t, local_K) / (self.head_dim ** 0.5)
            local_weights = F.softmax(local_scores, dim=-1)
            local_out = torch.einsum('bhqk,bhkd->bhqd', local_weights, local_V)

            global_scores = torch.einsum('bhqd,bhkd->bhqk', q_t, K_proj) / (self.low_rank_dim ** 0.5)
            top_k_scores, top_k_indices = global_scores.topk(self.top_k, dim=-1)
            top_k_weights = F.softmax(top_k_scores, dim=-1)
            top_k_indices = top_k_indices.squeeze(2)
            top_k_V = torch.gather(V_proj, 2, top_k_indices.unsqueeze(-1).expand(-1, -1, -1, self.low_rank_dim))
            sparse_out = torch.einsum('bhqk,bhkd->bhqd', top_k_weights, top_k_V)

            sparse_out = self.value_proj(sparse_out.reshape(-1, self.low_rank_dim)).reshape(
                batch_size, self.num_heads, 1, self.head_dim)
            out_t = (local_out + sparse_out) / 2
            outputs.append(out_t)

        output = torch.cat(outputs, dim=2)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(output)
        output = self.norm(output + x)
        return output

# 计时函数
def measure_time(model, x, device, num_runs=100):
    model = model.to(device)
    x = x.to(device)
    torch.cuda.synchronize() if device.type == 'cuda' else None  # 同步 GPU

    # 预热运行
    for _ in range(10):
        _ = model(x)

    # 正式计时
    start = time.time()
    for _ in range(num_runs):
        _ = model(x)
    torch.cuda.synchronize() if device.type == 'cuda' else None  # 同步 GPU
    end = time.time()

    avg_time = (end - start) / num_runs
    return avg_time

# 主测试代码
if __name__ == "__main__":
    # 参数设置
    batch_size, seq_len, input_dim = 2, 128, 64
    hidden_dim = 128
    num_heads = 4
    window_size = 5
    top_k = 10
    low_rank_dim = 32

    # 输入数据
    x = torch.randn(batch_size, seq_len, input_dim)

    # 模型实例化
    self_attn = SelfAttention(input_dim, hidden_dim, num_heads)
    fault_attn = FaultAttention(input_dim, hidden_dim, window_size, num_heads, top_k, low_rank_dim)

    # 测试设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 测量时间
    num_runs = 100
    self_attn_time = measure_time(self_attn, x, device, num_runs)
    fault_attn_time = measure_time(fault_attn, x, device, num_runs)

    # 输出结果
    print(f"Self-Attention avg time per run: {self_attn_time:.6f} seconds")
    print(f"FaultAttention avg time per run: {fault_attn_time:.6f} seconds")
    print(f"Speedup (Self-Attention / FaultAttention): {self_attn_time / fault_attn_time:.2f}x")
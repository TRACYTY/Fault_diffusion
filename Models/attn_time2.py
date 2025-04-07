import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# 标准自注意力模块
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
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        out = self.out_proj(out)
        return out

# 优化后的 FaultAttention 模块
class FaultAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, window_size=5, num_heads=4, top_k=10):
        super(FaultAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.top_k = top_k
        self.head_dim = hidden_dim // num_heads

        # 多尺度卷积层
        base_channels = hidden_dim // 3
        remainder = hidden_dim % 3
        self.conv1 = nn.Conv1d(input_dim, base_channels + (remainder > 0), kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_dim, base_channels + (remainder > 1), kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(input_dim, base_channels, kernel_size=7, padding=3)

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
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

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.multi_scale_conv(x)

        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 局部注意力
        half_window = self.window_size // 2
        K_padded = F.pad(K, (0, 0, half_window, half_window), mode='constant')
        V_padded = F.pad(V, (0, 0, half_window, half_window), mode='constant')
        local_K = K_padded.unfold(2, self.window_size, 1)  # [B, H, N, head_dim, window_size]
        local_V = V_padded.unfold(2, self.window_size, 1)
        Q_expanded = Q.unsqueeze(-1)  # [B, H, N, head_dim, 1]
        local_scores = torch.einsum('bhndw,bhndv->bhnwv', local_K, Q_expanded) / (self.head_dim ** 0.5)
        local_weights = F.softmax(local_scores, dim=-2)
        local_out = torch.einsum('bhnwv,bhndv->bhnd', local_weights, local_V)

        # Top-K 稀疏注意力
        global_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)
        top_k_scores, top_k_indices = global_scores.topk(self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_scores, dim=-1)

        # 修正 top_k_V
        top_k_indices_expanded = top_k_indices.unsqueeze(-1).repeat(1, 1, 1, 1, self.head_dim)
        top_k_V = torch.gather(V, 2, top_k_indices_expanded)
        sparse_out = torch.einsum('bhnt,bhntd->bhnd', top_k_weights, top_k_V)

        # 融合
        out = (local_out + sparse_out) / 2
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        out = self.out_proj(out)
        out = self.norm(out + x)
        return out

# 计时函数
def measure_time(model, x, device, num_runs=100):
    model = model.to(device)
    x = x.to(device)
    torch.cuda.synchronize() if device.type == 'cuda' else None

    for _ in range(10):  # 预热
        _ = model(x)

    start = time.time()
    for _ in range(num_runs):
        _ = model(x)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end = time.time()

    return (end - start) / num_runs

# 测试代码
if __name__ == "__main__":
    batch_size, seq_len, input_dim = 2, 128, 64
    hidden_dim = 128
    num_heads = 4
    window_size = 5
    top_k = 10

    x = torch.randn(batch_size, seq_len, input_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    self_attn = SelfAttention(input_dim, hidden_dim, num_heads)
    fault_attn = FaultAttention(input_dim, hidden_dim, window_size, num_heads, top_k)

    self_attn_time = measure_time(self_attn, x, device)
    fault_attn_time = measure_time(fault_attn, x, device)

    print(f"Self-Attention avg time: {self_attn_time:.6f} seconds")
    print(f"FaultAttention avg time: {fault_attn_time:.6f} seconds")
    print(f"Speedup (Self-Attention / FaultAttention): {self_attn_time / fault_attn_time:.2f}x")
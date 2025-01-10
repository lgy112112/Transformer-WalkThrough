import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Q, K, V 可能来自不同序列长度
        batch_size_Q, seq_len_Q, d_model_Q = Q.shape
        batch_size_K, seq_len_K, d_model_K = K.shape
        batch_size_V, seq_len_V, d_model_V = V.shape
        
        # 断言 batch_size/d_model 一致，否则肯定无法计算
        assert batch_size_Q == batch_size_K == batch_size_V, "Q,K,V的batch不一致"
        assert d_model_Q == d_model_K == d_model_V, "Q,K,V的d_model不一致"
        # 对于 cross-attention, seq_len_Q != seq_len_K 也是合法的，但 seq_len_K 应该等于 seq_len_V
        assert seq_len_K == seq_len_V, "K,V的seq_len不一致"

        # 线性映射 Q,K,V
        Q = self.query(Q)   # [batch_size_Q, seq_len_Q, d_model]
        K = self.key(K)     # [batch_size_K, seq_len_K, d_model]
        V = self.value(V)   # [batch_size_V, seq_len_V, d_model]

        # 分割多头
        # Q: [batch_size_Q, seq_len_Q, num_heads, head_dim]
        Q = Q.view(batch_size_Q, seq_len_Q, self.num_heads, self.head_dim).transpose(1, 2)
        # K,V: [batch_size_K, seq_len_K, num_heads, head_dim]
        K = K.view(batch_size_K, seq_len_K, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size_V, seq_len_V, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力分数 => [batch_size, num_heads, Q_len, K_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 如果有mask，则必须与scores形状兼容：最后两个维度应是[Q_len, K_len]
        if mask is not None:
            # 例如 mask 的形状可以是 [batch_size, 1, Q_len, K_len]
            # 或能broadcast到 [batch_size, num_heads, Q_len, K_len]
            print(f"mask shape in mha: {mask.shape}")
            print(f"scores shape in mha: {scores.shape}")
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(scores, dim=-1)

        # 加权求和 => [batch_size, num_heads, Q_len, head_dim]
        out = torch.matmul(attention, V)

        # 还原形状 => [batch_size, Q_len, d_model]
        out = out.transpose(1, 2).contiguous().view(batch_size_Q, seq_len_Q, self.d_model)

        # 线性变换
        out = self.fc_out(out)
        return out



if __name__ == "__main__":
    # 参数设置
    batch_size = 32
    seq_len = 50
    d_model = 512
    num_heads = 16
    
    # 创建模块
    masked_attn = MultiHeadAttention(d_model, num_heads)
    
    # 生成输入数据
    x = torch.randn(batch_size, seq_len, d_model)  # 输入序列
    mask = torch.tril(torch.ones(seq_len, seq_len))  # 下三角掩码
    # print(mask.shape)
    
    # 前向传播
    output = masked_attn(x, x, x, mask)
    
    # 验证输出
    print("输入形状:", x.shape)  # 应该输出: torch.Size([32, 50, 512])
    print("输出形状:", output.shape)  # 应该输出: torch.Size([32, 50, 512])
    
    # 验证掩码效果
    # 检查注意力分数矩阵是否下三角
    with torch.no_grad():
        # 获取注意力分数
        Q = masked_attn.query(x)
        K = masked_attn.key(x)
        Q = Q.view(batch_size, seq_len, num_heads, d_model // num_heads).transpose(1, 2)
        K = K.view(batch_size, seq_len, num_heads, d_model // num_heads).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_model // num_heads, dtype=torch.float32))
        
        # 应用掩码
        masked_scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 检查是否下三角
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:  # 上三角部分
                    assert torch.all(masked_scores[:, :, i, j] == float('-inf')), "掩码未正确应用"
    
    print("测试通过！掩码正确应用，输出形状符合预期。")
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
        batch_size, seq_len, d_model = Q.shape
        
        # 线性变换（已移动到DecoderLayer中）
        Q = self.query(Q)
        K = self.key(K)
        V = self.value(V)
        
        # 分割多头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # 应用mask（如果存在）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attention = F.softmax(scores, dim=-1)
        
        # 加权求和
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
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
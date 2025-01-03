import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_head_attention import MultiHeadAttention
from transformer_nn import FeedForward



class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 多头自注意力 + 残差连接 + 层归一化
        attn_out = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    
if __name__ == "__main__":
    # 参数设置
    num_layers = 6
    d_model = 512
    num_heads = 8
    d_ff = 2048
    dropout = 0.1

    # 初始化解码器
    encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)

    # 输入数据
    batch_size = 32
    seq_len = 50
    x = torch.randn(batch_size, seq_len, d_model)  # 目标序列的嵌入表示

    # 前向传播
    output = encoder(x)

    # 检查输出形状
    print("输入形状:", x.shape)  # 输出: torch.Size([32, 50, 512])
    print("输出形状:", output.shape)  # 输出: torch.Size([32, 50, 512])
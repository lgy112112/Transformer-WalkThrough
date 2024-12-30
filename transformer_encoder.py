import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_head_attention import MultiHeadAttention


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # 第一层全连接
        self.linear2 = nn.Linear(d_ff, d_model)  # 第二层全连接
        self.activation = nn.ReLU()  # 激活函数

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = self.linear1(x)  # (batch_size, seq_len, d_ff)
        x = self.activation(x)  # 非线性变换
        x = self.linear2(x)  # (batch_size, seq_len, d_model)
        return x



class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

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
        attn_out = self.attention(x)
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
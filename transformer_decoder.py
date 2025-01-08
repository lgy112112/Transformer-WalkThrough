import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from transformer_nn import FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads)
        self.encoder_decoder_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, tgt_mask, src_mask):
        # 掩码多头自注意力
        attn_output = self.masked_self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 编码器-解码器注意力 + 残差连接 + 层归一化
        attn_output = self.encoder_decoder_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    
    
class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, encoder_output, tgt_mask, src_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        return x


if __name__ == "__main__":
    # 参数设置
    num_layers = 6
    d_model = 512
    num_heads = 8
    d_ff = 2048
    dropout = 0.1

    # 初始化解码器
    decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)

    # 输入数据
    batch_size = 32
    seq_len = 50
    x = torch.randn(batch_size, seq_len, d_model)  # 目标序列的嵌入表示
    encoder_output = torch.randn(batch_size, seq_len, d_model)  # 编码器的输出

    # 掩码
    tgt_mask = torch.tril(torch.ones(seq_len, seq_len))  # 目标序列的掩码
    src_mask = torch.ones(batch_size, seq_len, seq_len)  # 源序列的掩码（全1，假设没有填充）

    # 将掩码扩展为所需形状
    tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
    src_mask = src_mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]

    # 前向传播
    output = decoder(x, encoder_output, tgt_mask, src_mask)

    # 检查输出形状
    print("输入形状:", x.shape)  # 输出: torch.Size([32, 50, 512])
    print("输出形状:", output.shape)  # 输出: torch.Size([32, 50, 512])

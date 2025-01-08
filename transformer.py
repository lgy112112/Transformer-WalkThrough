import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_decoder import Decoder
from transformer_encoder import Encoder
from transformer_preprocess import TransformerPreprocessor
from transformer_nn import FeedForward

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout=0.1):
        super(Transformer, self).__init__()
        self.preprocessor = TransformerPreprocessor(vocab_size, d_model, max_seq_len)
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)  # 输出层，用于预测下一个词的概率

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 预处理输入
        src_emb = self.preprocessor(src)
        tgt_emb = self.preprocessor(tgt)

        # 编码器前向传播
        encoder_output = self.encoder(src_emb)

        # 解码器前向传播
        decoder_output = self.decoder(tgt_emb, encoder_output, tgt_mask, src_mask)

        # 输出层
        output = self.fc_out(decoder_output)
        return output

    def make_src_mask(self, src, src_pad_idx):
        src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(torch.bool)  # 使用 bool 类型

    def make_trg_mask(self, trg, trg_pad_idx):
        trg_pad_mask = (trg != trg_pad_idx).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, seq_len, 1]
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).to(trg.device)  # [seq_len, seq_len]
        trg_sub_mask = trg_sub_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        trg_sub_mask = trg_sub_mask.to(torch.bool)  # 将 trg_sub_mask 转换为 bool 类型
        trg_mask = trg_pad_mask & trg_sub_mask  # [batch_size, 1, seq_len, seq_len]
        return trg_mask.to(torch.bool)  # 使用 bool 类型

if __name__ == "__main__":
    # 参数设置
    vocab_size = 10000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_len = 100
    dropout = 0.1
    batch_size = 32
    seq_len = 50
    src_pad_idx = 0  # 假设填充索引为0
    trg_pad_idx = 0  # 假设填充索引为0

    # 初始化Transformer模型
    transformer = Transformer(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout)

    # 生成输入数据
    src = torch.randint(0, vocab_size, (batch_size, seq_len))  # 源序列
    tgt = torch.randint(0, vocab_size, (batch_size, seq_len))  # 目标序列

    # 生成掩码
    src_mask = transformer.make_src_mask(src, src_pad_idx)  # 源序列的掩码
    tgt_mask = transformer.make_trg_mask(tgt, trg_pad_idx)  # 目标序列的掩码

    # 前向传播
    output = transformer(src, tgt, src_mask, tgt_mask)

    # 检查输出形状
    print("输出形状:", output.shape)  # 应该输出: torch.Size([32, 50, 10000])
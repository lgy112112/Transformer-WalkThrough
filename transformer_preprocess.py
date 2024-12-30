import torch
import torch.nn as nn
import math

class InputPreprocessor(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len):
        super(InputPreprocessor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = self._generate_position_encoding(max_seq_len, d_model)
        
    def _generate_position_encoding(self, max_seq_len, d_model):
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_seq_len, d_model)
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        seq_len = x.size(1)
        embeddings = self.embedding(x)  # (batch_size, seq_len, d_model)
        position_encodings = self.position_encoding[:, :seq_len, :]  # (1, seq_len, d_model)
        return embeddings + position_encodings  # (batch_size, seq_len, d_model)

# 使用示例
vocab_size = 10000
d_model = 512
max_seq_len = 100
batch_size = 32
seq_len = 50

preprocessor = InputPreprocessor(vocab_size, d_model, max_seq_len)
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))  # 随机生成输入
output = preprocessor(input_ids)
print(output.shape)  # 输出: torch.Size([32, 50, 512])
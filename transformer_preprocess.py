import torch
import torch.nn as nn
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformerPreprocessor(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len):
        super(TransformerPreprocessor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, max_seq_len)
        
    def forward(self, x):
        # x: (batch_size, seq_len)
        embeddings = self.embedding(x)  # (batch_size, seq_len, d_model)
        output = self.position_encoding(embeddings)  # (batch_size, seq_len, d_model)
        return output

class PositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        self.max_len = max_len
        self.pe = self._generate_position_encoding()
        
    def _generate_position_encoding(self):
        position = torch.arange(self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * 
                           -(math.log(10000.0) / self.d_model))
        pe = torch.zeros(self.max_len, self.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)
    
    def __call__(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x.to(device)
        seq_len = x.size(1)
        pe_result = self.pe[:, :seq_len, :]
        pe_result = pe_result.to(device)
        return x + pe_result


if __name__ == "__main__":
    # 使用示例
    vocab_size = 10000
    d_model = 512
    max_seq_len = 100
    batch_size = 32
    seq_len = 50

    preprocessor = TransformerPreprocessor(vocab_size, d_model, max_seq_len)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))  # 随机生成输入
    output = preprocessor(input_ids)
    print(output.shape)  # 输出: torch.Size([32, 50, 512])
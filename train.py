import requests
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import GPT2Tokenizer
from tqdm import tqdm
import csv
from transformer import Transformer

# 1. 下载数据（如果文件不存在）
if not os.path.exists('shakespeare.txt'):
    print("下载 Shakespeare 数据集...")
    url = 'https://www.gutenberg.org/files/100/100-0.txt'
    response = requests.get(url)
    text = response.content.decode('utf-8-sig')  # 移除 UTF-8 BOM
    with open('shakespeare.txt', 'w', encoding='utf-8') as file:
        file.write(text)
    print("下载完成并保存为 shakespeare.txt")

# 2. 加载数据
with open('shakespeare.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# 3. 使用 GPT2 的分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokens = tokenizer.encode(text)

# 4. 创建数据集
class ShakespeareDataset(Dataset):
    def __init__(self, tokens, seq_length):
        self.tokens = tokens
        self.seq_length = seq_length

    def __len__(self):
        return len(self.tokens) // self.seq_length

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length
        return torch.tensor(self.tokens[start:end], dtype=torch.long)

# 5. 划分训练集和验证集
seq_length = 128
dataset = ShakespeareDataset(tokens, seq_length)
train_size = int(0.8 * len(dataset))  # 80% 训练集
val_size = len(dataset) - train_size  # 20% 验证集
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8)

# 6. 定义模型、损失函数和优化器
model = Transformer(
    vocab_size=tokenizer.vocab_size,
    d_model=512,
    num_heads=8,
    num_layers=2,
    d_ff=2048,
    max_seq_len=seq_length
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 7. 初始化变量
best_val_loss = float('inf')  # 用于保存最佳验证损失
loss_log = []  # 用于保存每轮的训练和验证损失

# 8. 训练循环
for epoch in range(10):
    model.train()  # 切换到训练模式
    epoch_train_loss = 0  # 用于累积每个 epoch 的训练 loss

    # 训练阶段
    with tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", unit="batch") as pbar:
        for batch in pbar:
            optimizer.zero_grad()
            output = model(batch[:, :-1], batch[:, 1:])
            loss = criterion(output.view(-1, tokenizer.vocab_size), batch[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()

            # 更新 epoch_train_loss
            epoch_train_loss += loss.item()

            # 实时显示当前 batch 的 loss
            pbar.set_postfix({"Batch Loss": loss.item()})

    # 计算平均训练 loss
    avg_train_loss = epoch_train_loss / len(train_loader)

    # 验证阶段
    model.eval()  # 切换到验证模式
    epoch_val_loss = 0  # 用于累积每个 epoch 的验证 loss

    with torch.no_grad():  # 禁用梯度计算
        for batch in val_loader:
            output = model(batch[:, :-1], batch[:, 1:])
            loss = criterion(output.view(-1, tokenizer.vocab_size), batch[:, 1:].reshape(-1))
            epoch_val_loss += loss.item()

    # 计算平均验证 loss
    avg_val_loss = epoch_val_loss / len(val_loader)

    # 打印训练和验证 loss
    print(f"Epoch [{epoch+1}/10], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # 保存损失到日志
    loss_log.append([epoch + 1, avg_train_loss, avg_val_loss])

    # 保存最佳模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_weights = model.state_dict()  # 暂存最佳模型权重
        print(f"验证损失降低，暂存最佳模型权重")

    # 保存最终的最佳模型
    if epoch == 9:  # 最后一轮训练结束后
        if best_model_weights is not None:
            best_model_filename = f'best_model_val_loss={best_val_loss:.4f}.pth'
            torch.save(best_model_weights, best_model_filename)
            print(f"训练完成，保存最佳模型为 {best_model_filename}")

# 9. 保存最终模型
torch.save(model.state_dict(), 'final_model.pth')
print("训练完成，保存最终模型为 final_model.pth")

# 10. 保存损失日志到 CSV
with open('loss_log.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Loss", "Val Loss"])  # 写入表头
    writer.writerows(loss_log)  # 写入数据
print("损失日志已保存为 loss_log.csv")
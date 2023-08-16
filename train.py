import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score
from torch.utils.data import DataLoader
from transformers import AdamW
from model.OffensivenessDetector import OffensivenessDetector, tokenizer
from utils.CustomDataset import CustomDataset

# 加载数据
train_data = pd.read_csv("./COLDataset/train.csv")
test_data = pd.read_csv("./COLDataset/test.csv")
dev_data = pd.read_csv("./COLDataset/dev.csv")

# 定义超参数
max_length = 128
batch_size = 16
epochs = 10
learning_rate = 1e-5

# 初始化模型和优化器
offensivenessDetector = OffensivenessDetector()
optimizer = AdamW(offensivenessDetector.parameters(), lr=learning_rate)

# 创建数据集和数据加载器
train_dataset = CustomDataset(train_data, tokenizer, max_length)
test_dataset = CustomDataset(test_data, tokenizer, max_length)
dev_dataset = CustomDataset(dev_data, tokenizer, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

# 定义损失函数
criterion = nn.BCELoss()

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
offensivenessDetector.to(device)

for epoch in range(epochs):
    offensivenessDetector.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = offensivenessDetector(input_ids, attention_mask)
        loss = criterion(outputs.squeeze(), labels.float())
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(train_loader)

    # 验证模型
    offensivenessDetector.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].tolist()

            outputs = offensivenessDetector(input_ids, attention_mask)
            preds = torch.round(outputs.squeeze()).tolist()

            all_labels.extend(labels)
            all_preds.extend(preds)

    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Average Loss: {average_loss:.4f}, "
        f"Accuracy: {accuracy:.4f}, "
        f"Recall: {recall:.4f}, F1: {f1:.4f}"
    )

# 在测试集上评估模型
offensivenessDetector.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].tolist()

        outputs = offensivenessDetector(input_ids, attention_mask)
        preds = torch.round(outputs.squeeze()).tolist()

        all_labels.extend(labels)
        all_preds.extend(preds)

accuracy = accuracy_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f"Test Accuracy: {accuracy:.4f}, Test Recall: {recall:.4f}, Test F1: {f1:.4f}")

# 保存模型
torch.save(offensivenessDetector.state_dict(), "offensiveness_model.pth")

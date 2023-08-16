import torch
from transformers import AutoTokenizer
from model.OffensivenessDetector import OffensivenessDetector
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# 使用训练完的模型进行分类
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = OffensivenessDetector().to('cpu')
model.load_state_dict(torch.load("offensiveness_model.pth", map_location=torch.device('cpu')))
model.eval()

# 读取测试数据
data = pd.read_csv("./COLDataset/test.csv")

# 初始化评估指标列表
accuracies = []
recalls = []
precisions = []
f1_scores = []

for label_value in [0, 1, 2, 3]:
    subset = data[data['fine-grained-label'] == label_value]
    print(subset.shape[0])
    texts = subset['TEXT'].tolist()
    labels = subset['label'].tolist()

    encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    input_ids = encoded_inputs['input_ids']
    attention_mask = encoded_inputs['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    preds = torch.round(outputs.squeeze()).tolist()

    # 计算准确率
    accuracy = accuracy_score(labels, preds)
    accuracies.append(accuracy)

    # 计算召回率
    recall = recall_score(labels, preds)
    recalls.append(recall)

    # 计算精确率
    precision = precision_score(labels, preds)
    precisions.append(precision)

    # 计算F1值
    f1 = f1_score(labels, preds)
    f1_scores.append(f1)

# 输出每个类别下的评估指标
for i, (accuracy, recall, precision, f1) in enumerate(zip(accuracies, recalls, precisions, f1_scores)):
    print(f'Class {i}: Accuracy = {accuracy:.4f}, Recall = {recall:.4f}, Precision = {precision:.4f}, F1 = {f1:.4f}')

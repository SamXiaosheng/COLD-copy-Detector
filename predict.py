# 使用训练完的模型进行分类
import torch
from transformers import AutoTokenizer
from model.OffensivenessDetector import OffensivenessDetector

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = OffensivenessDetector().to('cpu')
model.load_state_dict(torch.load("offensiveness_model.pth", map_location=torch.device('cpu')))
model.eval()
text = "黑人不行啊"
encoding = tokenizer.encode_plus(
    text,
    truncation=True,
    padding="max_length",
    max_length=128,
    return_tensors="pt"
)

input_ids = encoding["input_ids"].squeeze().unsqueeze(0)
attention_mask = encoding["attention_mask"].squeeze().unsqueeze(0)

# 将输入传递给模型进行预测
with torch.no_grad():
    output = model(input_ids, attention_mask)

# 对输出进行后处理
prediction = torch.round(output).item()

print(int(prediction))

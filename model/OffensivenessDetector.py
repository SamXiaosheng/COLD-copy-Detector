import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")


# 定义OffensivenessDetector模型
class OffensivenessDetector(nn.Module):
    def __init__(self):
        super(OffensivenessDetector, self).__init__()
        self.bert = model.bert
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.linear(cls_output)
        probs = self.sigmoid(logits)
        return probs

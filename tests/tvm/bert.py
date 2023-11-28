import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

# Define the BERT model
class BertForCustomTask(nn.Module):
    def __init__(self, num_labels):
        super(BertForCustomTask, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Example usage
num_labels = 2  # Number of labels for your custom task

# Instantiate the BERT model for your custom task
model = BertForCustomTask(num_labels)

# Save the model weights
torch.save(model.state_dict(), 'bert_model_weights.pth')
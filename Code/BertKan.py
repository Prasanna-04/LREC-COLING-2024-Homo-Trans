## loading of important libraries
import torch
from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration, AdamW

import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch import nn
from functions import *
import yaml

with open('config.yaml') as file:
    data = yaml.safe_load(file)

## importing pre-trained model and tokenizer from hugging face
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("l3cube-pune/kannada-bert")
model = AutoModelForMaskedLM.from_pretrained("l3cube-pune/kannada-bert")

## definining where the model weights will be stored
# model_name = "Bert_Kan"
wts_path = data['model_file']

num_epochs = data['num_epochs']
batch_size = data['batch_size']

train_loader, test_loader, dev_loader = get_loaders("Kannada")

## creating Pytorch model, modifying output classification head
class Xlmr(nn.Module):
    def __init__(self, hf_mod):
        super(Xlmr, self).__init__()
        self.model = hf_mod.bert
        self.pooler = nn.Linear(in_features=768, out_features=192, bias=True)
        self.dropout = nn.Dropout(p = 0.1, inplace = False)
        self.out_proj = nn.Linear(in_features=192, out_features=3, bias=True)

    def forward(self, x, attention_mask):
        out = self.model(x, attention_mask = attention_mask).last_hidden_state
        out = torch.mean(out, dim = 1)
        out = self.pooler(out)
        out = self.dropout(out)
        out = self.out_proj(out)

        return out

xlmr = Xlmr(model).to('cuda') ### instantiating model, loading to cuda

optimizer = torch.optim.Adam(xlmr.parameters(), lr = data['learning_rate'], betas = (0.9, 0.999), eps = 1e-8)
criterion = nn.CrossEntropyLoss()

## fixing encoder part parameters of the model
for p in xlmr.parameters():
    p.requires_grad = False

for p in xlmr.pooler.parameters():
    p.requires_grad = True

for p in xlmr.out_proj.parameters():
    p.requires_grad = True

for p in xlmr.dropout.parameters():
    p.requires_grad = True

xlmr.load_state_dict(torch.load(wts_path))

# losses, xlmr = train(xlmr, optimizer, criterion, wts_path, train_loader, dev_loader)

acc, y_true, y_preds = test(xlmr, criterion, wts_path, test_loader)
print(acc)

torch.cuda.empty_cache()
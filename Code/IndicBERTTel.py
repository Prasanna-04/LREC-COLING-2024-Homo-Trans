import torch
from transformers import T5Tokenizer
from transformers import T5ForConditionalGeneration, AdamW

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import pandas as pd
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch import nn
import matplotlib.pyplot as plt
from functions import *
import yaml

with open('config.yaml') as file:
    data = yaml.safe_load(file)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from transformers import AutoModel
tokenizer = AutoTokenizer.from_pretrained("IIIT-L/indic-bert-finetuned-code-mixed-DS")
model = AutoModel.from_pretrained("ai4bharat/indic-bert")

train_loader, test_loader, dev_loader = get_loaders("Telugu")

# model_name = "IndicTelugu"
wts_path = data['model_file']

class Xlmr(nn.Module):
    def __init__(self, hf_mod):
        super(Xlmr, self).__init__()
        self.model = hf_mod
        self.dropout = nn.Dropout(p = 0.1, inplace = False)
        self.out_proj = nn.Linear(in_features=768, out_features=3, bias=True)

    def forward(self, x, attention_mask):
        out = self.model(x, attention_mask = attention_mask).pooler_output
        out = self.dropout(out)
        out = self.out_proj(out)

        return out

xlmr = Xlmr(model).to('cuda') ### instantiating model, loading to cuda

optimizer = torch.optim.Adam(xlmr.parameters(), lr = data['learning_rate'], betas = (0.9, 0.999), eps = 1e-8)
criterion = nn.CrossEntropyLoss()
num_epochs = data['num_epochs']
batch_size = data['batch_size']

for p in xlmr.model.parameters():
    p.requires_grad = False

for p in xlmr.model.pooler.parameters():
    p.requires_grad = True

for p in xlmr.out_proj.parameters():
    p.requires_grad = True

for p in xlmr.model.embeddings.parameters():
    p.requires_grad = True

# xlmr.load_state_dict( torch.load(wts_path))

losses, xlmr = train(xlmr, optimizer, criterion, wts_path, train_loader, dev_loader)
torch.save(xlmr.state_dict(), wts_path)

acc, y_true, y_preds = test(xlmr, criterion, wts_path, test_loader)
print(acc)

# torch.cuda.empty_cache()
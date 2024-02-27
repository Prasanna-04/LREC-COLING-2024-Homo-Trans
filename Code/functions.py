import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import os
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoTokenizer
import yaml

with open('config.yaml') as file:
	data = yaml.safe_load(file)

tokenizer = AutoTokenizer.from_pretrained(data['tokenizerHF'])

num_epochs = data['num_epochs']
batch_size = data['batch_size']

def get_loaders(language):
	## loading dataset as dataframe
	lan_data1 = os.path.join("../Dataset", language + "_train.csv")
	lan_data2 = os.path.join("../Dataset", language + "_test.csv")
	lan_data3 = os.path.join("../Dataset", language + "_dev.csv")
	lddf_tr = pd.read_csv(lan_data1)
	lddf_te = pd.read_csv(lan_data2)
	lddf_dev = pd.read_csv(lan_data3)

	text_tr = lddf_tr["Text"]
	text_te = lddf_te["Text"]
	text_dev = lddf_dev["Text"]

	## converting labels to categorical -- 0 1 or 2, converintf
	def words2cat(tddf):
		cats = sorted(list(set(tddf["Category"])))
		cat_vars = {}
		for i in range(len(cats)):
			cat_vars[cats[i]] = i
		
		for i in range(len(tddf["Category"])):
			tddf["Category"][i] = cat_vars[tddf["Category"][i]]

		return tddf, cat_vars

	lddf_tr, cat_vars = words2cat(lddf_tr)
	lddf_te, cat_vars = words2cat(lddf_te)
	lddf_dev, cat_vars = words2cat(lddf_dev)

	size_tr = len(lddf_tr)
	size_te = len(lddf_te)
	size_dev = len(lddf_dev)
	text_all = lddf_tr["Text"].tolist() + lddf_te["Text"].tolist() + lddf_dev["Text"].tolist()

	tot_toks = tokenizer(text_all, truncation=True, max_length=100, padding = 'longest', return_tensors = 'pt')

	x_tr = tot_toks.input_ids[0: size_tr]
	x_te = tot_toks.input_ids[size_tr: size_te + size_tr]
	x_dev = tot_toks.input_ids[size_te + size_tr: size_te + size_tr + size_dev]

	am_tr = tot_toks.attention_mask[0: size_tr]
	am_te = tot_toks.attention_mask[size_tr: size_te + size_tr]
	am_dev = tot_toks.attention_mask[size_te + size_tr: size_te + size_tr + size_dev]

	y_tr = lddf_tr["Category"].tolist()
	y_te = lddf_te["Category"].tolist()
	y_dev = lddf_dev["Category"].tolist()

	## Pytorch dataset class
	class lan_data(Dataset):
		def __init__(self, x, am, y):
			super(lan_data, self).__init__()
			self.toks = x
			self.ams = am
			self.cats = y

		def __len__(self):
			return len(self.toks)

		def __getitem__(self, ix):
			return (self.toks[ix], self.ams[ix], torch.tensor(self.cats[ix]))

	## Pytorch dataloaders
	train_set = lan_data(x_tr, am_tr, y_tr)
	train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, drop_last = True)
	test_set = lan_data(x_te, am_te, y_te)
	test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = True, drop_last = True)
	dev_set = lan_data(x_dev, am_dev, y_dev)
	dev_loader = DataLoader(dev_set, batch_size = batch_size, shuffle = True, drop_last = True)

	return train_loader, test_loader, dev_loader

def train(model, optimizer, criterion, wts_path, train_loader, dev_loader):
	losses = []

	for epoch in range(0, num_epochs + 1):
		model.train()
		model = model.to('cuda')
		epoch_loss = 0
		best_loss = 1_000_000.
		its = 0
		
		with tqdm(train_loader, unit = "batch") as train_epoch:
			for com, mask, lab in train_epoch:
				train_epoch.set_description(f"Epoch: {epoch}")
				lab, com, mask = lab.to('cuda'), com.to('cuda'), mask.to('cuda')

				batch_loss = 0
				optimizer.zero_grad()
				lab_hat = model(com, mask)
				loss = criterion(lab_hat, lab)

				loss.backward()
				optimizer.step()
		
				epoch_loss += (loss.item())
				its += 1

		if (epoch%2 == 0):
			losses.append(epoch_loss)

		# model.eval()
		# model = model.to('cpu')
		# for i, (vcom, vmask, vlab) in enumerate(dev_loader):
		# 	vlab, vcom, vmask = lab.to('cpu'), com.to('cpu'), mask.to('cpu')
		# 	dev_loss = 0
		# 	dlab_hat = model(vcom, vmask)
		# 	loss = criterion(dlab_hat, vlab)
		# 	dev_loss += loss.item()

		# dev_loss = dev_loss / (i + 1)
		# if (dev_loss < best_loss):
		# 	best_loss = dev_loss
		# 	torch.save(model.state_dict(), wts_path) # save checkpoint
		# model.train()
		# model = model.to('cuda')

		print("Average loss over epoch: ", epoch_loss / float(its))

	return losses, model

def test(model, criterion, wts_path, loader):
	## testing code
	test_loss = 0
	acc = 0
	it = 0
	model = model.to('cuda')
	model.eval()
	y_true = []
	y_preds = []

	with torch.no_grad():
	    with tqdm(loader, unit = "batch") as te:
	        for com, mask, lab in te:
	            lab, com, mask = lab.to('cuda'), com.to('cuda'), mask.to('cuda')
	    
	            lab_hat = model(com, mask)
	            loss = criterion(lab_hat, lab)
	            test_loss += loss.item()
	    
	            preds = F.softmax(lab_hat, dim = 1).argmax(dim = 1)
	            y_preds.extend(preds.tolist())
	            y_true.extend(lab.tolist())
	            acc += (preds == lab).sum().item() / float(batch_size)
	            it += 1

	test_acc = (acc / it) * 100

	return test_acc, y_true, y_preds
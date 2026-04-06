#!/usr/bin/python3
import torch
import torch.nn as nn
# debugging
import psutil, os
process = psutil.Process(os.getpid())

### job handles ###
input_data_dir="/gpfs01/usfcc/asciandra/tokenization/"
n_classes=7
# training of Transformer-based classifier
m_epochs=20
# number of PF features
K=64

#########################################
#### STEP 1: LOAD TOKENIZED DATASETS ####
#########################################

#load data
print("==> Loading the dataset.")
import torch
# set how tensors are shared between processes (DataLoader workers) 
# -> uses disk-backed files (no sharing, N.B. not strictly needed if num_workers=0)
##torch.multiprocessing.set_sharing_strategy('file_system')
# more threads → more memory arenas → higher VIRT
# reduce: fragmentation, VIRT growth, hidden allocations
torch.set_num_threads(1)

#Load
data = torch.load(input_data_dir+"tokenized_dataset.pt", map_location="cpu")
TOKENS = data["tokens"]
MASKS  = data["mask"]
LABELS = data["labels"]
from torch.utils.data import Dataset
class TokenDataset(Dataset):
    def __init__(self, tokens, mask, labels):
        self.tokens = tokens
        self.mask = mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.tokens[idx],
            self.mask[idx],
            self.labels[idx]
        )
from torch.utils.data import random_split

dataset = TokenDataset(TOKENS, MASKS, LABELS)

n_total = len(dataset)
n_train = int(0.9 * n_total)
n_val   = n_total - n_train

train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0,      # keep this
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

#################################################
#### STEP 2: TRAIN A TRANSFORMER CLASSIFIER  ####
#################################################

#Model definition
class JetTransformer(nn.Module):
    def __init__(self, num_tokens, d_model=128, nhead=4, num_layers=4, num_classes=5):
        super().__init__()

        self.embedding = nn.Embedding(num_tokens, d_model)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4*d_model,
                batch_first=True
            ),
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, tokens, mask):
        # tokens: [B, N]
        x = self.embedding(tokens)   # [B, N, d_model]

        # attention mask (True = ignore)
        attn_mask = ~mask.bool()

        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # masked mean pooling
        x = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)

        logits = self.classifier(x)
        return logits

#Traning loop

tf_model = JetTransformer(num_tokens=K, num_classes=n_classes).cuda()

optimizer = torch.optim.AdamW(tf_model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

print("==> Run transformer classifier training.")
# clear garbage after each epoch
import gc
for epoch in range(m_epochs):
    tf_model.train()

    print("\ttorch.cuda.memory_allocated() / 1024**2 = ", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("\ttorch.cuda.max_memory_allocated() / 1024**2 = ", torch.cuda.max_memory_allocated() / 1024**2, "MB")
    print("RAM (GB):", process.memory_info().rss / 1024**3)
    for tokens, mask, labels in train_loader:
        with torch.no_grad():
            tokens = tokens.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        optimizer.zero_grad()

        logits = tf_model(tokens, mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
    print("\ttorch.cuda.memory_allocated() / 1024**2 = ", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("\ttorch.cuda.max_memory_allocated() / 1024**2 = ", torch.cuda.max_memory_allocated() / 1024**2, "MB")
    print("RAM (GB):", process.memory_info().rss / 1024**3)

    print(f"\tEpoch {epoch}: loss = {loss.item():.4f}")
    gc.collect()
    del loss, logits

torch.save(tf_model.state_dict(),"TransformerClassifier_model.pt")

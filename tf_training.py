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
m_epochs=100
# Early stopping
patience = 5
epochs_no_improve = 0
min_delta = 1e-4
# codebook size
#K=64
#K=128
K=256
#K=512
# VQ-VAE config
vqvaeconfig="selfAttention_decay099_sup_detachedClassifier_lambda001_K256_D64"#"sup_K512_D128"#"sup_K512_D64"#"K512_D64"#"K512_D128"
N_FEAT=35
# RANDOMIZED TOKENS CHECK?
dorandomtokens=False#True

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
data = torch.load(input_data_dir+vqvaeconfig+"_tokenized_dataset.pt", map_location="cpu")
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

# Compute class weights
# to compensate class imbalance
counts = torch.bincount(LABELS)
weights = 1.0 / counts.float()
weights = weights / weights.sum() * len(counts)
print("==> Weights to compensate class imbalance")
print("\t",weights)

#################################################
#### STEP 2: TRAIN A TRANSFORMER CLASSIFIER  ####
#################################################

#Model definition
class JetTransformer(nn.Module):
    def __init__(self, num_tokens, d_model=128, nhead=4, num_layers=4, num_classes=7):
        super().__init__()

        self.embedding = nn.Embedding(num_tokens, d_model)
        # test continuous PF features
        # as performance ceriling of pipeline
        self.input_proj = nn.Linear(N_FEAT, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            # add dropout 
            # to prevent overfitting
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_classes)
        )

    # standard token model
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

    # continuous model
    # FIXME need x (continuous features) in dataset for this
    #def forward(self, x, mask):
    #    # test continuous PF features
    #    # as performance ceriling of pipeline
    #    x = self.input_proj(tokens)

    #    # attention mask (True = ignore)
    #    attn_mask = ~mask.bool()

    #    x = self.transformer(x, src_key_padding_mask=attn_mask)

    #    # masked mean pooling
    #    x = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)

    #    logits = self.classifier(x)
    #    return logits

# Add validation
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for tokens, mask, labels in loader:
            tokens = tokens.cuda()
            mask   = mask.cuda()
            labels = labels.cuda()
            # RANDOMIZE CHECK -> take tokens & destroy structure
            if dorandomtokens:
                tokens = tokens[torch.randperm(tokens.size(0))]

            logits = model(tokens, mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()

    return total_loss / len(loader)

# Training loop

tf_model = JetTransformer(num_tokens=K, num_classes=n_classes).cuda()

optimizer = torch.optim.AdamW(tf_model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss(weight=weights.cuda())

# Add Gradient Clipping 
# to stabilize the training
nn.utils.clip_grad_norm_(tf_model.parameters(), 1.0)

print("==> Run transformer classifier training.")
# clear garbage after each epoch
import gc
# select epoch with lowest val loss
best_val = float("inf")
for epoch in range(m_epochs):
    tf_model.train()
    train_loss = 0.0

    # debugging
    #print("\ttorch.cuda.memory_allocated() / 1024**2 = ", torch.cuda.memory_allocated() / 1024**2, "MB")
    #print("\ttorch.cuda.max_memory_allocated() / 1024**2 = ", torch.cuda.max_memory_allocated() / 1024**2, "MB")
    #print("RAM (GB):", process.memory_info().rss / 1024**3)
    for tokens, mask, labels in train_loader:
        tokens = tokens.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        # RANDOMIZE CHECK -> take tokens & destroy structure
        if dorandomtokens:
            tokens = tokens[torch.randperm(tokens.size(0))]

        optimizer.zero_grad()

        logits = tf_model(tokens, mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss = evaluate(tf_model, val_loader)

    print(f"\tEpoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")
    # debugging
    #print("\ttorch.cuda.memory_allocated() / 1024**2 = ", torch.cuda.memory_allocated() / 1024**2, "MB")
    #print("\ttorch.cuda.max_memory_allocated() / 1024**2 = ", torch.cuda.max_memory_allocated() / 1024**2, "MB")
    #print("RAM (GB):", process.memory_info().rss / 1024**3)

    gc.collect()
    del loss, logits

    if val_loss < best_val - min_delta:
        best_val = val_loss
        torch.save(tf_model.state_dict(), vqvaeconfig+"_best_tf.pt")
        print("  → Saved new best model")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"  → No improvement ({epochs_no_improve}/{patience})")

        if epochs_no_improve >= patience:
            print("⛔ Early stopping triggered")
            break

torch.save(tf_model.state_dict(),vqvaeconfig+"_last_tf.pt")

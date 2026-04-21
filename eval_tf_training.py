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
#K=64
#K=128
#K=256
K=512
# VQ-VAE config
vqvaeconfig="K512_D128"

#########################################
#### STEP 1: LOAD TOKENIZED DATASETS ####
#########################################

#load data
print("==> Loading the dataset.")
# set how tensors are shared between processes (DataLoader workers) 
# -> uses disk-backed files (no sharing, N.B. not strictly needed if num_workers=0)
##torch.multiprocessing.set_sharing_strategy('file_system')
# more threads → more memory arenas → higher VIRT
# reduce: fragmentation, VIRT growth, hidden allocations
torch.set_num_threads(1)

#Load
data = torch.load(input_data_dir+vqvaeconfig+"_val_tokenized_dataset.pt", map_location="cpu")
#data = torch.load(input_data_dir+vqvaeconfig+"_tokenized_dataset.pt", map_location="cpu")
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
n_train = int(0.99999 * n_total)
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

# Retrieve transformer model
print("==> Retrieving TF model.")
from tf_model import JetTransformer
tf_model = JetTransformer(num_tokens=K, num_classes=n_classes).cuda()
tf_model.load_state_dict(torch.load(vqvaeconfig+"_best_tf.pt"))

all_logits = []
all_labels = []
# Collect predictions
def evaluate_perf(model, loader):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for tokens, mask, labels in loader:
            tokens = tokens.cuda()
            mask   = mask.cuda()
            labels = labels.cuda()

            logits = model(tokens, mask)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    return torch.cat(all_logits),torch.cat(all_labels)

all_logits,all_labels = evaluate_perf(tf_model, train_loader)

# Convert to probabilities
probs = torch.softmax(all_logits, dim=1)   # [N, 7]
labels = all_labels                        # [N]

# Compute ROC + AUC (with sklearn)
# Plot ROC curves
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

class_names = ["tau", "g", "u", "d", "s", "c", "b"]
n_classes = probs.shape[1]
labels_np = labels.cpu().numpy()
probs_np = probs.numpy()

print("==> Plotting ROCs.")
for c in range(n_classes):  # signal class

    plt.figure()

    for b in range(n_classes):  # background class
        if b == c:
            continue

        # select only events of class c or b
        mask_cb = (labels_np == c) | (labels_np == b)

        y_true = (labels_np[mask_cb] == c).astype(int)
        y_score = probs_np[mask_cb, c]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.plot(
            tpr,
            1.0 / np.maximum(fpr, 1e-6),
            label=f"{class_names[c]} vs {class_names[b]} (AUC={roc_auc:.3f})"
        )

    plt.yscale("log")
    plt.xlabel(f"{class_names[c]} efficiency")
    plt.ylabel("Background rejection")
    plt.title(f"{class_names[c]} vs others (pairwise)")
    plt.legend()
    plt.grid()

    plt.savefig(f"roc_pairwise_{class_names[c]}.png")
    plt.close()

# compute token freq vs. class
token_counts = torch.zeros(n_classes, K)

for tokens, mask, labels in train_loader:
    for c in range(n_classes):
        mask_c = (labels == c)

        if mask_c.sum() == 0:
            continue

        t = tokens[mask_c]           # [Nc, N]
        m = mask[mask_c]

        t = t[m.bool()]              # flatten valid tokens
        counts = torch.bincount(t, minlength=K)

        token_counts[c] += counts

# Normalize ->  probability
token_probs = token_counts / token_counts.sum(dim=1, keepdim=True)

# Perplexity (standard in VQ-VAE)
# metrics for codebook efficiency
p = token_counts / token_counts.sum()
perplexity = torch.exp(-(p * torch.log(p + 1e-10)).sum())

print("==> Perplexity: torch.exp(-(p * torch.log(p + 1e-10)).sum())")
print("\t",perplexity)
print("\tIs it better than 10% of K=",K ," ?")

# Compare two classes (e.g. b vs c)
c1, c2 = 6, 5  # example: b vs c
score = token_probs[c1] / (token_probs[c2] + 1e-6)
# Most discriminating tokens
top_tokens = torch.argsort(score, descending=True)[:10]
print("==> Top 10 tokens for b vs. c discrimination:")
print("\t", top_tokens)
# s vs. d
c1, c2 = 4, 3
score = token_probs[c1] / (token_probs[c2] + 1e-6)
top_tokens = torch.argsort(score, descending=True)[:10]
print("==> Top 10 tokens for s vs. d discrimination:")
print("\t", top_tokens)
# g vs. d
c1, c2 = 1, 3
score = token_probs[c1] / (token_probs[c2] + 1e-6)
top_tokens = torch.argsort(score, descending=True)[:10]
print("==> Top 10 tokens for g vs. d discrimination:")
print("\t", top_tokens)

### FIXME TO-DO 
### Extract PF features assigned to top-ranked token
##selected_token = top_tokens[0]
##
##features_list = 
##    [
##	    "pfcand_erel_log",
##	    "pfcand_thetarel",
##	    "pfcand_phirel",
##	    "pfcand_dptdpt",
##	    "pfcand_detadeta",
##	    "pfcand_dphidphi",
##	    "pfcand_dxydxy",
##	    "pfcand_dzdz",
##	    "pfcand_dxydz",
##	    "pfcand_charge",
##	    "pfcand_dphidxy",
##	    "pfcand_dlambdadz",
##	    "pfcand_dxyc",
##	    "pfcand_dxyctgtheta",
##	    "pfcand_phic",
##	    "pfcand_phidz",
##	    "pfcand_phictgtheta",
##	    "pfcand_cdz",
##	    "pfcand_cctgtheta",
##	    "pfcand_mtof",
##	    "pfcand_dndx",
##	    "pfcand_isMu",
##	    "pfcand_isEl",
##	    "pfcand_isChargedHad",
##	    "pfcand_isGamma",
##	    "pfcand_isNeutralHad",
##	    "pfcand_dxy",
##	    "pfcand_dz",
##	    "pfcand_btagSip2dVal",
##	    "pfcand_btagSip2dSig",
##	    "pfcand_btagSip3dVal",
##	    "pfcand_btagSip3dSig",
##	    "pfcand_btagJetDistVal",
##	    "pfcand_btagJetDistSig",
##	    "pfcand_type"
##    ]
##
##with torch.no_grad():
##    for x, mask, _ in train_loader:  # PF features
##        x = x.cuda()
##        mask = mask.cuda()
##
##        _, tokens, _ = vqvae(x, mask)
##
##        mask_sel = (tokens == selected_token) & mask.bool()
##
##        feats = x[mask_sel]
##        features_list.append(feats.cpu())
##
##features = torch.cat(features_list)
##
###  Plot feature distributions
##for i in range(features.shape[1]):
##    plt.hist(features[:, i].numpy(), bins=50)
##    plt.title(f"Feature {i} for token {selected_token}")
##    plt.savefig("feature_token.png")

### Transformer-level importance
### Gradient-based importance
##tokens = tokens.cuda()
##tokens.requires_grad = False
##
##emb = model.embedding(tokens)
##emb.requires_grad_(True)
##
##logits = model.forward_from_embedding(emb, mask)
##loss = criterion(logits, labels)
##
##loss.backward()
##
##importance = emb.grad.abs().sum(dim=-1)  # [B, N]
### Aggregate per token
##token_importance = torch.zeros(K)
##
##for t, imp, m in zip(tokens, importance, mask):
##    for token_id in range(K):
##        token_importance[token_id] += imp[(t == token_id) & m.bool()].sum()

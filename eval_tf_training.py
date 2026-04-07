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

# Retrieve transformer model
from tf_model import JetTransformer
tf_model = JetTransformer(num_tokens=K, num_classes=n_classes).cuda()
tf_model.load_state_dict(torch.load("best_tf.pt"))

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

all_logits,all_labels = evaluate_perf(tf_model, val_loader)

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

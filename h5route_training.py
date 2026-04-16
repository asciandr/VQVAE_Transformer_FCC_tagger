#!/usr/bin/python3

### job handles ###
# input datasets
input_data_dir="/gpfs01/usfcc/asciandra/tokenization/"
#input_data_dir="/gpfs01/usfcc/asciandra/tokenization/TEST/"
train_file  = "prealloc_fcc_ee_7classes_35features_5_6Mjets_pf.h5"
val_file    = "val_prealloc_fcc_ee_7classes_35features_700kjets_pf.h5"
#train_file  = "val_prealloc_fcc_ee_7classes_35features_700kjets_pf.h5"
#val_file    = "val_prealloc_fcc_ee_7classes_35features_70kjets_pf.h5"
# training of unsupervised VQ-VAE tokenizer
IO_BATCH = 4096     # efficient disk read
TRAIN_BATCH = 256   # good for VQ-VAE
#n_epochs=1
n_epochs=15
#n_epochs=10
# number of PF features
N_FEAT=35
# want to use already standardized
# dataset in input_data_dir?
use_std_data=True
# need to compute mean & std 
# for input standardization?
import os
donot_std = os.path.exists(input_data_dir + "norm.pt")
### job handles ###

###########################################
#### STEP 1: TRAIN TOKENIZER JetVQVAE) ####
###########################################
import torch
from torch.utils.data import Dataset
import h5py

# Dataset class
class H5JetDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, io_batch_size):
        self.file = h5py.File(file_path, "r")

        self.X = self.file["X"]
        self.mask = self.file["mask"]
        self.jet_pt = self.file["jet_pt"]
        self.labels = self.file["labels"]

        self.io_batch_size = io_batch_size
        self.N = self.X.shape[0]

    def __len__(self):
        return self.N // self.io_batch_size

    def __getitem__(self, idx):
        start = idx * self.io_batch_size
        end = start + self.io_batch_size

        return (
            torch.from_numpy(self.X[start:end]),
            torch.from_numpy(self.mask[start:end]),
            torch.from_numpy(self.jet_pt[start:end]),
            torch.from_numpy(self.labels[start:end])
        )

# use already standardized
# dataset from previous run?
if not use_std_data:
    #load data
    print("==> Loading the dataset.")
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    dataset     = H5JetDataset(input_data_dir+train_file, io_batch_size=IO_BATCH)
    val_dataset = H5JetDataset(input_data_dir+val_file, io_batch_size=IO_BATCH)
    
    print("==> DataLoaders...")
    # DataLoaders
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,   # IMPORTANT
        shuffle=False,
        pin_memory=True,
        num_workers=0      # IMPORTANT
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=None,   # IMPORTANT
        shuffle=False,
        pin_memory=True,
        num_workers=0      # IMPORTANT
    )

    
    if not donot_std:
        print("==> Compute mean and std.")
        
        def compute_mean_std_h5_fast(file_path, chunk_size=50000):
            sum_ = None
            sumsq = None
            count = 0
        
            with h5py.File(file_path, "r") as f:
                X = f["X"]
                MASK = f["mask"]
        
                N = X.shape[0]
        
                for i in range(0, N, chunk_size):
                    if i % (chunk_size * 10) == 0:
                        print(f"\t{i}/{N}")
        
                    x = torch.from_numpy(X[i:i+chunk_size]).float()      # (B, P, F)
                    mask = torch.from_numpy(MASK[i:i+chunk_size]).bool() # (B, P)
        
                    mask = mask.unsqueeze(-1)  # (B, P, 1)
        
                    # zero padding
                    x_masked = x * mask
        
                    # accumulate
                    batch_sum = x_masked.sum(dim=(0, 1))
                    batch_sumsq = (x_masked ** 2).sum(dim=(0, 1))
                    batch_count = mask.sum()
        
                    if sum_ is None:
                        sum_ = batch_sum
                        sumsq = batch_sumsq
                        count = batch_count
                    else:
                        sum_ += batch_sum
                        sumsq += batch_sumsq
                        count += batch_count
        
            mean = sum_ / count
            var = sumsq / count - mean**2
            std = torch.sqrt(var) + 1e-6
        
            return mean, std
        
        mean, std = compute_mean_std_h5_fast(
            input_data_dir + train_file,
            chunk_size=50000   # tune this!
        )
        print("\tMean:", mean)
        print("\tStd:", std)
    else:
        print("==> Standardization already available.")
        mean_std_data = torch.load(input_data_dir + "norm.pt")
        mean    = mean_std_data["mean"]
        std     = mean_std_data["std"]
        print("\tMean:", mean)
        print("\tStd:", std)
    
    print("==> Standardize...")
    def standardize_h5_fast(input_path, output_path, mean, std, chunk_size=50000):
        with h5py.File(input_path, "r") as fin, \
             h5py.File(output_path, "w") as fout:
    
            X_in    = fin["X"]
            MASK_in = fin["mask"]
    
            # copy auxiliary datasets (fast, no loop)
            fout.create_dataset("mask", data=MASK_in, compression="gzip")
            fout.create_dataset("jet_pt", data=fin["jet_pt"], compression="gzip")
            fout.create_dataset("labels", data=fin["labels"], compression="gzip")
    
            # output dataset
            X_out = fout.create_dataset(
                "X",
                shape=X_in.shape,
                dtype="f4",
                compression="gzip"
            )
    
            N = X_in.shape[0]
    
            for i in range(0, N, chunk_size):
                print(f"\t\tProcessing {i}/{N}")
    
                x = torch.from_numpy(X_in[i:i+chunk_size]).float()       # (B, P, F)
                mask = torch.from_numpy(MASK_in[i:i+chunk_size]).bool()  # (B, P)
    
                mask_exp = mask.unsqueeze(-1)  # (B, P, 1)
    
                # GPU acceleration
                x = x.cuda(non_blocking=True)
                mask_exp = mask_exp.cuda(non_blocking=True)
    
                # === FULLY VECTORIZED ===
                x = (x - mean) / std
                x = torch.clamp(x, -5, 5)
                x = x * mask_exp  # zero padding (FASTER than indexing)
                # ========================
    
                x = x.cpu()
    
                X_out[i:i+chunk_size] = x.numpy()
    
    print("\tTraining sample")
    standardize_h5_fast(
        input_data_dir + train_file,
        input_data_dir + "train_standardized.h5",
        mean.cuda(non_blocking=True),
        std.cuda(non_blocking=True),
        chunk_size=50000
    )
    print("\tValidation sample")
    standardize_h5_fast(
        input_data_dir + val_file,
        input_data_dir + "val_standardized.h5",
        mean.cuda(non_blocking=True),
        std.cuda(non_blocking=True),
        chunk_size=50000
    )
    
    if not donot_std:
        torch.save({
            "mean": mean,
            "std": std
        }, input_data_dir + "norm.pt")
else:
    print("==> NB using existing standardized train and val samples!")
    assert (os.path.exists(input_data_dir + "train_standardized.h5") and os.path.exists(input_data_dir + "val_standardized.h5"))

print("==> Standardized Datasets...")
# Replace with standardized Datasets
dataset     = H5JetDataset(input_data_dir + "train_standardized.h5",    io_batch_size=IO_BATCH)
val_dataset = H5JetDataset(input_data_dir + "val_standardized.h5",      io_batch_size=IO_BATCH)

print("==> Standardized DataLoaders...")
# Replace with standardized DataLoaders
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=None,   # IMPORTANT
    shuffle=False,
    pin_memory=True,
    num_workers=0      # IMPORTANT
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=None,   # IMPORTANT
    shuffle=False,
    pin_memory=True,
    num_workers=0      # IMPORTANT
)

# clean-up
import gc
gc.collect()

import torch.nn as nn
import torch.nn.functional as FNC

class VectorQuantizerEMA(nn.Module):
    def __init__(self, K, D, beta=0.5, decay=0.95, eps=1e-5):
        super().__init__()
        self.K = K
        self.D = D
        self.beta = beta
        self.decay = decay
        self.eps = eps

        self.codebook = nn.Parameter(torch.randn(K, D))
        self.register_buffer("ema_cluster_size", torch.zeros(K))
        self.register_buffer("ema_codebook", torch.randn(K, D))

    def forward(self, z):
        # z: [B, N, D]
        z_flat = z.reshape(-1, self.D)

        # distances
        dist = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ self.codebook.t()
            + self.codebook.pow(2).sum(dim=1)
        )

        indices = dist.argmin(dim=1)
        z_q = self.codebook[indices].view(z.shape)

        # EMA updates (training only)
        if self.training:
            with torch.no_grad():
                onehot = FNC.one_hot(indices, self.K).float()
                cluster_size = onehot.sum(dim=0)

                self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

                embed_sum = onehot.t() @ z_flat
                self.ema_codebook.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + self.eps)
                    / (n + self.K * self.eps) * n
                )

                #self.codebook.data = self.ema_codebook / cluster_size.unsqueeze(1)
                self.codebook.data.copy_(self.ema_codebook / cluster_size.unsqueeze(1))

        # losses
        commit_loss = self.beta * FNC.mse_loss(z_q.detach(), z)
        z_q = z + (z_q - z).detach()

        return z_q, indices.view(z.shape[:-1]), commit_loss

class JetVQVAE(nn.Module):
    #def __init__(self, D=32, K=256):
    #def __init__(self, D=32, K=64):
    #def __init__(self, D=16, K=16):
    #def __init__(self, D=16, K=32):
    #def __init__(self, D=16, K=64):
    #def __init__(self, D=16, K=128):
    #def __init__(self, D=32, K=256):
    def __init__(self, D=64, K=256):
    # NB K=256 proves to saturate
    # codebook size efficiency
    # ==> re-check with much larger stats
    #def __init__(self, D=32, K=512):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(N_FEAT, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, D),
            nn.LayerNorm(D),
        )

        self.vq = VectorQuantizerEMA(K, D)

        self.decoder = nn.Sequential(
            nn.Linear(D, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, N_FEAT),
        )

    def forward(self, x, mask):
        # x: [B, N, N_FEAT], mask: [B, N]
        z = self.encoder(x)
        z_q, tokens, vq_loss = self.vq(z)
        x_rec = self.decoder(z_q)

        # masked reconstruction loss
        num_valid = mask.sum() * N_FEAT
        rec_loss = ((x_rec - x)**2 * mask.unsqueeze(-1)).sum() / num_valid

        return x_rec, tokens, rec_loss + vq_loss

print("==> Defining model.")
model = JetVQVAE().cuda()
# debugging
# VQ model EMA buffers must live on the same device as the model
#for name, buf in model.named_buffers():
#    print(name, buf.device, buf.numel())
# debugging
opt = torch.optim.AdamW(model.parameters(), lr=5e-5)

# check
print("==> Running single-batch smoke test.")
x, mask, jet_pt, labels = next(iter(loader))
x = x.cuda()
mask = mask.cuda()

torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    _ = model(x, mask)

print("\ttorch.cuda.max_memory_allocated() / 1024**2 = ", torch.cuda.max_memory_allocated() / 1024**2, "MB")
print("\tDone with single-batch smoke test.")
# check

# best validation loss (may add entropy in the future?)
# to define "best" model
best_val_loss = float("inf")
model_best_val_loss = model

print("==> Starting VQ-VAE training loop.")
for epoch in range(n_epochs):

    # ---- TRAINING ----
    model.train()
    train_loss = 0.0
    n_train_steps = 0   # 🔥 IMPORTANT

    for X_big, M_big, jet_pt_big, labels_big in loader:
        # move entire macro-batch to GPU once
        X_big = X_big.cuda(non_blocking=True)
        M_big = M_big.cuda(non_blocking=True)

        # shuffle inside macro-batch
        perm = torch.randperm(X_big.size(0))
        X_big = X_big[perm]
        M_big = M_big[perm]

        # split into micro-batches
        for i in range(0, X_big.size(0), TRAIN_BATCH):

            x = X_big[i:i+TRAIN_BATCH]
            mask = M_big[i:i+TRAIN_BATCH]
            # skip empty batches 
            # (with >~10M jets it can happen to have jets with 0 constituents after cuts/padding)
            if mask.sum().item() == 0:
                continue

            opt.zero_grad(set_to_none=True)

            #debugging
            if not torch.isfinite(x).all():
                print("NaNs in X!")
                exit()
            
            if not torch.isfinite(mask).all():
                print("NaNs in MASK!")
                exit()

            _, tokens, loss = model(x, mask)
            loss.backward()
            opt.step()

            train_loss += loss.item()
            n_train_steps += 1

            if loss.item() > 10:
                print("WARNING: very large train loss:", loss.item())
            if not torch.isfinite(loss):
                print("LOSS IS NAN")
                print("x stats:", x.mean().item(), x.std().item(), x.abs().max().item())
                print("mask sum:", mask.sum().item())
                exit()

    train_loss /= n_train_steps   # -> FIXED

    if train_loss > 10:
        print("WARNING: very large AVERAGE LOSS:", train_loss)

    # ---- VALIDATION ----
    model.eval()
    val_loss = 0.0
    n_val_steps = 0   # 🔥 IMPORTANT

    with torch.no_grad():
        for X_big, M_big, jet_pt_big, labels_big in val_loader:
            # move entire macro-batch to GPU once
            X_big = X_big.cuda(non_blocking=True)
            M_big = M_big.cuda(non_blocking=True)

            # no shuffle for validation

            for i in range(0, X_big.size(0), TRAIN_BATCH):

                x = X_big[i:i+TRAIN_BATCH]
                mask = M_big[i:i+TRAIN_BATCH]

                _, tokens, loss = model(x, mask)

                val_loss += loss.item()
                n_val_steps += 1

    val_loss /= n_val_steps   # -> FIXED

    # ---- SAVE BEST ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "JetVQVAE_best.pt")
        model_best_val_loss = model
        print("  → Saved new best model")

    print(f"\tEpoch {epoch}: train = {train_loss:.4f}, val = {val_loss:.4f}")

print("==> Save last model too.")
torch.save(model.state_dict(), "JetVQVAE_last.pt")

# evaluation and sanity checks
print("==> Evaluate and check distributions.")
# use best model for plots
model_best_val_loss.eval()

print("\tNow token frequency.")
all_tokens = []
all_features = []
all_jet_pt = []

with torch.no_grad():
    for x, mask, jet_pt, labels in val_loader:
        x = x.cuda()
        mask = mask.cuda()
    
        z = model_best_val_loss.encoder(x)
        z_real = z[mask.bool()]
    
        _, tokens, _ = model_best_val_loss.vq(z_real)
    
        all_tokens.append(tokens.cpu())
        all_features.append(x[mask.bool()].cpu())
        ncounts = (mask.sum(dim=1).long()).cpu()
        all_jet_pt.append(jet_pt.repeat_interleave(ncounts).cpu())

tokens = torch.cat(all_tokens)       # [N_pf_total]
features = torch.cat(all_features)   # [N_pf_total, F]
jet_pt_pf = torch.cat(all_jet_pt)    # [N_pf_total]

import matplotlib.pyplot as plt

K = model_best_val_loss.vq.K

counts = torch.bincount(tokens, minlength=K).float()
freq = counts / counts.sum()

plt.figure(figsize=(6,4))
plt.bar(range(K), freq.numpy())
plt.xlabel("Token ID")
plt.ylabel("Frequency")
plt.yscale("log")
plt.title("Token usage histogram")
plt.savefig('token_freq.png')

print("\tNow token <-> charge.")

charge = features[:, 9]

token_charge = []
for k in range(K):
    sel = tokens == k
    if sel.sum() > 0:
        token_charge.append(charge[sel].mean().item())
    else:
        token_charge.append(0.0)

plt.figure(figsize=(6,4))
plt.scatter(range(K), token_charge)
plt.axhline(0, color="gray", linestyle="--")
plt.xlabel("Token ID")
plt.ylabel("Mean charge")
plt.title("Token charge")
plt.savefig('token_charge.png')

# NB first feature is not momentum anymore!
#print("Now token <-> momentum scale.")
#logpt = features[:, 0]
#
#token_logpt = []
#for k in range(K):
#    sel = tokens == k
#    if sel.sum() > 0:
#        token_logpt.append(logpt[sel].mean().item())
#    else:
#        token_logpt.append(0.0)
#
#plt.figure(figsize=(6,4))
#plt.scatter(range(K), token_logpt)
#plt.axhline(0, color="gray", linestyle="--")
#plt.xlabel("Token ID")
#plt.ylabel("Mean log10(p/p_jet)")
#plt.title("Token momentum scale")
#plt.savefig('token_momentum_scale.png')

print("\tNow token frequency per jet energy bin.")
freq_per_bin = {}
bins = torch.tensor([0, 5, 10, 15, 20, 30, 40, 50])  # GeV
bin_ids = torch.bucketize(jet_pt_pf, bins)

for b in bin_ids.unique():
    sel = bin_ids == b
    counts = torch.bincount(tokens[sel], minlength=K).float()
    freq_per_bin[int(b)] = counts / counts.sum()

plt.figure(figsize=(6,4))
for b, freq in freq_per_bin.items():
    plt.plot(freq.numpy(), "o", label=f"bin {b}")

plt.xlabel("Token ID")
plt.ylabel("Frequency")
plt.legend()
plt.title("Token stability vs jet momentum")
plt.savefig('token_stability_jet_p.png')

# Token entropy - entropy ~ log(K) -> full usage, entropy <<  log(K) -> collapse
entropy = -(freq * torch.log(freq + 1e-8)).sum()
print("==> Token entropy:", entropy.item())
print("==> Max entropy:", torch.log(torch.tensor(K)).item())

##################################################
#### STEP 2: CONVERT JETS TO TOKEN SEQUENCES  ####
##################################################

#Tokenization function
def tokenize_batch(model, x, mask):
    with torch.no_grad():
        z = model.encoder(x)
        z_real = z[mask.bool()]
        _, tokens, _ = model.vq(z_real)

        # rebuild padded token tensor
        token_tensor = torch.zeros_like(mask, dtype=torch.long)
        token_tensor[mask.bool()] = tokens

    return token_tensor

#Build token dataset (offline step)
print("==> Build token dataset.")
all_tokens = []
all_masks = []
all_labels = []

# at this point no need for "last" model,
# just use "best" model for transformer
model = model_best_val_loss
model.eval()

for x, mask, jet_pt, labels in loader:
    x = x.cuda()
    mask = mask.cuda()

    tokens = tokenize_batch(model, x, mask)

    all_tokens.append(tokens.cpu())
    all_masks.append(mask.cpu())
    all_labels.append(labels)

TOKENS = torch.cat(all_tokens)
MASKS  = torch.cat(all_masks)
LABELS = torch.cat(all_labels)

#Save
torch.save({
    "tokens": TOKENS,
    "mask": MASKS,
    "labels": LABELS
}, input_data_dir+"tokenized_dataset.pt")

#Token dataset
class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, mask, labels):
        self.tokens = tokens
        self.mask = mask
        self.labels = labels

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        return self.tokens[i], self.mask[i], self.labels[i]


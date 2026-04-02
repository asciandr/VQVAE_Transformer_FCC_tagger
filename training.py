#!/usr/bin/python3

### job handles ###
n_classes=2
# training of unsupervised VQ-VAE tokenizer
#n_epochs=1
#n_epochs=5
n_epochs=20
# training of Transformer-based classifier
train_transformer=False
m_epochs=20
# number of PF features
N_FEAT=10

###########################################
#### STEP 1: TRAIN TOKENIZER JetVQVAE) ####
###########################################

#load data
print("==> Loading the dataset.")
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
input_data_dir="/gpfs01/usfcc/asciandra/tokenization/"
data = torch.load(input_data_dir+"fcc_ee_7classes_10features_1_4Mjets_pf.pt", map_location="cpu")
#data = torch.load(input_data_dir+"fcc_ee_7classes_1_4Mjets_pf.pt", map_location="cpu")
#data = torch.load(input_data_dir+"fcc_ee_7classes_16Mjets_pf.pt", map_location="cpu")
#data = torch.load(input_data_dir+"fcc_ee_Hbb_Hcc_4_6Mjets_pf.pt", map_location="cpu")
#data = torch.load(input_data_dir+"fcc_ee_7classes_70kjets_pf.pt", map_location="cpu")

val_data = torch.load(input_data_dir+"fcc_ee_7classes_10features_70kjets_pf.pt", map_location="cpu")
#val_data = torch.load(input_data_dir+"fcc_ee_7classes_70kjets_pf.pt", map_location="cpu")
#val_data = torch.load(input_data_dir+"fcc_ee_Hbb_Hcc_20kjets_pf.pt", map_location="cpu")

X           = data["X"]        # [N, N_max, N_FEAT]
MASK        = data["mask"] # [N, N_max]
LABELS      = data["labels"]
JET_PT      = data["jet_pt"]

val_X       = val_data["X"]        # [N, N_max, N_FEAT]
val_MASK    = val_data["mask"] # [N, N_max]
val_LABELS  = val_data["labels"]
val_JET_PT  = val_data["jet_pt"]

print("==> Standardize input features.")
print("\tBefore standardization.")
print("\t\tX.mean():\t", X.mean())
print("\t\tX.std():\t", X.std())
print("\t\tX.abs().max():\t", X.abs().max())
# standardization of input features
# flatten only valid PFs
X_flat  = X[MASK.bool()]
mean    = X_flat.mean(dim=0)
std     = X_flat.std(dim=0) + 1e-6
# apply to train and validation features
# FIXME: will need to store values for inference!
X     = (X - mean) / std
val_X = (val_X - mean) / std
# THEN clip
X     = torch.clamp(X, -5, 5)
val_X = torch.clamp(val_X, -5, 5)
# After normalization + clipping, 
# zero out padding again 
# padding contains non-zero junk
# model sees fake PF candidates
X[~MASK.bool()]         = 0.0
val_X[~val_MASK.bool()] = 0.0
# debugging
print("\tAfter standardization.")
print("\t\tX.mean():\t", X.mean())
print("\t\tX.std():\t", X.std())
print("\t\tX.abs().max():\t", X.abs().max())
X_valid = X[MASK.bool()]
print("\t\tmean (valid PFs):", X_valid.mean())
print("\t\tstd  (valid PFs):", X_valid.std())
# check max() of diff features
#for i in range(N_FEAT):
#    print(i, X[..., i].abs().max())
# where are X and MASK sitting?
#print(X.device, MASK.device)
# debugging

from torch.utils.data import Dataset

class TensorJetDataset(Dataset):
    def __init__(self, X, mask, jet_pt, labels=None):
        self.X = X
        self.mask = mask
        self.jet_pt = jet_pt
        self.labels = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        if self.labels is None:
            return self.X[i], self.mask[i], self.jet_pt[i]
        return self.X[i], self.mask[i], self.jet_pt[i], self.labels[i]

dataset = TensorJetDataset(X, MASK, JET_PT, LABELS)
val_dataset = TensorJetDataset(val_X, val_MASK, val_JET_PT, val_LABELS)
loader = torch.utils.data.DataLoader(
    dataset,
#    batch_size=1,
#    batch_size=32,
#    batch_size=64,
    batch_size=256,
    shuffle=True,
    num_workers=0,
    pin_memory=False
    #pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=256,
    shuffle=False,     # IMPORTANT
    num_workers=0,
    pin_memory=False
    #pin_memory=True
)

import torch
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
    def __init__(self, D=16, K=16):
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
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

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

    for x, mask, jet_pt, labels in loader:
        x = x.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)

        opt.zero_grad(set_to_none=True)
        _, tokens, loss = model(x, mask)
        loss.backward()
        opt.step()

        train_loss += loss.item()
        # debugging
        #print(loss.shape)
        if loss.item()>10:
            print("WARNING: very large train loss:", loss.item())
        #print("train mask sum:", mask.sum().item())
        #print("train x mean:", x.abs().mean().item())
        #print("train loss per element:", loss.item() / (mask.sum().item() * N_FEAT))

    train_loss /= len(loader)
    if train_loss>10:
        print("WARNING: very large AVERAGE LOSS:", train_loss)

    # ---- VALIDATION ----
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x, mask, jet_pt, labels in val_loader:
            x = x.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            _, tokens, loss = model(x, mask)
            val_loss += loss.item()
            # debugging
            #print("val loss:", loss.item())
            #print("val mask sum:", mask.sum().item())
            #print("val x mean:", x.abs().mean().item())

    val_loss /= len(val_loader)
    # Save best epoch
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
    plt.plot(freq.numpy(), label=f"bin {b}")
plt.xlabel("Token ID")
plt.ylabel("Frequency")
plt.legend()
plt.title("Token stability vs jet momentum")
plt.savefig('token_stability_jet_p.png')

# Token entropy - entropy ~ log(K) -> full usage, entropy <<  log(K) -> collapse
entropy = -(freq * torch.log(freq + 1e-8)).sum()
print("==> Token entropy:", entropy.item())
print("==> Max entropy:", torch.log(torch.tensor(K)).item())

### Make post-VQ-VAE part
### of trainings optional
if train_transformer:

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
    
    #################################################
    #### STEP 3: TRAIN A TRANSFORMER CLASSIFIER  ####
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
    dataset = TokenDataset(TOKENS, MASKS, LABELS)
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    tf_model = JetTransformer(num_tokens=K, num_classes=n_classes).cuda()
    
    optimizer = torch.optim.AdamW(tf_model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    print("==> Run transformer classifier training.")
    for epoch in range(m_epochs):
        tf_model.train()
    
        for tokens, mask, labels in train_loader:
            tokens = tokens.cuda()
            mask = mask.cuda()
            labels = labels.cuda()
    
            optimizer.zero_grad()
    
            logits = tf_model(tokens, mask)
            loss = criterion(logits, labels)
    
            loss.backward()
            optimizer.step()
    #    print(torch.cuda.memory_allocated() / 1024**3)
    
        print(f"\tEpoch {epoch}: loss = {loss.item():.4f}")
    
    torch.save(tf_model.state_dict(),"TransformerClassifier_model.pt")

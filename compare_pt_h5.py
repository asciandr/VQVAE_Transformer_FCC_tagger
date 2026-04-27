import torch
import h5py
import numpy as np

# =========================
# CONFIG
# =========================
input_data_dir="/gpfs01/usfcc/asciandra/tokenization/"
#pt_file = input_data_dir+"fcc_ee_7classes_35features_700kjets_pf.pt"
#h5_file = input_data_dir+"val_prealloc_fcc_ee_7classes_35features_700kjets_pf.h5"
pt_file = input_data_dir+"standardized_fcc_ee_7classes_35features_700kjets_pf.pt"
h5_file = input_data_dir+"TEST/train_standardized.h5"

N_CHECK = 50  # number of jets to compare

# =========================
# LOAD PT
# =========================
pt = torch.load(pt_file, map_location="cpu")

X_pt = pt["X"].numpy()
MASK_pt = pt["mask"].numpy()
PT_pt = pt["jet_pt"].numpy()
LABELS_pt = pt["labels"].numpy()

# =========================
# LOAD H5
# =========================
f = h5py.File(h5_file, "r")

X_h5 = f["X"]
MASK_h5 = f["mask"]
PT_h5 = f["jet_pt"]
LABELS_h5 = f["labels"]
print("PRINTING STATS")
print("PT shape:", X_pt.shape)
print("H5 shape:", X_h5.shape)
print("==========================")
print("PT mean:", X_pt[:10000].mean())
print("H5 mean:", X_h5[:10000].mean())

# =========================
# COMPARE FUNCTION
# =========================
def compare(name, a, b):
    diff = np.abs(a - b)

    print(f"\n=== {name} ===")
    print("shape:", a.shape, b.shape)
    print("max abs diff:", diff.max())
    print("mean abs diff:", diff.mean())
    print("allclose:", np.allclose(a, b, atol=1e-6))
    print("a: ",a)
    print("b: ",b)

# =========================
# LOOP OVER FEW ENTRIES
# =========================
for i in range(N_CHECK):
    print(f"\n########################")
    print(f"Entry {i}")
    print(f"########################")

    x_pt = X_pt[i]
    x_h5 = X_h5[i]

    m_pt = MASK_pt[i]
    m_h5 = MASK_h5[i]

    pt_pt = PT_pt[i]
    pt_h5 = PT_h5[i]

    l_pt = LABELS_pt[i]
    l_h5 = LABELS_h5[i]

    compare("X", x_pt, x_h5)
    compare("MASK", m_pt, m_h5)
    compare("JET_PT", pt_pt, pt_h5)
    compare("LABELS", l_pt, l_h5)

    # 🔍 extra: check padding region explicitly
    mask_bool = m_pt.astype(bool)

    if mask_bool.any():
        diff_valid = np.abs(x_pt[mask_bool] - x_h5[mask_bool])
        print("valid region max diff:", diff_valid.max())

    if (~mask_bool).any():
        print("padding pt unique:", np.unique(x_pt[~mask_bool]))
        print("padding h5 unique:", np.unique(x_h5[~mask_bool]))

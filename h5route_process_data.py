import uproot
import awkward as ak
import numpy as np
import h5py
import gc
import psutil, os

import ctypes
libc = ctypes.CDLL("libc.so.6")

# =========================
# CONFIG
# =========================
file_name = "/atlasgpfs01/usatlas/workarea/asciandra/training/reduced100kjets_inputs/FSR_studies_IDEA_lighterBP_50pc_7labels_out_Hbb_cc_ss_dd_uu_gg_tautau.root"
#file_name = "/atlasgpfs01/usatlas/workarea/asciandra/training/reduced3Mjets_inputs/FSR_studies_IDEA_lighterBP_50pc_7labels_out_Hbb_cc_ss_dd_uu_gg_tautau.root"
#file_name = "/atlasgpfs01/usatlas/workarea/asciandra/training/reduced800kjets_inputs/FSR_studies_IDEA_lighterBP_50pc_7labels_out_Hbb_cc_ss_dd_uu_gg_tautau.root"
tree_name = "tree"
output_file = "/gpfs01/usfcc/asciandra/tokenization/val_prealloc_fcc_ee_7classes_35features_700kjets_pf.h5"
#output_file = "/gpfs01/usfcc/asciandra/tokenization/prealloc_fcc_ee_7classes_35features_21Mjets_pf.h5"
#output_file = "/gpfs01/usfcc/asciandra/tokenization/prealloc_fcc_ee_7classes_35features_5_6Mjets_pf.h5"
N_MAX = 75
CHUNK_SIZE = 20000   # safe value
# total number of jets to be processed
file = uproot.open(file_name)
tree = file[tree_name]
TOTAL = tree.num_entries
#TOTAL = 700_000   # or tree.num_entries
#TOTAL = 5_600_000   # or tree.num_entries
#TOTAL = 21_000_000   # or tree.num_entries
#TOTAL = 42_099_532   # or tree.num_entries
file.close()

feature_branches = [   # per-particle features (jagged)
	"pfcand_erel_log",
	"pfcand_thetarel",
	"pfcand_phirel",
	"pfcand_dptdpt",
	"pfcand_detadeta",
	"pfcand_dphidphi",
	"pfcand_dxydxy",
	"pfcand_dzdz",
	"pfcand_dxydz",
	"pfcand_charge",
	"pfcand_dphidxy",
	"pfcand_dlambdadz",
	"pfcand_dxyc",
	"pfcand_dxyctgtheta",
	"pfcand_phic",
	"pfcand_phidz",
	"pfcand_phictgtheta",
	"pfcand_cdz",
	"pfcand_cctgtheta",
	"pfcand_mtof",
	"pfcand_dndx",
	"pfcand_isMu",
	"pfcand_isEl",
	"pfcand_isChargedHad",
	"pfcand_isGamma",
	"pfcand_isNeutralHad",
	"pfcand_dxy",
	"pfcand_dz",
	"pfcand_btagSip2dVal",
	"pfcand_btagSip2dSig",
	"pfcand_btagSip3dVal",
	"pfcand_btagSip3dSig",
	"pfcand_btagJetDistVal",
	"pfcand_btagJetDistSig",
	"pfcand_type"
]
jet_branches = ["jet_p","jet_theta","jet_phi"]   # per-jet features
# class labels
# fix a consistent mapping
class_branches = [
    "recojet_isTAU",
    "recojet_isG",
    "recojet_isU",
    "recojet_isD",
    "recojet_isS",
    "recojet_isC",
    "recojet_isB",
]

N_FEAT = len(feature_branches)

# =========================
# PRECOMPUTE (outside loop)
# =========================
idx = np.arange(N_MAX)[None, :]

def print_ram():
    mem = psutil.Process(os.getpid()).memory_info().rss / 1e9
    print(f"RAM: {mem:.2f} GB")

# =========================
# CREATE HDF5 DATASET
# =========================
with h5py.File(output_file, "w", rdcc_nbytes=0) as f:

    dset_X = f.create_dataset(
        "X",
        shape=(TOTAL, N_MAX, N_FEAT),
        dtype="f4",
        chunks=(10000, N_MAX, N_FEAT),
        compression="lzf"
    )

    dset_mask = f.create_dataset(
        "mask",
        shape=(TOTAL, N_MAX),
        dtype="f4",
        chunks=(10000, N_MAX),
        compression="lzf"
    )

    dset_pt = f.create_dataset(
        "jet_pt",
        shape=(TOTAL,),
        dtype="f4",
        chunks=(10000,),
        compression="lzf"
    )

    dset_labels = f.create_dataset(
        "labels",
        shape=(TOTAL,),
        dtype="i8",
        chunks=(10000,),
        compression="lzf"
    )

    start = 0

    # =========================
    # MAIN LOOP
    # =========================
    for arrays in uproot.iterate(
        f"{file_name}:{tree_name}",
        feature_branches + jet_branches + class_branches,
        library="ak",
        step_size=CHUNK_SIZE,
        num_workers=1
    ):
        njets = len(arrays[jet_branches[0]])
        end = start + njets

        # =========================
        # MASK (counts-based)
        # =========================
        counts = ak.num(arrays[feature_branches[0]], axis=1)
        counts_np = ak.to_numpy(counts)

        MASK = (idx < counts_np[:, None]).astype(np.float32)
        dset_mask[start:end] = MASK

        # =========================
        # JET PT
        # =========================
        jet_pt = ak.to_numpy(arrays["jet_p"]).astype(np.float32)
        dset_pt[start:end] = jet_pt

        # =========================
        # LABELS (no stack!)
        # =========================
        labels_np = np.empty((njets, len(class_branches)), dtype=np.float32)
        for i, b in enumerate(class_branches):
            labels_np[:, i] = ak.to_numpy(arrays[b])

        labels_idx = np.argmax(labels_np, axis=1).astype(np.int64)
        dset_labels[start:end] = labels_idx

        # =========================
        # FEATURES (CRITICAL PART)
        # =========================
        # pad once (awkward, cheap)
        padded = [
            ak.pad_none(arrays[b], N_MAX, axis=1, clip=True)
            for b in feature_branches
        ]

        # convert once (grouped)
        padded_np = [
            np.nan_to_num(ak.to_numpy(p), copy=False).astype(np.float32)
            for p in padded
        ]

        # write feature-by-feature
        for j, arr in enumerate(padded_np):
            dset_X[start:end, :, j] = arr

        # =========================
        # CLEANUP
        # =========================
        del padded, padded_np
        del arrays, counts, counts_np, labels_np, labels_idx, jet_pt, MASK

        gc.collect()

        start = end
        # Force malloc trimming
        if start % 100000 == 0:
            libc.malloc_trim(0)

        print(f"Processed {start} jets")
        print_ram()

print("Done.")

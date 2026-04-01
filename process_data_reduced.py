import torch
import uproot
import numpy as np
from tqdm import trange

N_MAX = 64

file = uproot.open("/atlasgpfs01/usatlas/workarea/asciandra/training/reduced10kjets_inputs/FSR_studies_IDEA_lighterBP_50pc_7labels_out_Hbb_cc_ss_dd_uu_gg_tautau.root")
#file = uproot.open("/atlasgpfs01/usatlas/workarea/asciandra/training/reduced10kjets_inputs/FSR_studies_IDEA_lighterBP_50pc_7labels_out_Hbb_Hcc.root")
#file = uproot.open("/atlasgpfs01/usatlas/workarea/asciandra/training/FSR_studies_IDEA_7labels_out_Hbb_Hcc.root")
#file = uproot.open("/atlasgpfs01/usatlas/workarea/asciandra/training/FSR_studies_IDEA_7labels_out_Hcc.root")
#file = uproot.open("/atlasgpfs01/usatlas/workarea/asciandra/training/FSR_studies_IDEA_7labels_out_Hbb.root")
tree = file["tree"]

print("tree arrays")
arrays = tree.arrays(
    [
        "jet_p", "jet_theta", "jet_phi",
        "pfcand_p", "pfcand_theta", "pfcand_phi",
        "pfcand_charge", "pfcand_erel_log",
        "pfcand_thetarel", "pfcand_phirel",
        "pfcand_dptdpt",
        "pfcand_detadeta",
        "pfcand_dphidphi",
        "pfcand_dxydxy",
        "pfcand_dzdz",
        "pfcand_dxydz",
        "pfcand_dphidxy",
        # NB need 7 classes
        "recojet_isB",
        "recojet_isC",
        "recojet_isS",
        "recojet_isD",
        "recojet_isU",
        "recojet_isG",
        "recojet_isTAU"
    ],
    library="np"
)

njets = len(arrays["jet_p"])
# test small n. of jets
#njets = 20000
#njets = 100000
print("njets:\t"+str(njets))

print("torch zeros")
X = torch.zeros((njets, N_MAX, 5), dtype=torch.float32)
MASK = torch.zeros((njets, N_MAX), dtype=torch.float32)
JET_PT = torch.tensor(arrays["jet_p"], dtype=torch.float32)
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
# stack into array -> shape = [N_jets, N_classes]
labels_np = np.stack([arrays[b] for b in class_branches], axis=1)
# sanity check: do jets have multiple or no label true?
assert np.all(labels_np.sum(axis=1) == 1)
# convert to class index -> labels_idx.shape = [N_jets], values {0, ..., N_classes-1}
labels_idx = np.argmax(labels_np, axis=1)
# check class balance
print(np.bincount(labels_idx))
# convert to torch
LABELS = torch.tensor(labels_idx, dtype=torch.long)

def wrap_phi(dphi):
    return (dphi + np.pi) % (2*np.pi) - np.pi

def jet_to_tensor(arrays, i, N_max=64, eps=1e-10):
    jet_pt  = arrays["jet_p"][i]
    jet_eta = arrays["jet_theta"][i]
    jet_phi = arrays["jet_phi"][i]

    pt   = arrays["pfcand_p"][i]
    eta  = arrays["pfcand_theta"][i]
    phi  = arrays["pfcand_phi"][i]
    q    = arrays["pfcand_charge"][i]
    ecal = arrays["pfcand_erel_log"][i]
    pfcand_dptdpt   = arrays["pfcand_dptdpt"][i]
    pfcand_thetarel	= arrays["pfcand_thetarel"][i]
    pfcand_phirel	= arrays["pfcand_phirel"][i]
    pfcand_erel_log	= arrays["pfcand_erel_log"][i]

    # sort by p
    # NB do not sort, as baseline graph transformer does not do it
    #order = np.argsort(pt)[::-1]
    #pt, eta, phi, q, ecal = (
    #    pt[order], eta[order], phi[order],
    #    q[order], ecal[order]
    #)

    n = min(len(pfcand_dptdpt), N_max)

    x = np.zeros((N_max, 5), dtype=np.float32)
    mask = np.zeros(N_max, dtype=np.float32)

    if n == 0:
        return x, mask

    x[:n, 0] = pfcand_dptdpt[:n]#np.log10(pt[:n] / jet_pt + eps)
    x[:n, 1] = pfcand_thetarel[:n]#eta[:n] - jet_eta
    x[:n, 2] = pfcand_phirel[:n]#wrap_phi(phi[:n] - jet_phi)
    x[:n, 3] = q[:n]
    x[:n, 4] = pfcand_erel_log[:n]#ecal[:n]

    mask[:n] = 1.0
    return x, mask

print("for loop")
for i in trange(njets):
    x, mask = jet_to_tensor(arrays, i, N_MAX)
    X[i] = torch.from_numpy(x)
    MASK[i] = torch.from_numpy(mask)

# save to disk

torch.save(
    {
        "X": X,
        "mask": MASK,
        "jet_pt": JET_PT,
        "labels": LABELS
    },
    #"valHcc_fcc_ee_jets_pf.pt"
    #"fcc_ee_jets_pf.pt"
    "/gpfs01/usfcc/asciandra/tokenization/fcc_ee_7classes_70kjets_pf.pt"
#    "/gpfs01/usfcc/asciandra/tokenization/fcc_ee_Hbb_Hcc_20kjets_pf.pt"
)


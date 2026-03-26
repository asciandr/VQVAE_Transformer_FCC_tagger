import torch
import uproot
import numpy as np
from tqdm import trange

N_MAX = 64

file = uproot.open("/atlasgpfs01/usatlas/workarea/asciandra/training/FSR_studies_IDEA_7labels_out_Hbb_Hcc.root")
#file = uproot.open("/atlasgpfs01/usatlas/workarea/asciandra/training/FSR_studies_IDEA_7labels_out_Hcc.root")
#file = uproot.open("/atlasgpfs01/usatlas/workarea/asciandra/training/FSR_studies_IDEA_7labels_out_Hbb.root")
tree = file["tree"]

print("tree arrays")
arrays = tree.arrays(
    [
        "jet_p", "jet_theta", "jet_phi",
        "pfcand_p", "pfcand_theta", "pfcand_phi",
        "pfcand_charge", "pfcand_erel_log",
        # for later classification
        "recojet_isB"#,
        #"recojet_isC"
    ],
    library="np"
)

njets = len(arrays["jet_p"])
# test small n. of jets
#njets = 4644300
#njets = 100000
print("njets:\t"+str(njets))

print("torch zeros")
X = torch.zeros((njets, N_MAX, 6), dtype=torch.float32)
MASK = torch.zeros((njets, N_MAX), dtype=torch.float32)
LABELS = torch.tensor(arrays["recojet_isB"], dtype=torch.long)

def wrap_phi(dphi):
    return (dphi + np.pi) % (2*np.pi) - np.pi

def jet_to_tensor(arrays, i, N_max=64, eps=1e-6):
    jet_pt  = arrays["jet_p"][i]
    jet_eta = arrays["jet_theta"][i]
    jet_phi = arrays["jet_phi"][i]

    pt   = arrays["pfcand_p"][i]
    eta  = arrays["pfcand_theta"][i]
    phi  = arrays["pfcand_phi"][i]
    q    = arrays["pfcand_charge"][i]
    ecal = arrays["pfcand_erel_log"][i]

    # sort by pT
    order = np.argsort(pt)[::-1]
    pt, eta, phi, q, ecal = (
        pt[order], eta[order], phi[order],
        q[order], ecal[order]
    )

    n = min(len(pt), N_max)

    x = np.zeros((N_max, 6), dtype=np.float32)
    mask = np.zeros(N_max, dtype=np.float32)

    if n == 0:
        return x, mask

    x[:n, 0] = np.log(pt[:n] / jet_pt + eps)
    x[:n, 1] = eta[:n] - jet_eta
    x[:n, 2] = wrap_phi(phi[:n] - jet_phi)
    x[:n, 3] = q[:n]
    x[:n, 4] = ecal[:n]

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
        "labels": LABELS
    },
    #"valHcc_fcc_ee_jets_pf.pt"
    #"fcc_ee_jets_pf.pt"
    "/gpfs01/usfcc/asciandra/tokenization/fcc_ee_Hbb_Hcc_4_6Mjets_pf.pt"
)


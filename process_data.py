import torch
import uproot
import numpy as np
from tqdm import trange

# max n. of PF candidates per jet
N_MAX = 75
# n. of PF features
N_FEAT = 35

file = uproot.open("/atlasgpfs01/usatlas/workarea/asciandra/training/reduced100kjets_inputs/FSR_studies_IDEA_lighterBP_50pc_7labels_out_Hbb_cc_ss_dd_uu_gg_tautau.root")
#file = uproot.open("/atlasgpfs01/usatlas/workarea/asciandra/training/reduced200kjets_inputs/FSR_studies_IDEA_lighterBP_50pc_7labels_out_Hbb_cc_ss_dd_uu_gg_tautau.root")
#file = uproot.open("/atlasgpfs01/usatlas/workarea/asciandra/training/FSR_studies_IDEA_lighterBP_50pc_7labels_out_Hbb_cc_ss_dd_uu_gg_tautau.root")
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
        "pfcand_type",
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
#njets = 4644300
#njets = 100000
print("njets:\t"+str(njets))

print("torch zeros")
X = torch.zeros((njets, N_MAX, N_FEAT), dtype=torch.float32)
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
    #ecal = arrays["pfcand_erel_log"][i]

    pfcand_erel_log	= arrays["pfcand_erel_log"][i]
    pfcand_thetarel	= arrays["pfcand_thetarel"][i]
    pfcand_phirel	= arrays["pfcand_phirel"][i]
    pfcand_dptdpt   = arrays["pfcand_dptdpt"][i]
    pfcand_detadeta	= arrays["pfcand_detadeta"][i]
    pfcand_dphidphi	= arrays["pfcand_dphidphi"][i]
    pfcand_dxydxy	= arrays["pfcand_dxydxy"][i]
    pfcand_dzdz	    = arrays["pfcand_dzdz"][i]
    pfcand_dxydz	= arrays["pfcand_dxydz"][i]
    pfcand_charge   = arrays["pfcand_charge"][i]
    pfcand_dphidxy	= arrays["pfcand_dphidxy"][i]
    pfcand_dlambdadz= arrays["pfcand_dlambdadz"][i]
    pfcand_dxyc	    = arrays["pfcand_dxyc"][i]
    pfcand_dxyctgtheta	= arrays["pfcand_dxyctgtheta"][i]
    pfcand_phic	    = arrays["pfcand_phic"][i]
    pfcand_phidz	= arrays["pfcand_phidz"][i]
    pfcand_phictgtheta	= arrays["pfcand_phictgtheta"][i]
    pfcand_cdz	= arrays["pfcand_cdz"][i]
    pfcand_cctgtheta	= arrays["pfcand_cctgtheta"][i]
    pfcand_mtof	= arrays["pfcand_mtof"][i]
    pfcand_dndx	= arrays["pfcand_dndx"][i]
    pfcand_isMu	= arrays["pfcand_isMu"][i]
    pfcand_isEl	= arrays["pfcand_isEl"][i]
    pfcand_isChargedHad	= arrays["pfcand_isChargedHad"][i]
    pfcand_isGamma	= arrays["pfcand_isGamma"][i]
    pfcand_isNeutralHad	= arrays["pfcand_isNeutralHad"][i]
    pfcand_dxy	= arrays["pfcand_dxy"][i]
    pfcand_dz	= arrays["pfcand_dz"][i]
    pfcand_btagSip2dVal	= arrays["pfcand_btagSip2dVal"][i]
    pfcand_btagSip2dSig	= arrays["pfcand_btagSip2dSig"][i]
    pfcand_btagSip3dVal	= arrays["pfcand_btagSip3dVal"][i]
    pfcand_btagSip3dSig	= arrays["pfcand_btagSip3dSig"][i]
    pfcand_btagJetDistVal	= arrays["pfcand_btagJetDistVal"][i]
    pfcand_btagJetDistSig	= arrays["pfcand_btagJetDistSig"][i]
    pfcand_type	= arrays["pfcand_type"][i]




    # sort by p
    # NB do not sort, as baseline graph transformer does not do it
    #order = np.argsort(pt)[::-1]
    #pt, eta, phi, q, ecal = (
    #    pt[order], eta[order], phi[order],
    #    q[order], ecal[order]
    #)

    n = min(len(pfcand_dptdpt), N_max)

    x = np.zeros((N_max, N_FEAT), dtype=np.float32)
    mask = np.zeros(N_max, dtype=np.float32)

    if n == 0:
        return x, mask

    x[:n, 0] = pfcand_erel_log[:n]
    x[:n, 1] = pfcand_thetarel[:n]
    x[:n, 2] = pfcand_phirel[:n]
    x[:n, 3] = pfcand_dptdpt[:n]
    x[:n, 4] = pfcand_detadeta[:n]
    x[:n, 5] = pfcand_dphidphi[:n]
    x[:n, 6] = pfcand_dxydxy[:n]
    x[:n, 7] = pfcand_dzdz[:n]
    x[:n, 8] = pfcand_dxydz[:n]
    x[:n, 9] = pfcand_charge[:n]
    x[:n, 10] = pfcand_dphidxy[:n]
    x[:n, 11] = pfcand_dlambdadz[:n]
    x[:n, 12] = pfcand_dxyc[:n]
    x[:n, 13] = pfcand_dxyctgtheta[:n]
    x[:n, 14] = pfcand_phic[:n]
    x[:n, 15] = pfcand_phidz[:n]
    x[:n, 16] = pfcand_phictgtheta[:n]
    x[:n, 17] = pfcand_cdz[:n]
    x[:n, 18] = pfcand_cctgtheta[:n]
    x[:n, 19] = pfcand_mtof[:n]
    x[:n, 20] = pfcand_dndx[:n]
    x[:n, 21] = pfcand_isMu[:n]
    x[:n, 22] = pfcand_isEl[:n]
    x[:n, 23] = pfcand_isChargedHad[:n]
    x[:n, 24] = pfcand_isGamma[:n]
    x[:n, 25] = pfcand_isNeutralHad[:n]
    x[:n, 26] = pfcand_dxy[:n]
    x[:n, 27] = pfcand_dz[:n]
    x[:n, 28] = pfcand_btagSip2dVal[:n]
    x[:n, 29] = pfcand_btagSip2dSig[:n]
    x[:n, 30] = pfcand_btagSip3dVal[:n]
    x[:n, 31] = pfcand_btagSip3dSig[:n]
    x[:n, 32] = pfcand_btagJetDistVal[:n]
    x[:n, 33] = pfcand_btagJetDistSig[:n]
    x[:n, 34] = pfcand_type[:n]

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
    "/gpfs01/usfcc/asciandra/tokenization/fcc_ee_7classes_35features_700kjets_pf.pt"
    #"/gpfs01/usfcc/asciandra/tokenization/fcc_ee_7classes_35features_1_4Mjets_pf.pt"
    #"/gpfs01/usfcc/asciandra/tokenization/fcc_ee_7classes_16Mjets_pf.pt"
#    "/gpfs01/usfcc/asciandra/tokenization/fcc_ee_Hbb_Hcc_4_6Mjets_pf.pt"
)


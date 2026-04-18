import os
from time import sleep
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors

from rdkit.Chem.QED import qed
from easydict import EasyDict
#from utils.reconstruct import reconstruct_from_generated_with_edges
#from sascorer import compute_sa_score
#from docking import QVinaDockingTask
#from utils.datasets import get_dataset
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.FilterCatalog import *
from typing import Dict, Any, Optional
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def is_valid_smiles(smiles: str) -> bool:
    if not smiles: 
        return False
    if '.' in smiles:
        return False
    m = Chem.MolFromSmiles(smiles, sanitize=False)
    if m is None:
        return False
    try:
        Chem.SanitizeMol(m)  # SANITIZE_ALL
        # Optional strict roundtrip:
        Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(m, isomericSmiles=True), sanitize=True))
        return True
    except Exception:
        return False

def murcko_scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaf, isomericSmiles=True) if scaf else ""

def obey_lipinski(mol):
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    rule_4 = (logp:=Crippen.MolLogP(mol)>=-2) & (logp<=5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])
    

def get_basic(mol):
    n_atoms = len(mol.GetAtoms())
    n_bonds = len(mol.GetBonds())
    n_rings = len(Chem.GetSymmSSSR(mol))
    weight = Descriptors.ExactMolWt(mol)
    return n_atoms, n_bonds, n_rings, weight


def get_rdkit_rmsd(mol, n_conf=20, random_seed=42):
    """
    Calculate the alignment of generated mol and rdkit predicted mol
    Return the rmsd (max, min, median) of the `n_conf` rdkit conformers
    """
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    mol3d = Chem.AddHs(mol)
    rmsd_list = []
    # predict 3d
    confIds = AllChem.EmbedMultipleConfs(mol3d, n_conf, randomSeed=random_seed)
    for confId in confIds:
        AllChem.UFFOptimizeMolecule(mol3d, confId=confId)
        rmsd = Chem.rdMolAlign.GetBestRMS(mol, mol3d, refId=confId)
        rmsd_list.append(rmsd)
    # mol3d = Chem.RemoveHs(mol3d)
    rmsd_list = np.array(rmsd_list)
    return [np.max(rmsd_list), np.min(rmsd_list), np.median(rmsd_list)]


def get_logp(mol):
    return Crippen.MolLogP(mol)

def get_flexibility_index(mol):
    """
    Compute a normalized flexibility index = NumRotatableBonds / NumHeavyAtoms.
    Returns 0 if mol is None or empty.

    drug-like molecules: 0.1~0.3
    >0.4: too flexible
    """
    # Safety guard
    if mol is None:
        return 0.0

    try:
        # RDKit definition of rotatable bond: single, non-ring bond,
        # not terminal (and excludes amide C–N by default)
        rb = rdMolDescriptors.CalcNumRotatableBonds(
            mol, rdMolDescriptors.NumRotatableBondsOptions.Default
        )
    except Exception:
        # fall back to legacy descriptor if necessary
        rb = Descriptors.NumRotatableBonds(mol)

    ha = mol.GetNumHeavyAtoms()
    if ha == 0:
        return 0.0

    return rb / ha


def get_chem(mol):
    qed_score = qed(mol)
    sa_score = compute_sa_score(mol)
    logp_score = Crippen.MolLogP(mol)
    hacc_score = Lipinski.NumHAcceptors(mol)
    hdon_score = Lipinski.NumHDonors(mol)
    return qed_score, sa_score, logp_score, hacc_score, hdon_score

def summarize_smiles(smiles: str, sa_fn=None) -> Dict[str, Any]:
    """
    Summarize one SMILES into:
      - valid: RDKit-sanitizable AND connected (no '.')
      - sa: synthetic accessibility score (float) if valid and sa_fn provided
      - scaffold: Murcko scaffold SMILES ("" if invalid)
      - n_rotatable: RDKit rotatable bond count (int, 0 if invalid)
      - flexibility: normalized rotatable/heavy atoms (float, 0 if invalid)
    """
    out = {
        "smiles": smiles,
        "valid": False,
        "sa": None,
        "scaffold": "",
        "n_rotatable": 0,
        "flexibility": 0.0,
        "n_heavy_atoms": 0,
    }

    if not is_valid_smiles(smiles):
        return out

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return out

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return out

    out["valid"] = True
    out["scaffold"] = murcko_scaffold(smiles)
    out["n_heavy_atoms"] = mol.GetNumHeavyAtoms()

    # rigidity proxies
    try:
        out["n_rotatable"] = int(
            rdMolDescriptors.CalcNumRotatableBonds(
                mol, rdMolDescriptors.NumRotatableBondsOptions.Default
            )
        )
    except Exception:
        out["n_rotatable"] = int(rdMolDescriptors.CalcNumRotatableBonds(mol))

    out["flexibility"] = float(get_flexibility_index(mol))

    # SA
    if sa_fn is not None:
        try:
            out["sa"] = float(sa_fn(mol))
        except Exception:
            out["sa"] = None

    return out





class SimilarityWithTrain:
    def __init__(self) -> None:
        self.cfg_dataset = EasyDict({
            'name': 'pl',
            'path': './data/crossdocked_pocket10', 
            'split': './data/split_by_name.pt', 
            'fingerprint': './data/crossdocked_pocket10_fingerprint.pt',
            'smiles': './data/crossdocked_pocket10_smiles.pt', 
        })
        self.train_smiles = None
        self.train_fingers = None
        
    def _get_train_mols(self):
        file_not_exists = (not os.path.exists(self.cfg_dataset.fingerprint)) or (not os.path.exists(self.cfg_dataset.smiles))
        if file_not_exists:
            _, subsets = get_dataset(config = self.cfg_dataset)
            train_set = subsets['train']
            self.train_smiles = []
            self.train_fingers = []
            for data in tqdm(train_set):  # calculate fingerprint and smiles of train data
                data.ligand_context_pos = data.ligand_pos
                data.ligand_context_element = data.ligand_element
                data.ligand_context_bond_index = data.ligand_bond_index
                data.ligand_context_bond_type = data.ligand_bond_type
                mol = reconstruct_from_generated_with_edges(data, sanitize=True)
                mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))  # automately sanitize 
                smiles = Chem.MolToSmiles(mol)
                fg = Chem.RDKFingerprint(mol)
                self.train_fingers.append(fg)
                self.train_smiles.append(smiles)
            self.train_smiles = np.array(self.train_smiles)
            # self.train_fingers = np.array(self.train_fingers)
            torch.save(self.train_smiles, self.cfg_dataset.smiles)
            torch.save(self.train_fingers, self.cfg_dataset.fingerprint)
        else:
            self.train_smiles = torch.load(self.cfg_dataset.smiles)
            self.train_fingers = torch.load(self.cfg_dataset.fingerprint)
            self.train_smiles = np.array(self.train_smiles)
            # self.train_fingers = np.array(self.train_fingers)

    def _get_uni_mols(self):
        self.train_uni_smiles, self.index_in_train = np.unique(self.train_smiles, return_index=True)
        self.train_uni_fingers = [self.train_fingers[idx] for idx in self.index_in_train]

    def get_similarity(self, mol):
        if self.train_fingers is None:
            self._get_train_mols()
            self._get_uni_mols()
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))  # automately sanitize 
        fp_mol = Chem.RDKFingerprint(mol)
        sims = [DataStructs.TanimotoSimilarity(fp, fp_mol) for fp in self.train_uni_fingers]
        return np.array(sims)


    def get_top_sims(self, mol, top=3):
        similarities = self.get_similarity(mol)
        idx_sort = np.argsort(similarities)[::-1]
        top_scores = similarities[idx_sort[:top]]
        top_smiles = self.train_uni_smiles[idx_sort[:top]]
        return top_scores, top_smiles


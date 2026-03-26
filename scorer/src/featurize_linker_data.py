import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

ATOM2IDX = {'C': 0, 'O': 1, 'N': 2, 'F': 3, 'S': 4, 'Cl': 5, 'Br': 6, 'I': 7, 'P':8}
IDX2ATOM = {0: 'C', 1: 'O', 2: 'N', 3: 'F', 4: 'S', 5: 'Cl', 6: 'Br', 7: 'I', 8: 'P'}
VALID_ATOMS = set(ATOM2IDX.keys())

class LINKER:
    """
    Process Enamine's Linker Dataset
    :param csv_path: csv file with headers: ID, SMILES and SA
    :return features: per-smiles feature with the following items:

    positions: (N atoms, 3), representing x,y,z positions of each atom
    atom_types: (N_atoms, 9), representing one of {C,O,N,F,S,Cl,Br,I,P} atom
    atom_mask: (N_atoms, 1), mask for prediction
    y: (1), score value
    """
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path).copy()
        df['is_valid'] = df['SMILES'].apply(is_valid_linker) # remove linkers that have atoms not in the vocabulary
        df = df[df['is_valid']].copy()
        df["label"] = df["SA"]
        self.data = df
    
    def create_data(self):
        dataset = []
        for idx, (smiles, label) in tqdm(enumerate(zip(self.data['SMILES'], self.data["label"]))):
            feature = self.smiles_to_3d_features(smiles, label)
            if feature is not None:
                dataset.append(feature)
        return dataset

    def smiles_to_3d_features(self, smiles, label):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
                
            mol = Chem.AddHs(mol)
            
            # Try multiple embedding strategies
            embed_success = AllChem.EmbedMolecule(mol, randomSeed=42)
            if embed_success == -1:
                for seed in [0, 123, 456]:
                    embed_success = AllChem.EmbedMolecule(mol, randomSeed=seed)
                    if embed_success != -1:
                        break
            
            if embed_success == -1:
                print(f"Embedding failed for: {smiles[:50]}...")
                return None
            
            # Optimize
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            except:
                pass
            
            mol = Chem.RemoveHs(mol)
            
            # Extract features
            positions, atom_types = [], []
            conf = mol.GetConformer()
            
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                positions.append([pos.x, pos.y, pos.z])
                
                # One-hot encode atom type
                atom_type_vec = [0] * 9  # follow DiffPROTACs' atom vocabulary
                atom_symbol = atom.GetSymbol()
                
                if atom_symbol in ATOM2IDX:
                    atom_type_vec[ATOM2IDX[atom_symbol]] = 1
                else:
                    # Unknown atom 
                    print(f"Unknown atom type '{atom_symbol}' in {smiles[:50]}...")
                    return None  
                atom_types.append(atom_type_vec)
            
            # sanity check
            if len(positions) == 0:
                return None
            
            return Data(
                pos=torch.tensor(positions, dtype=torch.float),
                x=torch.tensor(atom_types, dtype=torch.float),
                y=torch.tensor([float(label)], dtype=torch.float),
                smiles=smiles
            )
            
        except Exception as e:
            print(f"Unexpected error for '{smiles[:50]}...': {e}")
            return None

def is_valid_linker(smiles):
    """Check if molecule only contains valid atom types"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in VALID_ATOMS:
            return False
    return True


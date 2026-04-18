from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import (
    CalcNumAromaticRings, CalcNumRings, CalcNumAliphaticRings, CalcFractionCSP3,
    CalcNumHeteroatoms, CalcNumSaturatedRings, CalcNumHeterocycles,
    CalcNumAromaticHeterocycles, CalcNumSaturatedHeterocycles,
    CalcNumAliphaticHeterocycles
)

from sascorer import calculateScore

# BertzCT is in Descriptors module
try:
    from rdkit.Chem.Descriptors import BertzCT
except ImportError:
    from rdkit.Chem.rdMolDescriptors import CalcBertzCT as BertzCT
# Import fragments with extensive fallbacks for different RDKit versions
fragment_functions = {}

# Core fragments that should be available in most versions
try:
    from rdkit.Chem.Fragments import fr_ester, fr_amide, fr_ketone, fr_aldehyde
    fragment_functions.update({
        'fr_ester': fr_ester,
        'fr_amide': fr_amide, 
        'fr_ketone': fr_ketone,
        'fr_aldehyde': fr_aldehyde
    })
except ImportError as e:
    print(f"Warning: Could not import basic fragments: {e}")

# Try to import other fragments with fallbacks
fragment_mappings = {
    'fr_phenol': ['fr_phenol'],
    'fr_alcohol': ['fr_alcohol', 'fr_Al_OH'],
    'fr_ether': ['fr_ether', 'fr_Ar_OH'], 
    'fr_halogen': ['fr_halogen', 'fr_halogen'],
    'fr_nitro': ['fr_nitro', 'fr_nitro_arom'],
    'fr_nitrile': ['fr_nitrile'],
    'fr_sulfone': ['fr_sulfone'],
    'fr_benzene': ['fr_benzene'],
    'fr_pyridine': ['fr_pyridine'],
    'fr_lactam': ['fr_lactam'],
    'fr_lactone': ['fr_lactone'],
    'fr_carboacid': ['fr_COO', 'fr_carboxylic_acid', 'fr_carboacid']
}

# Try to import each fragment function
for func_name, possible_names in fragment_mappings.items():
    fragment_functions[func_name] = None
    for name in possible_names:
        try:
            from rdkit.Chem.Fragments import *
            if name in globals():
                fragment_functions[func_name] = globals()[name]
                break
        except:
            continue

# Define SMARTS patterns as fallbacks
smarts_patterns = {
    'fr_ester': '[CX3](=O)[OX2H0]',
    'fr_amide': '[CX3](=O)[NX3]',
    'fr_lactone': '[CX3](=O)[OX2H0]',  # Note: this is simplified
    'fr_lactam': '[CX3](=O)[NX3H0]',   # Note: this is simplified  
    'fr_carboacid': '[CX3](=O)[OX2H1]',
    'fr_alcohol': '[OX2H]',
    'fr_phenol': '[OX2H][cX3]',
    'fr_ether': '[OD2]([#6])[#6]',
    'fr_nitro': '[NX3](=O)=O',
    'fr_nitrile': '[NX1]#[CX2]',
    'fr_sulfone': '[SX4](=O)(=O)',
    'fr_halogen': '[F,Cl,Br,I]',
    'fr_benzene': 'c1ccccc1',
    'fr_pyridine': 'c1ccccn1',
    'fr_ketone': '[CX3]=[OX1]',
    'fr_aldehyde': '[CX3H1](=O)'
}

# Universal fragment counting function
def count_fragments(mol, fragment_name):
    """
    Count fragments using RDKit function if available, otherwise use SMARTS pattern
    """
    if mol is None:
        return 0
    
    # Try RDKit fragment function first
    if fragment_name in fragment_functions and fragment_functions[fragment_name] is not None:
        try:
            return fragment_functions[fragment_name](mol)
        except:
            pass
    
    # Fallback to SMARTS pattern
    if fragment_name in smarts_patterns:
        try:
            pattern = Chem.MolFromSmarts(smarts_patterns[fragment_name])
            if pattern:
                return len(mol.GetSubstructMatches(pattern))
        except:
            pass
    
    return 0
# Additional imports with fallbacks
try:
    from rdkit.Chem.rdMolDescriptors import CalcNumBridgeheadAtoms, CalcNumSpiroAtoms
except ImportError:
    # Fallback for older RDKit versions
    def CalcNumBridgeheadAtoms(mol):
        return 0
    def CalcNumSpiroAtoms(mol):
        return 0

try:
    from rdkit.Chem.QED import qed
except ImportError:
    # Fallback if QED is not available
    def qed(mol):
        return 0.5
import numpy as np

def calculate_structural_descriptors(mol):
    """Calculate structural descriptors."""
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    if mol is None:
        return {}
            
    return {
        'ring_count': CalcNumRings(mol),
        'aromatic_rings': CalcNumAromaticRings(mol),
        'aliphatic_rings': CalcNumAliphaticRings(mol),
        'saturated_rings': CalcNumSaturatedRings(mol),
        'heterocycles': CalcNumHeterocycles(mol),
        'aromatic_heterocycles': CalcNumAromaticHeterocycles(mol),
        'saturated_heterocycles': CalcNumSaturatedHeterocycles(mol),
        'aliphatic_heterocycles': CalcNumAliphaticHeterocycles(mol),
        'sp3_fraction': CalcFractionCSP3(mol),
        'heavy_atom_count': mol.GetNumHeavyAtoms(),
        'atom_count': mol.GetNumAtoms(),
        'heteroatom_count': CalcNumHeteroatoms(mol),
        'carbon_count': sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6),
        'bertz_complexity': BertzCT(mol)  # Molecular complexity index
    }

def calculate_lipinski_descriptors(mol):
    """Calculate Lipinski's Rule of Five descriptors."""
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    if mol is None:
        return {}
            
    return {
        'molecular_weight': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'h_bond_donors': Lipinski.NumHDonors(mol),
        'h_bond_acceptors': Lipinski.NumHAcceptors(mol),
        'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'tpsa': Descriptors.TPSA(mol),  # Topological polar surface area
        'molar_refractivity': Descriptors.MolMR(mol)
    }

def calculate_degradability_descriptors(mol):
    """
    Calculate descriptors specifically relevant to molecular degradability.
    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    if mol is None:
        return {}
    
    try:
        # Hydrolyzable bond indicators
        ester_bonds = count_fragments(mol, 'fr_ester')
        amide_bonds = count_fragments(mol, 'fr_amide')
        lactone_bonds = count_fragments(mol, 'fr_lactone')
        lactam_bonds = count_fragments(mol, 'fr_lactam')
        
        # Total hydrolyzable bonds
        total_hydrolyzable = ester_bonds + amide_bonds + lactone_bonds + lactam_bonds
        
        # Functional groups affecting stability
        carboxylic_acids = count_fragments(mol, 'fr_carboacid')
        alcohols = count_fragments(mol, 'fr_alcohol')
        phenols = count_fragments(mol, 'fr_phenol')
        ethers = count_fragments(mol, 'fr_ether')
        
        # Electron-withdrawing groups (can affect degradation)
        nitro_groups = count_fragments(mol, 'fr_nitro')
        nitrile_groups = count_fragments(mol, 'fr_nitrile')
        sulfone_groups = count_fragments(mol, 'fr_sulfone')
        halogen_count = count_fragments(mol, 'fr_halogen')
        
        # Aromatic systems (affect stability)
        benzene_rings = count_fragments(mol, 'fr_benzene')
        pyridine_rings = count_fragments(mol, 'fr_pyridine')
        
        return {
            'ester_bonds': ester_bonds,
            'amide_bonds': amide_bonds,
            'lactone_bonds': lactone_bonds,
            'lactam_bonds': lactam_bonds,
            'total_hydrolyzable_bonds': total_hydrolyzable,
            'carboxylic_acids': carboxylic_acids,
            'alcohols': alcohols,
            'phenols': phenols,
            'ethers': ethers,
            'nitro_groups': nitro_groups,
            'nitrile_groups': nitrile_groups,
            'sulfone_groups': sulfone_groups,
            'halogen_count': halogen_count,
            'benzene_rings': benzene_rings,
            'pyridine_rings': pyridine_rings,
            'hydrolyzable_ratio': total_hydrolyzable / max(1, mol.GetNumBonds())  # Fraction of hydrolyzable bonds
        }
        
    except Exception as e:
        print(f"Error calculating degradability descriptors: {e}")
        return {}

def calculate_electronic_descriptors(mol):
    """
    Calculate electronic and reactivity-related descriptors.
    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    if mol is None:
        return {}
    
    try:
        return {
            'formal_charge': Chem.rdmolops.GetFormalCharge(mol),
            'radical_electrons': sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()),
            'valence_electrons': sum(atom.GetTotalValence() for atom in mol.GetAtoms()),
            'aromatic_atoms': sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic()),
            'chiral_centers': len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
            'bridgehead_atoms': CalcNumBridgeheadAtoms(mol),
            'spiro_atoms': CalcNumSpiroAtoms(mol)
        }
        
    except Exception as e:
        print(f"Error calculating electronic descriptors: {e}")
        return {}

def calculate_pharmacokinetic_descriptors(mol):
    """
    Calculate ADMET-related descriptors that might correlate with degradability.
    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    if mol is None:
        return {}
    
    try:
        # Lipinski's Rule of Five violations
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        
        lipinski_violations = (
            (mw > 500) + (logp > 5) + (hbd > 5) + (hba > 10)
        )
        
        # Additional ADMET descriptors
        return {
            'lipinski_violations': lipinski_violations,
            'qed': qed(mol) if 'qed' in dir(Descriptors) else qed(mol),  # Quantitative Estimate of Drug-likeness
            'sas': calculate_synthetic_accessibility(mol),  # Synthetic accessibility score
            'flexibility': calculate_flexibility_index(mol),
            'rigidity': calculate_rigidity_index(mol),
            'polar_surface_area_ratio': Descriptors.TPSA(mol) / max(1, Descriptors.MolWt(mol)),
            'lipophilicity_efficiency': logp / max(1, mol.GetNumHeavyAtoms()),
            'molecular_density': mol.GetNumHeavyAtoms() / max(1, Descriptors.MolWt(mol))
        }
        
    except Exception as e:
        print(f"Error calculating pharmacokinetic descriptors: {e}")
        return {}

def calculate_synthetic_accessibility(mol):
    """
    Simplified synthetic accessibility score.
    Lower values indicate easier synthesis (potentially more degradable).
    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    if mol is None:
        return None
    return calculateScore(mol)

def calculate_flexibility_index(mol):
    """
    Calculate molecular flexibility index.
    More flexible molecules might be more susceptible to degradation.
    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    if mol is None:
        return None
    
    try:
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        total_bonds = mol.GetNumBonds()
        
        if total_bonds == 0:
            return 0
            
        flexibility = rotatable_bonds / total_bonds
        return flexibility
        
    except Exception as e:
        return None

def calculate_rigidity_index(mol):
    """
    Calculate molecular rigidity index (inverse of flexibility).
    More rigid molecules might be less susceptible to degradation.
    """
    flexibility = calculate_flexibility_index(mol)
    if flexibility is None:
        return None
    return 1 - flexibility

def calculate_bond_descriptors(mol):
    """
    Calculate descriptors related to chemical bonds.
    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    if mol is None:
        return {}
    
    try:
        single_bonds = 0
        double_bonds = 0
        triple_bonds = 0
        aromatic_bonds = 0
        
        for bond in mol.GetBonds():
            if bond.GetIsAromatic():
                aromatic_bonds += 1
            elif bond.GetBondType() == Chem.BondType.SINGLE:
                single_bonds += 1
            elif bond.GetBondType() == Chem.BondType.DOUBLE:
                double_bonds += 1
            elif bond.GetBondType() == Chem.BondType.TRIPLE:
                triple_bonds += 1
        
        total_bonds = mol.GetNumBonds()
        
        return {
            'single_bonds': single_bonds,
            'double_bonds': double_bonds,
            'triple_bonds': triple_bonds,
            'aromatic_bonds': aromatic_bonds,
            'total_bonds': total_bonds,
            'double_bond_ratio': double_bonds / max(1, total_bonds),
            'aromatic_bond_ratio': aromatic_bonds / max(1, total_bonds),
            'unsaturation_index': (double_bonds + triple_bonds + aromatic_bonds) / max(1, total_bonds)
        }
        
    except Exception as e:
        print(f"Error calculating bond descriptors: {e}")
        return {}

def calculate_atom_type_descriptors(mol):
    """
    Calculate descriptors based on atom types and their environments.
    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    if mol is None:
        return {}
    
    try:
        atom_counts = {}
        hybridization_counts = {'SP': 0, 'SP2': 0, 'SP3': 0, 'OTHER': 0}
        
        for atom in mol.GetAtoms():
            # Count atom types
            symbol = atom.GetSymbol()
            atom_counts[symbol] = atom_counts.get(symbol, 0) + 1
            
            # Count hybridization states
            hybridization = atom.GetHybridization()
            if hybridization == Chem.HybridizationType.SP:
                hybridization_counts['SP'] += 1
            elif hybridization == Chem.HybridizationType.SP2:
                hybridization_counts['SP2'] += 1
            elif hybridization == Chem.HybridizationType.SP3:
                hybridization_counts['SP3'] += 1
            else:
                hybridization_counts['OTHER'] += 1
        
        # Calculate ratios
        total_atoms = mol.GetNumAtoms()
        
        result = {
            'carbon_atoms': atom_counts.get('C', 0),
            'nitrogen_atoms': atom_counts.get('N', 0),
            'oxygen_atoms': atom_counts.get('O', 0),
            'sulfur_atoms': atom_counts.get('S', 0),
            'phosphorus_atoms': atom_counts.get('P', 0),
            'fluorine_atoms': atom_counts.get('F', 0),
            'chlorine_atoms': atom_counts.get('Cl', 0),
            'bromine_atoms': atom_counts.get('Br', 0),
            'iodine_atoms': atom_counts.get('I', 0),
            'sp_carbons': hybridization_counts['SP'],
            'sp2_carbons': hybridization_counts['SP2'],
            'sp3_carbons': hybridization_counts['SP3'],
            'heteroatom_ratio': (total_atoms - atom_counts.get('C', 0)) / max(1, total_atoms),
            'cn_ratio': atom_counts.get('C', 0) / max(1, atom_counts.get('N', 1)),
            'co_ratio': atom_counts.get('C', 0) / max(1, atom_counts.get('O', 1))
        }
        
        return result
        
    except Exception as e:
        print(f"Error calculating atom type descriptors: {e}")
        return {}

def calculate_energy(mol):
    """
    Calculate the MMFF (Merck Molecular Force Field) energy for a molecule.
    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    if mol is None:
        return None
            
    try:
        # Make a copy of the molecule to avoid modifying the original
        mol_copy = Chem.Mol(mol)
            
        # Add explicit hydrogens
        mol_copy = Chem.AddHs(mol_copy)
            
        # Generate 3D coordinates if they don't exist
        if mol_copy.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol_copy, randomSeed=42)
                
        # Optimize the geometry with MMFF
        AllChem.MMFFOptimizeMolecule(mol_copy)
                
        # Get MMFF properties and calculate energy
        mp = AllChem.MMFFGetMoleculeProperties(mol_copy)
        if mp is None:  # Some molecules might not be parameterized in MMFF
            return None
                
        ff = AllChem.MMFFGetMoleculeForceField(mol_copy, mp, confId=0)
        if ff is None:
            return None
                
        energy = ff.CalcEnergy()
        return energy
    except Exception as e:
        return None

def calculate_all_properties(smiles):
    """
    Calculate all molecular properties for a given SMILES string.
    
    Args:
        smiles (str): SMILES string of the molecule
        
    Returns:
        dict: Dictionary containing all calculated properties
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Could not parse SMILES: {smiles}")
        return {}
    
    # Calculate all descriptor categories
    properties = {}
    
    # Original descriptors
    properties.update(calculate_structural_descriptors(mol))
    properties.update(calculate_lipinski_descriptors(mol))
    
    # New descriptors
    properties.update(calculate_degradability_descriptors(mol))
    properties.update(calculate_electronic_descriptors(mol))
    properties.update(calculate_pharmacokinetic_descriptors(mol))
    properties.update(calculate_bond_descriptors(mol))
    properties.update(calculate_atom_type_descriptors(mol))
    
    # Energy calculation (optional, can be slow)
    energy = calculate_energy(mol)
    if energy is not None:
        properties['mmff_energy'] = energy
    
    # Add SMILES for reference
    properties['smiles'] = smiles
    
    return properties

def get_degradability_relevant_features():
    """
    Return a list of features most relevant for degradability prediction.
    """
    return [
        # Hydrolyzable bonds (key for degradability)
        'ester_bonds', 'amide_bonds', 'lactone_bonds', 'lactam_bonds', 
        'total_hydrolyzable_bonds', 'hydrolyzable_ratio',
        
        # Structural stability indicators
        'aromatic_rings', 'benzene_rings', 'rigidity', 'bertz_complexity',
        
        # Functional groups affecting degradation
        'alcohols', 'phenols', 'carboxylic_acids', 'nitro_groups',
        
        # Electronic properties
        'formal_charge', 'aromatic_atoms', 'unsaturation_index',
        
        # Atom types and ratios
        'heteroatom_ratio', 'oxygen_atoms', 'nitrogen_atoms',
        
        # Physical properties
        'molecular_weight', 'logp', 'tpsa', 'flexibility'
    ]

def get_all_feature_names():
    """
    Return a list of all available feature names.
    """
    # Generate a dummy molecule to get all possible feature names
    dummy_smiles = "CCO"  # Simple ethanol
    dummy_properties = calculate_all_properties(dummy_smiles)
    return [key for key in dummy_properties.keys() if key != 'smiles']
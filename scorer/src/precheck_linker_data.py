# precheck_linker_data.py
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse
import warnings
warnings.filterwarnings('ignore')

ATOM2IDX = {'C': 0, 'O': 1, 'N': 2, 'F': 3, 'S': 4, 'Cl': 5, 'Br': 6, 'I': 7, 'P': 8}
VALID_ATOMS = set(ATOM2IDX.keys())

class LinkerValidator:
    def __init__(self):
        self.stats = {
            'total': 0,
            'invalid_smiles': 0,
            'invalid_atoms': 0,
            'embedding_failed': 0,
            'optimization_failed': 0,
            'valid': 0
        }
        self.invalid_atom_types = set()
    
    def check_smiles(self, smiles):
        """Check if SMILES is valid"""
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None, mol
    
    def check_atoms(self, mol):
        """Check if all atoms are in vocabulary"""
        invalid = []
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol not in VALID_ATOMS:
                invalid.append(symbol)
                self.invalid_atom_types.add(symbol)
        return len(invalid) == 0, invalid
    
    def check_3d_generation(self, smiles):
        """Check if 3D structure can be generated"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            
            # Try embedding
            result = AllChem.EmbedMolecule(mol, randomSeed=42)
            if result < 0:
                return False, "embedding_failed"
            
            # Try optimization
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except:
                return False, "optimization_failed"
            
            mol = Chem.RemoveHs(mol)
            
            # Verify we can extract features
            if mol.GetNumAtoms() == 0:
                return False, "no_atoms_after_processing"
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    def validate_entry(self, smiles, label):
        """Full validation pipeline"""
        self.stats['total'] += 1
        
        # Check 1: Valid SMILES
        is_valid_smiles, mol = self.check_smiles(smiles)
        if not is_valid_smiles:
            self.stats['invalid_smiles'] += 1
            return False, "invalid_smiles"
        
        # Check 2: Valid atom types
        has_valid_atoms, invalid_atoms = self.check_atoms(mol)
        if not has_valid_atoms:
            self.stats['invalid_atoms'] += 1
            return False, f"invalid_atoms: {invalid_atoms}"
        
        # Check 3: 3D generation
        can_generate_3d, error = self.check_3d_generation(smiles)
        if not can_generate_3d:
            if "embedding" in error:
                self.stats['embedding_failed'] += 1
            elif "optimization" in error:
                self.stats['optimization_failed'] += 1
            return False, error
        
        self.stats['valid'] += 1
        return True, None
    
    def print_stats(self):
        """Print validation statistics"""
        print("\n" + "="*60)
        print("VALIDATION STATISTICS")
        print("="*60)
        print(f"Total entries:           {self.stats['total']}")
        print(f"Valid entries:           {self.stats['valid']} ({100*self.stats['valid']/max(self.stats['total'],1):.1f}%)")
        print(f"\nFailure breakdown:")
        print(f"  Invalid SMILES:        {self.stats['invalid_smiles']}")
        print(f"  Invalid atom types:    {self.stats['invalid_atoms']}")
        print(f"  Embedding failed:      {self.stats['embedding_failed']}")
        print(f"  Optimization failed:   {self.stats['optimization_failed']}")
        
        if self.invalid_atom_types:
            print(f"\nInvalid atom types found: {sorted(self.invalid_atom_types)}")
        print("="*60)


def precheck_linker_csv(input_csv, output_csv, sa_column='SA', smiles_column='SMILES'):
    """
    Precheck and filter linker CSV file
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output cleaned CSV file
        sa_column: Name of SA score column
        smiles_column: Name of SMILES column
    """
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Check required columns
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in CSV")
    if sa_column not in df.columns:
        raise ValueError(f"Column '{sa_column}' not found in CSV")
    
    print(f"Found {len(df)} entries")
    print(f"Columns: {list(df.columns)}")
    
    validator = LinkerValidator()
    
    # Validate each entry
    valid_indices = []
    failure_reasons = []
    
    print("\nValidating entries...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        smiles = row[smiles_column]
        sa_score = row[sa_column]
        
        is_valid, reason = validator.validate_entry(smiles, sa_score)
        
        if is_valid:
            valid_indices.append(idx)
        else:
            failure_reasons.append(reason)
    
    # Create cleaned dataframe
    df_clean = df.iloc[valid_indices].copy()
    
    # Print statistics
    validator.print_stats()
    
    # Save cleaned data
    df_clean.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df_clean)} valid entries to {output_csv}")
    
    # Optional: save rejection report
    rejection_report = output_csv.replace('.csv', '_rejection_report.txt')
    with open(rejection_report, 'w') as f:
        f.write("REJECTION REPORT\n")
        f.write("="*60 + "\n\n")
        
        # Count failure types
        from collections import Counter
        failure_counts = Counter(failure_reasons)
        
        f.write("Failure type distribution:\n")
        for failure_type, count in failure_counts.most_common():
            f.write(f"  {failure_type}: {count}\n")
        
        f.write(f"\nInvalid atom types encountered: {sorted(validator.invalid_atom_types)}\n")
    
    print(f"Rejection report saved to {rejection_report}")
    
    return df_clean


def main():
    parser = argparse.ArgumentParser(description='Precheck and filter linker CSV data')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output cleaned CSV file')
    parser.add_argument('--sa_column', type=str, default='SA', help='SA score column name')
    parser.add_argument('--smiles_column', type=str, default='SMILES', help='SMILES column name')
    
    args = parser.parse_args()
    
    precheck_linker_csv(
        input_csv=args.input,
        output_csv=args.output,
        sa_column=args.sa_column,
        smiles_column=args.smiles_column
    )


if __name__ == "__main__":
    main()

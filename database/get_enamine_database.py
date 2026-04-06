import os
import csv
import re
import glob
import argparse
import pandas as pd
try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

import sys
sys.path.append("evaluation/")
from sascorer import  calculateScore

class DBParser:
    def __init__(
            self,
            data_path = 'database/ENAMINE/Comprehensive/'):
        self.data_path = data_path

    def process_data(self):
        sdf_list = glob.glob(f'{self.data_path}/*.sdf')
        df_list = []
        for sdf in sdf_list:
            fname = os.path.basename(sdf)
            df = self.sdf_to_csv(sdf)
            df['sdf_path'] = f'{self.data_path}/{fname}'
            df_list.append(df)
        df = pd.concat(df_list, axis=0)
        print(f'processed {len(sdf_list)} sdf files, {len(df)} linkers in total')
        return df


    def sdf_to_csv(self, sdf_file, include_smiles=True):
        """
        Convert SDF file to CSV, using the field names in angle brackets as column names.
        
        Parameters:
        -----------
        sdf_file : str
            Path to the input SDF file
        include_smiles : bool, optional
            Whether to include SMILES strings as a column (requires RDKit)
        """
        if include_smiles and not RDKIT_AVAILABLE:
            print("Warning: RDKit is not installed. SMILES strings will not be included.")
            include_smiles = False
        
        molecules = []
        all_properties = set()
        
        # For SMILES generation with RDKit
        mol_objects = []
        if include_smiles and RDKIT_AVAILABLE:
            # Read directly from the file for RDKit
            suppl = Chem.SDMolSupplier(sdf_file)
            mol_objects = [mol for mol in suppl if mol]
        
        # Read SDF file in text mode for property extraction
        with open(sdf_file, 'r') as f:
            sdf_content = f.read()
        
        # First pass: collect all possible property names
        pattern = r'>  <([^>]+)>'
        all_property_names = set(re.findall(pattern, sdf_content))
        
        # If SMILES is requested, add it to the property names
        if include_smiles:
            all_property_names.add('SMILES')
        
        # Split file into individual molecules
        molecule_blocks = sdf_content.split('$$$$')
        
        # Process each molecule
        for i, block in enumerate(molecule_blocks):
            if not block.strip():
                continue
            
            # Extract properties using regex pattern matching
            properties = {}
            pattern = r'>  <([^>]+)>\n(.+?)(?=\n\n|\n>|\Z)'
            matches = re.findall(pattern, block, re.DOTALL)
            for prop_name, prop_value in matches:
                prop_name = prop_name.strip()
                prop_value = prop_value.strip()
                properties[prop_name] = prop_value
            
            # Add SMILES if requested and available
            if include_smiles and RDKIT_AVAILABLE and i < len(mol_objects):
                try:
                    properties['SMILES'] = Chem.MolToSmiles(mol_objects[i])
                except Exception as e:
                    print(f"Warning: Failed to generate SMILES for molecule {i+1}: {e}")
                    properties['SMILES'] = ''
            
            molecules.append(properties)
        
        
        # Use all properties found in the SDF file
        fieldnames = sorted(list(all_property_names))
        if include_smiles and 'SMILES' not in fieldnames:
            fieldnames.append('SMILES')
        
        # Create DataFrame with only the desired columns
        df = pd.DataFrame(molecules)
        if "ID" in df.columns:
            cols = ["ID", "SMILES", "mw"]
        elif "Catalog_ID" in df.columns:
            cols = ["Catalog_ID", "SMILES", "mw"]
        else:
            df["ID"] = df.index
            cols = ["ID", "SMILES", "mw"]
        f_df = df[cols].copy()
        f_df.rename(columns={"Catalog_ID": "ID"}, inplace=True)
        return f_df
    
def main(args):
    df_list = []
    for cate in ['Comprehensive', 'MADE', 'Stock']:
        db = DBParser(data_path=f'{args.data_path}/{cate}/')
        df = db.process_data()
        df["Group"] = cate
        cols = ["ID", "SMILES", "mw", "Group"]
        df_list.append(df[cols])
    merged_df = pd.concat(df_list, axis=0)
    uni_df = merged_df.drop_duplicates(subset=["SMILES"], keep="first").copy()

    uni_df["SA"] = uni_df['SMILES'].apply(lambda x: calculateScore(Chem.MolFromSmiles(x)))
    uni_df.to_csv(f'{args.save_dir}/enamine_database.csv', header=True, index=False)
    print(f'final linkers with unique SMILES={len(uni_df)}/{len(merged_df)}')
    print(f'saved file to {args.save_dir}/enamine_database.csv')

def parse_args():
    parser = argparse.ArgumentParser(description='Process Enamine datasets')
    parser.add_argument('--data_path', type=str, default='database/ENAMINE/', help='Path to Emamine sdf files')
    parser.add_argument('--save_dir', type=str, default='database/', help='Path to output file')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
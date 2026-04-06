"""
Return processed_protacdb.csv


Note: xlsx files were requested directly from PROTAC-DB,
not downloaded from the database
"""
import os
import re
import json
import argparse
import pandas as pd

class DBParser:
    def __init__(self, data_path = 'database/PROTAC-DB/'):
        self.data_path = data_path
        self.warheads_df = pd.read_excel(f'{self.data_path}/warhead.xlsx')
        self.e3_ligands_df = pd.read_excel(f'{self.data_path}/e3_ligand.xlsx')
        self.protacs_df = pd.read_excel(f'{self.data_path}/protac.xlsx') 

    def parse_data(self):
        # Create mapping dictionaries for fast lookup
        wsmi_wid_dict = self.warheads_df.set_index('ligand_canonical')['id_in_database'].to_dict()
        esmi_eid_dict = self.e3_ligands_df.set_index('ligand_canonical')['id_in_database'].to_dict()
        ename_eut_dict = self.e3_ligands_df.set_index('short_target_name')['uniprot'].to_dict()

        # Map values 
        self.protacs_df['id_warhead'] = self.protacs_df['warhead_canonical'].map(wsmi_wid_dict)
        self.protacs_df['id_e3_ligand'] = self.protacs_df['e3_ligand_canonical'].map(esmi_eid_dict)
        self.protacs_df['e3_ligase_uniprot'] = self.protacs_df['e3_ligase'].map(ename_eut_dict)

        cols = ['id_in_database',
                'id_warhead',
                'id_linker',
                'id_e3_ligand',
                'uniprot', #POI target
                'short_target_name', #POI target
                'e3_ligase',
                'e3_ligase_uniprot',
                'dc50',
                'dmax',
                'smiles_canonical',
                'warhead_canonical', 
                'linker_canonical', 
                'e3_ligand_canonical', 
                #'percent_degradation',
                #'percent_degradation_assay',
                #'kd_pt',
                #'kd_pe',
                #'kd_ce'
                ]
        df = self.protacs_df[cols].copy()

        # Clean DC50 and DMax
        df['dc50_parsed'] = df['dc50'].apply(lambda x: self._parse_multiple_values(x, self._parse_single_dc50))
        df['dmax_parsed'] = df['dmax'].apply(lambda x: self._parse_multiple_values(x, self._parse_single_dmax))

        # Rename columns
        df = df.rename(columns={
            'dc50_parsed': 'DC50 (nM)',
            'dmax_parsed': 'Dmax (%)',
            #'percent_degradation': 'Percent degradation (%)',
            #'percent_degradation_assay': 'Assay (Percent degradation)',
            #'kd_pt': 'Kd (nM, Protac to Target)',
            #'kd_pe': 'Kd (nM, Protac to E3)',
            #'kd_ce': 'Kd (nM, Ternary complex)',
            'id_in_database': 'id_protac',
            'uniprot': 'target_uniprot',
            'short_target_name': 'target_name',
            'smiles_canonical': 'protac_smiles',
            'warhead_canonical': 'warhead_smiles',
            'linker_canonical': 'linker_smiles',
            'e3_ligand_canonical': 'e3_ligand_smiles',
        })

        
        
        # Remove missing smiles
        rdf = df[df['protac_smiles'].notna() & df['warhead_smiles'].notna() & df['linker_smiles'].notna() & df['e3_ligand_smiles'].notna()]
        return rdf

    # Helper function to parse individual DC50 (nM) values
    def _parse_single_dc50(self, value):
        if isinstance(value, str):
            value = value.strip()
            # Check for numeric strings
            if re.match(r'^\d+(\.\d+)?$', value):
                return float(value)
            # Check for inequalities or ranges
            elif re.match(r'^(<=|>=|<|>)\d+(\.\d+)?$', value):
                return float(re.sub(r'[<>=]', '', value))  # Extract the numeric part
            elif re.match(r'^\d+-\d+$', value):
                # For ranges like '51-70', take the average
                start, end = map(float, value.split('-'))
                return (start + end) / 2
            else:
                # For other strings like 'unknown', 'N.D.', return None
                return None
        return value  # In case it's already a numeric value

    # Helper function to parse individual Dmax (%) values
    def _parse_single_dmax(self, value):
        if isinstance(value, str):
            value = value.strip()
            # Check for numeric strings
            if re.match(r'^\d+(\.\d+)?$', value):
                return float(value)
            # Check for inequalities
            elif re.match(r'^(<=|>=|<|>)\d+(\.\d+)?$', value):
                return float(re.sub(r'[<>=]', '', value))  # Extract the numeric part
            else:
                # For other strings like 'unknown', 'N.D.', return None
                return None
        return value  # In case it's already a numeric value

    def _parse_multiple_values(self, value, parse_function):
        if isinstance(value, str) and '/' in value:
            values = value.split('/')
            parsed_values = [parse_function(v) for v in values]
            return parsed_values[0] if parsed_values else None  # first parsed value
        else:
            parsed = parse_function(value)
            return parsed  # single value only, not wrapped in a list

    # Function to extract concentration part specifically before "nM"
    def _extract_concentration(self, assay_text):
        match = re.search(r'(\d+(?:/\d+)*)(?=\s*nM)', assay_text)
        return match.group(1) if match else None
       
if __name__ == "__main__":
    db = DBParser(data_path="database/PROTAC-DB/")
    df = db.parse_data()
    df.to_csv('database/protacdb_database.csv', header=True, index=False)
    




    



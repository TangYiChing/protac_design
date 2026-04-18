# mol_converter.py

"""
OpenBabel-based Molecule Converter
XYZ → SDF/Mol
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np

try:
    from openbabel import openbabel as ob
    from openbabel import pybel
    OPENBABEL_AVAILABLE = True
except ImportError:
    OPENBABEL_AVAILABLE = False
    print("OpenBabel not available. Install with: conda install -c conda-forge openbabel")


class OpenBabelConverter:
   
    def __init__(self, 
                 perceive_bonds: bool = True,
                 add_hydrogens: bool = True,
                 ph: float = 7.4):
        
        if not OPENBABEL_AVAILABLE:
            raise ImportError("OpenBabel not installed!")
        
        self.perceive_bonds = perceive_bonds
        self.add_hydrogens = add_hydrogens
        self.ph = ph
        
        print("\n" + "="*70)
        print("OpenBabel Converter Initialized")
        print("="*70)
        print(f"  Version: {ob.OBReleaseVersion()}")
        print(f"  Perceive Bonds: {perceive_bonds}")
        print(f"  Add Hydrogens: {add_hydrogens}")
        print(f"  pH: {ph}")
        print("="*70 + "\n")
    
    def xyz_to_mol(self, xyz_file: str) -> Optional[pybel.Molecule]:
        """
        Returns:
            pybel.Molecule (None, if fails)
        """
        
        try:
            mol = next(pybel.readfile("xyz", xyz_file))
            
            # infer bonds
            if self.perceive_bonds:
                mol.OBMol.PerceiveBondOrders()
            
            # add Hs
            if self.add_hydrogens:
                mol.OBMol.AddHydrogens(False, True, self.ph)
            
            return mol
            
        except StopIteration:
            print(f"Empty XYZ file: {xyz_file}")
            return None
        except Exception as e:
            print(f"Error reading {xyz_file}: {e}")
            return None
    
    def mol_to_sdf(self, 
                   mol: pybel.Molecule, 
                   output_file: str,
                   add_properties: Dict = None) -> bool:
        
        try:
            if add_properties:
                for key, value in add_properties.items():
                    data = ob.OBPairData()
                    data.SetAttribute(key)
                    data.SetValue(str(value))
                    mol.OBMol.CloneData(data)
            mol.write("sdf", output_file, overwrite=True)
            
            return True   
        except Exception as e:
            print(f"Error writing SDF {output_file}: {e}")
            return False
    def xyz_to_sdf(self,
                   xyz_file: str,
                   sdf_file: str,
                   properties: Dict = None) -> bool:
        mol = self.xyz_to_mol(xyz_file)
        
        if mol is None:
            return False
        
        return self.mol_to_sdf(mol, sdf_file, properties)
    
    def batch_convert(self,
                     input_dir: str,
                     output_dir: str,
                     input_format: str = "xyz",
                     output_format: str = "sdf",
                     properties_dict: Dict[str, Dict] = None) -> Dict:
        """
        Batch convertion of xyz2mol
        
        Args:
            input_dir: path to xyz files
            output_dir: path to outputs
            input_format: can be xyz, pdb, mol2
            output_format: can be sdf, mol2, pdb
            properties_dict: {filename: {prop: value}} 
            
        Returns:
            summary_dict
        """
        
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # load inputs
        input_files = list(input_dir.glob(f"*.{input_format}"))
        
        #print(f"\nBatch Converting {len(input_files)} files...")
        #print(f"   {input_format.upper()} → {output_format.upper()}")
        #print(f"   Input:  {input_dir}")
        #print(f"   Output: {output_dir}\n")
        
        stats = {
            'total': len(input_files),
            'successful': 0,
            'failed': 0,
            'failed_files': []
        }
        
        for i, input_file in enumerate(input_files, 1):
            output_file = output_dir / f"{input_file.stem}.{output_format}"
            
            # properties
            props = None
            if properties_dict and input_file.name in properties_dict:
                props = properties_dict[input_file.name]
            
            # conversion
            try:
                mol = next(pybel.readfile(input_format, str(input_file)))
                
                if self.perceive_bonds:
                    mol.OBMol.PerceiveBondOrders()
                
                if self.add_hydrogens:
                    mol.OBMol.AddHydrogens(False, True, self.ph)
                
                # add props
                if props:
                    for key, value in props.items():
                        data = ob.OBPairData()
                        data.SetAttribute(key)
                        data.SetValue(str(value))
                        mol.OBMol.CloneData(data)
                
                # save to file
                mol.write(output_format, str(output_file), overwrite=True)
                
                stats['successful'] += 1
                
                #if i % 10 == 0:
                #    print(f"   Progress: {i}/{len(input_files)}")
                
            except Exception as e:
                print(f"Failed: {input_file.name} - {e}")
                stats['failed'] += 1
                stats['failed_files'].append(input_file.name)
        
        # Summary
        #print(f"\n{'='*70}")
        #print(f"Batch Conversion Complete!")
        #print(f"   Successful: {stats['successful']}/{stats['total']}")
        #print(f"   Failed: {stats['failed']}/{stats['total']}")
        
        #if stats['failed'] > 0:
        #    print(f"\n   Failed files:")
        #    for fname in stats['failed_files'][:10]:  
        #        print(f"     - {fname}")
        #    if len(stats['failed_files']) > 10:
        #        print(f"     ... and {len(stats['failed_files']) - 10} more")
        
        #print(f"{'='*70}\n")
        
        return stats
    
    def xyz_to_smiles(self, xyz_file: str) -> Optional[str]:
        """
        XYZ → SMILES
        """
        
        mol = self.xyz_to_mol(xyz_file)
        
        if mol is None:
            return None
        
        try:
            smiles = mol.write("smi").strip()
            return smiles.split("\t")[0]
        except:
            return None
    
    def validate_molecule(self, mol: pybel.Molecule) -> Dict:
        validation = {
            'valid': True,
            'num_atoms': mol.OBMol.NumAtoms(),
            'num_bonds': mol.OBMol.NumBonds(),
            'num_heavy_atoms': mol.OBMol.NumHvyAtoms(),
            'molecular_weight': mol.molwt,
            'formula': mol.formula,
            'has_3d': mol.OBMol.Has3D(),
            'warnings': []
        }
        
        # check atoms
        if validation['num_atoms'] == 0:
            validation['valid'] = False
            validation['warnings'].append("No atoms")
        
        # check heavy atoms
        if validation['num_heavy_atoms'] == 0:
            validation['valid'] = False
            validation['warnings'].append("No heavy atoms")
        
        # check molecular weight
        if validation['molecular_weight'] < 10 or validation['molecular_weight'] > 2000:
            validation['warnings'].append(f"Unusual molecular weight: {validation['molecular_weight']:.1f}")
        
        # check 3d coordinations
        if not validation['has_3d']:
            validation['warnings'].append("No 3D coordinates")
        
        return validation


def quick_xyz_to_sdf(xyz_file: str, sdf_file: str = None) -> str:
    """
    single case xyz to sdf
    
    Returns:
        path
    """
    
    if sdf_file is None:
        sdf_file = str(Path(xyz_file).with_suffix('.sdf'))
    
    converter = OpenBabelConverter()
    success = converter.xyz_to_sdf(xyz_file, sdf_file)
    
    if success:
        print(f"Converted: {Path(xyz_file).name} → {Path(sdf_file).name}")
        return sdf_file
    else:
        print(f"Failed to convert: {xyz_file}")
        return None


def batch_xyz_to_sdf(xyz_dir: str, sdf_dir: str = None, sa_scores: Dict = None):
    """
    batch xyz to sdf
    
    Args:
        xyz_dir: XYZ folders
        sdf_dir: SDF folders (default: xyz_dir + '_sdf')
        sa_scores: {filename: sa_score}
    """
    
    if sdf_dir is None:
        sdf_dir = str(Path(xyz_dir).parent / f"{Path(xyz_dir).name}_sdf")
    
    # props
    properties_dict = None
    if sa_scores:
        properties_dict = {
            fname: {'SA_Score': score} 
            for fname, score in sa_scores.items()
        }
    
    converter = OpenBabelConverter()
    stats = converter.batch_convert(
        input_dir=xyz_dir,
        output_dir=sdf_dir,
        properties_dict=properties_dict
    )
    return stats

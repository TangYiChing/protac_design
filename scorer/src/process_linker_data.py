import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize


PROTAC_DATA = {
    "protacdb": "../database/protacdb_database.csv",
    "protacpedia": "../database/PROTACpedia_database.csv"
}


def canonicalize_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        params = rdMolStandardize.CleanupParameters()
        rdMolStandardize.Cleanup(mol)
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except:
        return None


def ecfp4_fingerprint(smi, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    gen = GetMorganGenerator(radius=radius, fpSize=nbits)
    return gen.GetFingerprint(mol)


def max_tanimoto_to_set(fp, fp_set):
    if fp is None or not fp_set:
        return 0.0
    sims = DataStructs.BulkTanimotoSimilarity(fp, fp_set)
    return max(sims) if sims else 0.0


def nearest_neighbor_tanimoto(fps):
    n = len(fps)
    sims = np.zeros(n, dtype=float)
    for i, fp in enumerate(fps):
        if fp is None:
            sims[i] = 0.0
            continue
        all_sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, fps))
        if np.isfinite(all_sims).any():
            all_sims[i] = -np.inf
            sims[i] = float(np.nanmax(all_sims))
        else:
            sims[i] = 0.0
    return sims


def murcko_scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaf, isomericSmiles=True) if scaf else ""


def stratified_sample_by_decile(df, size, label_col="SA", seed=42):
    df = df.copy()
    df["_decile"] = pd.qcut(df[label_col], 10, labels=False, duplicates="drop")
    parts = []
    for dec, g in df.groupby("_decile"):
        k = max(1, round(len(g) * size / len(df)))
        k = min(k, len(g))
        parts.append(g.sample(n=k, random_state=seed))
    out = pd.concat(parts).sample(frac=1.0, random_state=seed)
    if len(out) > size:
        out = out.sample(n=size, random_state=seed)
    return out.drop(columns=["_decile"]).reset_index(drop=True)


def remove_protac_linkers(df, threshold=0.95):
    """Remove molecules similar to known PROTAC linkers to prevent data leakage."""
    linker_smiles = []
    
    for name, path in PROTAC_DATA.items():
        protac_df = pd.read_csv(path)
        if name == "protacdb":
            protac_df = protac_df.dropna(subset=["linker_smiles"])
            linker_smiles.extend(protac_df["linker_smiles"].unique())
        elif name == "protacpedia":
            protac_df = protac_df.dropna(subset=["Linker"])
            linker_smiles.extend(protac_df["Linker"].unique())
    
    linker_smiles = [canonicalize_smiles(s) for s in linker_smiles]
    linker_smiles = [s for s in linker_smiles if s]
    linker_fps = [ecfp4_fingerprint(s) for s in linker_smiles]
    linker_fps = [fp for fp in linker_fps if fp is not None]
    
    keep_mask = []
    for smi in df["SMILES"].tolist():
        fp = ecfp4_fingerprint(smi)
        max_sim = max_tanimoto_to_set(fp, linker_fps)
        keep_mask.append(max_sim < threshold)
    
    return df.loc[keep_mask].reset_index(drop=True)


def create_splits(df, ood_threshold=0.20, id_fraction=0.20, target_ood_frac=0.10, k_folds=5, seed=42):
    """
    Create OOD test, ID test, and k-fold CV splits with SA stratification.
    
    OOD test: molecules with nearest-neighbor Tanimoto < threshold
    ID test: scaffold-based split maintaining SA distribution
    CV folds: scaffold-aware GroupKFold on remaining data
    """
    rng = np.random.RandomState(seed)
    df = df.copy().reset_index(drop=True)
    
    fps = [ecfp4_fingerprint(s) for s in df["SMILES"].tolist()]
    df["_nn_tanimoto"] = nearest_neighbor_tanimoto(fps)
    df["scaffold"] = df["SMILES"].map(murcko_scaffold)
    
    N = len(df)
    
    # OOD test set
    ood_mask = df["_nn_tanimoto"] < ood_threshold
    df_ood_pool = df[ood_mask].copy()
    target_ood = max(0, int(round(target_ood_frac * N)))
    if target_ood > 0 and len(df_ood_pool) > target_ood:
        df_ood = stratified_sample_by_decile(df_ood_pool, size=target_ood, label_col="SA", seed=seed)
    else:
        df_ood = df_ood_pool.copy()
    
    remaining = df[~df.index.isin(df_ood.index)].copy().reset_index(drop=True)
    
    # ID test set (scaffold split)
    target_id = int(round(id_fraction * len(remaining)))
    rem_with_dec = remaining.copy()
    
    if target_id > 0 and len(rem_with_dec) > 0:
        rem_with_dec["_decile"] = pd.qcut(rem_with_dec["SA"], 10, labels=False, duplicates="drop")
        scaffolds = rem_with_dec["scaffold"].fillna("").unique().tolist()
        rng.shuffle(scaffolds)
        
        id_indices = []
        for scaf in scaffolds:
            cand = rem_with_dec[rem_with_dec["scaffold"] == scaf]
            if len(id_indices) + len(cand) <= target_id or len(id_indices) == 0:
                id_indices.extend(cand.index.tolist())
            if len(id_indices) >= target_id:
                break
        
        df_id = rem_with_dec.loc[sorted(set(id_indices))].drop(columns=["_decile"], errors="ignore")
        train_val = rem_with_dec.drop(index=df_id.index).drop(columns=["_decile"], errors="ignore")
    else:
        df_id = remaining.iloc[0:0].copy()
        train_val = remaining
    
    df_id = df_id.reset_index(drop=True)
    train_val = train_val.reset_index(drop=True)
    
    # CV folds
    groups = train_val["scaffold"].fillna("")
    gkf = GroupKFold(n_splits=k_folds)
    cv_splits = []
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(train_val, groups=groups)):
        cv_splits.append({
            "fold": fold_idx + 1,
            "train": train_val.iloc[tr_idx].reset_index(drop=True),
            "valid": train_val.iloc[va_idx].reset_index(drop=True)
        })
    
    return df_ood, df_id, train_val, cv_splits


def main(args):
    enamine_df = pd.read_csv(args.data_path)
    enamine_df = enamine_df.drop_duplicates(subset=["SMILES"], keep="first")
    enamine_df["SMILES"] = enamine_df["SMILES"].map(canonicalize_smiles)
    enamine_df = enamine_df.dropna(subset=["SMILES", "SA"]).reset_index(drop=True)
    
    print(f"Initial molecules: {len(enamine_df)}")
    
    clean_df = remove_protac_linkers(enamine_df, threshold=0.95)
    print(f"After removing PROTAC linkers: {len(clean_df)}")
    
    df_ood, df_id, train_val, cv_splits = create_splits(
        clean_df,
        ood_threshold=args.ood_threshold,
        id_fraction=args.id_fraction,
        target_ood_frac=args.ood_frac,
        k_folds=args.k_folds,
        seed=args.seed
    )
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = save_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    
    df_ood.to_csv(splits_dir / "test_ood.csv", index=False)
    df_id.to_csv(splits_dir / "test_id.csv", index=False)
    train_val.to_csv(splits_dir / "train_val_merged.csv", index=False)
    
    for split in cv_splits:
        fold_dir = splits_dir / f"cv_fold_{split['fold']}"
        fold_dir.mkdir(exist_ok=True)
        split["train"].to_csv(fold_dir / "train.csv", index=False)
        split["valid"].to_csv(fold_dir / "valid.csv", index=False)
    
    manifest = {
        "sizes": {
            "total": len(clean_df),
            "ood_test": len(df_ood),
            "id_test": len(df_id),
            "train_val": len(train_val)
        },
        "cv_folds": args.k_folds,
        "ood_threshold": args.ood_threshold,
        "id_fraction": args.id_fraction,
        "seed": args.seed
    }
    
    with open(splits_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nSplit sizes:")
    print(f"  OOD test: {len(df_ood)}")
    print(f"  ID test: {len(df_id)}")
    print(f"  Train/Val: {len(train_val)}")
    print(f"  CV folds: {args.k_folds}")


def parse_args():
    parser = argparse.ArgumentParser(description="Process linker data with OOD/ID splits and SA stratification")
    parser.add_argument("--data_path", type=str, default="data/enamine_clean.csv")
    parser.add_argument("--save_dir", type=str, default="data/")
    parser.add_argument("--ood_threshold", type=float, default=0.20)
    parser.add_argument("--id_fraction", type=float, default=0.20)
    parser.add_argument("--ood_frac", type=float, default=0.10)
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
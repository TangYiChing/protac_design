#!/usr/bin/env python3
"""
Primary metrics:
1. validity: total number of valid & connected SMILES / total predicted samples
2. synthetic accessibility: average SA scores
3. uniqueness: total number of unique scaffolds / total scaffolds
4. rigidity: total number of rotatable counts

NEW:
5. end-to-end distance (Å): linker end-to-end distance aggregated per UUID
   - extracted from each *_linker_report.json produced by linker_extractor.py
   - valid-only: uses reports where sanitized_ok == True

Also reports:
- linker_e2e_coverage = linker_n_with_e2e / linker_n
"""

import os, sys, json, glob, argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

from linker_extractor import LinkerExtractor, batch_extract_linkers

sys.path.append("evaluation/")
from scoring_func import obey_lipinski, get_flexibility_index, is_valid_smiles, summarize_smiles
from sascorer import calculateScore
from collections import defaultdict


def _read_smiles_files(paths: List[str]) -> List[str]:
    out = []
    for p in paths:
        try:
            with open(p, "r") as f:
                s = f.read().strip()
            if s:
                out.append(s)
        except Exception:
            pass
    return out


def _read_json_files(paths: List[str]) -> List[Dict[str, Any]]:
    out = []
    for p in paths:
        try:
            with open(p, "r") as f:
                d = json.load(f)
            if isinstance(d, dict):
                out.append(d)
        except Exception:
            pass
    return out


def _extract_e2e_from_reports(reports: List[Dict[str, Any]], valid_only: bool = True) -> List[float]:
    """
    Extract linker end-to-end distances from *_linker_report.json dictionaries.

    Expected structure:
      report["compactness"]["end_to_end_dist"]  -> float (Å) or None

    valid_only:
      - if True: only include reports where report["sanitized_ok"] == True
    """
    e2e = []
    for r in reports:
        try:
            if valid_only and (not bool(r.get("sanitized_ok", False))):
                continue
            comp = r.get("compactness", None)
            if not isinstance(comp, dict):
                continue
            val = comp.get("end_to_end_dist", None)
            if val is None:
                continue
            val_f = float(val)
            if np.isfinite(val_f):
                e2e.append(val_f)
        except Exception:
            continue
    return e2e


def _summarize_set(smiles_list: List[str], sa_fn):
    """
    Returns:
      n_total
      valid_rate
      mean_sa (valid-only)
      unique_scaffold_rate (valid-only)
      mean_rotatable (valid-only)
      mean_flexibility (valid-only)
    """
    rows = []
    scaffolds = []
    for smi in smiles_list:
        r = summarize_smiles(smi, sa_fn=sa_fn)
        rows.append(r)
        if r["valid"] and r["scaffold"] != "":
            scaffolds.append(r["scaffold"])

    n_total = len(rows)
    n_valid = sum(int(r["valid"]) for r in rows)
    valid_rate = (n_valid / n_total) if n_total > 0 else 0.0

    sa_vals = [r["sa"] for r in rows if r["valid"] and (r["sa"] is not None)]
    mean_sa = float(np.mean(sa_vals)) if len(sa_vals) > 0 else np.nan

    rot_vals = [r["n_rotatable"] for r in rows if r["valid"]]
    mean_rot = float(np.mean(rot_vals)) if len(rot_vals) > 0 else np.nan

    flex_vals = [r["flexibility"] for r in rows if r["valid"]]
    mean_flex = float(np.mean(flex_vals)) if len(flex_vals) > 0 else np.nan

    uniq_scaf_rate = (len(set(scaffolds)) / len(scaffolds)) if len(scaffolds) > 0 else 0.0

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "valid_rate": valid_rate,
        "mean_sa": mean_sa,
        "unique_scaffold_rate": uniq_scaf_rate,
        "mean_rotatable": mean_rot,
        "mean_flexibility": mean_flex,
    }


def main():
    args = parse_args()

    # 1) linker extraction (writes *_linker_report.json and *_linker.smiles into inference_path/UUID/)
    _ = batch_extract_linkers(
        results_dir=args.inference_path,
        output_dir=args.inference_path,
        n_samples=args.n_samples,
    )

    # 2) load PROTAC smiles from json outputs
    with open(args.json_path, "r") as fin:
        data_dict = json.load(fin)
    row_ids = list(data_dict.keys())

    per_uuid = []

    for rid in row_ids:
        # --- PROTAC smiles from inference json ---
        protac_smiles = list(data_dict[rid].get("predicted_protac_smiles", {}).values())

        # --- Linker smiles (only exists if linker RDKit sanitize succeeds) ---
        linker_smiles_files = glob.glob(f"{args.inference_path}/{rid}/*_linker.smiles")
        linker_smiles = _read_smiles_files(linker_smiles_files)

        # --- Linker reports (exists regardless of sanitize success, unless extraction crashed) ---
        linker_report_files = glob.glob(f"{args.inference_path}/{rid}/*_linker_report.json")
        linker_reports = _read_json_files(linker_report_files)

        protac_stats = _summarize_set(protac_smiles, sa_fn=calculateScore)
        linker_stats = _summarize_set(linker_smiles, sa_fn=calculateScore)

        # --- NEW: per-UUID end-to-end distance stats (valid-only) ---
        e2e_vals = _extract_e2e_from_reports(linker_reports, valid_only=True)
        linker_n_with_e2e = int(len(e2e_vals))
        linker_mean_e2e = float(np.mean(e2e_vals)) if linker_n_with_e2e > 0 else np.nan
        linker_std_e2e = float(np.std(e2e_vals)) if linker_n_with_e2e > 0 else np.nan

        linker_n = int(linker_stats["n_total"])
        linker_e2e_coverage = float(linker_n_with_e2e / linker_n) if linker_n > 0 else 0.0

        per_uuid.append({
            "uuid": rid,

            # PROTAC-level
            "protac_n": protac_stats["n_total"],
            "protac_valid_rate": protac_stats["valid_rate"],
            "protac_mean_sa": protac_stats["mean_sa"],
            "protac_unique_scaffold_rate": protac_stats["unique_scaffold_rate"],
            "protac_mean_rotatable": protac_stats["mean_rotatable"],
            "protac_mean_flexibility": protac_stats["mean_flexibility"],

            # Linker-level
            "linker_n": linker_stats["n_total"],
            "linker_valid_rate": linker_stats["valid_rate"],
            "linker_mean_sa": linker_stats["mean_sa"],
            "linker_unique_scaffold_rate": linker_stats["unique_scaffold_rate"],
            "linker_mean_rotatable": linker_stats["mean_rotatable"],
            "linker_mean_flexibility": linker_stats["mean_flexibility"],

            # NEW: geometry / compactness metric from extraction reports
            "linker_n_with_e2e": linker_n_with_e2e,
            "linker_e2e_coverage": linker_e2e_coverage,
            "linker_mean_end_to_end_dist": linker_mean_e2e,
            "linker_std_end_to_end_dist": linker_std_e2e,
        })

    per_uuid_df = pd.DataFrame(per_uuid)

    # 3) overall aggregate across uuids (macro-average)
    def macro_avg(col: str) -> float:
        x = per_uuid_df[col].to_numpy()
        return float(np.nanmean(x)) if len(x) else np.nan

    overall = pd.DataFrame([{
        "protac_valid_rate": macro_avg("protac_valid_rate"),
        "protac_mean_sa": macro_avg("protac_mean_sa"),
        "protac_unique_scaffold_rate": macro_avg("protac_unique_scaffold_rate"),
        "protac_mean_rotatable": macro_avg("protac_mean_rotatable"),
        "protac_mean_flexibility": macro_avg("protac_mean_flexibility"),

        "linker_valid_rate": macro_avg("linker_valid_rate"),
        "linker_mean_sa": macro_avg("linker_mean_sa"),
        "linker_unique_scaffold_rate": macro_avg("linker_unique_scaffold_rate"),
        "linker_mean_rotatable": macro_avg("linker_mean_rotatable"),
        "linker_mean_flexibility": macro_avg("linker_mean_flexibility"),

        # NEW: macro-average of per-UUID e2e summaries
        "linker_n_with_e2e": macro_avg("linker_n_with_e2e"),
        "linker_e2e_coverage": macro_avg("linker_e2e_coverage"),
        "linker_mean_end_to_end_dist": macro_avg("linker_mean_end_to_end_dist"),
        "linker_std_end_to_end_dist": macro_avg("linker_std_end_to_end_dist"),
    }])

    # 5) valid protac and linker
    record_list = []
    uuid_list = glob.glob(f"{args.inference_path}/row*")
    for uuid in uuid_list:
        protacs = sorted(glob.glob(f"{uuid}/*.smi"))
        linkers = sorted(glob.glob(f"{uuid}/*_linker.smiles"))
        for idx, (protac, linker) in enumerate(zip(protacs, linkers)):
            rid = os.path.basename(os.path.dirname(protac))
            pid = os.path.basename(protac).replace(".smi", ".xyz")
            #print(idx, rid, pid, protac, linker)
            if os.path.exists(protac) and os.path.exists(linker):
                with open(protac) as fin:
                    p_smi = fin.readline().strip()
                with open(linker) as fin:
                    l_smi = fin.readline().strip()
                if is_valid_smiles(p_smi) and is_valid_smiles(l_smi):
                    record = (rid, pid, p_smi, l_smi)
                    record_list.append(record)
    cols = ["uuid", "protac_id", "protac_smiles", "linker_smiles"]
    valid_df = pd.DataFrame(record_list, columns=cols)

    # 4) save
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    per_uuid_path = os.path.join(save_dir, "per_uuid_metrics.csv")
    overall_path = os.path.join(save_dir, "overall_metrics.csv")
    valid_path = os.path.join(save_dir, "valid_smiles.csv")

    per_uuid_df.to_csv(per_uuid_path, index=False)
    overall.to_csv(overall_path, index=False)
    valid_df.to_csv(valid_path,index=False)

    print(f"[OK] Wrote: {per_uuid_path}")
    print(f"[OK] Wrote: {overall_path}")
    print(f"[OK] Wrote: {valid_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate model outputs (validity, SA, uniqueness, rigidity, end-to-end distance)"
    )
    parser.add_argument("--inference_path", required=True,
                        help="Path to model outputs folder (e.g., guided_sampling_svdd_pm/)")
    parser.add_argument("--json_path", required=True,
                        help="Path to model output in json format (e.g., guided_sampling_svdd_pm.results.json)")
    parser.add_argument("--save_dir", default="reports",
                        help="Output folder (default: reports)")
    parser.add_argument("-n", "--n_samples", type=int, default=30,
                        help="Number of prediction samples per molecule (batch mode of linker extraction)")
    return parser.parse_args()


if __name__ == "__main__":
    main()


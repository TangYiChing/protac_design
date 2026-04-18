#!/usr/bin/env python3
"""
Linker Extractor - Enhanced Approach A:
- 3D RMSD Alignment + Atom Mapping (Hungarian + iterative Kabsch refinement)
- Linker extraction by removing fragment-mapped atoms
- Bond reconstruction using covalent radii (single bonds only) + valence guarding
- Sanitization gating + extraction report JSON

This version is designed to improve RDKit-validity and reduce "nonsense" linkers.

Date: 2025
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from rdkit import Chem
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment


@dataclass
class ExtractionConfig:
    # Atom mapping / alignment
    distance_threshold: float = 2.0          # initial assignment distance gate (Å), used loosely
    inlier_threshold: float = 1.2            # post-align inlier gate (Å) for refinement
    max_refine_iters: int = 3                # assignment <-> alignment refinement iters
    min_matches: int = 3                     # minimum matched atoms to attempt alignment
    match_rmsd_threshold: float = 1.5        # fail extraction if alignment rmsd too large

    # Bond building
    covalent_scale: float = 1.25             # scale factor for covalent radii cutoffs
    max_bonds_per_atom: Optional[int] = None # optionally cap neighbors per atom (None disables)

    # IO/debug
    write_unsanitized_sdf: bool = True       # keep SDF for debugging even if sanitize fails
    debug_dir: Optional[str] = None          # where to copy failed samples (xyz + report)


class LinkerExtractor:
    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.cfg = config or ExtractionConfig()
        self.ptable = Chem.GetPeriodicTable()

    # ============================================================
    # STEP 1: Load XYZ files
    # ============================================================
    def load_xyz(self, filepath: str) -> Tuple[np.ndarray, List[str]]:
        """
        Load XYZ file with format:
            N
            comment
            Element x y z
        """
        atoms: List[str] = []
        coords: List[List[float]] = []
        with open(filepath, "r") as f:
            lines = f.readlines()

        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            atom = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append(atom)
            coords.append([x, y, z])

        return np.array(coords, dtype=np.float64), atoms

    # ============================================================
    # STEP 2: Kabsch alignment
    # ============================================================
    def kabsch_alignment(self, mobile: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """
        Align mobile to target. Returns:
          aligned_mobile, rmsd, R, t
        where aligned = (mobile - mobile_center) @ R + target_center
        """
        if mobile.shape != target.shape or mobile.shape[1] != 3:
            raise ValueError("kabsch_alignment expects (N,3) arrays with same shape")

        mobile_center = mobile.mean(axis=0)
        target_center = target.mean(axis=0)

        X = mobile - mobile_center
        Y = target - target_center

        H = X.T @ Y
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # reflection fix
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        aligned = X @ R + target_center
        diff = aligned - target
        rmsd = float(np.sqrt((diff * diff).sum() / mobile.shape[0]))

        # translation is implicit: mobile_center -> target_center with rotation
        t = target_center - mobile_center @ R
        return aligned, rmsd, R, t

    # ============================================================
    # STEP 3: Atom correspondence (Hungarian + iterative refine)
    # ============================================================
    def _hungarian_correspondence(
        self,
        frag_coords: np.ndarray,
        frag_atoms: List[str],
        pred_coords: np.ndarray,
        pred_atoms: List[str],
        relaxed_gate: float,
    ) -> Dict[int, int]:
        """
        Returns mapping frag_idx -> pred_idx based on cost matrix.
        Only same element pairs allowed. Others cost = huge.
        """
        dmat = distance_matrix(frag_coords, pred_coords)
        huge = 1e6
        cost = dmat.copy()

        # element constraint
        for i, a in enumerate(frag_atoms):
            for j, b in enumerate(pred_atoms):
                if a != b:
                    cost[i, j] = huge

        row_ind, col_ind = linear_sum_assignment(cost)

        mapping: Dict[int, int] = {}
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] < relaxed_gate:
                mapping[i] = j
        return mapping

    def find_atom_correspondence(
        self,
        frag_coords: np.ndarray,
        frag_atoms: List[str],
        pred_coords: np.ndarray,
        pred_atoms: List[str],
    ) -> Tuple[Dict[int, int], float, Optional[float]]:
        """
        Robust correspondence:
          1) initial Hungarian match with relaxed gate
          2) refine: Kabsch align pred->frag on matches
          3) re-run Hungarian on aligned coords
          4) inlier filter + repeat a few times

        Returns:
          mapping frag_idx->pred_idx,
          correspondence_ratio,
          final_rmsd (or None if not aligned)
        """
        # Step 1: initial correspondence with relaxed gate
        relaxed_gate = self.cfg.distance_threshold * 5.0
        mapping = self._hungarian_correspondence(
            frag_coords, frag_atoms, pred_coords, pred_atoms, relaxed_gate=relaxed_gate
        )

        if len(mapping) < self.cfg.min_matches:
            ratio = len(mapping) / max(len(frag_atoms), 1)
            return mapping, ratio, None

        aligned_pred_coords = pred_coords.copy()
        final_rmsd: Optional[float] = None

        # iterative refinement
        for _ in range(self.cfg.max_refine_iters):
            frag_idx = np.array(list(mapping.keys()), dtype=int)
            pred_idx = np.array([mapping[i] for i in frag_idx], dtype=int)

            matched_frag = frag_coords[frag_idx]
            matched_pred = aligned_pred_coords[pred_idx]  # align using current pred coords

            # Align pred->frag using matched pairs
            try:
                _, rmsd, R, t = self.kabsch_alignment(matched_pred, matched_frag)
            except Exception:
                break

            # Apply transform to *all* pred coords
            # (p - c)@R + frag_center is equivalent to p@R + t, where t computed above
            aligned_pred_coords = pred_coords @ R + t
            final_rmsd = rmsd

            # Recompute correspondence on aligned coords with a tighter gate
            new_mapping = self._hungarian_correspondence(
                frag_coords, frag_atoms, aligned_pred_coords, pred_atoms,
                relaxed_gate=self.cfg.distance_threshold * 3.0
            )
            if len(new_mapping) < self.cfg.min_matches:
                mapping = new_mapping
                break

            # Inlier filtering on assigned pairs
            frag_idx2 = np.array(list(new_mapping.keys()), dtype=int)
            pred_idx2 = np.array([new_mapping[i] for i in frag_idx2], dtype=int)
            dists = np.linalg.norm(frag_coords[frag_idx2] - aligned_pred_coords[pred_idx2], axis=1)

            inliers = dists < self.cfg.inlier_threshold
            if inliers.sum() >= self.cfg.min_matches:
                # keep only inliers
                mapping = {int(fi): int(pi) for fi, pi, ok in zip(frag_idx2, pred_idx2, inliers) if ok}
            else:
                mapping = new_mapping

            # stop if stable
            if new_mapping == mapping:
                break

        ratio = len(mapping) / max(len(frag_atoms), 1)
        return mapping, ratio, final_rmsd

    # ============================================================
    # STEP 4: Extract linker atoms (pred minus mapped fragment atoms)
    # ============================================================
    def extract_linker_atoms(
        self,
        pred_coords: np.ndarray,
        pred_atoms: List[str],
        mapping: Dict[int, int],
    ) -> Tuple[np.ndarray, List[str], List[int]]:
        mapped_pred_indices = set(mapping.values())
        linker_indices = [i for i in range(len(pred_atoms)) if i not in mapped_pred_indices]
        linker_coords = pred_coords[linker_indices]
        linker_atoms = [pred_atoms[i] for i in linker_indices]
        return linker_coords, linker_atoms, linker_indices

    # ============================================================
    # STEP 5: Build RDKit molecule from atoms+coords (covalent radii)
    # ============================================================
    def _max_valence(self, symbol: str) -> int:
        # conservative caps to reduce overbonding; RDKit will still sanitize-check
        caps = {
            "H": 1, "B": 3, "C": 4, "N": 3, "O": 2, "F": 1,
            "P": 5, "S": 6, "Cl": 1, "Br": 1, "I": 1
        }
        return caps.get(symbol, 4)

    def build_molecule(
        self,
        coords: np.ndarray,
        atoms: List[str],
    ) -> Tuple[Chem.Mol, bool, Optional[str]]:
        """
        Build molecule using covalent radii cutoffs -> SINGLE bonds only.
        Includes a basic valence/neighbors guard to avoid overbonding.

        Returns:
          mol, sanitized_ok, sanitize_error
        """
        mol = Chem.RWMol()
        for a in atoms:
            mol.AddAtom(Chem.Atom(a))

        # Build candidate bonds by distance cutoff based on covalent radii
        n = len(atoms)
        if n >= 2:
            dist = distance_matrix(coords, coords)

            # precompute radii
            radii = np.zeros(n, dtype=np.float64)
            for i, sym in enumerate(atoms):
                z = self.ptable.GetAtomicNumber(sym)
                r = self.ptable.GetRcovalent(z)
                # fallback if missing
                radii[i] = float(r if r > 1e-6 else 0.75)

            # create list of candidate edges with distances
            candidates: List[Tuple[float, int, int]] = []
            for i in range(n):
                for j in range(i + 1, n):
                    cutoff = self.cfg.covalent_scale * (radii[i] + radii[j])
                    if dist[i, j] < cutoff:
                        candidates.append((float(dist[i, j]), i, j))

            # Sort by shortest first (greedy helps reduce overbonding)
            candidates.sort(key=lambda x: x[0])

            # Track current degrees
            degree = [0] * n

            for d, i, j in candidates:
                if self.cfg.max_bonds_per_atom is not None:
                    if degree[i] >= self.cfg.max_bonds_per_atom or degree[j] >= self.cfg.max_bonds_per_atom:
                        continue

                # valence guard
                if degree[i] >= self._max_valence(atoms[i]) or degree[j] >= self._max_valence(atoms[j]):
                    continue

                # add bond if not already
                if mol.GetBondBetweenAtoms(int(i), int(j)) is None:
                    mol.AddBond(int(i), int(j), Chem.BondType.SINGLE)
                    degree[i] += 1
                    degree[j] += 1

        mol = mol.GetMol()

        # Add conformer with coords
        conf = Chem.Conformer(n)
        for i in range(n):
            conf.SetAtomPosition(i, (float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])))
        mol.AddConformer(conf)

        # Try sanitize
        try:
            Chem.SanitizeMol(mol)
            return mol, True, None
        except Exception as e:
            return mol, False, str(e)

    # ===========================================================
    # Helper functions for compactness
    # ===========================================================
    def _radius_of_gyration(self, coords: np.ndarray) -> float:
        if coords.shape[0] == 0:
            return float("nan")
        com = coords.mean(axis=0)
        return float(np.sqrt(((coords - com) ** 2).sum(axis=1).mean()))

    def _pca_anisotropy(self, coords: np.ndarray) -> float:
        # returns lambda1/lambda3 (>=1). If too few points, returns nan.
        if coords.shape[0] < 3:
            return float("nan")
        X = coords - coords.mean(axis=0, keepdims=True)
        C = (X.T @ X) / max(coords.shape[0], 1)
        evals = np.linalg.eigvalsh(C)  # ascending
        if evals[0] <= 1e-12:
            return float("nan")
        return float(evals[-1] / evals[0])

    def _mean_pairwise_distance(self, coords: np.ndarray) -> float:
        if coords.shape[0] < 2:
            return float("nan")
        D = distance_matrix(coords, coords)
        n = coords.shape[0]
        # mean of upper triangle (excluding diagonal)
        return float(D[np.triu_indices(n, k=1)].mean())

    def _build_adjacency_by_covalent_radii(
        self, coords: np.ndarray, atoms: List[str], scale: float
    ) -> np.ndarray:
        n = len(atoms)
        if n == 0:
            return np.zeros((0, 0), dtype=bool)
        dist = distance_matrix(coords, coords)
        radii = np.zeros(n, dtype=np.float64)
        for i, sym in enumerate(atoms):
            z = self.ptable.GetAtomicNumber(sym)
            r = self.ptable.GetRcovalent(z)
            radii[i] = float(r if r > 1e-6 else 0.75)

        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                cutoff = scale * (radii[i] + radii[j])
                if dist[i, j] < cutoff:
                    adj[i, j] = True
                    adj[j, i] = True
        np.fill_diagonal(adj, False)
        return adj

    def _connected_components(self, adj: np.ndarray) -> List[List[int]]:
        n = adj.shape[0]
        seen = np.zeros(n, dtype=bool)
        comps: List[List[int]] = []
        for i in range(n):
            if seen[i]:
                continue
            stack = [i]
            seen[i] = True
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                nbrs = np.where(adj[u])[0]
                for v in nbrs:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
            comps.append(comp)
        return comps

    def _infer_end_to_end_distance(
        self,
        pred_coords: np.ndarray,
        linker_indices: List[int],
        frag_pred_indices: List[int],
        frag_coords: np.ndarray,
        frag_atoms: List[str],
    ) -> Tuple[Optional[float], Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """
        Returns:
        end_to_end_dist (Å) or None,
        (linker_idx_a, linker_idx_b) in *linker-local indexing* (0..n_linker-1) or None,
        (frag_comp_a, frag_comp_b) or None
        """
        # Need linker + fragment
        if len(linker_indices) == 0 or len(frag_pred_indices) == 0:
            return None, None, None

        # Identify fragment components in the *fragment* xyz (more stable than pred)
        adj_frag = self._build_adjacency_by_covalent_radii(
            frag_coords, frag_atoms, scale=self.cfg.covalent_scale
        )
        comps = self._connected_components(adj_frag)

        # If fragment is a single component (rare for your use case), end-to-end not defined
        if len(comps) < 2:
            return None, None, None

        # Work in pred-coordinates for linker, and map frag component atoms into pred indices.
        linker_pos = pred_coords[np.array(linker_indices, dtype=int)]  # [L,3]

        # For each fragment component, find the closest linker atom (by Euclidean distance)
        # We approximate fragment component positions by the mapped fragment atoms in pred coords
        frag_pred_pos = pred_coords[np.array(frag_pred_indices, dtype=int)]  # [F,3]

        # Need mapping from frag-local index -> pred index ordering used in frag_pred_indices.
        # We assume frag_pred_indices is aligned to frag atom indices (see how we build it below).
        # So component indices refer to frag atom indices.
        best = []
        for comp_id, comp_atoms in enumerate(comps):
            comp_atoms = np.array(comp_atoms, dtype=int)
            # gather those frag atoms' positions from pred space
            comp_pos = pred_coords[np.array([frag_pred_indices[a] for a in comp_atoms], dtype=int)]
            # distances: [L, |comp|]
            D = distance_matrix(linker_pos, comp_pos)
            li, _ = np.unravel_index(np.argmin(D), D.shape)
            best.append((comp_id, int(li), float(D.min())))

        # Pick two distinct components: choose the two with smallest linker distance
        best.sort(key=lambda x: x[2])
        comp_a, li_a, _ = best[0]
        # find next different component
        li_b = None
        comp_b = None
        for comp_id, li, _d in best[1:]:
            if comp_id != comp_a:
                comp_b = comp_id
                li_b = li
                break

        if li_b is None:
            return None, None, None

        # end-to-end on linker atoms (linker-local indices)
        d = float(np.linalg.norm(linker_pos[li_a] - linker_pos[li_b]))
        return d, (li_a, li_b), (comp_a, comp_b)


    # ============================================================
    # STEP 6: Save outputs
    # ============================================================
    def save_xyz(self, coords: np.ndarray, atoms: List[str], filepath: str) -> None:
        with open(filepath, "w") as f:
            f.write(f"{len(atoms)}\n")
            f.write("Extracted linker\n")
            for a, c in zip(atoms, coords):
                f.write(f"{a} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")

    def save_sdf(self, mol: Chem.Mol, filepath: str) -> None:
        writer = Chem.SDWriter(filepath)
        writer.write(mol)
        writer.close()

    def save_smiles(self, mol: Chem.Mol, filepath: str) -> str:
        smiles = Chem.MolToSmiles(mol)
        with open(filepath, "w") as f:
            f.write(smiles + "\n")
        return smiles

    # ============================================================
    # MAIN WORKFLOW
    # ============================================================
    def extract_from_files(
        self,
        pred_xyz_path: str,
        frag_xyz_path: str,
        output_dir: str,
        output_prefix: str = "linker",
    ) -> Optional[Dict[str, Any]]:
        """
        Extract linker from predicted xyz and fragment xyz.

        Returns dict with:
          - smiles (only if sanitized_ok)
          - sanitized_ok
          - correspondence_ratio
          - alignment_rmsd
          - n_atoms
          - n_components
          - output_files
          - report_path
        """
        os.makedirs(output_dir, exist_ok=True)

        pred_coords, pred_atoms = self.load_xyz(pred_xyz_path)
        frag_coords, frag_atoms = self.load_xyz(frag_xyz_path)

        mapping, corr_ratio, rmsd = self.find_atom_correspondence(
            frag_coords, frag_atoms, pred_coords, pred_atoms
        )

        expected_linker_atoms = max(len(pred_atoms) - len(frag_atoms), 0)

        # Require reasonable alignment if we had enough matches to align
        if rmsd is not None and rmsd > self.cfg.match_rmsd_threshold:
            # treat as failure
            report = {
                "status": "fail_alignment_rmsd",
                "pred_xyz_path": pred_xyz_path,
                "frag_xyz_path": frag_xyz_path,
                "correspondence_ratio": corr_ratio,
                "alignment_rmsd": rmsd,
                "expected_linker_atoms": expected_linker_atoms,
            }
            report_path = os.path.join(output_dir, f"{output_prefix}_report.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            self._maybe_debug_copy(pred_xyz_path, report_path, output_prefix)
            return None

        linker_coords, linker_atoms, linker_indices = self.extract_linker_atoms(
            pred_coords, pred_atoms, mapping
        )

        # build RDKit mol
        mol, sanitized_ok, sanitize_err = self.build_molecule(linker_coords, linker_atoms)

        # components
        try:
            frags = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
            n_components = len(frags)
        except Exception:
            n_components = None

        # save
        out_files: Dict[str, str] = {}
        xyz_path = os.path.join(output_dir, f"{output_prefix}.xyz")
        self.save_xyz(linker_coords, linker_atoms, xyz_path)
        out_files["xyz"] = xyz_path

        sdf_path = os.path.join(output_dir, f"{output_prefix}.sdf")
        if sanitized_ok or self.cfg.write_unsanitized_sdf:
            try:
                self.save_sdf(mol, sdf_path)
                out_files["sdf"] = sdf_path
            except Exception:
                pass

        smiles: Optional[str] = None
        smiles_path = os.path.join(output_dir, f"{output_prefix}.smiles")
        if sanitized_ok:
            try:
                smiles = self.save_smiles(mol, smiles_path)
                out_files["smiles"] = smiles_path
            except Exception as e:
                sanitized_ok = False
                sanitize_err = f"MolToSmiles failed: {e}"

        # --- Compactness metrics (geometry-only) ---
        compactness = {
            "linker_heavy_atoms": int(len(linker_atoms)),
            "rg": self._radius_of_gyration(linker_coords),
            "anisotropy_l1_l3": self._pca_anisotropy(linker_coords),
            "mean_pairwise_dist": self._mean_pairwise_distance(linker_coords),
            "end_to_end_dist": None,
            "anchor_linker_indices": None,
            "anchor_frag_component_ids": None,
        }

        # Build frag_pred_indices aligned to frag atom order:
        # mapping is frag_idx -> pred_idx, so we need a list where index=frag_idx
        frag_pred_indices = [None] * len(frag_atoms)
        for fi, pi in mapping.items():
            if 0 <= fi < len(frag_pred_indices):
                frag_pred_indices[fi] = int(pi)

        # If some frag atoms failed to map, end-to-end will be less reliable; require full mapping for anchors
        if all(x is not None for x in frag_pred_indices) and len(linker_indices) > 0:
            d_end, anchor_pair, comp_pair = self._infer_end_to_end_distance(
                pred_coords=pred_coords,
                linker_indices=linker_indices,
                frag_pred_indices=frag_pred_indices,   # now frag_idx -> pred_idx
                frag_coords=frag_coords,
                frag_atoms=frag_atoms,
            )
            compactness["end_to_end_dist"] = d_end
            if anchor_pair is not None:
                compactness["anchor_linker_indices"] = [int(anchor_pair[0]), int(anchor_pair[1])]
            if comp_pair is not None:
                compactness["anchor_frag_component_ids"] = [int(comp_pair[0]), int(comp_pair[1])]


        # report
        report = {
            "status": "ok" if sanitized_ok else "fail_sanitize",
            "pred_xyz_path": pred_xyz_path,
            "frag_xyz_path": frag_xyz_path,
            "correspondence_ratio": corr_ratio,
            "alignment_rmsd": rmsd,
            "n_pred_atoms": len(pred_atoms),
            "n_frag_atoms": len(frag_atoms),
            "expected_linker_atoms": expected_linker_atoms,
            "n_linker_atoms": len(linker_atoms),
            "n_components": n_components,
            "sanitized_ok": sanitized_ok,
            "sanitize_error": sanitize_err,
            "output_files": out_files,
            "compactness": compactness,
        }
        report_path = os.path.join(output_dir, f"{output_prefix}_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        if not sanitized_ok:
            self._maybe_debug_copy(pred_xyz_path, report_path, output_prefix)

        return {
            "smiles": smiles,
            "sanitized_ok": sanitized_ok,
            "correspondence_ratio": corr_ratio,
            "alignment_rmsd": rmsd,
            "n_atoms": len(linker_atoms),
            "n_components": n_components,
            "output_files": out_files,
            "report_path": report_path,
        }

    def _maybe_debug_copy(self, pred_xyz_path: str, report_path: str, output_prefix: str) -> None:
        if not self.cfg.debug_dir:
            return
        os.makedirs(self.cfg.debug_dir, exist_ok=True)
        try:
            # copy the pred xyz and report
            import shutil
            shutil.copy2(pred_xyz_path, os.path.join(self.cfg.debug_dir, os.path.basename(pred_xyz_path)))
            shutil.copy2(report_path, os.path.join(self.cfg.debug_dir, os.path.basename(report_path)))
        except Exception:
            pass


def batch_extract_linkers(
    results_dir: str,
    output_dir: str,
    n_samples: int = 10,
    frag_filename: str = "frag_.xyz",
    config: Optional[ExtractionConfig] = None,
) -> Dict[str, Dict[int, Optional[Dict[str, Any]]]]:
    """
    Batch process:
      results_dir/
        UUID/
          frag_.xyz
          0_.xyz, 1_.xyz, ...
    Writes outputs into output_dir/UUID/

    Returns:
      all_results[uuid][sample_idx] = result dict or None
    """
    cfg = config or ExtractionConfig()
    extractor = LinkerExtractor(cfg)

    all_results: Dict[str, Dict[int, Optional[Dict[str, Any]]]] = {}

    for mol_dir in sorted(os.listdir(results_dir)):
        mol_path = os.path.join(results_dir, mol_dir)
        if not os.path.isdir(mol_path):
            continue

        frag_path = os.path.join(mol_path, frag_filename)
        if not os.path.exists(frag_path):
            # skip
            continue

        mol_output_dir = os.path.join(output_dir, mol_dir)
        os.makedirs(mol_output_dir, exist_ok=True)

        mol_results: Dict[int, Optional[Dict[str, Any]]] = {}

        for i in range(n_samples):
            pred_path = os.path.join(mol_path, f"{i}_.xyz")
            if not os.path.exists(pred_path):
                continue

            try:
                result = extractor.extract_from_files(
                    pred_xyz_path=pred_path,
                    frag_xyz_path=frag_path,
                    output_dir=mol_output_dir,
                    output_prefix=f"{i}_linker",
                )
                mol_results[i] = result
            except Exception as e:
                # write minimal failure report
                report_path = os.path.join(mol_output_dir, f"{i}_linker_report.json")
                try:
                    with open(report_path, "w") as f:
                        json.dump(
                            {
                                "status": "exception",
                                "error": str(e),
                                "pred_xyz_path": pred_path,
                                "frag_xyz_path": frag_path,
                            },
                            f,
                            indent=2,
                        )
                except Exception:
                    pass
                mol_results[i] = None

        all_results[mol_dir] = mol_results

        # summary
        smiles_list = [r["smiles"] for r in mol_results.values() if r and r.get("smiles")]
        unique_smiles = set(smiles_list)
        ok_count = sum(1 for r in mol_results.values() if r and r.get("sanitized_ok"))
        print(f"{mol_dir}: extracted={len(mol_results)} sanitized_ok={ok_count} unique_smiles={len(unique_smiles)}")

    return all_results


if __name__ == "__main__":
    print("Enhanced LinkerExtractor ready.")

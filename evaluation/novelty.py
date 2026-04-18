from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Dict, Any, Tuple

import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


@dataclass
class NoveltyConfig:
    fp_radius: int = 2
    fp_nbits: int = 2048
    use_chirality: bool = True
    # If True, canonicalize SMILES to improve dedup + matching consistency
    canonicalize: bool = True


class SmilesNoveltyScorer:
    """
    Novelty = 1 - max_tanimoto(design_fp, db_fps)

    - Higher novelty => farther from the closest database molecule.
    - Also supports:
        * unique SMILES extraction (canonical dedup)
        * returning "novel" subset under a similarity threshold
    """

    def __init__(self, db_smiles: Iterable[str], cfg: NoveltyConfig = NoveltyConfig()):
        self.cfg = cfg
        self.db_smiles_raw = [s for s in db_smiles if isinstance(s, str) and s.strip()]
        self.db_smiles = []
        self.db_fps = []
        self._build_db_index()

    # ---------------------------
    # Core utilities
    # ---------------------------
    def _to_mol(self, smi: str) -> Optional[Chem.Mol]:
        try:
            mol = Chem.MolFromSmiles(smi)
            return mol
        except Exception:
            return None

    def _canon(self, mol: Chem.Mol) -> str:
        # isomericSmiles=True keeps stereochem when possible
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    def _fp(self, mol: Chem.Mol):
        return AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=self.cfg.fp_radius,
            nBits=self.cfg.fp_nbits,
            useChirality=self.cfg.use_chirality,
        )

    def _build_db_index(self) -> None:
        seen = set()
        for smi in self.db_smiles_raw:
            mol = self._to_mol(smi)
            if mol is None:
                continue
            if self.cfg.canonicalize:
                smi_c = self._canon(mol)
            else:
                smi_c = smi.strip()

            if smi_c in seen:
                continue
            seen.add(smi_c)

            self.db_smiles.append(smi_c)
            self.db_fps.append(self._fp(mol))

    # ---------------------------
    # Public API
    # ---------------------------
    def unique_smiles(self, smiles: Iterable[str]) -> List[str]:
        """
        Return unique SMILES (canonicalized if enabled).
        """
        out = []
        seen = set()
        for smi in smiles:
            if not isinstance(smi, str) or not smi.strip():
                continue
            mol = self._to_mol(smi)
            if mol is None:
                continue
            smi2 = self._canon(mol) if self.cfg.canonicalize else smi.strip()
            if smi2 not in seen:
                seen.add(smi2)
                out.append(smi2)
        return out

    def score(self, designs: Iterable[str], return_unique: bool = False) -> pd.DataFrame:
        """
        Score novelty for designs.

        Returns a DataFrame with:
          - smiles_input
          - smiles (canonical if enabled)
          - valid
          - max_sim_to_db
          - novelty (1 - max_sim)
          - closest_db_smiles (best hit)
        """
        designs_list = list(designs)
        if return_unique:
            designs_list = self.unique_smiles(designs_list)

        rows: List[Dict[str, Any]] = []
        for smi_in in designs_list:
            if not isinstance(smi_in, str) or not smi_in.strip():
                rows.append({
                    "smiles_input": smi_in,
                    "smiles": None,
                    "valid": False,
                    "max_sim_to_db": None,
                    "novelty": None,
                    "closest_db_smiles": None,
                })
                continue

            mol = self._to_mol(smi_in)
            if mol is None:
                rows.append({
                    "smiles_input": smi_in,
                    "smiles": None,
                    "valid": False,
                    "max_sim_to_db": None,
                    "novelty": None,
                    "closest_db_smiles": None,
                })
                continue

            smi = self._canon(mol) if self.cfg.canonicalize else smi_in.strip()
            fp = self._fp(mol)

            if len(self.db_fps) == 0:
                # No DB => everything is maximally novel (but say so)
                rows.append({
                    "smiles_input": smi_in,
                    "smiles": smi,
                    "valid": True,
                    "max_sim_to_db": 0.0,
                    "novelty": 1.0,
                    "closest_db_smiles": None,
                })
                continue

            sims = DataStructs.BulkTanimotoSimilarity(fp, self.db_fps)
            max_sim = float(max(sims)) if sims else 0.0
            best_j = int(max(range(len(sims)), key=lambda j: sims[j])) if sims else -1
            closest = self.db_smiles[best_j] if best_j >= 0 else None

            rows.append({
                "smiles_input": smi_in,
                "smiles": smi,
                "valid": True,
                "max_sim_to_db": max_sim,
                "novelty": 1.0 - max_sim,
                "closest_db_smiles": closest,
            })

        return pd.DataFrame(rows)

    def novel_subset(
        self,
        designs: Iterable[str],
        sim_threshold: float = 0.5,
        return_unique: bool = True
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Return (scored_df, novel_smiles_list) where novel means max_sim_to_db < sim_threshold.
        """
        df = self.score(designs, return_unique=return_unique)
        df_valid = df[df["valid"]].copy()
        novel_df = df_valid[df_valid["max_sim_to_db"] < float(sim_threshold)].copy()
        return df, novel_df["smiles"].tolist()


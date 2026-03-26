# quantile_normalizer.py
import numpy as np
import torch
import joblib
from sklearn.preprocessing import QuantileTransformer
from pathlib import Path
import json

class SAQuantileNormalizer:
    def __init__(
        self, 
        n_quantiles=1000,
        output_distribution='uniform',  # 'uniform' or 'normal'
        subsample=100000
    ):
        self.transformer = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            subsample=subsample,
            random_state=42
        )
        self.is_fitted = False
        self.stats = {}
    
    def fit(self, sa_scores):
        """
        fit_transform training data
        
        Args:
            sa_scores: array-like, shape (n_samples,)
        """
        sa_array = np.array(sa_scores).reshape(-1, 1)
        
        print("\n" + "="*70)
        print("🔧 Fitting Quantile Transformer")
        print("="*70)
        
        # Original stat
        print("\n📊 Distribution of original SA scores:")
        print(f"  size: {len(sa_array)}")
        print(f"  range: [{sa_array.min():.3f}, {sa_array.max():.3f}]")
        print(f"  mean: {sa_array.mean():.3f} ± {sa_array.std():.3f}")
        print(f"  median: {np.median(sa_array):.3f}")
        
        # IQRs
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        print("\n  quantiles:")
        for q in quantiles:
            val = np.quantile(sa_array, q)
            print(f"    {int(q*100):>3}%: {val:.3f}")
        
        # Fit
        self.transformer.fit(sa_array)
        self.is_fitted = True
        
        # After normalization
        normalized = self.transformer.transform(sa_array).flatten()
        
        print("\n📊 Distribution of quantile-normalized scores:")
        print(f"  range: [{normalized.min():.4f}, {normalized.max():.4f}]")
        print(f"  mean: {normalized.mean():.4f} ± {normalized.std():.4f}")
        print(f"  median: {np.median(normalized):.4f}")
        
        # Save to file
        self.stats = {
            'n_samples': len(sa_array),
            'original': {
                'min': float(sa_array.min()),
                'max': float(sa_array.max()),
                'mean': float(sa_array.mean()),
                'std': float(sa_array.std()),
                'median': float(np.median(sa_array))
            },
            'normalized': {
                'min': float(normalized.min()),
                'max': float(normalized.max()),
                'mean': float(normalized.mean()),
                'std': float(normalized.std()),
                'median': float(np.median(normalized))
            }
        }
        
        # sanity check
        if normalized.std() < 0.15:
            print("\n WARNING: variantion is stil < 0.15")
            print("     check outliers")
        else:
            print(f"\n healthy variation! ({normalized.std():.4f} >= 0.15)")
        
        print("\nQuantile Transformer fitted successfully!")
        print("="*70)
        
        return self
    
    def transform(self, sa_scores):
        """
        Transform SA score in [0, 1]
        
        Args:
            sa_scores: array-like or torch.Tensor
            
        Returns:
            normalized scores in [0, 1]
        """
        if not self.is_fitted:
            raise RuntimeError("Transformer not fitted! Call .fit() first.")
        
        # handle different input types
        is_tensor = torch.is_tensor(sa_scores)
        if is_tensor:
            device = sa_scores.device
            sa_scores = sa_scores.cpu().numpy()
        
        sa_array = np.array(sa_scores).reshape(-1, 1)
        normalized = self.transformer.transform(sa_array).flatten()
        
        if is_tensor:
            normalized = torch.tensor(normalized, dtype=torch.float32).to(device)
        
        return normalized
    
    def inverse_transform(self, normalized_scores):
        """
        from normalized to original SA score
        """
        if not self.is_fitted:
            raise RuntimeError("Transformer not fitted!")
        
        is_tensor = torch.is_tensor(normalized_scores)
        if is_tensor:
            device = normalized_scores.device
            normalized_scores = normalized_scores.cpu().numpy()
        
        norm_array = np.array(normalized_scores).reshape(-1, 1)
        original = self.transformer.inverse_transform(norm_array).flatten()
        
        if is_tensor:
            original = torch.tensor(original, dtype=torch.float32).to(device)
        
        return original
    
    def save(self, save_dir='./checkpoints'):
        """Save to file"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # save transformer
        transformer_path = save_dir / 'quantile_transformer.pkl'
        joblib.dump(self.transformer, transformer_path)
        
        # save stats
        stats_path = save_dir / 'normalization_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"\n Saved:")
        print(f"  Transformer: {transformer_path}")
        print(f"  Statistics: {stats_path}")
    
    @classmethod
    def load(cls, save_dir='./checkpoints'):
        """load transformer"""
        save_dir = Path(save_dir)
        
        transformer_path = save_dir / 'quantile_transformer.pkl'
        stats_path = save_dir / 'normalization_stats.json'
        
        if not transformer_path.exists():
            raise FileNotFoundError(f"Transformer not found at {transformer_path}")
        
        # initiate an instance
        normalizer = cls()
        normalizer.transformer = joblib.load(transformer_path)
        normalizer.is_fitted = True
        
        # load stats
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                normalizer.stats = json.load(f)
        
        print(f"\n Loaded:")
        print(f"  Transformer: {transformer_path}")
        if normalizer.stats:
            print(f"  Original SA range: [{normalizer.stats['original']['min']:.2f}, "
                  f"{normalizer.stats['original']['max']:.2f}]")
            print(f"  Normalized std: {normalizer.stats['normalized']['std']:.4f}")
        
        return normalizer

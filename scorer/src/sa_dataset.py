import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import numpy as np
from quantile_normalizer import SAQuantileNormalizer

class SADatasetWithQuantile(Dataset):
    def __init__(
        self, 
        data_list,           # List of PyG Data objects
        normalizer=None,     # SAQuantileNormalizer instance
        split='train'        # 'train', 'val', or 'test'
    ):
        super().__init__()
        self.data_list = data_list
        self.split = split
        
        print(f"\n{'='*70}")
        print(f"Initializing {split.upper()} Dataset")
        print(f"{'='*70}")
        print(f"Number of samples: {len(data_list)}")
        
        # Handle empty dataset
        if len(data_list) == 0:
            print("WARNING: Empty dataset detected. Skipping normalization.")
            self.normalizer = normalizer
            print(f"{'='*70}\n")
            return
        
        # original SA values
        original_sa = np.array([data.y.item() for data in data_list])
        print(f"Original SA values:")
        print(f"    Range: [{original_sa.min():.3f}, {original_sa.max():.3f}]")
        print(f"    Mean: {original_sa.mean():.3f} ± {original_sa.std():.3f}")
        
        # Quantile normalization
        if split == 'train':
            # fit a normalizer
            if normalizer is None:
                print("Creating new Quantile Normalizer...")
                self.normalizer = SAQuantileNormalizer(
                    n_quantiles=min(len(data_list), 1000)
                )
                self.normalizer.fit(original_sa)
            else:
                print("Using provided Quantile Normalizer...")
                self.normalizer = normalizer
        else:
            # must use train data's normalizer
            if normalizer is None:
                raise ValueError(
                    f"{split} set requires a fitted normalizer from training set!"
                )
            print(f"Using training set's Quantile Normalizer...")
            self.normalizer = normalizer
        
        # Transform SA values
        normalized_sa = self.normalizer.transform(original_sa)
        
        # Update labe y value for better training
        for i, data in enumerate(self.data_list):
            data.y = torch.tensor([normalized_sa[i]], dtype=torch.float32)
        
        # summary
        print(f"Quantile normalization!")
        print(f"  Range: [{normalized_sa.min():.4f}, {normalized_sa.max():.4f}]")
        print(f"  Mean: {normalized_sa.mean():.4f} ± {normalized_sa.std():.4f}")
        
        if normalized_sa.std() < 0.15:
            print(f"WARNNING: Low variation: ({normalized_sa.std():.4f})")
        else:
            print(f"Pass y variable value variation checking({normalized_sa.std():.4f})")
        
        print(f"{'='*70}\n")
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]
    
    def get_normalizer(self):
        """Return normalizer"""
        return self.normalizer


def create_dataloaders(
    train_data, 
    val_data, 
    test_data,
    batch_size=32,
    num_workers=4,
    normalizer=None
):
    """
    Args:
        train_data: List of PyG Data for training
        val_data: List of PyG Data for validation  
        test_data: List of PyG Data for testing
        batch_size: Batch size
        num_workers: Number of workers for DataLoader
        normalizer: Optional pre-fitted normalizer
    
    Returns:
        train_loader, val_loader, test_loader, normalizer
    """
    
    print("Creating DataLoaders with Quantile Normalization")
    print("="*70)
    
    # for training set
    train_dataset = SADatasetWithQuantile(
        train_data, 
        normalizer=normalizer,
        split='train'
    )
    
    # normalizer
    normalizer = train_dataset.get_normalizer()
    
    # validation set（required normalizer from train set）
    val_dataset = SADatasetWithQuantile(
        val_data,
        normalizer=normalizer,
        split='val'
    )
    
    test_dataset = SADatasetWithQuantile(
        test_data,
        normalizer=normalizer,
        split='test'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print("All DataLoaders created successfully!")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print("="*70 + "\n")
    
    return train_loader, val_loader, test_loader, normalizer

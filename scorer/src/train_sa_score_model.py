import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import numpy as np
from pathlib import Path
from tqdm import tqdm
import wandb 
from datetime import datetime
import torch.nn.functional as F

from featurize_linker_data import LINKER
from sa_dataset import create_dataloaders
from quantile_normalizer import SAQuantileNormalizer

from scipy.stats import spearmanr

import sys 
sys.path.append("models/") 
from LinkerScorer import SAScorer

config = {
    'atom_feat_dim': 9,
    'hidden_dim': 256,
    'num_layers': 5,      
    'dropout': 0.1,         
    'batch_size': 32,
    'num_epochs': 100,
    'learning_rate': 5e-4, 
    'device': 'cuda:6' if torch.cuda.is_available() else 'cpu',
    'num_folds': 5,
    'pool': 'mean',
    'radius': 2.0
}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_data(pt_path):
    """Load data from .pt file"""
    pt_p = Path(pt_path)
    if not pt_p.exists():
        raise FileNotFoundError(
            f"Missing precomputed dataset '{pt_p}'. Please run: python glue/scorer/src/featurize_training_data.py"
        )
    obj = torch.load(pt_p, map_location='cpu', weights_only=False)
    if isinstance(obj, dict) and 'data_list' in obj:
        return obj['data_list']
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unsupported dataset format in '{pt_p}'")

def load_merged_train_val_data(csv_path: str):
    csv_p = Path(csv_path)
    pt_p = csv_p.with_suffix(".pt")

    if pt_p.exists():
        print(f"Loading pre-featurized dataset from {pt_p}")
        obj = torch.load(pt_p, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and 'data_list' in obj:
            data_list = obj['data_list']
        elif isinstance(obj, list):
            data_list = obj
        else:
            raise ValueError(f"Unexpected data format in {pt_p}")
        print(f"Loaded {len(data_list)} molecules from cache")
        return data_list

    print(f"Featurized .pt not found. Creating from CSV: {csv_p}")
    data_list = LINKER(str(csv_p)).create_data()
    print(f"Featurized {len(data_list)} molecules")

    torch.save({'data_list': data_list, 'csv_path': str(csv_p)}, pt_p)
    print(f"Saved featurized dataset to {pt_p}")

    return data_list
 
def train_single_fold(fold_num, device):
    print(f"\n{'='*80}")
    print(f"TRAINING FOLD {fold_num}")
    print(f"{'='*80}")    

    train_path = f"data/splits/cv_fold_{fold_num}/train.pt"
    val_path = f"data/splits/cv_fold_{fold_num}/valid.pt"
    test_path = "data/splits/test_id.pt"

    train_data_list = load_data(train_path)
    val_data_list = load_data(val_path)
    test_data_list = load_data(test_path)

    print(f"Data loaded - Train: {len(train_data_list)}, Val: {len(val_data_list)}, Test: {len(test_data_list)}")

    model = SAScorer(
        atom_feat_dim=config['atom_feat_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        radius = config['radius'],
        build_edges='radius',
        pool=config['pool']
        )

    model = model.to(device)

    train_loader, val_loader, test_loader, normalizer = create_dataloaders(
    train_data=train_data_list,
    val_data=val_data_list,
    test_data=test_data_list,
    batch_size=config['batch_size'],
    num_workers=0
)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = config['learning_rate'],
        weight_decay = 1e-5
    )

    def run(split, loader, return_preds = False):
        model.train(split == 'train')
        total_loss = 0.0
        total_n = 0
        all_pred, all_y = [], []
        
        for batch in loader:
            batch = batch.to(device) 

            pred = model(batch)
            y = batch.y.view(-1).to(pred.dtype)
            loss = F.mse_loss(pred, y)
            
            if split == 'train':
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            total_loss += loss.item() * batch.num_graphs
            total_n += batch.num_graphs
            all_pred.append(pred.detach().cpu())
            all_y.append(y.detach().cpu())
        
        all_pred = torch.cat(all_pred) if len(all_pred) > 0 else torch.tensor([])
        all_y = torch.cat(all_y) if len(all_y) > 0 else torch.tensor([])

        mae = F.l1_loss(all_pred, all_y).item()

        try:
            from scipy.stats import spearmanr
            sp = spearmanr(all_pred.numpy(), all_y.numpy()).correlation
        except Exception:
            sp = float('nan')

        # add spearman correlation for original SA score
        if normalizer is not None:
            pred_raw = normalizer.inverse_transform(all_pred.numpy())
            y_raw = normalizer.inverse_transform(all_y.numpy())
            sp_raw = spearmanr(pred_raw, y_raw).correlation
        else:
            sp_raw = float("nan")

        avg_loss = total_loss / max(total_n, 1)

        if return_preds:
            return avg_loss, mae, sp, sp_raw, all_pred.numpy(), all_y.numpy()
        
        return avg_loss, mae, sp, sp_raw

    # Training loop
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    patience = 25
    patience_counter = 0

    train_loss = []
    train_loss_hist, val_loss_hist, val_mae_hist = [], [], []

    for epoch in range(1, config['num_epochs'] + 1):
        tr_loss, tr_mae, tr_sp, tr_sp_raw = run('train', train_loader)
        va_loss, va_mae, va_sp, va_sp_raw = run('val', val_loader)
        train_loss.append(tr_loss)

        train_loss_hist.append(tr_loss)
        val_loss_hist.append(va_loss)
        val_mae_hist.append(va_mae)

        # if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Train MSE: {tr_loss:.4f} MAE: {tr_mae:.4f} | Val MSE: {va_loss:.4f} MAE: {va_mae:.4f}")

        if va_mae < best_val_mae:
            best_val_loss = va_loss
            best_val_mae = va_mae
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Final evaluation
    va_loss, va_mae, va_sp, va_sp_raw, val_pred, val_true = run('val',  val_loader,  return_preds=True)
    te_loss, te_mae, te_sp, te_sp_raw, test_pred, test_true = run('test', test_loader, return_preds=True)

    print(f"Fold {fold_num} Results:")
    print(f"Val MSE: {va_loss:.4f} MAE: {va_mae:.4f} SP: {va_sp:.3f} SP (original SA): {va_sp_raw:.3f}")
    print(f"Test MSE: {te_loss:.4f} MAE: {te_mae:.4f} SP: {te_sp:.3f} SP (original SA): {te_sp_raw:.3f}")
    
    return {
    'fold': fold_num,
    'val_loss': va_loss,
    'val_mae': va_mae,
    'val_spearman': va_sp,
    'val_spearman_raw_sa': va_sp_raw,
    'test_loss': te_loss,
    'test_mae': te_mae,
    'test_spearman': te_sp,
    'test_spearman_raw_sa': te_sp_raw,
    'train_loss_history': train_loss_hist,
    'val_loss_history': val_loss_hist,
    'val_mae_history': val_mae_hist,
    'val_predictions': val_pred, 
    'val_targets': val_true,
    'test_predictions': test_pred,
    'test_targets': test_true,
}


def train_on_merged_data(csv_path, device, test_pt_path, checkpoint_name='best_model'):
    """
    Train model on all merged train_val data and evaluate on test set.
    
    Args:
        csv_path: Path to train_val_merged.csv
        device: Device to train on
        test_pt_path: Path to test_id.pt
        checkpoint_name: Name for the checkpoint file (without .pt extension)
    
    Returns:
        dict with training results and model info
    """
    print(f"\n{'='*80}")
    print("TRAINING ON ALL MERGED DATA")
    print(f"{'='*80}")
    
    # Load merged train_val data
    print(f"\nLoading training data from: {csv_path}")
    train_data_list = load_merged_train_val_data(csv_path)
    print(f"Loaded {len(train_data_list)} molecules from merged train_val data")
    
    # Load test data from .pt file
    print(f"\nLoading test data for final evaluation: {test_pt_path}")
    test_data_list = load_data(test_pt_path)
    print(f"Loaded {len(test_data_list)} molecules from test data")
    
    # Initialize model
    model = SAScorer(
        atom_feat_dim=config['atom_feat_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        radius=config['radius'],
        build_edges='radius',
        pool=config['pool']
    )
    model = model.to(device)
    
    train_loader, val_loader, test_loader, normalizer = create_dataloaders(
        train_data=train_data_list,
        val_data=[],
        test_data=test_data_list,
        batch_size=config['batch_size'],
        num_workers=0
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=1e-6
    )
    
    def run(split, loader, return_preds=False):
        model.train(split == 'train')
        total_loss = 0.0
        total_n = 0
        all_pred, all_y = [], []
        
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            y = batch.y.view(-1).to(pred.dtype)
            loss = F.mse_loss(pred, y)
            
            if split == 'train':
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            total_n += batch.num_graphs
            all_pred.append(pred.detach().cpu())
            all_y.append(y.detach().cpu())
        
        all_pred = torch.cat(all_pred) if len(all_pred) > 0 else torch.tensor([])
        all_y = torch.cat(all_y) if len(all_y) > 0 else torch.tensor([])

        # Handle empty predictions
        if len(all_pred) == 0 or len(all_y) == 0:
            avg_loss = total_loss / max(total_n, 1)
            if return_preds:
                return avg_loss, float('inf'), float('nan'), all_pred.numpy() if len(all_pred) > 0 else np.array([]), all_y.numpy() if len(all_y) > 0 else np.array([])
            return avg_loss, float('inf'), float('nan')
        
        mae = F.l1_loss(all_pred, all_y).item()
        
        try:
            from scipy.stats import spearmanr
            sp = spearmanr(all_pred.numpy(), all_y.numpy()).correlation
        except Exception:
            sp = float('nan')

         # add spearman correlation for original SA score
        if normalizer is not None:
            pred_raw = normalizer.inverse_transform(all_pred.numpy())
            y_raw = normalizer.inverse_transform(all_y.numpy())
            sp_raw = spearmanr(pred_raw, y_raw).correlation
        else:
            sp_raw = float("nan")
        
        avg_loss = total_loss / max(total_n, 1)
        
        if return_preds:
            return avg_loss, mae, sp, sp_raw, all_pred.numpy(), all_y.numpy()
        
        return avg_loss, mae, sp, sp_raw
    
    best_train_loss = float('inf')
    train_loss_hist = []
    
    # Create checkpoints directory
    checkpoint_dir = Path('./checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    print("\nStarting training...")
    
    for epoch in range(1, config['num_epochs'] + 1):
        tr_loss, tr_mae, tr_sp, tr_sp_raw = run('train', train_loader)
        
        train_loss_hist.append(tr_loss)
        
        print(f"Epoch {epoch:03d} | Train MSE: {tr_loss:.4f} MAE: {tr_mae:.4f}")
        
        if tr_loss < best_train_loss:
            best_train_loss = tr_loss
            best_epoch = epoch
            
            # Save checkpoint based on training loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'normalizer': normalizer,
                'config': config,
                'train_loss': tr_loss,
                'train_mae': tr_mae,
                'train_loss_hist': train_loss_hist
            }
            
            checkpoint_path = checkpoint_dir / f'{checkpoint_name}.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE - Best model at epoch {best_epoch}")
    print(f"{'='*80}")
    print(f"Best Training Loss: {best_train_loss:.4f}")
    print(f"Checkpoint saved: {checkpoint_dir / f'{checkpoint_name}.pt'}")
    print(f"{'='*80}\n")
    
    # Final evaluation on test set
    print("Evaluating on test set...")
    test_loss, test_mae, test_sp, test_sp_raw, test_pred, test_true = run('test', test_loader, return_preds=True)
    
    print(f"\nTest Set Results:")
    print(f"MSE: {test_loss:.4f}")
    print(f"MAE: {test_mae:.4f}")
    print(f"Spearman: {test_sp:.3f}")
    print(f"Spearman: {test_sp_raw:.3f}")
    
    results = {
        'best_epoch': best_epoch,
        'best_train_loss': best_train_loss,
        'test_loss': test_loss,
        'test_mae': test_mae,
        'test_spearman': test_sp,
        'test_spearman_raw_sa': test_sp_raw,
        'train_loss_history': train_loss_hist,
        'test_predictions': test_pred,
        'test_targets': test_true
    }
    
    return results

def create_cv_visualizations(fold_results, config):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score

    os.makedirs('./outputs/train_results', exist_ok=True)

    num_folds = len(fold_results)
    colors = plt.cm.tab10(np.linspace(0, 1, num_folds))
    
    # Calculate metrics across all folds
    val_maes = [r['val_mae'] for r in fold_results]

    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # (A) Training / Validation loss curves (all folds)
    axA = fig.add_subplot(gs[0, 0])
    for i, r in enumerate(fold_results):
        epochs = np.arange(1, len(r['train_loss_history']) + 1)
        axA.plot(epochs, r['train_loss_history'], color=colors[i], alpha=0.6, label=f"Fold {r['fold']} Train")
        axA.plot(epochs, r['val_loss_history'],   color=colors[i], lw=2)
    axA.set_title('Training and Validation Loss', fontsize=13)
    axA.set_xlabel('Epoch'); axA.set_ylabel('MSE')
    axA.grid(True, alpha=0.3)
    axA.legend(ncol=2, fontsize=9)

    # (B) Predictions vs Truth (Test, all folds combined)
    axB = fig.add_subplot(gs[0, 1])
    all_t = np.concatenate([r['test_targets'] for r in fold_results])
    all_p = np.concatenate([r['test_predictions'] for r in fold_results])
    axB.scatter(all_t, all_p, s=8, alpha=0.25)
    axB.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect prediction')
    axB.set_title('Test Set: Predictions vs Truth', fontsize=13)
    axB.set_xlabel('True Normalized SA'); axB.set_ylabel('Predicted Normalized SA')
    axB.grid(True, alpha=0.3); axB.legend()
    axB.text(0.04, 0.95, f"R² = {r2_score(all_t, all_p):.3f}\nMAE = {np.mean(np.abs(all_p-all_t)):.3f}",
             transform=axB.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
             va='top', fontsize=10)

    # (C) Distribution comparison (Validation, all folds)
    axC = fig.add_subplot(gs[1, 0])
    v_t = np.concatenate([r['val_targets'] for r in fold_results])
    v_p = np.concatenate([r['val_predictions'] for r in fold_results])
    axC.hist(v_t, bins=40, density=True, alpha=0.5, label='True')
    axC.hist(v_p, bins=40, density=True, alpha=0.5, label='Pred')
    axC.set_title('Validation: Distribution Comparison', fontsize=13)
    axC.set_xlabel('Normalized SA'); axC.set_ylabel('Density')
    axC.grid(True, alpha=0.3); axC.legend()

    # (D) Residual plot (Test, all folds)
    axD = fig.add_subplot(gs[1, 1])
    resid = all_p - all_t
    axD.scatter(all_p, resid, s=8, alpha=0.25)
    axD.axhline(0.0, color='r', ls='--', lw=2)
    axD.set_title('Residual Plot (Test)', fontsize=13)
    axD.set_xlabel('Predicted SA'); axD.set_ylabel('Pred - True')
    axD.grid(True, alpha=0.3)

    fig.suptitle(f"SA Scorer Cross-Validation Results ({num_folds} folds; "
                 f"mean Val MAE={np.mean(val_maes):.3f}±{np.std(val_maes):.3f})",
                 fontsize=15, y=0.995, fontweight='bold')

    out = './outputs/cv_results.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Cross-validation visualization saved to '{out}'")

def main():
    parser = argparse.ArgumentParser(description='Train SA Scorer with 5-Fold CV or on merged data')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--train_merged', action='store_true', help='Train on merged train_val data instead of CV')
    parser.add_argument('--merged_csv', type=str, default='data/splits/train_val_merged.csv', help='Path to merged CSV file')
    parser.add_argument('--test_pt', type=str, default='data/splits/test_id.pt', help='Path to test_id.pt file')
    parser.add_argument('--checkpoint_name', type=str, default='best_model', help='Name for checkpoint file (without .pt extension)')
    args = parser.parse_args()
    
    device = "cpu" if args.gpu == -1 else f"cuda:{args.gpu}"
    set_seed(args.seed)    
    
    start_time = datetime.now()

    print(f"Device: {device}")
    print(f"Model parameters: {config}")

    if args.train_merged:
        # Train on merged data
        print(f"\nTraining on merged data: {args.merged_csv}")
        print(f"Checkpoint will be saved as: {args.checkpoint_name}.pt")
        result = train_on_merged_data(
            csv_path=args.merged_csv,
            device=device,
            test_pt_path=args.test_pt,
            checkpoint_name=args.checkpoint_name
        )
        
        end_time = datetime.now()
        print("\n" + "="*80)
        print("TRAINING COMPLETED!")
        print("="*80)
        print(f"Total time: {end_time - start_time}")
        print(f"Test MAE: {result['test_mae']:.4f}")
        print(f"Test MSE: {result['test_loss']:.4f}")
        print(f"Test Spearman: {result['test_spearman']:.3f}")
        print(f"Test Spearman (original SA): {result['test_spearman_raw_sa']:.3f}")
        print("="*80)
    else:
        # Cross-validation mode
        fold_results = []
        print(f"Starting 5-Fold Cross-Validation")

        # Train on all folds
        for fold in range(1, config['num_folds'] + 1):
            try:
                result = train_single_fold(fold, device)
                fold_results.append(result)
            except Exception as e:
                print(f"\nError training fold {fold}: {e}")
                continue
        
        if len(fold_results) == 0:
            print("\nNo folds completed successfully!")
            return
        
        # Print summary
        print("\n" + "="*80)
        print("CROSS-VALIDATION SUMMARY")
        print("="*80)
        
        val_maes = [r['val_mae'] for r in fold_results]
        val_losses = [r['val_loss'] for r in fold_results]
        test_losses = [r['test_loss'] for r in fold_results]
        
        print(f"\nCompleted {len(fold_results)} out of {config['num_folds']} folds\n")
        
        print("="*60)
        print("VALIDATION METRICS (Best Checkpoint)")
        print("="*60)
        print(f"MAE: {np.mean(val_maes):.4f} ± {np.std(val_maes):.4f}")
        print(f"Range: [{np.min(val_maes):.4f}, {np.max(val_maes):.4f}]")
        print(f"Loss:{np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")

        print("\n" + "="*60)
        print("PER-FOLD RESULTS")
        print("="*60)
        print(f"{'Fold':<6} {'Val MAE':<10} {'Val Loss':<10} {'Test MAE':<10} {'Test Loss':<10}")
        print("-" * 60)
        for r in fold_results:
            print(f"{r['fold']:<6} {r['val_mae']:<10.4f} {r['val_loss']:<10.4f} {r['test_mae']:<10.4f} {r['test_loss']:<10.4f}")
        
        # Create visualizations
        print("\n" + "="*80)
        print("Creating visualizations...")
        print("="*80)
        create_cv_visualizations(fold_results, config)
        
        end_time = datetime.now()
        print("\n" + "="*80)
        print("ALL FOLDS COMPLETED!")
        print("="*80)
        print(f"Total time: {end_time - start_time}")
        print(f"\nFinal Results:")
        print(f"  Validation MAE: {np.mean(val_maes):.4f} ± {np.std(val_maes):.4f}")
        print("="*80)


if __name__ == '__main__':
    main()
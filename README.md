# Official Repository for SVDD-PROTAC

## Installation
This repo mainly depends on
- Python 3.9
- PyTorch
- PyTorch Geometric
- RDKit
- Open Babel
- NumPy / SciPy / pandas / scikit-learn
- NetworkX
- wandb

```bash
conda env create -f env.yaml
conda activate svdd-protac
```
## Quick start
Obtain two checkpoints:
a. The base Diffusion generator: https://github.com/Fenglei104/DiffPROTACs/blob/main/checkpoints/protacs_best.ckpt, 
b. The Linker SA scorer is available at scorer/checkpoints/best_ckpt.pt

### 1. Guided inference
```bash
python make_inference_w_guidance.py \
  --test path/to/test.json \
  --generator path/to/generator.ckpt \
  --scorer path/to/scorer.pt \
  --normalizer_path path/to/quantile_transformer.pkl \
  --mode svdd_pm \
  --sample_M 10 \
  --samples 10 \
  --save_dir outputs/inference
```
### 2. Evaluate outputs
```bash
python evaluate.py \
  --inference_path outputs/inference/guided_sampling_svdd_pm \
  --json_path outputs/inference/guided_sampling_svdd_pm.results.json \
  --save_dir outputs/reports
```
### Example run: SVDD-Both
```bash
python make_inference_w_guidance.py \
  --test database/independent_PROTACpedia_linker.json \
  --generator your-path-to-base-model/protacs_best.ckpt \
  --scorer scorer/checkpoints/best_ckpt.pt \
  --mode svdd_pm \
  --samples 30 \
  --temp 0.5 \
  --branch_start 0 \
  --branch_end 120 \
  --w_rigid 0.5 \ # SVDD-SA: 0.0, SVDD-Geom: 1.0
  --no_wandb
  --save_dir outputs/inference
```


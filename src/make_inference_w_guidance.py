import os, sys, json, glob, random, argparse, wandb, torch, copy
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader

from featurize_protac import PROTAC
from dataset import PROTACDataset, collate
from trainer import Trainer
from mol_converter import OpenBabelConverter


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir) 
diffusion_models_dir = os.path.join(parent_dir, 'models')
scorer_src_dir = os.path.join(parent_dir, 'scorer', 'src')
scorer_models_dir = os.path.join(parent_dir, 'scorer', 'models')

sys.path.insert(0, scorer_src_dir)
sys.path.insert(0, scorer_models_dir)
sys.path.insert(0, diffusion_models_dir)
print(f"diffusion_models_dir:{diffusion_models_dir}")
print(f"scorer_src_dir:{scorer_src_dir}")
print(f"scorer_models_dir:{scorer_models_dir}")

from GuidedDiffPROTACs import EDM
from LinkerScorer import SAScorer

GENERATOR_PARAMS = {
    'epochs': 10,
    'batch_size': 4, #64,
    'num_workers': 4,
    'lr': 2e-4,
    'in_node_nf': 9,
    'hidden_nf': 128,
    'diffusion_loss_type': 'l2',
    'noise_schedule': 'polynomial_2',
    'noise_precision': 1e-5,
    'timesteps': 500,
    'n_layers': 6,
    'ffn_embedding_dim': 1024,
    'attention_heads': 32,
    'tanh': False,
    'coords_range': 10.,
    'dropout': 0.05,
    'activation_dropout': 0.05,
    'log_dir': 'logs',
    'norm_values': [1,4,10],
    'norm_biases': [None, 0., 0.],
    'sampling_size': 10 # get 10 candidates from each item
}


SCORER_PARAMS = {
    'atom_feat_dim': 9,
    'hidden_dim': 256,
    'num_layers': 5,
    'dropout': 0.1,
    'use_distance': True,
    'rbf_k': 32,
    'build_edges': "radius",  # "radius" | "knn" | "none"
    'radius': 2.0,
    'k': None,
    'pool':  "mean", # "mean" | "add" | "max"
    'lr': 2e-4,
    'batch_size': 32,
    'epochs': 200
}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def convert_xyz(input_json, output_path):
    """Collect predicted SMILES written during inference.
    """
    with open(input_json, "r") as f:
        items = json.load(f) # will append predicted_protac_smiles

    # Convert frag_.xyz to frag_.sdf, true_.xyz to true_.sdf
    row_ids = items.keys()
    for row_id in row_ids:
        frag = f"{output_path}/{row_id}/frag_.xyz"
        true = f"{output_path}/{row_id}/true_.xyz"
        for xyz in [frag, true]:
            # xyz to sdf
            converter = OpenBabelConverter()
            converter.batch_convert(
                input_dir=f"{output_path}/{row_id}/", 
                output_dir=f"{output_path}/{row_id}/",
                input_format="xyz",
                output_format="sdf")
    
    # Output items
    fout_dict = copy.deepcopy(items) # {row_id:{}}

    # Retrieve model predictions
    fails = []
    for row_id in row_ids:
        pred_xyz = sorted(glob.glob(f"{output_path}/{row_id}/*.xyz"))
        if len(pred_xyz) > 2: # in addition to true_.xyz and frag_.xyz
            smi_files = sorted(glob.glob(os.path.join(output_path, row_id, "*.smi")))
            smi_dict = {}
            for smi_path in smi_files:
                base = os.path.basename(smi_path)      # e.g., "0.smi"
                stem = os.path.splitext(base)[0]       # "0"
                key = f"{stem}.xyz"                   # backward-compatible key
                with open(smi_path, "r") as f:
                    smi = f.readline().strip()
                if smi:
                    smi_dict[key] = smi
            fout_dict[row_id]["predicted_protac_smiles"] = smi_dict
        else:
            fails.append(row_id)
            print(f"WARNNING!!! {row_id} has no predictions")

    return fout_dict


def make_inference(args):
    # set device
    device = "cpu" if args.gpu == -1 else f"cuda:{args.gpu}"

    # set seed for reproducibility
    set_seed(args.seed)

    # create output folder
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # load generative model
    generator = EDM(
        in_node_nf = GENERATOR_PARAMS['in_node_nf'],
        hidden_nf = GENERATOR_PARAMS['hidden_nf'],
        timesteps = GENERATOR_PARAMS['timesteps'],
        noise_schedule = GENERATOR_PARAMS['noise_schedule'],
        noise_precision = GENERATOR_PARAMS['noise_precision'],
        loss_type = GENERATOR_PARAMS['diffusion_loss_type'],
        norm_values = GENERATOR_PARAMS['norm_values'],
        norm_biases = GENERATOR_PARAMS['norm_biases'],
        ffn_embedding_dim = GENERATOR_PARAMS['ffn_embedding_dim'],
        attention_heads = GENERATOR_PARAMS['attention_heads'],
        tanh = GENERATOR_PARAMS['tanh'],
        n_layers = GENERATOR_PARAMS['n_layers'],
        coords_range = GENERATOR_PARAMS['coords_range'],
        dropout = GENERATOR_PARAMS['dropout'],
        activation_dropout = GENERATOR_PARAMS['activation_dropout'],
        device = device
    )
    print(f"Loading pretrained generative model...: {args.generator}")
    generator.load_state_dict( torch.load(args.generator) )
    generator = generator.to(device)

    # load scoring model
    scorer = SAScorer(
        atom_feat_dim = SCORER_PARAMS['atom_feat_dim'],
        hidden_dim = SCORER_PARAMS['hidden_dim'],
        num_layers = SCORER_PARAMS['num_layers'],
        dropout = SCORER_PARAMS['dropout'],
        use_distance = SCORER_PARAMS['use_distance'],
        rbf_k = SCORER_PARAMS['rbf_k'],
        build_edges = SCORER_PARAMS['build_edges'],
        radius = SCORER_PARAMS['radius'],
        k = SCORER_PARAMS['k'],
        pool = SCORER_PARAMS['pool'],
        use_quantile_norm = True,
        normalizer_path = args.normalizer_path
    )
    print(f"Loading pretrained scorer model...")
    ckpt = torch.load(args.scorer, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    scorer.load_state_dict(state, strict=True)
    scorer = scorer.to(device)

     # process data
    print(f"Processing input data...")
    fin = args.test
    if fin.endswith(".pt"):
        f = torch.load(fin)
    else:
        p = PROTAC(fin)#, sdf_json=args.conf, use_conformers=bool(args.conf))
        f = p.create_data()

        pt_name = os.path.splitext(os.path.basename(fin))[0] + ".pt"
        pt_path = os.path.join(save_dir, pt_name)
        torch.save(f, pt_path)


    # load data
    test_dataset = PROTACDataset(data=f)
    test_loader = DataLoader(test_dataset, GENERATOR_PARAMS['batch_size'], shuffle=False, collate_fn=collate)

    # inference mode
    optimizer = None
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    run = wandb.init(
        project="guided_diffprotacs",
        config=GENERATOR_PARAMS,
    )
    trainer = Trainer(
        model = generator,
        device = device,
        epochs = GENERATOR_PARAMS['epochs'],
        analyze_epochs=5,
        loss_type = GENERATOR_PARAMS['diffusion_loss_type'],
        n_stability_samples = args.samples, #GENERATOR_PARAMS['sampling_size'],
        save_path=save_dir,
        save_prefix="inference",
        optimizer = optimizer,
        run = run
    )
    
    start_time = datetime.now()
    if args.mode == "regular_sampling":
        print(f"inference using {args.mode}")
        out_dir = f"{save_dir}/regular_sampling"
        os.makedirs(out_dir, exist_ok=True)
        trainer.test_epoch(test_loader, out_dir=out_dir)  # Vanilla diffusion
        fout_dict = convert_xyz(args.test, out_dir) # xyz to smiles and sdfs
        fout_path = f"{save_dir}/regular_sampling.results.json"
        with open(fout_path, "w") as fout:
            json.dump(fout_dict, fout, indent=2)

    elif args.mode == "regular_sampling_w_gate":
        print(f"inference using {args.mode}")
        out_dir = f"{save_dir}/regular_sampling_w_gate"
        os.makedirs(out_dir, exist_ok=True)
        trainer.pred_svdd(
            dataloader=test_loader,
            output_dir=out_dir,
            sampler_fn=lambda data, **kw: trainer.sample_chain(
                data, sample_fn=None, keep_frames=1
            ),
            guided_max_tries=args.guided_max_tries,
            debug_disconnected=args.debug_disconnected,
            debug_max_saves_per_uuid=args.debug_max_saves_per_uuid,
        ) # Vanilla diffusion & post-sampling connectivity filter
        fout_dict = convert_xyz(args.test, out_dir) # xyz to smiles and sdfs
        fout_path = f"{save_dir}/regular_sampling.results.json"
        with open(fout_path, "w") as fout:
            json.dump(fout_dict, fout, indent=2)
    
    elif args.mode == "svdd_pm":
        print(f"inference using {args.mode}")
        out_dir=f"{args.save_dir}/guided_sampling_svdd_pm"
        os.makedirs(out_dir, exist_ok=True)
        trainer.test_epoch_svdd_pm(
            test_loader,
            scorer=scorer,                   # SAScorer is the reward/value
            sample_M=args.sample_M,
            select=args.select,
            temperature=args.temperature,
            w_rigid=max(0.0, min(1.0, args.w_rigid)),
            branch_start=args.branch_start,
            branch_end=args.branch_end,
            out_dir=out_dir,
            guided_max_tries=args.guided_max_tries,
            debug_disconnected=args.debug_disconnected,
            debug_max_saves_per_uuid=args.debug_max_saves_per_uuid,
        )
        fout_dict = convert_xyz(args.test, out_dir) # xyz to smiles and sdfs
        fout_path = f"{save_dir}/guided_sampling_svdd_pm.results.json"
        with open(fout_path, "w") as fout:
            json.dump(fout_dict, fout, indent=2)

    elif args.mode == "svdd_mc":
        print(f"inference using {args.mode}")
        out_dir = f"{args.save_dir}/guided_sampling_svdd_mc"
        os.makedirs(out_dir, exist_ok=True)
        trainer.test_epoch_svdd_mc(
            test_loader,
            scorer=scorer,
            sample_M=args.sample_M,
            select=args.select,
            temperature=args.temperature,
            partial_steps=args.svdd_partial_step,
            branch_start=args.branch_start,
            branch_end=args.branch_end,
            out_dir=out_dir,
            guided_max_tries=args.guided_max_tries,
            debug_disconnected=args.debug_disconnected,
            debug_max_saves_per_uuid=args.debug_max_saves_per_uuid,
        )
        fout_dict = convert_xyz(args.test, out_dir) # xyz to smiles and sdfs
        fout_path = f"{save_dir}/guided_sampling_svdd_mc.results.json"
        with open(fout_path, "w") as fout:
            json.dump(fout_dict, fout, indent=2)


    else:
         raise ValueError(f"Unknown mode: {args.mode}")

    end_time = datetime.now()
    print(f'Inference takes {(end_time-start_time)}')

def parse_args():
    parser = argparse.ArgumentParser(description='Make inference with trained model')
    parser.add_argument('--test', type=str, required=True, help='Path to test data json file')
    parser.add_argument('--samples', type=int, default=10, help='Sampling size (default: 10)')
    parser.add_argument('--generator', type=str, default='/isilon/ytang4/PROTAC_Project/src/DiffPROTACs/checkpoints/protacs_best.ckpt', help='Path to pretrained generator checkpoint')
    parser.add_argument('--scorer', type=str, default='/isilon/ytang4/protac_design/glue/scorer/checkpoints/best_ckpt.pt', help='Path to pretrained scorer checkpoint')
    parser.add_argument('--normalizer_path', type=str, default='/isilon/ytang4/protac_design/glue/scorer/checkpoints/quantile_transformer.pkl', help='Path to quantile normalizer directory')
    parser.add_argument('--mode', type=str, default='regular_sampling', choices=['regular_sampling', 'svdd_pm', 'svdd_mc', 'regular_sampling_w_gate'],
                       help='Guidance mode at inference')
    parser.add_argument('--sample_M', type=int, default=10,
                       help='Number of branches per step for SVDD (default: 10)')
    parser.add_argument('--branch_start', type=int, default=0,
                       help='First reverse-time step to start branching (default: 0 if None)')
    parser.add_argument('--branch_end', type=int, default=30,
                       help='Last reverse-time step to branch (default: 30, T-1 if None)')
    parser.add_argument('--select', type=str, default='softmax',
                       choices=['argmax', 'softmax'],
                       help='Branch selection rule in SVDD'),
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Softmax temperature (SVDD)')
    parser.add_argument('--svdd_partial_step', type=int, default=10,
                       help='For svdd_mc: extra denoise steps to estimate reward before selection')
    parser.add_argument("--guided_max_tries", type=int, default=10,
                       help="Guided sampling attempt budget multiplier: total attempts per UUID = guided_max_tries * n_stability_samples. "
                            "(default: 30)")
    parser.add_argument("--debug_disconnected", action="store_true",
                       help="If set, save disconnected samples for debugging under {uuid}/debug_disconnected/.")
    parser.add_argument("--debug_max_saves_per_uuid", type=int, default=5,
                       help="Max number of disconnected attempts to save per UUID for debugging.")
    parser.add_argument("--w_rigid", type=float, default=0.5,
                       help="Geometry weight for SVDD-PM composite reward. 0.0 = SA-only (backward compatible). (default: 0.5)")

    parser.add_argument('--conf', type=str, default=None, help='Path to conformation data json file')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID (default: 0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--save_dir', default="outputs/inference_w_guidance", help='Path to outputs')
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    make_inference(args)
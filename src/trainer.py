import os
import torch 
import const
import torch.nn.functional as F
from dataset import create_templates_for_linker_generation
from guidance_utils import _softmax_select, _argmax_select, _select, get_reward_fn

import numpy as np
from rdkit import Chem

class Trainer:
    def __init__(
        self,
        model,
        device,
        epochs,
        analyze_epochs,
        optimizer,
        run,
        loss_type,
        save_path,
        save_prefix,
        n_stability_samples=10, 
    ) -> None:
        self.device = device
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.run = run
        self.save_path = save_path
        self.save_prefix = save_prefix
        self.loss_type = loss_type
        self.analyze_epochs = analyze_epochs
        self.n_stability_samples = n_stability_samples 

        # Track loss history
        self.train_losses = []
        self.val_losses = []
    
    def pred(self, dataloader, output_dir, sample_fn=None):

        for data in dataloader:
            uuids = []
            true_names = []
            frag_names = []
            for uuid in data['uuid']:
                uuid = str(uuid)
                uuids.append(uuid)
                true_names.append(f'{uuid}/true')
                frag_names.append(f'{uuid}/frag')
                os.makedirs(os.path.join(output_dir, uuid), exist_ok=True)


            # Removing COM of fragment from the atom coordinates
            h, x, node_mask, frag_mask = data['one_hot'], data['positions'], data['atom_mask'], data['fragment_mask']
            
            center_of_mass_mask = data['fragment_mask']

            x = remove_partial_mean_with_mask(x, node_mask, center_of_mass_mask)
            assert_partial_mean_zero_with_mask(x, node_mask, center_of_mass_mask)

            # Saving ground-truth molecules
            save_xyz_file(output_dir, h, x, node_mask, true_names)

            # Saving fragments
            save_xyz_file(output_dir, h, x, frag_mask, frag_names)

            # Sampling and saving generated molecules
            for i in range(self.n_stability_samples):
                chain, node_mask = self.sample_chain(data, sample_fn, keep_frames=1)
                x = chain[0][:, :, :3]
                h = chain[0][:, :, 3:]

                pred_names = [f'{uuid}/{i}' for uuid in uuids]
                save_xyz_file(output_dir, h, x, node_mask, pred_names)
                save_sdf_and_smi_from_xh(output_dir, self.model, x, h, node_mask, pred_names)

    def train(self, train_loader, val_loader, test_loader):
        for epoch in range(self.epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.val_epoch(val_loader, epoch)

            # store losses
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])

            # print progress
            print(f"Epoch {epoch}/{self.epochs}: Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        
        # final test evaluation
        #print("\nEvaluating on test set...")
        #self.test_epoch(test_loader)
        
    
    def train_epoch(self, loader):
        self.model.train()
        step_outputs = []
        for data in loader:
            self.optimizer.zero_grad()
            output = self._step(data, training=True)       
            output['loss'].backward()
            self.optimizer.step()
            for metric in output.keys():
                self.run.log({f'{metric}/train_step': output[metric]})
            step_outputs.append(output)

        # Calculate epoch metrics
        epoch_metrics = {}
        with torch.no_grad():
            for metric in step_outputs[0].keys():
                avg_metric = Trainer.aggregate_metric(step_outputs, metric)
                self.run.log({f'{metric}/train_epoch': avg_metric}, commit=False)
                epoch_metrics[metric] = avg_metric
        return epoch_metrics
    
    def val_epoch(self, loader, epoch):
        self.model.eval()
        with torch.no_grad():
            step_outputs = []
            for data in loader:
                output = self._step(data)
                step_outputs.append(output)
            
            # Calculate validation metrics
            val_metrics = {}
            best_loss = getattr(self, 'best_loss', float('inf'))
            
            for metric in step_outputs[0].keys():
                avg_metric = Trainer.aggregate_metric(step_outputs, metric)
                self.run.log({f'{metric}/val': avg_metric})
                val_metrics[metric] = avg_metric.item()  # Convert to float
                
                if metric == 'loss':
                    if avg_metric < best_loss:
                        self.best_loss = avg_metric.item()
                        torch.save(self.model.state_dict(), 
                                f'{self.save_path}/{self.save_prefix}_best.ckpt')
                        print(f'  New best model saved! Loss: {avg_metric:.4f}')
            
            if (epoch + 1) % self.analyze_epochs == 0:
                torch.save(self.model.state_dict(), 
                        f'{self.save_path}/{self.save_prefix}_epoch{epoch}.ckpt')
        
        return val_metrics
        
    def test_epoch(self, loader, out_dir=None):
        self.model.eval()
        with torch.no_grad():
            step_outputs = []
            for data in loader:
                output = self._step(data)
                step_outputs.append(output)
            for metric in step_outputs[0].keys():
                avg_metric = Trainer.aggregate_metric(step_outputs, metric)
                print(f'{metric}/test: {avg_metric}')
            if out_dir is None:
                out_dir = f"{self.save_path}/test"
            self.pred(loader, out_dir)
    
    def _step(self, data, training=False):
        l2_loss = self.model(data, training)
        if self.loss_type == 'l2':
            loss, loss_x, loss_h = l2_loss
        else:
            raise NotImplementedError(self.loss_type)
        
        metrics = {
            'loss': loss,
            'loss_x': loss_x,
            'loss_h': loss_h,
        }

        return metrics

    def sample_chain(self, data, sample_fn=None, keep_frames=None):
        if sample_fn is None:
            linker_sizes = data['linker_mask'].sum(1).view(-1).int()
        else:
            linker_sizes = sample_fn(data)

        template_data = create_templates_for_linker_generation(data, linker_sizes)

        x = template_data['positions']
        node_mask = template_data['atom_mask']
        edge_mask = template_data['edge_mask']
        h = template_data['one_hot']
        fragment_mask = template_data['fragment_mask']
        linker_mask = template_data['linker_mask']
        context = fragment_mask
        center_of_mass_mask = fragment_mask

        x = remove_partial_mean_with_mask(x, node_mask, center_of_mass_mask)

        chain = self.model.sample_chain(
            x=x.to(self.device),
            h=h.to(self.device),
            node_mask=node_mask.to(self.device),
            edge_mask=edge_mask.to(self.device),
            fragment_mask=fragment_mask.to(self.device),
            linker_mask=linker_mask.to(self.device),
            context=context.to(self.device),
            keep_frames=keep_frames,
        )
        return chain, node_mask

    def sample_chain_svdd_pm(self, data, scorer, sample_M=8, select='argmax', temperature=1.0, w_rigid=0.0, keep_frames=None, branch_start=None, branch_end=None):
        linker_sizes = data['linker_mask'].sum(1).view(-1).int()
        template = create_templates_for_linker_generation(data, linker_sizes)
        x, h = template['positions'], template['one_hot']
        node_mask, edge_mask = template['atom_mask'], template['edge_mask']
        fragment_mask, linker_mask = template['fragment_mask'], template['linker_mask']
        context = fragment_mask
        x = remove_partial_mean_with_mask(x, node_mask, fragment_mask)

        chain = self.model.sample_chain_svdd_pm(
            x = x.to(self.device), h=h.to(self.device),
            node_mask=node_mask.to(self.device), fragment_mask=fragment_mask.to(self.device),
            linker_mask=linker_mask.to(self.device),edge_mask=edge_mask.to(self.device),
            context=context.to(self.device), scorer=scorer, sample_M=sample_M, select=select, 
            temperature=temperature, w_rigid=w_rigid, keep_frames=keep_frames,branch_start=branch_start, branch_end=branch_end
        )
        return chain, node_mask

    def sample_chain_svdd_mc(self, data, scorer, sample_M=8, select='argmax',
                         temperature=1.0, partial_steps=10, keep_frames=1, branch_start=None, branch_end=None):
        """
        SVDD-MC sampling with partial rollout.
        
        Args:
            partial_steps: Number of denoising steps in each MC rollout
                        - 0: immediate prediction (fastest)
                        - 1-20: partial rollout (recommended)
                        - >50: longer rollout (more accurate but slower)
        """
        x = data['positions'].to(self.device)
        h = data['one_hot'].to(self.device)
        node_mask = data['atom_mask'].to(self.device)
        fragment_mask = data['fragment_mask'].to(self.device)
        linker_mask = data['linker_mask'].to(self.device)
        edge_mask = data['edge_mask'].to(self.device)
        context = fragment_mask
        
        # IMPORTANT: center w.r.t. fragment COM (same as other sampling paths)
        x = remove_partial_mean_with_mask(x, node_mask, fragment_mask)
        
        chain = self.model.sample_chain_svdd_mc(
            x=x, h=h,
            node_mask=node_mask,
            fragment_mask=fragment_mask,
            linker_mask=linker_mask,
            edge_mask=edge_mask,
            context=context,
            scorer=scorer,
            sample_M=sample_M,
            rollout_K=4,  
            partial_steps=partial_steps,
            select=select,
            temperature=temperature,
            keep_frames=keep_frames,
            branch_start=branch_start,
            branch_end=branch_end
        )
        
        return chain, node_mask

    def pred_svdd(
        self,
        dataloader,
        output_dir,
        sampler_fn,
        guided_max_tries: int = 50,
        debug_disconnected: bool = True,
        debug_max_saves_per_uuid: int = 5,
        **sampler_kwargs,
    ):
        """
        Guided SVDD prediction with connectivity gating.

        Behavior:
        - Always saves true + frag xyz.
        - For guided samples: tries to save up to self.n_stability_samples CONNECTED samples per UUID.
        - Does NOT crash if target cannot be met within the attempt budget.
        - Optionally saves a few DISCONNECTED samples for debugging under:
                {output_dir}/{uuid}/debug_disconnected/
            including a rolling "best.xyz" (heuristic: most recent disconnected) and up to debug_max_saves_per_uuid attempt files.

        guided_max_tries:
        - Effort knob, NOT a guarantee. Total attempt budget per UUID is guided_max_tries * self.n_stability_samples.
        """
        for data in dataloader:
            uuids = [str(u) for u in data["uuid"]]
            B = len(uuids)

            # Create dirs
            for u in uuids:
                os.makedirs(os.path.join(output_dir, u), exist_ok=True)
                if debug_disconnected:
                    os.makedirs(os.path.join(output_dir, u, "debug_disconnected"), exist_ok=True)

            # Save true + frag (COM-centered to fragment mask)
            h, x = data["one_hot"], data["positions"]
            node_mask, frag_mask = data["atom_mask"], data["fragment_mask"]
            x_centered = remove_partial_mean_with_mask(x, node_mask, frag_mask)

            true_names = [f"{u}/true" for u in uuids]
            frag_names = [f"{u}/frag" for u in uuids]
            save_xyz_file(output_dir, h, x_centered, node_mask, true_names)
            save_xyz_file(output_dir, h, x_centered, frag_mask, frag_names)

            # Guided sampling: best-effort connected outputs
            target = int(self.n_stability_samples)
            max_total_attempts = int(guided_max_tries) * max(target, 1)

            saved_counts = [0] * B
            disconnected_saved = [0] * B
            attempts = 0

            # Track a rolling disconnected "best" (heuristic: last seen disconnected)
            last_disconnected_x = [None] * B
            last_disconnected_h = [None] * B
            last_disconnected_node_mask = [None] * B

            while attempts < max_total_attempts and any(c < target for c in saved_counts):
                attempts += 1

                chain, nm = sampler_fn(data, **sampler_kwargs, keep_frames=1)
                x_pred = chain[0][:, :, :3]
                h_pred = chain[0][:, :, 3:]

                connected = self.model.check_connected_final(
                    x=x_pred,
                    h=h_pred,
                    node_mask=nm.to(x_pred.device),
                    fragment_mask=data["fragment_mask"].to(x_pred.device),
                    linker_mask=data["linker_mask"].to(x_pred.device),
                )  # expected shape [B] bool

                for b in range(B):
                    # Already reached target for this uuid
                    if saved_counts[b] >= target:
                        continue

                    is_conn = bool(connected[b].item())

                    if is_conn:
                        # Save connected sample into main folder: {uuid}/{k}.xyz
                        k = saved_counts[b]
                        pred_name = [f"{uuids[b]}/{k}"]

                        xb = x_pred[b : b + 1]
                        hb = h_pred[b : b + 1]
                        nmb = nm[b : b + 1]

                        save_xyz_file(output_dir, hb, xb, nmb, pred_name)
                        save_sdf_and_smi_from_xh(output_dir, self.model, xb, hb, nmb, pred_name)

                        saved_counts[b] += 1
                    else:
                        if not debug_disconnected:
                            continue

                        # Cache "best" (heuristic: most recent disconnected)
                        last_disconnected_x[b] = x_pred[b : b + 1].detach()
                        last_disconnected_h[b] = h_pred[b : b + 1].detach()
                        last_disconnected_node_mask[b] = nm[b : b + 1].detach()

                        # Save a few disconnected attempts for debugging
                        if disconnected_saved[b] < debug_max_saves_per_uuid:
                            dbg_dir = os.path.join(output_dir, uuids[b], "debug_disconnected")
                            # write attempt_i.xyz under debug_disconnected/
                            dbg_name = [f"{uuids[b]}/debug_disconnected/attempt_{disconnected_saved[b]}"]
                            xb = x_pred[b : b + 1]
                            hb = h_pred[b : b + 1]
                            nmb = nm[b : b + 1]

                            save_xyz_file(output_dir, hb, xb, nmb, dbg_name)
                            save_sdf_and_smi_from_xh(output_dir, self.model, xb, hb, nmb, dbg_name)
                            disconnected_saved[b] += 1


                            # also update "best.xyz" (overwrite semantics via same name)
                            best_name = [f"{uuids[b]}/debug_disconnected/best"]
                            xb = x_pred[b : b + 1]
                            hb = h_pred[b : b + 1]
                            nmb = nm[b : b + 1]

                            save_xyz_file(output_dir, hb, xb, nmb, best_name)
                            save_sdf_and_smi_from_xh(output_dir, self.model, xb, hb, nmb, best_name)


            # If we ended without meeting target, still write one "best" disconnected if we never saved any debug files
            if debug_disconnected:
                for b in range(B):
                    if saved_counts[b] >= target:
                        continue
                    if disconnected_saved[b] == 0 and last_disconnected_x[b] is not None:
                        best_name = [f"{uuids[b]}/debug_disconnected/best"]
                        xb = last_disconnected_x[b]
                        hb = last_disconnected_h[b]
                        nmb = last_disconnected_node_mask[b]

                        save_xyz_file(output_dir, hb, xb, nmb, best_name)
                        save_sdf_and_smi_from_xh(output_dir, self.model, xb, hb, nmb, best_name)


            # Friendly summary (no exception)
            for b in range(B):
                if saved_counts[b] < target:
                    print(
                        f"[pred_svdd] WARNING: uuid={uuids[b]} saved {saved_counts[b]}/{target} "
                        f"connected samples after {attempts}/{max_total_attempts} attempts "
                        f"(guided_max_tries={guided_max_tries})."
                    )
                else:
                    print(
                        f"[pred_svdd] uuid={uuids[b]} saved {saved_counts[b]}/{target} connected samples "
                        f"in {attempts} attempts."
                    )

    def test_epoch_svdd_pm(self, loader, scorer, sample_M=8, select='argmax', temperature=1.0, w_rigid=0.0, branch_start=None, branch_end=None, out_dir=None,
            guided_max_tries=50, debug_disconnected=False, debug_max_saves_per_uuid=5):
        self.model.eval(); scorer.eval()
        #out_dir = out_dir or f"{self.save_path}/test_svdd_pm_M{sample_M}_{select}"
        #os.makedirs(out_dir, exist_ok=True)
        print(f"SVDD-PM sampling (M={sample_M}, select={select}, τ={temperature})")
        self.pred_svdd(loader, out_dir, sampler_fn=lambda data, **kw: self.sample_chain_svdd_pm(data, **kw),
                    scorer=scorer, sample_M=sample_M, select=select, temperature=temperature, w_rigid=w_rigid, branch_start=branch_start, branch_end=branch_end,
                    guided_max_tries=guided_max_tries, debug_disconnected=debug_disconnected, debug_max_saves_per_uuid=debug_max_saves_per_uuid)
    def test_epoch_svdd_mc(self, loader, scorer, sample_M=8, select='argmax', temperature=1.0, partial_steps=0,branch_start=None, branch_end=None, out_dir=None,
            guided_max_tries=50, debug_disconnected=False, debug_max_saves_per_uuid=5):
        self.model.eval(); scorer.eval()
        #out_dir = out_dir or f"{self.save_path}/test_svdd_mc_M{sample_M}_{select}_ps{partial_steps}"
        #os.makedirs(out_dir, exist_ok=True)
        print(f"SVDD-MC sampling (M={sample_M}, select={select}, τ={temperature}, partial_steps={partial_steps})")
        self.pred_svdd(loader, out_dir, sampler_fn=lambda data, **kw: self.sample_chain_svdd_mc(data, **kw),
                    scorer=scorer, sample_M=sample_M, select=select, temperature=temperature, partial_steps=partial_steps,
                    branch_start=branch_start, branch_end=branch_end, 
                    guided_max_tries=guided_max_tries, debug_disconnected=debug_disconnected, debug_max_saves_per_uuid=debug_max_saves_per_uuid)

    @staticmethod
    def aggregate_metric(step_outputs, metric):
        return torch.tensor([out[metric] for out in step_outputs]).mean()


    def plot_loss_curves(self, save_path=None):
        """
        Plot training and validation loss curves
        
        Args:
            save_path: Path to save the figure. If None, uses self.save_path
        """
        import matplotlib.pyplot as plt
        
        if save_path is None:
            save_path = os.path.join(self.save_path, f'{self.save_prefix}_loss_curve.png')
        
        epochs = range(len(self.train_losses))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epochs, self.train_losses, label='Train Loss', 
                marker='o', linewidth=2, markersize=4, alpha=0.8)
        ax.plot(epochs, self.val_losses, label='Validation Loss', 
                marker='s', linewidth=2, markersize=4, alpha=0.8)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add min validation loss marker
        min_val_epoch = self.val_losses.index(min(self.val_losses))
        min_val_loss = self.val_losses[min_val_epoch]
        ax.plot(min_val_epoch, min_val_loss, 'r*', markersize=15, 
                label=f'Best Val Loss: {min_val_loss:.4f} @ Epoch {min_val_epoch}')
        ax.legend(fontsize=10, loc='best')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Loss curve saved to {save_path}")
        
        return save_path

def remove_partial_mean_with_mask(x, node_mask, center_of_mass_mask):
    """
    Subtract center of mass of fragments from coordinates of all atoms
    """
    x_masked = x * center_of_mass_mask
    N = center_of_mass_mask.sum(1, keepdims=True)
    mean = torch.sum(x_masked, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x

def assert_partial_mean_zero_with_mask(x, node_mask, center_of_mass_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)
    x_masked = x * center_of_mass_mask
    largest_value = x_masked.abs().max().item()
    error = torch.sum(x_masked, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f'Partial mean is not zero, relative_error {rel_error}'

def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().max().item() < 1e-4, \
        'Variables not masked properly.'

def save_xyz_file(path, one_hot, positions, node_mask, names, suffix=''):
    idx2atom = const.IDX2ATOM

    for batch_i in range(one_hot.size(0)):
        mask = node_mask[batch_i].squeeze()
        n_atoms = mask.sum()
        atom_idx = torch.where(mask)[0]

        f = open(os.path.join(path, f'{names[batch_i]}_{suffix}.xyz'), "w")
        f.write("%d\n\n" % n_atoms)
        atoms = torch.argmax(one_hot[batch_i], dim=1)
        for atom_i in atom_idx:
            atom = atoms[atom_i].item()
            atom = idx2atom[atom]
            f.write("%s %.9f %.9f %.9f\n" % (
                atom, positions[batch_i, atom_i, 0], positions[batch_i, atom_i, 1], positions[batch_i, atom_i, 2]
            ))
        f.close()

def save_sdf_and_smi_from_xh(path, model, positions, one_hot, node_mask, names, suffix=''):
    """Write per-sample .sdf and .smi using RDKit built from predicted (x,h)."""
    if Chem is None:
        return

    for batch_i in range(one_hot.size(0)):
        mask = node_mask[batch_i].squeeze()
        mol = model.build_rdkit_mol_from_xh(
            pos=positions[batch_i],
            one_hot=one_hot[batch_i],
            atom_mask=mask,
        )
        if mol is None:
            continue

        out_base = os.path.join(path, f"{names[batch_i]}_{suffix}")
        os.makedirs(os.path.dirname(out_base), exist_ok=True)

        sdf_path = out_base + ".sdf"
        w = Chem.SDWriter(sdf_path)
        w.write(mol)
        w.close()

        smi_path = out_base + ".smi"
        with open(smi_path, "w") as f:
            f.write(Chem.MolToSmiles(mol) + "\n")

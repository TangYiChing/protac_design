import torch
import torch.nn.functional as F
import numpy as np
import math

import utils
from dynamics import Dynamics
from noise import GammaNetwork, PredefinedNoiseSchedule
from adapter import DiffusionToScorerAdapter

try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None


class EDM(torch.nn.Module):
    def __init__(
            self,
            in_node_nf = 9,
            n_dims = 3,
            hidden_nf=128, 
            timesteps: int = 500,
            noise_schedule='polynomial_2', # learned, cosine
            noise_precision=1e-5,
            loss_type='l2',
            norm_values=[1, 4, 10],
            norm_biases=(None, 0., 0.),
            device='cpu',
            ffn_embedding_dim=3072,
            attention_heads=8,
            tanh=True,
            n_layers=6,
            coords_range=10,
            dropout=0.1,
            activation_dropout= 0.1,
    ):
        super().__init__()
        if noise_schedule == 'learned':
            assert loss_type == 'vlb', 'A noise schedule can only be learned with a vlb objective'
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps, precision=noise_precision)

        self.dynamics = Dynamics(
            device=device,
            in_node_nf=in_node_nf,
            n_layers=n_layers,
            hidden_nf=hidden_nf, 
            ffn_embedding_dim=ffn_embedding_dim,
            attention_heads=attention_heads,
            tanh=tanh,
            coords_range=coords_range,
            dropout=dropout,
            activation_dropout= activation_dropout,
        )
        self.device = device
        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.T = timesteps
        self.norm_values = norm_values
        self.norm_biases = norm_biases

    def forward(self, data, training):
        x = data['positions']
        h = data['one_hot']
        node_mask = data['atom_mask']
        edge_mask = data['edge_mask']
        fragment_mask = data['fragment_mask']
        linker_mask = data['linker_mask']

        context = fragment_mask
        center_of_mass_mask = fragment_mask
            
        x = utils.remove_partial_mean_with_mask(x, node_mask, center_of_mass_mask)
        utils.assert_partial_mean_zero_with_mask(x, node_mask, center_of_mass_mask)

        x=x.to(self.device)
        h=h.to(self.device)
        node_mask=node_mask.to(self.device)
        fragment_mask=fragment_mask.to(self.device)
        linker_mask=linker_mask.to(self.device)
        edge_mask=edge_mask.to(self.device)
        context=context.to(self.device)

        # Normalization and concatenation
        x, h = self.normalize(x, h)
        xh = torch.cat([x, h], dim=2)

        # Sample t
        t_int = torch.randint(0, self.T + 1, size=(x.size(0), 1), device=x.device).float()
        t = t_int / self.T

        # Compute gamma_t and gamma_s according to the noise schedule
        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        # Compute alpha_t and sigma_t from gamma
        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)

        # Sample noise
        # Note: only for linker
        eps_t = self.sample_combined_position_feature_noise(n_samples=x.size(0), n_nodes=x.size(1), mask=linker_mask)

        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        # Note: keep fragments unchanged
        z_t = alpha_t * xh + sigma_t * eps_t
        z_t = xh * fragment_mask + z_t * linker_mask

        # Neural net prediction
        eps_t_hat = self.dynamics.forward(
            xh=z_t,
            t=t,
            node_mask=node_mask,
            linker_mask=linker_mask,
            context=context,
            edge_mask=edge_mask,
            training=training,
        )
        eps_t_hat = eps_t_hat * linker_mask

        # Computing basic error (further used for computing NLL and L2-loss)
        # error_t = self.sum_except_batch((eps_t - eps_t_hat) ** 2)
        error_x = self.sum_except_batch((eps_t[...,:3] - eps_t_hat[...,:3]) ** 2)
        error_h = self.sum_except_batch((eps_t[...,3:] - eps_t_hat[...,3:]) ** 2)

        # Computing L2-loss for t>0
        normalization = (self.n_dims + self.in_node_nf) * self.numbers_of_nodes(linker_mask)
        # l2_loss = (error_t / normalization).mean()
        loss_x = (error_x/normalization).mean()
        loss_h = (error_h/normalization).mean()

        l2_loss = loss_x + loss_h

        return l2_loss, loss_x, loss_h


    @torch.no_grad()
    def sample_chain_with_sa_guidance(self, x, h, node_mask, fragment_mask, linker_mask, 
                                    edge_mask, context, sa_scorer, guidance_scale=1.0, 
                                    keep_frames=None):
        """SA-guided sampling"""
        n_samples = x.size(0)
        n_nodes = x.size(1)

        # Normalization and concatenation
        x, h = self.normalize(x, h)
        xh = torch.cat([x, h], dim=2)

        # Initial linker sampling from N(0, I)
        z = self.sample_combined_position_feature_noise(n_samples, n_nodes, mask=linker_mask)
        z = xh * fragment_mask + z * linker_mask

        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + z.size(), device=z.device)

        # Sample p(z_s | z_t) with SA guidance
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            # 標準diffusion step
            z = self.sample_p_zs_given_zt_only_linker(
                s=s_array,
                t=t_array,
                z_t=z,
                node_mask=node_mask,
                fragment_mask=fragment_mask,
                linker_mask=linker_mask,
                edge_mask=edge_mask,
                context=context,
            )
            
            # SA guidance（關鍵新增部分）
            if guidance_scale > 0 and s > 0:  # 不在最後一步apply guidance
                z = self.apply_sa_guidance(
                    z=z,
                    fragment_mask=fragment_mask,
                    linker_mask=linker_mask,
                    sa_scorer=sa_scorer,
                    guidance_scale=guidance_scale
                )
            
            write_index = (s * keep_frames) // self.T
            chain[write_index] = self.unnormalize_z(z)

        # Finally sample p(x, h | z_0)
        x, h = self.sample_p_xh_given_z0_only_linker(
            z_0=z,
            node_mask=node_mask,
            fragment_mask=fragment_mask,
            linker_mask=linker_mask,
            edge_mask=edge_mask,
            context=context,
        )
        chain[0] = torch.cat([x, h], dim=2)

        return chain

    def apply_sa_guidance(self, z, fragment_mask, linker_mask, sa_scorer, guidance_scale):
        """SA gradient guidance"""
        from adapter import DiffusionToScorerAdapter
        
        # 分離x和h
        x = z[:, :, :self.n_dims]  # [B, N, 3]
        h = z[:, :, self.n_dims:]  # [B, N, 9]
        
        # 反歸一化以獲得真實座標和特徵
        x_unnorm, h_unnorm = self.unnormalize(x, h)
        
        # 準備scorer輸入
        current_state = {
            'positions': x_unnorm,
            'one_hot': h_unnorm,
            'linker_mask': linker_mask,
            #'fragment_mask': fragment_mask
        }
        
        # 轉換格式
        linker_batch = DiffusionToScorerAdapter.extract_linker_batch(current_state)
        
        # 計算SA gradient（需要enable grad）
        with torch.enable_grad():
            # 確保可以計算梯度
            linker_batch.pos.requires_grad_(True)
            
            # SA評分
            sa_score = sa_scorer(linker_batch)
            
            # 計算梯度
            sa_gradient = torch.autograd.grad(
                outputs=sa_score.sum(),
                inputs=linker_batch.pos,
                create_graph=False  # 在sampling時不需要保持graph
            )[0]
        
        # 將gradient映射回完整tensor
        full_gradient = torch.zeros_like(x_unnorm)
        batch_size = x_unnorm.shape[0]
        
        # 恢復每個sample的gradient
        ptr = 0
        for i in range(batch_size):
            # Count linker atoms for this sample
            mask_i = linker_mask[i].squeeze(-1).bool()
            n_linker = mask_i.sum().item()
            if n_linker > 0:
                # Extract gradient slice for this sample
                grad_slice = sa_gradient[ptr:ptr+n_linker]
                # Place back into full gradient tensor
                full_gradient[i, mask_i] = grad_slice
                ptr += n_linker
            else:
                print(f"Sample {i} has no linker atoms")
        
        # Apply guidance（在歸一化空間）
        x_guided = x + guidance_scale * (full_gradient / self.norm_values[0])
        
        # 重新組合
        z_guided = torch.cat([x_guided, h], dim=2)
        
        # 保持fragment不變
        z_guided = z * fragment_mask + z_guided * linker_mask
        
        return z_guided


    @torch.no_grad()
    def sample_chain(self, x, h, node_mask, fragment_mask, linker_mask, edge_mask, context, keep_frames=None):
        n_samples = x.size(0)
        n_nodes = x.size(1)

        # Normalization and concatenation
        x, h, = self.normalize(x, h)
        xh = torch.cat([x, h], dim=2)

        # Initial linker sampling from N(0, I)
        z = self.sample_combined_position_feature_noise(n_samples, n_nodes, mask=linker_mask)
        z = xh * fragment_mask + z * linker_mask

        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + z.size(), device=z.device)

        # Sample p(z_s | z_t)
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt_only_linker(
                s=s_array,
                t=t_array,
                z_t=z,
                node_mask=node_mask,
                fragment_mask=fragment_mask,
                linker_mask=linker_mask,
                edge_mask=edge_mask,
                context=context,
            )
            write_index = (s * keep_frames) // self.T
            chain[write_index] = self.unnormalize_z(z)

        # Finally sample p(x, h | z_0)
        x, h = self.sample_p_xh_given_z0_only_linker(
            z_0=z,
            node_mask=node_mask,
            fragment_mask=fragment_mask,
            linker_mask=linker_mask,
            edge_mask=edge_mask,
            context=context,
        )
        chain[0] = torch.cat([x, h], dim=2)

        return chain

    @torch.no_grad()
    def sample_chain_svdd_pm(
        self,
        x, h,
        node_mask, fragment_mask, linker_mask, edge_mask, context,
        scorer,
        sample_M: int = 8,
        select: str = 'argmax',       # or 'softmax'
        temperature: float = 1.0,
        w_rigid: float = 0.0,         # NEW: rigidity weight (0 disables; keeps old SA-only behavior),
        keep_frames: int = None,
        # NEW: Selective branching parameters
        branch_start: int = None,     # Start branching at this timestep (None = all steps)
        branch_end: int = None,       # Stop branching at this timestep (None = all steps)
        # NEW: Diagnostic parameters
        enable_diagnostics: bool = False,
        diagnostics_interval: int = 50,  # Log every N steps
        connectivity_checker: object = None  # Pass connectivity checker for diagnostics
        ):
        """
        SVDD-PM (Posterior-Mean) branch-and-select sampler with diagnostics.
        
        At each step s, propose M candidates, score r( x0_hat(z_s, s) ), pick one per item, and continue.
        
        NEW Features:
        - Selective branching: Only apply branching in [branch_start, branch_end] range
        - Connectivity diagnostics: Track when disconnections occur
        - Configurable logging interval
        
        Args:
            branch_start: First timestep to apply branching (None = 0, i.e., from beginning)
            branch_end: Last timestep to apply branching (None = T-1, i.e., to end)
            enable_diagnostics: Whether to log connectivity statistics
            diagnostics_interval: How often to log (every N steps)
            connectivity_checker: Connectivity checker object for diagnostics
            
        Example Usage:
            # Test different branching ranges
            for start, end in [(400, 500), (300, 400), (200, 300), (100, 200)]:
                chain = model.sample_chain_svdd_pm(
                    x, h, ...,
                    branch_start=start,
                    branch_end=end,
                    enable_diagnostics=True,
                    connectivity_checker=checker
                )
        """
        device = x.device
        B0, N = x.size(0), x.size(1)

        # Set default branching range (full range if not specified)
        if branch_start is None:
            branch_start = 0
        if branch_end is None:
            branch_end = self.T - 1
        
        # Validate branching range
        assert 0 <= branch_start <= branch_end < self.T, \
            f"Invalid branching range: [{branch_start}, {branch_end}] (must be in [0, {self.T-1}])"

        # Initialize diagnostics
        diagnostics_data = [] if enable_diagnostics else None
        if enable_diagnostics:
            print(f"\n{'='*80}")
            print(f"SVDD-PM with Connectivity Diagnostics")
            print(f"{'='*80}")
            print(f"Branching range: [{branch_start}, {branch_end}]")
            print(f"Sample M: {sample_M}")
            print(f"Selection: {select}")
            print(f"Logging interval: every {diagnostics_interval} steps")
            print(f"{'='*80}\n")

        # normalize inputs and init z on linker only
        x, h = self.normalize(x, h)
        xh = torch.cat([x, h], dim=2)                                # [B0, N, 3+F]
        z = self.sample_combined_position_feature_noise(B0, N, linker_mask)
        z = xh * fragment_mask + z * linker_mask                     # keep fragments; noise only on linker

        # frames to keep (full chain by default)
        if keep_frames is None:
            keep_frames = self.T
        else:
            assert 1 <= keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + z.size(), device=device)

        # main reverse process
        for s in reversed(range(0, self.T)):
            # scalar timestep vectors shaped [B0, 1] in [0,1]
            s_arr = (torch.full((B0,), s, device=device, dtype=torch.long).float() / self.T).unsqueeze(1)
            t_arr = ((torch.full((B0,), s, device=device, dtype=torch.long) + 1).float() / self.T).unsqueeze(1)

            # Decide whether to branch at this timestep
            use_branching = (branch_start <= s <= branch_end)
            actual_M = sample_M if use_branching else 1

            # repeat state and all graph-wise stuff M times (B0 -> B0*M)
            z_rep = self._repeat_batch(z, actual_M)
            tiled = self._tile_graph_inputs(
                actual_M,
                node_mask=node_mask, fragment_mask=fragment_mask,
                linker_mask=linker_mask, edge_mask=edge_mask,
                context=context, s_arr=s_arr, t_arr=t_arr
            )

            # propose z_s ~ p(z_s | z_t; s->t) on linker only
            z_s_all = self.sample_p_zs_given_zt_only_linker(
                s=tiled['s_arr'], t=tiled['t_arr'], z_t=z_rep,
                node_mask=tiled['node_mask'], fragment_mask=tiled['fragment_mask'],
                linker_mask=tiled['linker_mask'], edge_mask=tiled['edge_mask'],
                context=tiled['context']
            )  # [B0*M, N, 3+F]

            if use_branching:
                # score candidates via posterior-mean value r( x0_hat(z_s, s) )
                values = self._score_candidates_pm(
                    z_s=z_s_all, s_arr=tiled['s_arr'],
                    node_mask=tiled['node_mask'], fragment_mask=tiled['fragment_mask'],
                    linker_mask=tiled['linker_mask'], edge_mask=tiled['edge_mask'],
                    context=tiled['context'], scorer=scorer,
                    w_rigid=w_rigid
                ).view(B0, actual_M)  # per original item
                #print(f"[SVDD-PM] step {s:3d}: valid {100*self._last_valid_rate:5.1f}%")

                if (values <= -1e8).all():
                    # fallback to vanilla single-candidate step (no branching this timestep)
                    idx = torch.zeros(B0, dtype=torch.long, device=device)
                else:
                    # choose one branch per original sample
                    if select == 'softmax':
                        probs = torch.softmax(values / max(temperature, 1e-8), dim=1)  # [B0, M]
                        idx = torch.multinomial(probs, num_samples=1).squeeze(1)       # [B0]
                    else:
                        idx = torch.argmax(values, dim=1)                               # [B0]
                
                # DIAGNOSTICS: Log connectivity information
                if enable_diagnostics and (s % diagnostics_interval == 0 or s == branch_start or s == branch_end):
                    self._log_connectivity_diagnostics(
                        timestep=s,
                        z_s_all=z_s_all,
                        values=values,
                        selected_idx=idx,
                        B0=B0,
                        M=actual_M,
                        linker_mask=tiled['linker_mask'],
                        connectivity_checker=connectivity_checker,
                        diagnostics_data=diagnostics_data
                    )
            else:
                # No branching: just use the single candidate
                idx = torch.zeros(B0, dtype=torch.long, device=device)
                if enable_diagnostics and s % diagnostics_interval == 0:
                    print(f"  Step {s:3d}: No branching (outside range [{branch_start}, {branch_end}])")

            # gather chosen z_s and continue
            z_s_all = z_s_all.view(B0, actual_M, N, self.n_dims + self.in_node_nf)
            z = z_s_all[torch.arange(B0, device=device), idx]                   # [B0, N, 3+F]

            # save frame (uniformly downsample T -> keep_frames)
            write_index = (s * keep_frames) // self.T
            chain[write_index] = self.unnormalize_z(z)

        # final decode p(x,h | z0) with only-linker transition
        x_final, h_final = self.sample_p_xh_given_z0_only_linker(
            z_0=z, node_mask=node_mask, fragment_mask=fragment_mask,
            linker_mask=linker_mask, edge_mask=edge_mask, context=context
        )
        chain[0] = torch.cat([x_final, h_final], dim=2)
        
        # Print diagnostics summary
        if enable_diagnostics and diagnostics_data:
            self._print_diagnostics_summary(diagnostics_data, branch_start, branch_end)
        
        return chain

    @torch.no_grad()
    def sample_chain_svdd_mc(
        self,
        x, h, node_mask, fragment_mask, linker_mask, edge_mask, context,
        scorer,
        sample_M=8,
        select="argmax",
        temperature=1.0,
        rollout_K=4,
        partial_steps=10,
        keep_frames=1,
        branch_start=None,
        branch_end=None,
    ):
        device = x.device
        B0, N = x.size(0), x.size(1)

        if branch_start is None: branch_start = 0
        if branch_end is None: branch_end = self.T - 1

        # init z_T (same as your current code)
        z = self.sample_z_T_only_linker(x, h, node_mask, fragment_mask, linker_mask)

        chain = []
        for s in reversed(range(self.T)):
            s_arr = self._coerce_time_vec(torch.full((B0,), s, device=device), B0) / self.T
            #t_arr = self._coerce_time_vec(torch.full((B0,), s+1, device=device), B0) / self.T

            # Clamp t to avoid s+1 == T edge case
            t_val = min(s + 1, self.T -1)
            t_arr = self._coerce_time_vec(torch.full((B0,), t_val, device=device), B0) / self.T

            use_branching = (branch_start <= s <= branch_end)
            actual_M = sample_M if use_branching else 1

            z_rep = self._repeat_batch(z, actual_M)
            tiled = self._tile_graph_inputs(
                actual_M, node_mask=node_mask, fragment_mask=fragment_mask,
                linker_mask=linker_mask, edge_mask=edge_mask,
                context=context, s_arr=s_arr, t_arr=t_arr)

            z_s_all = self.sample_p_zs_given_zt_only_linker(
                s=tiled["s_arr"], t=tiled["t_arr"], z_t=z_rep,
                node_mask=tiled["node_mask"], fragment_mask=tiled["fragment_mask"],
                linker_mask=tiled["linker_mask"], edge_mask=tiled["edge_mask"],
                context=tiled["context"]
            )

            if use_branching:
                values = self._score_candidates_mc_partial(
                    z_s=z_s_all,
                    s_int=s,
                    node_mask=tiled["node_mask"], fragment_mask=tiled["fragment_mask"],
                    linker_mask=tiled["linker_mask"], edge_mask=tiled["edge_mask"],
                    context=tiled["context"], scorer=scorer,
                    rollout_K=rollout_K,partial_steps=partial_steps,
                ).view(B0, actual_M)

                # All-invalid fallback
                if (values <= -1e8).all():
                    idx = torch.zeros(B0, dtype=torch.long, device=device)
                else:
                    if select == "softmax":
                        probs = torch.softmax(values / max(temperature, 1e-8), dim=1)
                        idx = torch.multinomial(probs, num_samples=1).squeeze(1)
                    else:
                        idx = torch.argmax(values, dim=1)
            else:
                idx = torch.zeros(B0, dtype=torch.long, device=device)

            z_s_all = z_s_all.view(B0, actual_M, N, self.n_dims + self.in_node_nf)
            z = z_s_all[torch.arange(B0, device=device), idx]

            if keep_frames is not None and (s % keep_frames == 0):
                chain.append(z)

        # decode from z_0 (same as existing)
        x_final, h_final = self.sample_p_xh_given_z0_only_linker(
            z_0=z,
            node_mask=node_mask,
            fragment_mask=fragment_mask,
            linker_mask=linker_mask,
            edge_mask=edge_mask,
            context=context,
        )
        xh_final = torch.cat([x_final, h_final], dim=2)
        chain.insert(0, xh_final)
        return chain


    @torch.no_grad()
    def sample_chain_svdd_mc_original_no_start_end( # this design was to complement PM to improve robustness at early stage by looking ahead (partial_steps)
        self,
        x, h,
        node_mask, fragment_mask, linker_mask, edge_mask, context,
        scorer,
        sample_M: int = 8,           # Branch數量
        rollout_K: int = 4,          # 每個候選的MC重複次數
        partial_steps: int = 10,     # ← 新參數: 部分rollout深度
        select: str = 'argmax',      # 選擇策略
        temperature: float = 1.0,     # Softmax溫度
        keep_frames: int = None
    ):
        """
        SVDD-MC (Monte-Carlo) with partial rollout for efficiency.
        
        At each reverse step s:
        1. Propose M candidates
        2. For each candidate, run K partial rollouts (only 'partial_steps' steps)
        3. Score the predicted x_0 from partial rollout
        4. Aggregate scores (mean over K) and select best candidate
        
        Args:
            sample_M: Number of candidate branches per step
            rollout_K: Number of MC rollouts per candidate for scoring
            partial_steps: How many denoising steps to run in each rollout
                        (0 = immediate prediction, higher = more accurate but slower)
            select: 'argmax' for greedy selection, 'softmax' for stochastic
            temperature: Temperature for softmax selection
            keep_frames: Number of intermediate frames to save
        
        Returns:
            chain: Trajectory of samples [keep_frames, B, N, F]
        """
        device = x.device
        B0, N = x.size(0), x.size(1)

        # Normalize and initialize
        x, h = self.normalize(x, h)
        xh = torch.cat([x, h], dim=2)
        z = self.sample_combined_position_feature_noise(B0, N, linker_mask)
        z = xh * fragment_mask + z * linker_mask

        # Setup frame storage
        if keep_frames is None:
            keep_frames = self.T
        else:
            assert 1 <= keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + z.size(), device=device)

        # Main reverse diffusion loop
        for s in reversed(range(0, self.T)):
            # Create time tensors
            s_arr = (torch.full((B0,), s, device=device, dtype=torch.long).float() / self.T).unsqueeze(1)
            t_arr = ((torch.full((B0,), s, device=device, dtype=torch.long) + 1).float() / self.T).unsqueeze(1)

            # Branch to M candidates
            z_rep = self._repeat_batch(z, sample_M)
            tiled = self._tile_graph_inputs(
                sample_M,
                node_mask=node_mask, fragment_mask=fragment_mask,
                linker_mask=linker_mask, edge_mask=edge_mask,
                context=context, s_arr=s_arr, t_arr=t_arr
            )

            # Propose M candidates at time s
            z_s_all = self.sample_p_zs_given_zt_only_linker(
                s=tiled['s_arr'], t=tiled['t_arr'], z_t=z_rep,
                node_mask=tiled['node_mask'], fragment_mask=tiled['fragment_mask'],
                linker_mask=tiled['linker_mask'], edge_mask=tiled['edge_mask'],
                context=tiled['context']
            )  # [B0*M, N, 3+F]

            # Score candidates using MC with partial rollout
            values = self._score_candidates_mc_partial(
                z_s=z_s_all,
                s_int=s,
                node_mask=tiled['node_mask'],
                fragment_mask=tiled['fragment_mask'],
                linker_mask=tiled['linker_mask'],
                edge_mask=tiled['edge_mask'],
                context=tiled['context'],
                scorer=scorer,
                rollout_K=rollout_K,
                partial_steps=partial_steps  # ← 使用部分rollout
            ).view(B0, sample_M)  # [B0, M]
            #print(f"[SVDD-MC] step {s:3d}: valid {100*self._last_valid_rate:5.1f}%")

            # NEW
            if (values <= -1e8).all():
                idx = torch.zeros(B0, dtype=torch.long, device=device)
            else:
                # Select best candidate per original sample
                if select == 'softmax':
                    probs = torch.softmax(values / max(temperature, 1e-8), dim=1)
                    idx = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:  # argmax
                    idx = torch.argmax(values, dim=1)

            # Gather chosen candidate
            z_s_all = z_s_all.view(B0, sample_M, N, self.n_dims + self.in_node_nf)
            z = z_s_all[torch.arange(B0, device=device), idx]

            # Save downsampled frame
            write_index = (s * keep_frames) // self.T
            chain[write_index] = self.unnormalize_z(z)

        # Final decode from z0
        x_final, h_final = self.sample_p_xh_given_z0_only_linker(
            z_0=z, node_mask=node_mask, fragment_mask=fragment_mask,
            linker_mask=linker_mask, edge_mask=edge_mask, context=context
        )
        chain[0] = torch.cat([x_final, h_final], dim=2)
        
        return chain

    def sample_z_T_only_linker(self, x, h, node_mask, fragment_mask, linker_mask):
        """
        Initialize z_T for linker-only generation:
        - keep fragments fixed at their (normalized) x,h values
        - initialize linker part as Gaussian noise
        """
        # x: [B,N,n_dims], h: [B,N,in_node_nf]
        xh = torch.cat([x, h], dim=2)  # [B,N,n_dims+in_node_nf]

        B, N, _ = xh.shape
        noise = self.sample_combined_position_feature_noise(B, N, linker_mask)  # masked Gaussian

        z_T = xh * fragment_mask + noise * linker_mask

        # be safe: respect node_mask overall (if present)
        if node_mask is not None:
            z_T = z_T * node_mask

        return z_T

    def sample_p_zs_given_zt_only_linker(self, s, t, z_t, node_mask, fragment_mask, linker_mask, edge_mask, context):
        """Samples from zs ~ p(zs | zt). Only used during sampling. Samples only linker features and coords"""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, z_t)
        sigma_s = self.sigma(gamma_s, target_tensor=z_t)
        sigma_t = self.sigma(gamma_t, target_tensor=z_t)

        # Neural net prediction.
        eps_hat = self.dynamics.forward(
            xh=z_t,
            t=t,
            node_mask=node_mask,
            linker_mask=linker_mask,
            context=context,
            edge_mask=edge_mask,
            training=False,
        )
        eps_hat = eps_hat * linker_mask

        # Compute mu for p(z_s | z_t)
        mu = z_t / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_hat

        # Compute sigma for p(z_s | z_t)
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample z_s given the parameters derived from zt
        z_s = self.sample_normal(mu, sigma, linker_mask)
        z_s = z_t * fragment_mask + z_s * linker_mask

        return z_s

    def sample_p_xh_given_z0_only_linker(self, z_0, node_mask, fragment_mask, linker_mask, edge_mask, context):
        """Samples x ~ p(x|z0). Samples only linker features and coords"""
        zeros = torch.zeros(size=(z_0.size(0), 1), device=z_0.device)
        gamma_0 = self.gamma(zeros)

        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
        eps_hat = self.dynamics.forward(
            t=zeros,
            xh=z_0,
            node_mask=node_mask,
            linker_mask=linker_mask,
            edge_mask=edge_mask,
            context=context,
            training=False,
        )
        eps_hat = eps_hat * linker_mask

        mu_x = self.compute_x_pred(eps_t=eps_hat, z_t=z_0, gamma_t=gamma_0)
        xh = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=linker_mask)
        xh = z_0 * fragment_mask + xh * linker_mask

        x, h = xh[:, :, :self.n_dims], xh[:, :, self.n_dims:]
        x, h = self.unnormalize(x, h)
        h = F.one_hot(torch.argmax(h, dim=2), self.in_node_nf) * node_mask

        return x, h

    def compute_x_pred(self, eps_t, z_t, gamma_t):
        """Computes x_pred, i.e. the most likely prediction of x."""
        sigma_t = self.sigma(gamma_t, target_tensor=eps_t)
        alpha_t = self.alpha(gamma_t, target_tensor=eps_t)
        x_pred = 1. / alpha_t * (z_t - sigma_t * eps_t)
        return x_pred

    def kl_prior(self, xh, mask):
        """
        Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).
        This is essentially a lot of work for something that is in practice negligible in the loss.
        However, you compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T
        ones = torch.ones((xh.size(0), 1), device=xh.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, xh)

        # Compute means
        mu_T = alpha_T * xh
        mu_T_x, mu_T_h = mu_T[:, :, :self.n_dims], mu_T[:, :, self.n_dims:]

        # Compute standard deviations (only batch axis for x-part, inflated for h-part)
        sigma_T_x = self.sigma(gamma_T, mu_T_x).view(-1)  # Remove inflate, only keep batch dimension for x-part
        sigma_T_h = self.sigma(gamma_T, mu_T_h)

        # Compute KL for h-part
        zeros, ones = torch.zeros_like(mu_T_h), torch.ones_like(sigma_T_h)
        kl_distance_h = self.gaussian_kl(mu_T_h, sigma_T_h, zeros, ones)

        # Compute KL for x-part
        zeros, ones = torch.zeros_like(mu_T_x), torch.ones_like(sigma_T_x)
        d = self.dimensionality(mask)
        kl_distance_x = self.gaussian_kl_for_dimension(mu_T_x, sigma_T_x, zeros, ones, d=d)

        return kl_distance_x + kl_distance_h

    def log_constant_of_p_x_given_z0(self, x, mask):
        batch_size = x.size(0)
        degrees_of_freedom_x = self.dimensionality(mask)
        zeros = torch.zeros((batch_size, 1), device=x.device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0)
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (- log_sigma_x - 0.5 * np.log(2 * np.pi))

    def log_p_xh_given_z0_without_constants(self, h, z_0, gamma_0, eps, eps_hat, mask, epsilon=1e-10):
        # Discrete properties are predicted directly from z_0
        z_h = z_0[:, :, self.n_dims:]

        # Take only part over x
        eps_x = eps[:, :, :self.n_dims]
        eps_hat_x = eps_hat[:, :, :self.n_dims]

        # Compute sigma_0 and rescale to the integer scale of the data
        sigma_0 = self.sigma(gamma_0, target_tensor=z_0) * self.norm_values[1]

        # Computes the error for the distribution N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'
        log_p_x_given_z_without_constants = -0.5 * self.sum_except_batch((eps_x - eps_hat_x) ** 2)

        # Categorical features
        # Compute delta indicator masks
        h = h * self.norm_values[1] + self.norm_biases[1]
        estimated_h = z_h * self.norm_values[1] + self.norm_biases[1]

        # Centered h_cat around 1, since onehot encoded
        centered_h = estimated_h - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        # N(mean=centered_h_cat, stdev=sigma_0_cat)
        log_p_h_proportional = torch.log(
            self.cdf_standard_gaussian((centered_h + 0.5) / sigma_0) -
            self.cdf_standard_gaussian((centered_h - 0.5) / sigma_0) +
            epsilon
        )

        # Normalize the distribution over the categories
        log_Z = torch.logsumexp(log_p_h_proportional, dim=2, keepdim=True)
        log_probabilities = log_p_h_proportional - log_Z

        # Select the log_prob of the current category using the onehot representation
        log_p_h_given_z = self.sum_except_batch(log_probabilities * h * mask)

        # Combine log probabilities for x and h
        log_p_xh_given_z = log_p_x_given_z_without_constants + log_p_h_given_z

        return log_p_xh_given_z

    def sample_combined_position_feature_noise(self, n_samples, n_nodes, mask):
        z_x = utils.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims),
            device=mask.device,
            node_mask=mask
        )
        z_h = utils.sample_gaussian_with_mask(
            size=(n_samples, n_nodes, self.in_node_nf),
            device=mask.device,
            node_mask=mask
        )
        z = torch.cat([z_x, z_h], dim=2)
        return z

    def sample_normal(self, mu, sigma, node_mask):
        """Samples from a Normal distribution."""
        eps = self.sample_combined_position_feature_noise(mu.size(0), mu.size(1), node_mask)
        return mu + sigma * eps

    def normalize(self, x, h):
        new_x = x / self.norm_values[0]
        new_h = (h.float() - self.norm_biases[1]) / self.norm_values[1]
        return new_x, new_h

    def unnormalize(self, x, h):
        new_x = x * self.norm_values[0]
        new_h = h * self.norm_values[1] + self.norm_biases[1]
        return new_x, new_h

    def unnormalize_z(self, z):
        assert z.size(2) == self.n_dims + self.in_node_nf
        x, h = z[:, :, :self.n_dims], z[:, :, self.n_dims:]
        x, h = self.unnormalize(x, h)
        return torch.cat([x, h], dim=2)

    def delta_log_px(self, mask):
        return -self.dimensionality(mask) * np.log(self.norm_values[0])

    def dimensionality(self, mask):
        return self.numbers_of_nodes(mask) * self.n_dims

    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_tensor: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -self.expm1(self.softplus(gamma_s) - self.softplus(gamma_t)),
            target_tensor
        )

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(alpha_t_given_s, target_tensor)
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    @staticmethod
    def numbers_of_nodes(mask):
        return torch.sum(mask.squeeze(2), dim=1)

    @staticmethod
    def inflate_batch_array(array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,),
        or possibly more empty axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    @staticmethod
    def sum_except_batch(x):
        return x.view(x.size(0), -1).sum(-1)

    @staticmethod
    def expm1(x: torch.Tensor) -> torch.Tensor:
        return torch.expm1(x)

    @staticmethod
    def softplus(x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)

    @staticmethod
    def cdf_standard_gaussian(x):
        return 0.5 * (1. + torch.erf(x / math.sqrt(2)))

    @staticmethod
    def gaussian_kl(q_mu, q_sigma, p_mu, p_sigma):
        """
        Computes the KL distance between two normal distributions.
        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
        kl = torch.log(p_sigma / q_sigma) + 0.5 * (q_sigma ** 2 + (q_mu - p_mu) ** 2) / (p_sigma ** 2) - 0.5
        return EDM.sum_except_batch(kl)

    @staticmethod
    def gaussian_kl_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d):
        """
        Computes the KL distance between two normal distributions taking the dimension into account.
        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
            d: dimension
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
        mu_norm_2 = EDM.sum_except_batch((q_mu - p_mu) ** 2)
        return d * torch.log(p_sigma / q_sigma) + 0.5 * (d * q_sigma ** 2 + mu_norm_2) / (p_sigma ** 2) - 0.5 * d

    def _repeatM(self, x, M):
        return x.repeat_interleave(M, dim=0)
    
    def _repeat_batch(self, ten, K):
        # repeat along batch dimension ( B --> B*K)
        if ten is None: return None
        return ten.repeat_interleave(K, dim=0)

    def _assert_B0_match(self, *tensors):
        b0 = None
        for t in tensors:
            if t is None: continue
            if b0 is None:
                b0 = t.shape[0]
            else:
                assert t.shape[0] == b0, f"Batch mismatch: {b0} vs {t.shape[0]} for tensor with shape {t.shape}"

    def _coerce_time_vec(self, s_like, batch_expected):
        """
        Ensure s_like is a [batch, 1] float tensor with the expected batch size.
        Never reuse any flattened edge-wise view.
        """
        if s_like is None:
            raise ValueError("s_like cannot be None in PM/MC scoring.")
        s = s_like
        if s.dim() == 1:
            s = s.unsqueeze(1)
        elif s.dim() > 2:
            # if someone accidentally broadcasted with edge dims, flatten back to batch
            s = s.reshape(-1)[:batch_expected].unsqueeze(1)

        if s.size(0) != batch_expected:
            if s.size(0) > batch_expected:
                s = s[:batch_expected]
            else:
                raise ValueError(f"Time sensor batch:{s.size(0)} < expected {batch_expected}, cannot recover")
        return s.float()

    def _bm_from_state(self, z_s):
        BM = z_s.size(0)
        return BM


    def _tile_graph_inputs(self, K, *, node_mask, fragment_mask, linker_mask, edge_mask, context, s_arr=None, t_arr=None):
        rep = lambda x: None if x is None else x.repeat_interleave(K, dim=0)
        out = {
            'node_mask':     rep(node_mask),
            'fragment_mask': rep(fragment_mask),
            'linker_mask':   rep(linker_mask),
            'edge_mask':     rep(edge_mask),
            'context':       rep(context),
        }

        #NEW
        # edge_mask is flattened as [B*N*N, 1], so repeat per-graph, not per-edge
        if edge_mask is None:
            out['edge_mask'] = None
        else:
            if node_mask is None:
                # fallback: old behavior
                out['edge_mask'] = edge_mask.repeat_interleave(K, dim=0)
            else:
                B = node_mask.size(0)
                N = node_mask.size(1)
                if edge_mask.dim() == 2 and edge_mask.size(0) == B * N * N:
                    edge_mask_b = edge_mask.view(B, N * N, -1)                    # [B, N*N, 1]
                    edge_mask_rep = edge_mask_b.repeat_interleave(K, dim=0)       # [B*K, N*N, 1]
                    out['edge_mask'] = edge_mask_rep.view(B * K * N * N, -1)      # [B*K*N*N, 1]
                else:
                    # already batch-wise, or a different layout
                    out['edge_mask'] = edge_mask.repeat_interleave(K, dim=0)
        #NEW
        if s_arr is not None: out['s_arr'] = rep(s_arr)
        if t_arr is not None: out['t_arr'] = rep(t_arr)
        return out


    def _build_linker_batch(self, x_unnorm, h_unnorm, linker_mask):
        # build scorer batch via the same adapter used in apply_sa_guidance
        current_state = {
            'positions': x_unnorm, # [B or B*K, N, 3]
            'one_hot': h_unnorm, # [B or B*K, N, F]
            'linker_mask': linker_mask
        }
        return DiffusionToScorerAdapter.extract_linker_batch(current_state)

    # ---------------------------------------------------------------------
    # Connectivity / validity gate used ONLY during SVDD guided selection.
    # This is intentionally lightweight (no RDKit) and uses a distance-based
    # covalent-bond heuristic to avoid selecting disconnected candidates.
    # ---------------------------------------------------------------------
    def _covalent_radii_by_type(self, atom_type_idx: torch.Tensor) -> torch.Tensor:
        """Map atom type indices (0..8) to covalent radii in Å.

        Index order must match the model's one-hot atom types:
        C,O,N,F,S,Cl,Br,I,P.
        """
        radii = torch.tensor(
            [0.76, 0.66, 0.71, 0.57, 1.05, 1.02, 1.20, 1.39, 1.07],
            device=atom_type_idx.device,
            dtype=torch.float32,
        )
        return radii[atom_type_idx]

    def _adjacency_from_distance(
        self,
        pos: torch.Tensor,          # [N, 3]
        atom_type: torch.Tensor,    # [N] long
        mask: torch.Tensor,         # [N] bool
        scale: float = 1.25,
        min_dist: float = 0.4,
    ) -> torch.Tensor:
        """Build a boolean adjacency matrix [N,N] using a covalent-radius threshold."""
        idx = torch.where(mask)[0]
        n = int(idx.numel())
        N = int(pos.size(0))
        if n <= 1:
            return torch.zeros((N, N), device=pos.device, dtype=torch.bool)

        pos_m = pos[idx]                 # [n, 3]
        t_m = atom_type[idx].long()      # [n]
        r = self._covalent_radii_by_type(t_m)  # [n]

        d = torch.cdist(pos_m, pos_m)                 # [n, n]
        thr = scale * (r[:, None] + r[None, :])       # [n, n]
        adj_small = (d < thr) & (d > min_dist)
        adj_small.fill_diagonal_(False)

        adj = torch.zeros((N, N), device=pos.device, dtype=torch.bool)
        adj[idx[:, None], idx[None, :]] = adj_small
        return adj
    
    @torch.no_grad()
    def check_connected_final(self, x, h, node_mask, fragment_mask, linker_mask) -> torch.Tensor:
        """
        Check connectivity for a batch of FINAL decoded samples.
        Returns a boolean tensor [B] where True means "connected + linker attaches both fragments (if 2 comps)".
        Inputs are expected UNNORMALIZED:
        x: [B,N,3], h: [B,N,F] one-hot or logits (final decoding is usually one-hot)
        """
        B, N, _ = x.shape

        # Robust atom type extraction (works for one-hot or logits)
        atom_type = torch.argmax(h, dim=2).long()  # [B,N]

        atom_mask_bool = node_mask.squeeze(-1).bool()
        frag_mask_bool = fragment_mask.squeeze(-1).bool()
        link_mask_bool = linker_mask.squeeze(-1).bool()

        out = torch.zeros((B,), device=x.device, dtype=torch.bool)
        for i in range(B):
            out[i] = self._connectivity_gate(
                pos=x[i],
                atom_type=atom_type[i],
                atom_mask=atom_mask_bool[i],
                fragment_mask=frag_mask_bool[i],
                linker_mask=link_mask_bool[i],
            )
        return out

    @staticmethod
    def _num_components_from_adj(adj: torch.Tensor, mask: torch.Tensor) -> tuple:
        """Return (num_components, component_id_per_node).

        component ids are -1 for masked-out nodes.
        Uses a small BFS on CPU for robustness and minimal deps.
        """
        adj_cpu = adj.detach().cpu().numpy()
        mask_cpu = mask.detach().cpu().numpy().astype(bool)

        N = mask_cpu.shape[0]
        comp = -1 * np.ones(N, dtype=np.int32)

        cid = 0
        for i in range(N):
            if not mask_cpu[i] or comp[i] != -1:
                continue
            stack = [i]
            comp[i] = cid
            while stack:
                u = stack.pop()
                nbrs = np.where(adj_cpu[u])[0]
                for v in nbrs:
                    if mask_cpu[v] and comp[v] == -1:
                        comp[v] = cid
                        stack.append(v)
            cid += 1

        return cid, torch.from_numpy(comp)

    def _connectivity_gate(
        self,
        pos: torch.Tensor,          # [N, 3]
        atom_type: torch.Tensor,    # [N] long
        atom_mask: torch.Tensor,    # [N] bool
        fragment_mask: torch.Tensor,# [N] bool
        linker_mask: torch.Tensor,  # [N] bool
    ) -> bool:
        """Minimal must-pass gate for SVDD candidates.

        Conditions:
          1) Whole molecule (all atoms) is a single connected component.
          2) Linker atoms (if any) form a single connected component.
          3) If fragments have >=2 components, linker touches at least two of them.
        """
        # (1) whole-molecule connectivity
        adj_all = self._adjacency_from_distance(pos, atom_type, atom_mask)
        n_comp_all, _ = self._num_components_from_adj(adj_all, atom_mask)
        if n_comp_all != 1:
            return False

        # (2) linker connectivity (only if linker exists)
        if linker_mask.any():
            adj_link = self._adjacency_from_distance(pos, atom_type, linker_mask)
            n_comp_link, _ = self._num_components_from_adj(adj_link, linker_mask)
            if n_comp_link != 1:
                return False

        # (3) linker must connect to both warheads if fragments are disconnected
        if fragment_mask.any() and linker_mask.any():
            adj_frag = self._adjacency_from_distance(pos, atom_type, fragment_mask)
            n_comp_frag, frag_comp = self._num_components_from_adj(adj_frag, fragment_mask)
            if n_comp_frag >= 2:
                frag_comp = frag_comp.to(pos.device)

                # Edges from linker -> fragment in full adjacency
                lf = adj_all & (linker_mask[:, None] & fragment_mask[None, :])
                if not lf.any():
                    return False

                touched = set()
                linker_nodes = torch.where(linker_mask)[0]
                for u in linker_nodes:
                    nbrs = torch.where(lf[u])[0]
                    for v in nbrs:
                        cid = int(frag_comp[v].item())
                        if cid >= 0:
                            touched.add(cid)
                if len(touched) < 2:
                    return False

        return True

    def build_rdkit_mol_from_xh(
        self,
        pos: torch.Tensor,        # [N, 3]
        one_hot: torch.Tensor,    # [N, 9]
        atom_mask: torch.Tensor,  # [N] or [N,1]
        bond_scale: float = 1.25,
        min_dist: float = 0.4,
        max_repairs: int = 12,
    ):
        """Build an RDKit Mol from predicted (x,h) without XYZ→bond-order heuristics.

        - Bonds are inferred from distances using the same covalent-radii threshold
        as SVDD connectivity gating (conservative: SINGLE bonds only).
        - Coordinates are stored as a single conformer.
        - If sanitization fails (typically due to an extra spurious bond), we
        iteratively remove the longest bond and retry.

        Returns
        -------
        mol : rdkit.Chem.Mol or None
        """
        if Chem is None:
            return None

        if atom_mask.dim() > 1:
            atom_mask = atom_mask.squeeze(-1)
        atom_mask = atom_mask.bool()

        keep = torch.where(atom_mask)[0]
        if keep.numel() == 0:
            return None

        # Atom type indices (0..8) must match one-hot ordering: C,O,N,F,S,Cl,Br,I,P
        atom_type = torch.argmax(one_hot, dim=-1).long()

        # Build adjacency on the FULL indexing, then restrict to kept atoms
        adj = self._adjacency_from_distance(
            pos=pos,
            atom_type=atom_type,
            mask=atom_mask,
            scale=bond_scale,
            min_dist=min_dist,
        )

        # Map old indices -> new indices (compact mol)
        old_to_new = {int(old.item()): i for i, old in enumerate(keep)}
        atomic_nums = [6, 8, 7, 9, 16, 17, 35, 53, 15]

        rw = Chem.RWMol()
        for old in keep.tolist():
            z = atomic_nums[int(atom_type[old].item())]
            rw.AddAtom(Chem.Atom(z))

        # Add SINGLE bonds for inferred adjacency
        bond_edges = []
        keep_set = set(keep.tolist())
        for a_old in keep.tolist():
            row = adj[a_old]
            for b_old in torch.where(row)[0].tolist():
                if b_old <= a_old:
                    continue
                if b_old not in keep_set:
                    continue
                a = old_to_new[a_old]
                b = old_to_new[b_old]
                rw.AddBond(a, b, Chem.BondType.SINGLE)
                pa = pos[a_old]
                pb = pos[b_old]
                length = float(torch.norm(pa - pb).item())
                bond_edges.append((a, b, length))

        mol = rw.GetMol()

        # Add conformer coordinates
        conf = Chem.Conformer(mol.GetNumAtoms())
        for old in keep.tolist():
            new = old_to_new[old]
            x, y, z = [float(v) for v in pos[old].tolist()]
            conf.SetAtomPosition(new, (x, y, z))
        mol.RemoveAllConformers()
        mol.AddConformer(conf, assignId=True)

        def _try_sanitize(m):
            try:
                Chem.SanitizeMol(m)
                return True
            except Exception:
                return False

        if _try_sanitize(mol):
            return mol

        # Conservative repair: remove the longest bond(s) iteratively.
        bond_edges_sorted = sorted(bond_edges, key=lambda x: x[2], reverse=True)

        for _ in range(max_repairs):
            if not bond_edges_sorted:
                break
            a, b, _len = bond_edges_sorted.pop(0)

            rw2 = Chem.RWMol(mol)
            try:
                rw2.RemoveBond(int(a), int(b))
            except Exception:
                continue
            mol2 = rw2.GetMol()

            # preserve coordinates
            conf_old = mol.GetConformer()
            conf2 = Chem.Conformer(mol2.GetNumAtoms())
            for i in range(mol2.GetNumAtoms()):
                p = conf_old.GetAtomPosition(i)
                conf2.SetAtomPosition(i, p)
            mol2.RemoveAllConformers()
            mol2.AddConformer(conf2, assignId=True)

            if _try_sanitize(mol2):
                return mol2
            mol = mol2

        return None

    def _score_candidates_pm(
        self,
        z_s, s_arr,
        node_mask, fragment_mask, linker_mask, edge_mask, context,
        scorer,
        w_rigid: float = 0.0,
    ):
        """
        Posterior-mean scoring: r( x0_hat(z_s, s) ).
        1) predict eps_hat at time s on the *duplicated* batch,
        2) compute x_hat via compute_x_pred,
        3) unnormalize,
        4) extract the linker, score with `scorer`, and return a flat [B0*M] vector.
        """
        # ensure time has expected batch = z_s.size(0) and is [BM, 1]-float
        BM = z_s.size(0) # B0 * M

        s_arr = self._coerce_time_vec(s_arr, batch_expected=BM)

        # Extra safety: all graph tensors must match [BM, ...]
        self._assert_B0_match(z_s, node_mask, fragment_mask, linker_mask,  context, s_arr)

        # predict eps at time s
        eps_hat = self.dynamics.forward(
            xh=z_s,
            t=s_arr,
            node_mask=node_mask,
            linker_mask=linker_mask,
            edge_mask=edge_mask,
            context=context,
            training=False,
        )
        eps_hat = eps_hat * linker_mask

        # posterior-mean x_hat at time s
        gamma_s = self.gamma(s_arr)
        xh_pred = self.compute_x_pred(eps_t=eps_hat, z_t=z_s, gamma_t=gamma_s)

        # split & unnormalize
        x_pred, h_pred = xh_pred[:, :, :self.n_dims], xh_pred[:, :, self.n_dims:]
        x_unnorm, h_unnorm = self.unnormalize(x_pred, h_pred)

        # ---- NEW: connectivity gate (SVDD only) ----
        # Discretize atom types from (unnormalized) features for the distance-bond heuristic.
        atom_type = torch.argmax(h_unnorm, dim=2).long()  # [BM, N]
        atom_mask_bool = node_mask.squeeze(-1).bool()
        frag_mask_bool = fragment_mask.squeeze(-1).bool()
        link_mask_bool = linker_mask.squeeze(-1).bool()

        valid = torch.zeros((BM,), device=z_s.device, dtype=torch.bool)
        for i in range(BM):
            valid[i] = self._connectivity_gate(
                pos=x_unnorm[i],
                atom_type=atom_type[i],
                atom_mask=atom_mask_bool[i],
                fragment_mask=frag_mask_bool[i],
                linker_mask=link_mask_bool[i],
            )
        self._last_valid_rate = valid.float().mean().item()

        # build scorer batch and score
        batch = self._build_linker_batch(x_unnorm, h_unnorm, linker_mask)
        scores = scorer(batch)                     # shape: [BM] or [BM, 1]
        if scores.dim() > 1:
            scores = scores.squeeze(-1)
        scores = scores.view(-1)

        # ---- NEW: rigidity proxy reward (RDKit-free) ----
        # Compute a cheap "rotatable-like bond" proxy on the *linker subgraph* using the
        # same covalent-radius adjacency heuristic as the connectivity gate.
        # We treat an edge as "rotatable-like" if:
        #   - it is a linker-linker inferred SINGLE bond, and
        #   - both endpoints are non-terminal within the linker subgraph (degree > 1).
        # The rigidity score is the negative of this count (higher = more rigid).
        try:
            w = float(w_rigid)
        except Exception:
            w = 0.0
        if w > 0.0:
            w = max(0.0, min(1.0, w))
            rigid_scores = torch.zeros((BM,), device=z_s.device, dtype=scores.dtype)

            # boolean masks [BM, N]
            link_mask_bool = linker_mask.squeeze(-1).bool()

            for i in range(BM):
                if not bool(valid[i].item()):
                    # keep 0; will be masked to -1e9 below anyway
                    continue
                if link_mask_bool[i].sum().item() <= 1:
                    # no meaningful linker connectivity => no rotatable-like edges
                    rigid_scores[i] = 0.0
                    continue

                adj_link = self._adjacency_from_distance(
                    pos=x_unnorm[i],
                    atom_type=atom_type[i],
                    mask=link_mask_bool[i],
                    scale=1.25,
                    min_dist=0.4,
                )  # [N,N] bool (only linker-linker region populated)

                deg = adj_link.sum(dim=1)  # [N]
                # Count qualifying edges i<j to avoid double counting
                # non-terminal endpoints in linker-only graph
                non_terminal = deg > 1
                qual = adj_link & (non_terminal[:, None] & non_terminal[None, :])
                # upper triangle count
                rot_like = torch.triu(qual, diagonal=1).sum().to(scores.dtype)

                rigid_scores[i] = -rot_like

            # Mix SA scorer output with rigidity score.
            # Keep backward-compatibility: w_rigid=0 -> original SA-only scores.
            scores = (1.0 - w) * scores + w * rigid_scores

        # Penalize invalid candidates so SVDD never selects them.
        scores = torch.where(valid, scores, scores.new_full(scores.shape, -1e9))
        return scores


    def _rollout_for_mc(self, z_start, s_int, steps, node_mask, fragment_mask, linker_mask, edge_mask, context):
        """
        Rool ahead by 'steps' reverse steps (or until s==0)
        Returns (z_eval, s_eval_int)
        """        
        z_eval = z_start
        s_eval = s_int.clone()
        for _ in range(max(0, steps)):
            # stop if hit zero
            if (s_eval <= 0).all(): break
            s_next_int = torch.clamp(s_eval -1, min=0)
            # s, t are in [0,1]
            s_arr = ( s_next_int.float() / self.T ).unsqueeze(1)
            t_arr = ( s_eval.float() / self.T ).unsqueeze(1)
            s_arr = self._coerce_time_vec(s_arr, z_eval.size(0))
            t_arr = self._coerce_time_vec(t_arr, z_eval.size(0))
            z_eval = self.sample_p_zs_given_zt_only_linker(
                s = s_arr,
                t = t_arr,
                z_t = z_eval,
                node_mask = node_mask,
                fragment_mask = fragment_mask,
                linker_mask = linker_mask,
                edge_mask = edge_mask,
                context = context
            )
            s_eval = s_next_int
        return z_eval, s_eval

    def _score_candidates_mc(
        self,
        z_s,
        s_int: int,   # integer timestep s (0..T-1)
        node_mask, fragment_mask, linker_mask, edge_mask, context,
        scorer,
        rollout_K: int = 4
    ):
        """
        Monte-Carlo scoring: for each z_s (at reverse step s), run K independent rollouts to 0,
        decode x,h at terminal, score with `scorer`, and return the *mean* score per candidate.

        Returns: tensor [B0*M]   (expected value over K rollouts)
        """
        BM = z_s.size(0)
        B0M = BM  # naming clarity

        # Repeat everything K times for vectorized rollouts: [B0*M*K, ...]
        z_rep = self._repeat_batch(z_s, rollout_K)
        tiled = self._tile_graph_inputs(
            rollout_K,
            node_mask=node_mask, fragment_mask=fragment_mask,
            linker_mask=linker_mask, edge_mask=edge_mask, context=context
        )

        # Rollout from s -> 0 with stochastic transitions (linker-only)
        x_term, h_term = self._mc_rollout_from(
            z_start=z_rep,
            s_start=s_int,
            node_mask=tiled['node_mask'],
            fragment_mask=tiled['fragment_mask'],
            linker_mask=tiled['linker_mask'],
            edge_mask=tiled['edge_mask'],
            context=tiled['context']
        )  # each of shape [B0*M*K, N, ...]

        # Score terminals
        x_unnorm, h_unnorm = self.unnormalize(x_term, h_term)
        batch = self._build_linker_batch(x_unnorm, h_unnorm, tiled['linker_mask'])
        scores = scorer(batch)                       # [B0*M*K] or [B0*M*K, 1]
        if scores.dim() > 1:
            scores = scores.squeeze(-1)

        # Average over K for each candidate
        scores = scores.view(B0M, rollout_K).mean(dim=1)  # [B0*M]
        return scores

    @torch.no_grad()
    def _mc_rollout_from(
        self,
        z_start,
        s_start: int,            # integer step where candidates live
        node_mask, fragment_mask, linker_mask, edge_mask, context
        ):
        """
        Vectorized stochastic rollout from current z at step s_start down to step 0 (inclusive),
        returning decoded (x, h) at terminal.

        Inputs are already tiled to [B0*M*K, ...].
        """
        device = z_start.device
        BMK, N, _ = z_start.shape
        z = z_start

        # Traverse s-1, s-2, ..., 0
        for s in reversed(range(0, s_start)):
            s_arr = (torch.full((BMK,), s, device=device, dtype=torch.long).float() / self.T).unsqueeze(1)
            t_arr = ((torch.full((BMK,), s, device=device, dtype=torch.long) + 1).float() / self.T).unsqueeze(1)

            z = self.sample_p_zs_given_zt_only_linker(
                s=s_arr, t=t_arr, z_t=z,
                node_mask=node_mask, fragment_mask=fragment_mask,
                linker_mask=linker_mask, edge_mask=edge_mask, context=context
            )

        # At s = 0: decode p(x,h | z0) (linker-only)
        x_final, h_final = self.sample_p_xh_given_z0_only_linker(
            z_0=z,
            node_mask=node_mask, fragment_mask=fragment_mask,
            linker_mask=linker_mask, edge_mask=edge_mask, context=context
        )
        return x_final, h_final

    def _score_candidates_immediate(
        self,
        z_s, s_int,
        node_mask, linker_mask, edge_mask, context,
        scorer
    ):
        """
        Score candidates by immediately predicting x_0 from current z_s.
        No rollout, just direct epsilon prediction and x_0 computation.
        
        This is the fastest option (partial_steps=0).
        
        Returns:
            scores: [B0*M]
        """
        BM = z_s.size(0)
        device = z_s.device
        
        # Create time tensor
        s_arr = (torch.full((BM,), s_int, device=device, dtype=torch.long).float() / self.T).unsqueeze(1)
        gamma_s = self.gamma(s_arr)
        
        # Predict epsilon at current timestep
        eps_hat = self.dynamics.forward(
            xh=z_s,
            t=s_arr,
            node_mask=node_mask,
            linker_mask=linker_mask,
            edge_mask=edge_mask,
            context=context,
            training=False
        )
        eps_hat = eps_hat * linker_mask
        
        # Compute x_0 prediction
        xh_pred = self.compute_x_pred(eps_t=eps_hat, z_t=z_s, gamma_t=gamma_s)
        
        # Split and unnormalize
        x_pred = xh_pred[:, :, :self.n_dims]
        h_pred = xh_pred[:, :, self.n_dims:]
        x_unnorm, h_unnorm = self.unnormalize(x_pred, h_pred)
        
        # Discretize features
        h_discrete = F.one_hot(
            torch.argmax(h_unnorm, dim=2),
            self.in_node_nf
        ).float() * node_mask
        
        # ---- NEW: connectivity gate (SVDD only) ----
        atom_type = torch.argmax(h_discrete, dim=2).long()
        atom_mask_bool = node_mask.squeeze(-1).bool()
        # In immediate mode we don't have fragment_mask here; treat all non-linker as fragment.
        link_mask_bool = linker_mask.squeeze(-1).bool()
        frag_mask_bool = (atom_mask_bool & ~link_mask_bool)

        valid = torch.zeros((BM,), device=z_s.device, dtype=torch.bool)
        for i in range(BM):
            valid[i] = self._connectivity_gate(
                pos=x_unnorm[i],
                atom_type=atom_type[i],
                atom_mask=atom_mask_bool[i],
                fragment_mask=frag_mask_bool[i],
                linker_mask=link_mask_bool[i],
            )

        # Build linker batch and score
        batch = self._build_linker_batch(x_unnorm, h_discrete, linker_mask)
        scores = scorer(batch)
        
        if scores.dim() > 1:
            scores = scores.squeeze(-1)
        
        scores = scores.view(-1)
        scores = torch.where(valid, scores, scores.new_full(scores.shape, -1e9))
        return scores  # [B0*M]

    def _score_candidates_mc_partial(
        self,
        z_s,
        s_int: int,
        node_mask, fragment_mask, linker_mask, edge_mask, context,
        scorer,
        rollout_K: int = 4,
        partial_steps: int = 10
        ):
        """
        Score candidates using MC estimation with PARTIAL rollout.
        
        For each candidate z_s:
        1. Replicate K times
        2. Run partial rollout (only 'partial_steps' steps, not all the way to 0)
        3. Predict x_0 from the partially denoised state
        4. Score the predicted x_0
        5. Return mean score over K rollouts
        
        Args:
            z_s: Candidate states [B0*M, N, 3+F]
            s_int: Current integer timestep (0 to T-1)
            partial_steps: Number of denoising steps to run
                        - If 0: immediate x_0 prediction from current z_s
                        - If > s_int: will rollout to step 0 (full rollout)
            rollout_K: Number of independent rollouts per candidate
            
        Returns:
            scores: Mean scores [B0*M]
        """
        BM = z_s.size(0)  # B0 * M
        
        # Special case: if partial_steps is 0, just predict x_0 immediately
        if partial_steps == 0:
            return self._score_candidates_immediate(
                z_s=z_s, s_int=s_int,
                node_mask=node_mask, linker_mask=linker_mask,
                edge_mask=edge_mask, context=context, scorer=scorer
            )
        
        # Replicate each candidate K times for MC estimation
        z_rep = self._repeat_batch(z_s, rollout_K)
        tiled = self._tile_graph_inputs(
            rollout_K,
            node_mask=node_mask, fragment_mask=fragment_mask,
            linker_mask=linker_mask, edge_mask=edge_mask, context=context
        )

        # Partial rollout: denoise for 'partial_steps' steps
        # NOTE: _mc_partial_rollout returns UNNORMALIZED positions and DISCRETE one-hot features.
        x_unnorm, h_discrete = self._mc_partial_rollout(
            z_start=z_rep,
            s_start=s_int,
            steps=min(partial_steps, s_int),  # Can't rollout more than s_int steps
            node_mask=tiled['node_mask'],
            fragment_mask=tiled['fragment_mask'],
            linker_mask=tiled['linker_mask'],
            edge_mask=tiled['edge_mask'],
            context=tiled['context']
        )  # Returns (x_unnorm, h_discrete)

        # ---- NEW: connectivity gate (SVDD only) ----
        atom_type = torch.argmax(h_discrete, dim=2).long()
        atom_mask_bool = tiled['node_mask'].squeeze(-1).bool()
        frag_mask_bool = tiled['fragment_mask'].squeeze(-1).bool()
        link_mask_bool = tiled['linker_mask'].squeeze(-1).bool()

        valid = torch.zeros((x_unnorm.size(0),), device=x_unnorm.device, dtype=torch.bool)
        for i in range(x_unnorm.size(0)):
            valid[i] = self._connectivity_gate(
                pos=x_unnorm[i],
                atom_type=atom_type[i],
                atom_mask=atom_mask_bool[i],
                fragment_mask=frag_mask_bool[i],
                linker_mask=link_mask_bool[i],
            )
        self._last_valid_rate = valid.float().mean().item()

        # Score
        batch = self._build_linker_batch(x_unnorm, h_discrete, tiled['linker_mask'])
        scores = scorer(batch)  # [B0*M*K] or [B0*M*K, 1]
        
        if scores.dim() > 1:
            scores = scores.squeeze(-1)
        
        scores = scores.view(-1)
        scores = torch.where(valid, scores, scores.new_full(scores.shape, -1e9))
        # Average over K rollouts for each candidate
        scores = scores.view(BM, rollout_K).mean(dim=1)  # [B0*M]
        
        return scores

    @torch.no_grad()
    def _mc_partial_rollout(
        self,
        z_start,
        s_start: int,
        steps: int,
        node_mask, fragment_mask, linker_mask, edge_mask, context
    ):
        """
        Perform partial rollout: denoise for 'steps' steps, then predict x_0.
        
        Process:
        1. Start from z at timestep s_start
        2. Denoise for 'steps' iterations: s_start-1, s_start-2, ..., s_start-steps
        3. From the resulting z at timestep s_end = s_start - steps:
        - Predict epsilon
        - Compute x_0 prediction using the diffusion posterior mean formula
        
        Args:
            z_start: Starting latent [B*M*K, N, 3+F]
            s_start: Starting timestep (integer, 0 to T-1)
            steps: Number of denoising steps to take
            
        Returns:
            x_unnorm, h_discrete: Predicted clean samples in real units
            - x_unnorm: [B*M*K, N, 3]
            - h_discrete: [B*M*K, N, F] one-hot (masked)
        """
        device = z_start.device
        BMK = z_start.size(0)
        z = z_start
        
        # Calculate ending timestep (can't go below 0)
        s_end = max(0, s_start - steps)
        
        # Denoise from s_start-1 down to s_end
        for s in reversed(range(s_end, s_start)):
            s_arr = (torch.full((BMK,), s, device=device, dtype=torch.long).float() / self.T).unsqueeze(1)
            t_arr = ((torch.full((BMK,), s, device=device, dtype=torch.long) + 1).float() / self.T).unsqueeze(1)
            
            z = self.sample_p_zs_given_zt_only_linker(
                s=s_arr, t=t_arr, z_t=z,
                node_mask=node_mask,
                fragment_mask=fragment_mask,
                linker_mask=linker_mask,
                edge_mask=edge_mask,
                context=context
            )
        
        # Now z is at timestep s_end. Predict x_0 from z at s_end.
        s_end_arr = (torch.full((BMK,), s_end, device=device, dtype=torch.long).float() / self.T).unsqueeze(1)
        gamma_s = self.gamma(s_end_arr)
        
        # Predict noise
        eps_hat = self.dynamics.forward(
            xh=z,
            t=s_end_arr,
            node_mask=node_mask,
            linker_mask=linker_mask,
            edge_mask=edge_mask,
            context=context,
            training=False
        )
        eps_hat = eps_hat * linker_mask
        
        # Compute x_0 prediction using posterior mean formula
        # x_0 = (z_t - sigma_t * eps) / alpha_t
        xh_pred = self.compute_x_pred(eps_t=eps_hat, z_t=z, gamma_t=gamma_s)
        
        # Split into position and features
        x_pred = xh_pred[:, :, :self.n_dims]
        h_pred = xh_pred[:, :, self.n_dims:]

        # Unnormalize positions; discretize features (as in final decoding)
        x_unnorm, h_unnorm = self.unnormalize(x_pred, h_pred)
        h_discrete = F.one_hot(
            torch.argmax(h_unnorm, dim=2),
            self.in_node_nf
        ).float() * node_mask

        return x_unnorm, h_discrete

    def _log_connectivity_diagnostics(self,
        timestep, z_s_all, values, selected_idx, B0, M, 
        linker_mask, connectivity_checker, diagnostics_data
    ):
        """Log connectivity information for diagnostics."""
        # Split z_s_all back to x and h
        x_pred = z_s_all[:, :, :self.n_dims]
        h_pred = z_s_all[:, :, self.n_dims:]
        
        # Unnormalize
        x_unnorm, h_unnorm = self.unnormalize(x_pred, h_pred)
        
        # Discretize features for connectivity check
        h_discrete = F.one_hot(
            torch.argmax(h_unnorm, dim=2),
            self.in_node_nf
        ).float()
        
        # Check connectivity for first molecule in batch
        connectivity_flags = []
        scores_list = []
        
        for m in range(M):
            mol_pos = x_unnorm[m]  # [N, 3]
            mol_h = h_discrete[m]   # [N, F]
            mol_linker_mask = linker_mask[m]  # [N, 1]
            
            # Extract only linker atoms
            linker_atoms = mol_linker_mask.squeeze(-1).bool()
            if linker_atoms.sum() == 0:
                connectivity_flags.append(True)  # No linker = connected
                scores_list.append(values[0, m].item())
                continue
            
            linker_pos = mol_pos[linker_atoms]
            linker_h = mol_h[linker_atoms]
            
            # Check connectivity
            if connectivity_checker is not None:
                is_connected = connectivity_checker._is_connected_distance_based(linker_pos, linker_h)
            else:
                is_connected = True  # Assume connected if no checker
            
            connectivity_flags.append(is_connected)
            scores_list.append(values[0, m].item())
        
        # Count statistics
        n_connected = sum(connectivity_flags)
        selected_connected = connectivity_flags[selected_idx[0].item()]
        
        # Store data
        diagnostics_data.append({
            'timestep': timestep,
            'n_connected': n_connected,
            'n_total': M,
            'connectivity_flags': connectivity_flags,
            'scores': scores_list,
            'selected_idx': selected_idx[0].item(),
            'selected_connected': selected_connected,
            'selected_score': scores_list[selected_idx[0].item()]
        })
        
        # Print immediate feedback
        conn_str = "".join(["✓" if c else "✗" for c in connectivity_flags])
        sel_marker = " " * selected_idx[0].item() + "^"
        status = "✓" if selected_connected else "✗"
        
        print(f"  Step {timestep:3d}: [{conn_str}] {n_connected}/{M} connected, Selected: {status} (score: {scores_list[selected_idx[0].item()]:.2f})")
        print(f"            {sel_marker}")

    def _print_diagnostics_summary(self, diagnostics_data, branch_start, branch_end):
        """Print summary of diagnostics."""
        print(f"\n{'='*80}")
        print("CONNECTIVITY DIAGNOSTICS SUMMARY")
        print(f"{'='*80}")
        
        timesteps = [d['timestep'] for d in diagnostics_data]
        connectivity_rates = [d['n_connected'] / d['n_total'] for d in diagnostics_data]
        selected_connected = [d['selected_connected'] for d in diagnostics_data]
        
        print(f"\nTotal logged steps: {len(diagnostics_data)}")
        print(f"Branching range: [{branch_start}, {branch_end}]")
        print(f"\nConnectivity Statistics:")
        print(f"  Average connectivity rate: {np.mean(connectivity_rates):.1%}")
        print(f"  Selected connected: {sum(selected_connected)}/{len(selected_connected)} ({sum(selected_connected)/len(selected_connected):.1%})")
        print(f"  Selected disconnected: {len(selected_connected) - sum(selected_connected)}/{len(selected_connected)}")
        
        # Analyze by phase
        n_steps = len(diagnostics_data)
        if n_steps >= 3:
            early_idx = n_steps // 3
            late_idx = 2 * n_steps // 3
            
            early_phase = diagnostics_data[:early_idx]
            mid_phase = diagnostics_data[early_idx:late_idx]
            late_phase = diagnostics_data[late_idx:]
            
            print(f"\nBy Phase:")
            for phase_name, phase_data in [("Early", early_phase), ("Middle", mid_phase), ("Late", late_phase)]:
                if phase_data:
                    phase_conn_rate = np.mean([d['n_connected']/d['n_total'] for d in phase_data])
                    phase_sel_conn = sum([d['selected_connected'] for d in phase_data])
                    print(f"  {phase_name:10s}: Conn rate {phase_conn_rate:.1%}, Selected connected {phase_sel_conn}/{len(phase_data)}")
        
        # Find optimal timestep range (highest connectivity rate)
        best_timestep = max(diagnostics_data, key=lambda d: d['n_connected'] / d['n_total'])
        print(f"\nBest timestep: {best_timestep['timestep']} with {best_timestep['n_connected']}/{best_timestep['n_total']} connected")
        
        # Suggest optimal timestep range (highest connectivity rate)
        high_conn_steps = [d['timestep'] for d in diagnostics_data if (d['n_connected'] / d['n_total']) >= 0.75]
        if high_conn_steps:
            suggested_start = max(high_conn_steps)
            suggested_end = min(high_conn_steps)
            print(f"\nSuggested branching range: [{suggested_end}, {suggested_start}]")
            print(f"  (Timesteps where ≥75% candidates are connected)")
        
        print(f"{'='*80}\n")


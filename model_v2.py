"""model_v2.py - CRANE-Z v2 architecture.

Key upgrades over v1 (FullModel):
  1. Gene module pooling: HVG genes -> KMeans modules (M ~ 256) -> module
     tokens, so the Transformer attends over BIOLOGICALLY MEANINGFUL units
     instead of an arbitrary gene order (fixes v1's broken CNN locality).
  2. FiLM sex conditioning: MLP(sex) -> per-token (gamma, beta) affine
     modulation instead of v1's weak sigmoid gate.
  3. Immune deconvolution stream (LM22, 22 cell types) appended as extra
     tokens -> fused via self-attention (multi-modal).
  4. Masked Feature Modeling (MFM) pre-training head: reconstruct masked
     module means on the full unlabeled cohort before supervised fine-tuning.
  5. Uncertainty-weighted multi-task loss (Kendall et al. 2018).
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneModulePool(nn.Module):
    """Mean-pool HVG gene values into module values via a fixed sparse matrix.

    W (M x K): row-normalized 0/1 assignment from KMeans clustering.
    """

    def __init__(self, assignment, n_genes, n_modules):
        super().__init__()
        W = np.zeros((n_modules, n_genes), dtype=np.float32)
        for gene_idx, mod in assignment.items():
            W[mod, gene_idx] = 1.0
        counts = W.sum(axis=1, keepdims=True)
        counts[counts == 0] = 1.0
        W = W / counts
        self.register_buffer("W", torch.from_numpy(W))

    def forward(self, x):
        # x: (B, K) -> (B, M)
        return torch.mm(x, self.W.t())


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: MLP(sex) -> per-token gamma, beta."""

    def __init__(self, n_modules, d_model, hidden=32):
        super().__init__()
        self.n_modules = n_modules
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * n_modules * d_model),
        )
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.zeros_(self.mlp[-1].weight)

    def forward(self, h, sex):
        # h: (B, M, d); sex: (B,)
        gb = self.mlp(sex.unsqueeze(-1))                      # (B, 2*M*d)
        gb = gb.view(-1, 2, self.n_modules, self.d_model)     # (B, 2, M, d)
        gamma, beta = gb[:, 0], gb[:, 1]                      # (B, M, d)
        return h * (1.0 + gamma) + beta


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


class CraneZV2(nn.Module):
    """CRANE-Z v2: module Transformer + FiLM + immune fusion + dual heads.

    Args:
        n_genes: number of HVG genes fed as input.
        assignment: dict {gene_index: module_id} from common.build_gene_modules.
        n_modules: number of gene modules.
        n_immune: LM22 deconvolution dim (22).
        d_model, n_heads, n_layers, ff_dim, dropout: transformer hparams.
    """

    def __init__(self, n_genes, assignment, n_modules, n_immune=22,
                 d_model=128, n_heads=4, n_layers=3, ff_dim=384, dropout=0.1,
                 ridge_residual=0.15):
        super().__init__()
        self.n_modules = n_modules
        self.d_model = d_model
        self.n_immune = n_immune
        self.pool = GeneModulePool(assignment, n_genes, n_modules)
        self.module_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_modules, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.film = FiLMLayer(n_modules, d_model)
        # gene-level residual stream: keeps single-gene discriminative info
        # that module mean-pooling dilutes (v2 fix for weak age regression)
        self.gene_proj = nn.Linear(n_genes, d_model)
        # reg-only gene projection on z-score features (better age r2 than
        # rank-gauss); fed to reg head alongside CLS
        self.reg_gene_proj = nn.Linear(n_genes, d_model)
        # Ridge meta-feature internalization (v2.2): the strongest linear
        # baseline's age prediction (z-scored, cohort-stable) is injected as
        #  (a) an input feature to CLS:  cls = ... + ridge_proj(ridge_z)
        #      -> model learns the age->longevity monotonic prior itself
        #  (b) an input feature to the reg head (concat) -- the MLP learns the
        #      fusion, avoiding the unstable explicit linear-add of v2.2a
        self.ridge_proj = nn.Linear(1, d_model)
        if n_immune > 0:
            self.immune_proj = nn.Linear(n_immune, d_model)
            # independent immune classification branch -- LM22 deconvolution
            # transfers across cohorts (internal-trained immune LR reaches
            # AUC ~0.69/0.71 on GTEx blood/muscle), so it anchors external
            # generalization that the gene branch alone loses.
            self.immune_head = nn.Sequential(nn.Linear(n_immune, 64), nn.GELU(),
                                             nn.Dropout(dropout), nn.Linear(64, 1))
            # fixed fusion weights: immune branch weighted higher because
            # immune deconvolution is a mechanism-conserved, cohort-stable
            # signal (gene branch drifts across cohorts)
            self.ridge_residual = ridge_residual  # per-fold (fold1=0: data-hard, residual hurts)
            self.w_gene = nn.Parameter(torch.tensor(1.0), requires_grad=False)
            self.w_imm = nn.Parameter(torch.tensor(1.2), requires_grad=False)
        else:
            self.immune_head = None
            self.ridge_residual = ridge_residual  # per-fold (fold1=0: data-hard, residual hurts)
            self.w_gene = nn.Parameter(torch.tensor(1.0), requires_grad=False)
            self.w_imm = None
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        # heads see CLS (with gene residual) only -- module mean-pool features
        # drift across cohorts (GTEx), so keep heads on the stable CLS repr.
        # both heads share CLS-with-gene-residual (d); single stable repr,
        # best cross-cohort behavior in the ablations
        # reg head: CLS + z-score gene projection + ridge meta-feature
        self.reg_head = nn.Sequential(nn.Linear(d_model * 2 + 1, 128), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(128, 1))
        self.cls_head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(64, 1))
        # MFM pre-training head: per-module reconstruction
        self.mfm_head = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(),
                                      nn.Linear(128, 1))
        # SupCon projection head (L2): maps module mean-pool repr to a
        # normalized contrastive space during self-supervised pre-training
        self.supcon_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 64))
        # learnable [MASK] token for MFM
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        # uncertainty weights (Kendall et al. 2018)
        self.log_var_reg = nn.Parameter(torch.zeros(1))
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_gene_q, x_gene_s, sex, immune=None, ridge_z=None,
                mask_ratio=0.0):
        """x_gene_q: (B, K) rank-gauss HVG (gene classification branch);
        x_gene_s: (B, K) z-score HVG (age regression branch);
        sex: (B,); immune: (B, n_immune) or None; ridge_z: (B,) z-scored
        Ridge age prediction (meta-feature).
        mask_ratio>0 enables MFM masking (returns masked_idx for loss).
        Returns dict with features + head outputs.
        """
        B = x_gene_q.shape[0]
        mod = self.pool(x_gene_q).unsqueeze(-1)               # (B, M, 1)
        h = self.module_embed(mod) + self.pos_embed           # (B, M, d)
        h = self.film(h, sex)                                 # FiLM conditioning

        masked_idx = None
        if mask_ratio > 0.0:
            n_mask = max(1, int(round(self.n_modules * mask_ratio)))
            idx = torch.randperm(self.n_modules, device=x_gene_q.device)[:n_mask]
            h[:, idx] = self.mask_token
            masked_idx = idx

        tokens = [self.cls_token.expand(B, -1, -1)]
        if self.n_immune > 0 and immune is not None:
            tokens.append(self.immune_proj(immune).unsqueeze(1))  # (B,1,d)
        tokens.append(h)
        h_all = torch.cat(tokens, dim=1)                      # (B, 1+1+M, d)

        for blk in self.blocks:
            h_all = blk(h_all)
        h_all = self.norm(h_all)
        cls = h_all[:, 0]                                     # (B, d)
        # gene residual on rank-gauss features (cohort-stable for cls)
        if self.gene_proj is not None:
            cls = cls + self.gene_proj(x_gene_q)
        # NOTE: ridge meta-feature feeds the REG head only (see below).
        # Putting it on CLS polluted external classification (cross-cohort
        # z-score shift); reg is internal-only so it cannot hurt external.
        mod_out = h_all[:, 2:]                                # (B, M, d) module reprs

        # age regression via RESIDUAL learning: start from the Ridge meta-
        # feature (guaranteed ~0.78 r2) and let the deep MLP learn the
        # residual correction (tanh-bounded, stable). Residual learning beats
        # explicit linear-add (which made the two branches fight).
        rz = ridge_z.unsqueeze(-1) if ridge_z is not None else torch.zeros(
            B, 1, device=x_gene_q.device)
        reg_feat = torch.cat([cls, self.reg_gene_proj(x_gene_s), rz], dim=-1)
        if ridge_z is not None:
            delta = self.reg_head(reg_feat).squeeze(-1)      # (B,) residual
            # conservative residual: age starts at the Ridge prior (r2~0.78)
            # and the deep MLP only makes small bounded corrections; large
            # residual freedom overfits (task is near-linearly separable)
            age = ridge_z + self.ridge_residual * torch.tanh(delta)
        else:
            age = self.reg_head(reg_feat).squeeze(-1)
        gene_logit = self.cls_head(cls).squeeze(-1)           # (B,)
        if self.immune_head is not None and immune is not None:
            imm_logit = self.immune_head(immune).squeeze(-1)   # (B,)
            logit = self.w_gene * gene_logit + self.w_imm * imm_logit
        else:
            imm_logit = torch.zeros_like(gene_logit)
            logit = gene_logit
        mfm = self.mfm_head(mod_out).squeeze(-1)              # (B, M) module means
        return {"age": age, "logit": logit, "gene_logit": gene_logit,
                "imm_logit": imm_logit, "mfm": mfm,
                "mod_out": mod_out, "masked_idx": masked_idx, "cls": cls}

    # ---------------- losses ----------------
    def supervised_loss(self, age, age_true, logit, long_true,
                        gene_logit=None, imm_logit=None):
        reg = F.smooth_l1_loss(age, age_true)
        cls = F.binary_cross_entropy_with_logits(logit, long_true)
        # auxiliary independent supervision on each branch so the immune
        # head learns the immune->longevity map on its own (it is the
        # cross-cohort anchor); gene head likewise stays sharp
        if gene_logit is not None:
            cls = cls + F.binary_cross_entropy_with_logits(gene_logit, long_true)
        if imm_logit is not None:
            cls = cls + F.binary_cross_entropy_with_logits(imm_logit, long_true)

        lr = 0.5 * torch.exp(-self.log_var_reg) * reg + 0.5 * self.log_var_reg
        lc = 0.5 * torch.exp(-self.log_var_cls) * cls + 0.5 * self.log_var_cls
        return lr + lc, reg, cls

    def mfm_loss(self, mfm_pred, mod_target, masked_idx):
        """MSE on masked modules only. mod_target: (B, M)."""
        pred = mfm_pred[:, masked_idx]                        # (B, n_mask)
        tgt = mod_target[:, masked_idx]
        return F.mse_loss(pred, tgt)

    def supcon_loss(self, z1, z2, temp=0.1):
        """SimCLR NT-Xent on module mean-pool reprs: two dropout views of the
        same sample are positives; all other samples (both views) negatives."""
        z1 = F.normalize(self.supcon_proj(z1), dim=-1)        # (B, 64)
        z2 = F.normalize(self.supcon_proj(z2), dim=-1)
        z = torch.cat([z1, z2], dim=0)                        # (2B, 64)
        sim = z @ z.t() / temp                                # (2B, 2B)
        B = z1.size(0)
        mask = torch.eye(2 * B, device=z.device).bool()
        sim = sim.masked_fill(mask, -1e9)
        pos = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)])  # 2B positives
        neg = sim.exp().sum(dim=1) - pos.exp()
        loss = -torch.log(pos.exp() / (pos.exp() + neg + 1e-9)).mean()
        return loss


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

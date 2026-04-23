"""
Phase LCE: Understanding Why SMI-TED Pretraining Helps Predict
Battery Electrolyte Performance (LCE = Logarithmic Coulombic Efficiency).

Experiments:
  L1: Statistical Significance of the Paper's Claims
  L2a: Electrolyte Molecule Embedding Analysis (pretrained vs random)
  L2b: Sample Efficiency on ESOL (pretrained vs random at varying N)
  L3a: Cross-Molecule Attention Patterns
  L3b: Embedding Additivity Test
  L3c: Permutation Sensitivity
  L4a: Frozen Embeddings Performance on ESOL
  L4b: CKA Between Pretrained and Random

Adapted from:
    https://github.com/ChicagoHAI/interp_Molmformer

SMI-TED migration changes:
    1. Model loading: load_smi_ted() instead of HuggingFace AutoModel
    2. Token-to-atom mapping: SMI-TED regex + canonicalization
    3. Hook paths: model.encoder.tok_emb and
       model.encoder.blocks.layers[i]
    4. Forward pass: model.tokenize() + model.encoder()
    5. Molecule embedding: autoencoder latent z instead of mean pooling
    6. Random model: created via Smi_ted + MoLEncoder + MoLDecoder

Bug fixes from original MOLFormer code:
    1. L4a: evaluate FC head on test set, not validation set
    2. L3b/L3c: consistent separator format with L3a (no spaces)
"""

import os
import sys
import json
import random
import warnings
import logging
from pathlib import Path
from datetime import datetime
from itertools import permutations
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import regex as re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import wilcoxon
from scipy.spatial.distance import cosine as cosine_dist
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

SEED = 42
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

SMI_TED_PATH = ('/Users/xuzetong/projects/materials/models'
                '/smi_ted/inference')
CKPT_FILENAME = 'smi-ted-Light_40.pt'

ESOL_TRAIN_PATH = ('/Users/xuzetong/projects/materials/models'
                   '/smi_ted/finetune/smi_ted_light/esol')
ESOL_PATH = ('/Users/xuzetong/projects/materials/models'
             '/smi_ted/data/esol.csv')

RESULTS_DIR = Path('./results/smi_ted/lce')
FIGURES_DIR = Path('./results/smi_ted/lce/figures')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LAYERS_TO_ANALYZE = [0, 4, 6, 8, 12]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()

# ─────────────────────────────────────────────────────────────────────
# Table 1 data from Soares et al. NeurIPS 2023
# ─────────────────────────────────────────────────────────────────────

TABLE1_EXPERIMENTAL = np.array([
    1.094, 1.384, 1.468, 1.710, 1.832,
    2.104, 2.274, 1.071, 1.166, 1.335,
    1.129, 1.501, 1.663
])

TABLE1_MOLFORMER = np.array([
    1.198, 1.428, 1.340, 1.845, 1.763,
    1.816, 1.809, 1.058, 1.109, 1.727,
    0.982, 1.735, 1.565
])

TABLE1_MULTIMODAL = np.array([
    1.028, 1.267, 1.336, 1.823, 1.816,
    1.841, 1.897, 0.979, 0.971, 1.554,
    0.810, 1.599, 1.492
])

# ─────────────────────────────────────────────────────────────────────
# Electrolyte molecules
# ─────────────────────────────────────────────────────────────────────

ELECTROLYTE_MOLECULES = {
    'EC':     'C1COC(=O)O1',
    'DMC':    'COC(=O)OC',
    'DEC':    'CCOC(=O)OCC',
    'EMC':    'CCOC(=O)OC',
    'PC':     'CC1COC(=O)O1',
    'FEC':    'C1OC(=O)O[C@@H]1F',
    'VC':     'C1=COC(=O)O1',
    'LiPF6':  '[Li+].F[P-](F)(F)(F)(F)F',
    'LiBF4':  '[Li+].F[B-](F)(F)F',
    'LiTFSI': '[Li+].C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F',
}

ELECTROLYTE_GROUPS = {
    'linear_carbonates': ['DMC', 'DEC', 'EMC'],
    'cyclic_carbonates': ['EC', 'PC', 'FEC', 'VC'],
    'li_salts': ['LiPF6', 'LiBF4', 'LiTFSI'],
}


# ─────────────────────────────────────────────────────────────────────
# Model Loading
# Migration: load_smi_ted() instead of HuggingFace AutoModel
# ─────────────────────────────────────────────────────────────────────

def load_pretrained_model():
    """Load SMI-TED pretrained model."""
    sys.path.insert(0, SMI_TED_PATH)
    from smi_ted_light.load import load_smi_ted

    print("Loading pretrained SMI-TED...")
    model = load_smi_ted(
        folder=os.path.join(SMI_TED_PATH, 'smi_ted_light'),
        ckpt_filename=CKPT_FILENAME
    )
    model.encoder.eval()
    if torch.cuda.is_available():
        model.encoder.cuda()
    return model


def load_random_model(pretrained_model):
    """Create randomly initialized SMI-TED with same architecture."""
    sys.path.insert(0, SMI_TED_PATH)
    from smi_ted_light.load import Smi_ted, MoLEncoder, MoLDecoder

    print("Creating randomly initialized SMI-TED...")
    random_model = Smi_ted(pretrained_model.tokenizer)
    random_model.config = pretrained_model.config
    random_model.max_len = pretrained_model.max_len
    random_model.n_embd = pretrained_model.n_embd
    random_model.encoder = MoLEncoder(
        pretrained_model.config,
        len(pretrained_model.tokenizer.vocab)
    )
    random_model.decoder = MoLDecoder(
        len(pretrained_model.tokenizer.vocab),
        pretrained_model.max_len,
        pretrained_model.n_embd
    )
    random_model.encoder.eval()
    if torch.cuda.is_available():
        random_model.encoder.cuda()
    return random_model


# ─────────────────────────────────────────────────────────────────────
# Hook Mechanism
# Migration: hook paths updated for SMI-TED structure
# ─────────────────────────────────────────────────────────────────────

def register_hooks(model):
    hooks = []
    intermediate = {}

    def make_hook(idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                intermediate[idx] = output[0].detach().cpu()
            else:
                intermediate[idx] = output.detach().cpu()
        return hook_fn

    hooks.append(
        model.encoder.tok_emb.register_forward_hook(make_hook(0))
    )
    for i, layer in enumerate(model.encoder.blocks.layers):
        hooks.append(
            layer.register_forward_hook(make_hook(i + 1))
        )
    return hooks, intermediate


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ─────────────────────────────────────────────────────────────────────
# Embedding Extraction
# Migration: model.tokenize() + autoencoder latent z
# ─────────────────────────────────────────────────────────────────────

def get_mean_pooled_embedding(model, smiles,
                               hooks_intermediate=None):
    """
    Get mean-pooled embedding for a SMILES string.

    Migration change:
    - Uses model.tokenize() instead of HuggingFace tokenizer
    - Mean pools over non-padding tokens using mask
    - Also returns layer-wise embeddings if hooks provided
    """
    idx, mask = model.tokenize(smiles)

    with torch.no_grad():
        _ = model.encoder(idx, mask)

    # Mean pool using mask
    if torch.cuda.is_available():
        mask_cpu = mask.cpu().float()
    else:
        mask_cpu = mask.float()

    # Get final layer embedding from last_hidden_state equivalent
    # In SMI-TED, we use the autoencoder latent z as molecule embedding
    max_len = model.max_len
    n_embd = model.n_embd

    with torch.no_grad():
        token_embeddings = model.encoder(idx, mask)
        z = model.decoder.autoencoder.encoder(
            token_embeddings.view(-1, max_len * n_embd)
        )

    result = {'final': z.cpu().numpy()[0]}

    if hooks_intermediate is not None:
        for layer_idx, hs in hooks_intermediate.items():
            layer_hidden = hs[0] if hs.dim() == 3 else hs
            # Mean pool with mask
            mask_expanded = mask_cpu[0].unsqueeze(-1).expand(
                layer_hidden.shape)
            pooled = (layer_hidden * mask_expanded).sum(0) / \
                     mask_cpu[0].sum()
            result[layer_idx] = pooled.numpy()

    return result


def get_embeddings_for_molecules(model, molecules_dict,
                                  layers=None):
    """Extract embeddings for {name: SMILES} dict at multiple layers."""
    if layers is None:
        layers = LAYERS_TO_ANALYZE

    hooks, intermediate = register_hooks(model)
    embeddings = {l: {} for l in layers}
    embeddings['final'] = {}

    for name, smi in molecules_dict.items():
        intermediate.clear()
        result = get_mean_pooled_embedding(model, smi, intermediate)
        embeddings['final'][name] = result['final']
        for l in layers:
            if l in result:
                embeddings[l][name] = result[l]

    remove_hooks(hooks)
    return embeddings


def extract_frozen_embeddings(model, smiles_list, batch_size=32):
    """
    Extract molecule-level embeddings (autoencoder latent z).

    Migration change:
    - Uses SMI-TED's autoencoder latent z instead of mean pooling
    - This is SMI-TED's native molecule representation
    """
    model.encoder.eval()
    all_embs = []
    max_len = model.max_len
    n_embd = model.n_embd

    for i in range(0, len(smiles_list), batch_size):
        batch_smi = smiles_list[i:i + batch_size]
        idx, mask = model.tokenize(batch_smi)

        with torch.no_grad():
            token_embeddings = model.encoder(idx, mask)
            z = model.decoder.autoencoder.encoder(
                token_embeddings.view(-1, max_len * n_embd)
            )
        all_embs.append(z.cpu().numpy())

    return np.concatenate(all_embs, axis=0)


def extract_layerwise_embeddings(model, smiles_list, layers=None):
    """Extract mean-pooled embeddings at each layer."""
    if layers is None:
        layers = list(range(13))

    hooks, intermediate = register_hooks(model)
    layer_embs = {l: [] for l in layers}

    for smi in smiles_list:
        intermediate.clear()
        idx, mask = model.tokenize(smi)
        with torch.no_grad():
            model.encoder(idx, mask)

        mask_cpu = mask.cpu().float()
        for l in layers:
            if l in intermediate:
                h = intermediate[l]
                if h.dim() == 3:
                    h = h[0]
                mask_expanded = mask_cpu[0].unsqueeze(-1).expand(
                    h.shape)
                pooled = (h * mask_expanded).sum(0) / \
                         mask_cpu[0].sum()
                layer_embs[l].append(pooled.numpy())

    remove_hooks(hooks)
    return {l: np.array(v) for l, v in layer_embs.items()
            if len(v) > 0}


# ─────────────────────────────────────────────────────────────────────
# FC Head
# ─────────────────────────────────────────────────────────────────────

class FrozenEncoderHead(nn.Module):
    """2-layer FC head for downstream tasks."""
    def __init__(self, input_dim=768, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.relu1 = nn.GELU()
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.relu2 = nn.GELU()
        self.final = nn.Linear(input_dim, 1)

    def forward(self, x):
        h = self.fc1(x)
        h = self.dropout1(h)
        h = self.relu1(h)
        h = h + x
        z = self.fc2(h)
        z = self.dropout2(z)
        z = self.relu2(z)
        z = self.final(z + h)
        return z.squeeze(-1)


def train_head(X_train, y_train, X_val, y_val,
               n_epochs=100, lr=1e-3, batch_size=32,
               patience=15):
    """Train FC head, return (val_rmse, head)."""
    head = FrozenEncoderHead(
        input_dim=X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(
        head.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)

    X_tr = torch.tensor(
        X_train, dtype=torch.float32).to(DEVICE)
    y_tr = torch.tensor(
        y_train, dtype=torch.float32).to(DEVICE)
    X_v = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_v = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(n_epochs):
        head.train()
        perm = torch.randperm(len(X_tr))
        for j in range(0, len(X_tr), batch_size):
            idx = perm[j:j + batch_size]
            pred = head(X_tr[idx])
            loss = nn.functional.mse_loss(pred, y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        head.eval()
        with torch.no_grad():
            val_pred = head(X_v)
            val_loss = nn.functional.mse_loss(
                val_pred, y_v).item()

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in head.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()

    with torch.no_grad():
        val_pred = head(X_v).cpu().numpy()
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    return rmse, head


# ─────────────────────────────────────────────────────────────────────
# L1: Statistical Significance
# No migration changes needed (uses hardcoded Table 1 data)
# ─────────────────────────────────────────────────────────────────────

def run_experiment_l1():
    """Paired bootstrap CI on RMSE difference + Wilcoxon test."""
    print("\n" + "=" * 70)
    print("L1: Statistical Significance of Paper's Claims")
    print("=" * 70)

    exp = TABLE1_EXPERIMENTAL
    mol = TABLE1_MOLFORMER
    mm = TABLE1_MULTIMODAL
    n = len(exp)

    sq_err_mol = (exp - mol) ** 2
    sq_err_mm = (exp - mm) ** 2
    err_mol = np.abs(exp - mol)
    err_mm = np.abs(exp - mm)

    rmse_mol = np.sqrt(np.mean(sq_err_mol))
    rmse_mm = np.sqrt(np.mean(sq_err_mm))
    rmse_diff = rmse_mol - rmse_mm

    print(f"  MoLFormer RMSE:  {rmse_mol:.4f}")
    print(f"  MultiModal RMSE: {rmse_mm:.4f}")
    print(f"  RMSE difference: {rmse_diff:.4f}")

    n_boot = 10000
    set_seed()
    boot_diffs = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        boot_diffs.append(
            np.sqrt(np.mean(sq_err_mol[idx])) -
            np.sqrt(np.mean(sq_err_mm[idx]))
        )

    boot_diffs = np.array(boot_diffs)
    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)
    p_boot = np.mean(boot_diffs <= 0)

    print(f"\n  Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  P(MultiModal worse): {p_boot:.4f}")

    try:
        stat_w, p_wilcoxon = wilcoxon(
            err_mol, err_mm, alternative='greater')
    except Exception:
        stat_w, p_wilcoxon = np.nan, np.nan
    print(f"  Wilcoxon p: {p_wilcoxon:.4f}")

    mol_wins = int(np.sum(err_mol < err_mm))
    mm_wins = int(np.sum(err_mm < err_mol))
    print(f"  Point wins - MoLFormer: {mol_wins}, "
          f"MultiModal: {mm_wins}")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, pred, name, rmse in [
        (axes[0], mol, 'MoLFormer', rmse_mol),
        (axes[1], mm, 'MultiModal', rmse_mm)
    ]:
        ax.scatter(exp, pred, c='steelblue', s=50,
                   edgecolors='k', linewidths=0.5)
        lims = [min(exp.min(), pred.min()) - 0.1,
                max(exp.max(), pred.max()) + 0.1]
        ax.plot(lims, lims, 'k--', alpha=0.5)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('Experimental LCE')
        ax.set_ylabel('Predicted LCE')
        ax.set_title(f'{name}\nRMSE={rmse:.3f}')
        ax.set_aspect('equal')

    axes[2].hist(boot_diffs, bins=50, color='steelblue',
                 edgecolor='k', alpha=0.7, density=True)
    axes[2].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[2].axvline(rmse_diff, color='darkblue',
                    linestyle='-', linewidth=2)
    axes[2].set_xlabel(
        'RMSE(MoLFormer) - RMSE(MultiModal)')
    axes[2].set_ylabel('Density')
    axes[2].set_title(
        f'Bootstrap RMSE Diff\n'
        f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'l1_statistical_significance.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    results = {
        'rmse_molformer': float(rmse_mol),
        'rmse_multimodal': float(rmse_mm),
        'rmse_difference': float(rmse_diff),
        'bootstrap_ci_95': [float(ci_lower), float(ci_upper)],
        'bootstrap_p': float(p_boot),
        'wilcoxon_p': float(p_wilcoxon)
        if not np.isnan(p_wilcoxon) else None,
        'point_wins_molformer': mol_wins,
        'point_wins_multimodal': mm_wins,
    }

    with open(RESULTS_DIR / 'l1_statistical_significance.json',
              'w') as f:
        json.dump(results, f, indent=2)

    return results


# ─────────────────────────────────────────────────────────────────────
# L2a: Electrolyte Molecule Embedding Analysis
# ─────────────────────────────────────────────────────────────────────

def run_experiment_l2a(model, random_model=None):
    """Compare embeddings of electrolyte molecules."""
    print("\n" + "=" * 70)
    print("L2a: Electrolyte Molecule Embedding Analysis")
    print("=" * 70)

    print("  Extracting pretrained embeddings...")
    pre_emb = get_embeddings_for_molecules(
        model, ELECTROLYTE_MOLECULES)

    rand_emb = None
    if random_model is not None:
        print("  Extracting random embeddings...")
        rand_emb = get_embeddings_for_molecules(
            random_model, ELECTROLYTE_MOLECULES)

    names = list(ELECTROLYTE_MOLECULES.keys())
    n_mol = len(names)
    results = {}

    # Cosine similarity heatmaps
    layers_to_plot = [0, 6, 12, 'final']
    n_layers = len(layers_to_plot)
    n_rows = 2 if rand_emb else 1

    fig, axes = plt.subplots(
        n_rows, n_layers,
        figsize=(4 * n_layers, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for col, layer in enumerate(layers_to_plot):
        for row, (emb_dict, label) in enumerate([
            (pre_emb, 'Pretrained'),
            (rand_emb, 'Random'),
        ]):
            if emb_dict is None:
                continue
            if layer not in emb_dict:
                continue
            vecs = np.array(
                [emb_dict[layer][n] for n in names])
            from sklearn.metrics.pairwise import cosine_similarity
            sim = cosine_similarity(vecs)

            ax = axes[row, col]
            im = ax.imshow(sim, cmap='RdBu_r',
                           vmin=-1, vmax=1)
            ax.set_xticks(range(n_mol))
            ax.set_xticklabels(
                names, rotation=45, ha='right', fontsize=7)
            ax.set_yticks(range(n_mol))
            ax.set_yticklabels(names, fontsize=7)
            layer_label = (f'Layer {layer}'
                           if isinstance(layer, int) else 'Final')
            ax.set_title(
                f'{label} — {layer_label}', fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    fig.savefig(
        FIGURES_DIR / 'l2a_cosine_similarity_heatmaps.png',
        dpi=150, bbox_inches='tight')
    plt.close(fig)

    # PCA
    fig, axes = plt.subplots(
        1, 2 if rand_emb else 1,
        figsize=(6 * (2 if rand_emb else 1), 5))
    if rand_emb is None:
        axes = [axes]

    group_colors = {}
    colors = ['tab:blue', 'tab:green', 'tab:red']
    for i, (gname, members) in enumerate(
            ELECTROLYTE_GROUPS.items()):
        for m in members:
            group_colors[m] = colors[i]

    for ax_idx, (emb_dict, label) in enumerate([
        (pre_emb, 'Pretrained'), (rand_emb, 'Random')
    ]):
        if emb_dict is None:
            continue
        vecs = np.array(
            [emb_dict['final'][n] for n in names])
        pca = PCA(n_components=2)
        coords = pca.fit_transform(vecs)

        ax = axes[ax_idx]
        for i, name in enumerate(names):
            ax.scatter(
                coords[i, 0], coords[i, 1],
                c=group_colors[name],
                s=80, edgecolors='k',
                linewidths=0.5, zorder=5)
            ax.annotate(
                name, (coords[i, 0], coords[i, 1]),
                textcoords='offset points',
                xytext=(5, 5), fontsize=8)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors[i], label=gname)
            for i, gname in enumerate(
                ELECTROLYTE_GROUPS.keys())
        ]
        ax.legend(handles=legend_elements,
                  fontsize=8, loc='best')
        ax.set_xlabel(
            f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(
            f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title(
            f'SMI-TED {label} — '
            f'Electrolyte Embeddings PCA')

    plt.tight_layout()
    fig.savefig(
        FIGURES_DIR / 'l2a_pca_electrolyte_embeddings.png',
        dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Quantitative: within vs between group similarity
    for emb_dict, label in [
        (pre_emb, 'pretrained'), (rand_emb, 'random')
    ]:
        if emb_dict is None:
            continue
        vecs = np.array(
            [emb_dict['final'][n] for n in names])
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity(vecs)

        name_to_group = {}
        for gname, members in ELECTROLYTE_GROUPS.items():
            for m in members:
                name_to_group[m] = gname

        within_sims, between_sims = [], []
        for i in range(n_mol):
            for j in range(i + 1, n_mol):
                if name_to_group[names[i]] == \
                        name_to_group[names[j]]:
                    within_sims.append(sim[i, j])
                else:
                    between_sims.append(sim[i, j])

        within_mean = np.mean(within_sims) if within_sims else 0
        between_mean = (np.mean(between_sims)
                        if between_sims else 0)
        separation = within_mean - between_mean

        print(f"  {label}: within={within_mean:.4f}, "
              f"between={between_mean:.4f}, "
              f"separation={separation:.4f}")

        results[f'{label}_within'] = float(within_mean)
        results[f'{label}_between'] = float(between_mean)
        results[f'{label}_separation'] = float(separation)

    with open(RESULTS_DIR / 'l2a_electrolyte_embeddings.json',
              'w') as f:
        json.dump(results, f, indent=2)

    return results


# ─────────────────────────────────────────────────────────────────────
# L2b: Sample Efficiency on ESOL
# Bug fix: evaluate FC head on test set, not validation set
# ─────────────────────────────────────────────────────────────────────

def run_experiment_l2b(model, random_model=None):
    """Sample efficiency: pretrained vs random on ESOL."""
    print("\n" + "=" * 70)
    print("L2b: Sample Efficiency on ESOL")
    print("=" * 70)

    # Load ESOL splits
    esol_dir = Path(ESOL_TRAIN_PATH)
    try:
        train_df = pd.read_csv(esol_dir / 'train.csv')
        valid_df = pd.read_csv(esol_dir / 'valid.csv')
        test_df = pd.read_csv(esol_dir / 'test.csv')
        measure_col = train_df.columns[1]
    except FileNotFoundError:
        print("  ESOL train/val/test splits not found, "
              "using single file with manual split")
        df = pd.read_csv(ESOL_PATH)
        smiles_col = 'smiles' if 'smiles' in df.columns \
            else df.columns[0]
        # Explicitly use measured solubility, not predicted
        if 'measured log solubility in mols per litre' in df.columns:
            target_col = 'measured log solubility in mols per litre'
        else:
            target_col = [c for c in df.columns
                        if c != smiles_col
                        and df[c].dtype in
                        [np.float64, np.float32, float]][0]
        n = len(df)
        perm = np.random.permutation(n)
        train_df = df.iloc[perm[:int(0.7*n)]].rename(
            columns={smiles_col: 'smiles',
                     target_col: 'label'})
        valid_df = df.iloc[perm[int(0.7*n):int(0.85*n)]].rename(
            columns={smiles_col: 'smiles',
                     target_col: 'label'})
        test_df = df.iloc[perm[int(0.85*n):]].rename(
            columns={smiles_col: 'smiles',
                     target_col: 'label'})
        measure_col = 'label'

    all_train_smiles = train_df['smiles'].tolist()
    all_train_y = train_df[measure_col].values
    val_smiles = valid_df['smiles'].tolist()
    val_y = valid_df[measure_col].values
    test_smiles = test_df['smiles'].tolist()
    test_y = test_df[measure_col].values

    print("  Extracting pretrained embeddings...")
    all_smiles = all_train_smiles + val_smiles + test_smiles
    pre_emb = extract_frozen_embeddings(model, all_smiles)
    n_train = len(all_train_smiles)
    n_val = len(val_smiles)

    pre_train = pre_emb[:n_train]
    pre_val = pre_emb[n_train:n_train + n_val]
    pre_test = pre_emb[n_train + n_val:]

    rand_train = rand_val = rand_test = None
    if random_model is not None:
        print("  Extracting random embeddings...")
        rand_emb = extract_frozen_embeddings(
            random_model, all_smiles)
        rand_train = rand_emb[:n_train]
        rand_val = rand_emb[n_train:n_train + n_val]
        rand_test = rand_emb[n_train + n_val:]

    sample_sizes = [10, 50, 100, 200, 500, n_train]
    n_seeds = 5
    results = {
        'sample_sizes': sample_sizes,
        'pretrained': {},
        'random': {}
    }

    for model_type, tr_emb, v_emb, te_emb in [
        ('pretrained', pre_train, pre_val, pre_test),
        ('random', rand_train, rand_val, rand_test),
    ]:
        if tr_emb is None:
            continue

        means, stds = [], []
        for N in sample_sizes:
            seed_rmses = []
            for seed in range(n_seeds):
                set_seed(SEED + seed)
                if N < n_train:
                    idx = np.random.choice(
                        n_train, size=N, replace=False)
                else:
                    idx = np.arange(n_train)

                _, head = train_head(
                    tr_emb[idx], all_train_y[idx],
                    v_emb, val_y,
                    n_epochs=150, lr=1e-3, patience=20)

                # Bug fix: evaluate on test set, not val set
                head.eval()
                X_te = torch.tensor(
                    te_emb, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    test_pred = head(X_te).cpu().numpy()
                test_rmse = np.sqrt(
                    mean_squared_error(test_y, test_pred))
                seed_rmses.append(test_rmse)

            means.append(np.mean(seed_rmses))
            stds.append(np.std(seed_rmses))
            print(f"  {model_type} N={N}: "
                  f"RMSE={np.mean(seed_rmses):.4f} "
                  f"±{np.std(seed_rmses):.4f}")

        results[model_type] = {
            'means': [float(x) for x in means],
            'stds': [float(x) for x in stds],
        }

    # Figure
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label, color, marker in [
        ('pretrained', 'Pretrained + FC Head',
         'steelblue', 'o'),
        ('random', 'Random + FC Head', 'coral', 's'),
    ]:
        if not results[key]:
            continue
        m = np.array(results[key]['means'])
        s = np.array(results[key]['stds'])
        ax.plot(sample_sizes, m, f'-{marker}',
                color=color, label=label, markersize=6)
        ax.fill_between(sample_sizes, m - s, m + s,
                        alpha=0.15, color=color)

    ax.axvline(147, color='gray', linestyle='--',
               alpha=0.5, label='LCE dataset size (N≈147)')
    ax.set_xscale('log')
    ax.set_xlabel('Training Set Size (N)')
    ax.set_ylabel('Test RMSE')
    ax.set_title(
        'SMI-TED Sample Efficiency: Pretrained vs Random on ESOL')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'l2b_sample_efficiency.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    with open(RESULTS_DIR / 'l2b_sample_efficiency.json',
              'w') as f:
        json.dump(results, f, indent=2)

    return results


# ─────────────────────────────────────────────────────────────────────
# L3a: Cross-Molecule Attention Patterns
# Note: SMI-TED uses linear attention (FAVOR+), no explicit
# attention matrices. We analyze cross-molecule token patterns
# using hidden state similarity instead.
# ─────────────────────────────────────────────────────────────────────

def run_experiment_l3a(model, random_model=None):
    """
    Analyze cross-molecule patterns in concatenated SMILES input.

    Migration note: SMI-TED uses linear attention (FAVOR+) which
    does not produce explicit attention matrices. Instead of
    analyzing attention weights, we analyze how much the hidden
    states of tokens from different molecules influence each other
    by comparing token embeddings in concatenated vs individual
    inputs.
    """
    print("\n" + "=" * 70)
    print("L3a: Cross-Molecule Pattern Analysis")
    print("=" * 70)

    formulations = [
        ('EC + DMC + LiPF6',
         ['C1COC(=O)O1', 'COC(=O)OC',
          '[Li+].F[P-](F)(F)(F)(F)F']),
        ('EC + EMC + LiBF4',
         ['C1COC(=O)O1', 'CCOC(=O)OC',
          '[Li+].F[B-](F)(F)F']),
    ]

    sep_token = '.'
    results = {}

    for form_name, smiles_list in formulations:
        print(f"\n  Formulation: {form_name}")
        concat_smiles = sep_token.join(smiles_list)

        # Get individual molecule embeddings
        individual_embs = []
        for smi in smiles_list:
            idx, mask = model.tokenize(smi)
            with torch.no_grad():
                tok_emb = model.encoder(idx, mask)
            # Mean pool
            mask_cpu = mask.cpu().float()
            pooled = (tok_emb.cpu()[0] *
                      mask_cpu[0].unsqueeze(-1)).sum(0) / \
                     mask_cpu[0].sum()
            individual_embs.append(pooled.numpy())

        # Get concatenated embedding
        idx_c, mask_c = model.tokenize(concat_smiles)
        with torch.no_grad():
            tok_emb_c = model.encoder(idx_c, mask_c)
        mask_cpu_c = mask_c.cpu().float()
        concat_pooled = (tok_emb_c.cpu()[0] *
                         mask_cpu_c[0].unsqueeze(-1)).sum(0) / \
                        mask_cpu_c[0].sum()
        concat_pooled = concat_pooled.numpy()

        # Compare: cosine similarity between concat and average
        simple_avg = np.mean(individual_embs, axis=0)
        cos_sim = 1 - cosine_dist(concat_pooled, simple_avg)

        print(f"    cos(concat, simple_avg) = {cos_sim:.4f}")
        results[form_name] = {
            'cos_concat_simple_avg': float(cos_sim),
            'n_molecules': len(smiles_list),
        }

    # Also compare pretrained vs random
    if random_model is not None:
        for form_name, smiles_list in formulations:
            concat_smiles = sep_token.join(smiles_list)

            for mdl, label in [
                (model, 'pretrained'),
                (random_model, 'random')
            ]:
                idx_c, mask_c = mdl.tokenize(concat_smiles)
                with torch.no_grad():
                    tok_emb_c = mdl.encoder(idx_c, mask_c)
                mask_cpu_c = mask_c.cpu().float()
                concat_pooled = (
                    tok_emb_c.cpu()[0] *
                    mask_cpu_c[0].unsqueeze(-1)
                ).sum(0) / mask_cpu_c[0].sum()
                concat_pooled = concat_pooled.numpy()

                individual_embs = []
                for smi in smiles_list:
                    idx, mask = mdl.tokenize(smi)
                    with torch.no_grad():
                        tok_emb = mdl.encoder(idx, mask)
                    mask_cpu = mask.cpu().float()
                    pooled = (
                        tok_emb.cpu()[0] *
                        mask_cpu[0].unsqueeze(-1)
                    ).sum(0) / mask_cpu[0].sum()
                    individual_embs.append(pooled.numpy())

                simple_avg = np.mean(individual_embs, axis=0)
                cos_sim = 1 - cosine_dist(
                    concat_pooled, simple_avg)
                key = f'{form_name}_{label}'
                results[key] = {
                    'cos_concat_simple_avg': float(cos_sim)
                }
                print(f"    {form_name} ({label}): "
                      f"cos={cos_sim:.4f}")

    with open(RESULTS_DIR / 'l3a_cross_molecule.json',
              'w') as f:
        json.dump(results, f, indent=2)

    return results


# ─────────────────────────────────────────────────────────────────────
# L3b: Embedding Additivity Test
# Bug fix: consistent separator format '.' (no spaces)
# ─────────────────────────────────────────────────────────────────────

def run_experiment_l3b(model):
    """Test whether concatenated embedding ≈ weighted average."""
    print("\n" + "=" * 70)
    print("L3b: Embedding Additivity Test")
    print("=" * 70)

    formulations = [
        {
            'name': 'EC:DMC:LiPF6 (3:7 + 1M)',
            'components': [
                'C1COC(=O)O1', 'COC(=O)OC',
                '[Li+].F[P-](F)(F)(F)(F)F'],
            'weights': [0.3, 0.5, 0.2],
        },
        {
            'name': 'EC:EMC:LiBF4 (1:1 + 1M)',
            'components': [
                'C1COC(=O)O1', 'CCOC(=O)OC',
                '[Li+].F[B-](F)(F)F'],
            'weights': [0.35, 0.35, 0.3],
        },
    ]

    # Bug fix: use '.' without spaces (same as L3a)
    sep_token = '.'
    results = []

    for form in formulations:
        print(f"\n  Formulation: {form['name']}")

        # Individual embeddings
        individual_embs = []
        for smi in form['components']:
            idx, mask = model.tokenize(smi)
            with torch.no_grad():
                tok_emb = model.encoder(idx, mask)
            mask_cpu = mask.cpu().float()
            pooled = (tok_emb.cpu()[0] *
                      mask_cpu[0].unsqueeze(-1)).sum(0) / \
                     mask_cpu[0].sum()
            individual_embs.append(pooled.numpy())

        weights = np.array(form['weights'])
        weighted_avg = sum(w * e for w, e in
                           zip(weights, individual_embs))
        simple_avg = np.mean(individual_embs, axis=0)

        # Concatenated embedding
        concat_smiles = sep_token.join(form['components'])
        idx_c, mask_c = model.tokenize(concat_smiles)
        with torch.no_grad():
            tok_emb_c = model.encoder(idx_c, mask_c)
        mask_cpu_c = mask_c.cpu().float()
        concat_emb = (tok_emb_c.cpu()[0] *
                      mask_cpu_c[0].unsqueeze(-1)).sum(0) / \
                     mask_cpu_c[0].sum()
        concat_emb = concat_emb.numpy()

        cos_weighted = 1 - cosine_dist(concat_emb, weighted_avg)
        cos_simple = 1 - cosine_dist(concat_emb, simple_avg)

        rng = np.random.RandomState(SEED)
        random_vec = rng.randn(len(concat_emb))
        random_vec = random_vec / np.linalg.norm(random_vec) * \
                     np.linalg.norm(concat_emb)
        cos_random = 1 - cosine_dist(concat_emb, random_vec)

        print(f"    cos(concat, weighted_avg) = {cos_weighted:.4f}")
        print(f"    cos(concat, simple_avg)   = {cos_simple:.4f}")
        print(f"    cos(concat, random)       = {cos_random:.4f}")

        results.append({
            'name': form['name'],
            'cos_concat_weighted_avg': float(cos_weighted),
            'cos_concat_simple_avg': float(cos_simple),
            'cos_concat_random': float(cos_random),
        })

    # Figure
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(results))
    width = 0.25

    ax.bar(x - width,
           [r['cos_concat_weighted_avg'] for r in results],
           width, label='vs Weighted Avg', color='steelblue')
    ax.bar(x,
           [r['cos_concat_simple_avg'] for r in results],
           width, label='vs Simple Avg', color='seagreen')
    ax.bar(x + width,
           [r['cos_concat_random'] for r in results],
           width, label='vs Random', color='coral')

    ax.set_ylabel('Cosine Similarity')
    ax.set_title('SMI-TED Concatenated Embedding vs Averages')
    ax.set_xticks(x)
    ax.set_xticklabels([r['name'] for r in results],
                       fontsize=8, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'l3b_embedding_additivity.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    with open(RESULTS_DIR / 'l3b_embedding_additivity.json',
              'w') as f:
        json.dump(results, f, indent=2)

    return results


# ─────────────────────────────────────────────────────────────────────
# L3c: Permutation Sensitivity
# Bug fix: consistent separator format '.' (no spaces)
# ─────────────────────────────────────────────────────────────────────

def run_experiment_l3c(model):
    """Test whether molecule ordering affects embedding."""
    print("\n" + "=" * 70)
    print("L3c: Permutation Sensitivity")
    print("=" * 70)

    test_sets = [
        ('EC + DMC + LiPF6',
         ['C1COC(=O)O1', 'COC(=O)OC',
          '[Li+].F[P-](F)(F)(F)(F)F']),
        ('EC + EMC + FEC + LiPF6',
         ['C1COC(=O)O1', 'CCOC(=O)OC',
          'C1OC(=O)O[C@@H]1F',
          '[Li+].F[P-](F)(F)(F)(F)F']),
    ]

    # Bug fix: use '.' without spaces (same as L3a and L3b)
    sep_token = '.'
    results = []

    for name, smiles_list in test_sets:
        print(f"\n  Formulation: {name}")

        all_perms = list(permutations(range(len(smiles_list))))
        if len(all_perms) > 24:
            set_seed()
            all_perms = random.sample(all_perms, 24)

        perm_embeddings = []
        for perm in all_perms:
            ordered = [smiles_list[i] for i in perm]
            concat = sep_token.join(ordered)
            idx_c, mask_c = model.tokenize(concat)
            with torch.no_grad():
                tok_emb_c = model.encoder(idx_c, mask_c)
            mask_cpu_c = mask_c.cpu().float()
            emb = (tok_emb_c.cpu()[0] *
                   mask_cpu_c[0].unsqueeze(-1)).sum(0) / \
                  mask_cpu_c[0].sum()
            perm_embeddings.append(emb.numpy())

        perm_embeddings = np.array(perm_embeddings)

        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(perm_embeddings)
        triu_idx = np.triu_indices(len(all_perms), k=1)
        pairwise_sims = sim_matrix[triu_idx]

        print(f"    Pairwise cosine sim: "
              f"mean={pairwise_sims.mean():.4f}, "
              f"min={pairwise_sims.min():.4f}")

        results.append({
            'name': name,
            'n_permutations': len(all_perms),
            'cosine_sim_mean': float(pairwise_sims.mean()),
            'cosine_sim_min': float(pairwise_sims.min()),
            'cosine_sim_std': float(pairwise_sims.std()),
        })

    # Figure
    fig, axes = plt.subplots(
        1, len(results), figsize=(6 * len(results), 4))
    if len(results) == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        ax.bar(['Mean perm\nsimilarity', 'Min perm\nsimilarity'],
               [r['cosine_sim_mean'], r['cosine_sim_min']],
               color=['steelblue', 'lightblue'],
               edgecolor='k')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title(f"{r['name']}\n"
                     f"({r['n_permutations']} permutations)")
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'l3c_permutation_sensitivity.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    with open(RESULTS_DIR / 'l3c_permutation_sensitivity.json',
              'w') as f:
        json.dump(results, f, indent=2)

    return results


# ─────────────────────────────────────────────────────────────────────
# L4a: Frozen Embeddings Performance on ESOL
# Bug fix: evaluate FC head on test set, not validation set
# ─────────────────────────────────────────────────────────────────────

def run_experiment_l4a(model, random_model=None):
    """Compare frozen embeddings at each layer on ESOL."""
    print("\n" + "=" * 70)
    print("L4a: Frozen Embeddings Performance on ESOL")
    print("=" * 70)

    # Load ESOL
    esol_dir = Path(ESOL_TRAIN_PATH)
    try:
        train_df = pd.read_csv(esol_dir / 'train.csv')
        valid_df = pd.read_csv(esol_dir / 'valid.csv')
        test_df = pd.read_csv(esol_dir / 'test.csv')
        measure_col = train_df.columns[1]
    except FileNotFoundError:
        df = pd.read_csv(ESOL_PATH)
        smiles_col = 'smiles' if 'smiles' in df.columns \
            else df.columns[0]
        # Explicitly use measured solubility, not predicted
        if 'measured log solubility in mols per litre' in df.columns:
            target_col = 'measured log solubility in mols per litre'
        else:
            target_col = [c for c in df.columns
                        if c != smiles_col
                        and df[c].dtype in
                        [np.float64, np.float32, float]][0]   
        n = len(df)
        perm = np.random.permutation(n)
        train_df = df.iloc[perm[:int(0.7*n)]].rename(
            columns={smiles_col: 'smiles',
                     target_col: 'label'})
        valid_df = df.iloc[
            perm[int(0.7*n):int(0.85*n)]].rename(
            columns={smiles_col: 'smiles',
                     target_col: 'label'})
        test_df = df.iloc[perm[int(0.85*n):]].rename(
            columns={smiles_col: 'smiles',
                     target_col: 'label'})
        measure_col = 'label'

    all_train_smiles = train_df['smiles'].tolist()
    train_y = train_df[measure_col].values
    val_smiles = valid_df['smiles'].tolist()
    val_y = valid_df[measure_col].values
    test_smiles = test_df['smiles'].tolist()
    test_y = test_df[measure_col].values

    results = {}

    for model_obj, model_label in [
        (model, 'pretrained'),
        (random_model, 'random')
    ]:
        if model_obj is None:
            continue

        print(f"\n  Extracting {model_label} embeddings...")
        layer_embeddings = {
            l: {'train': [], 'val': [], 'test': []}
            for l in LAYERS_TO_ANALYZE
        }

        hooks, intermediate = register_hooks(model_obj)

        for split_name, smiles_list in [
            ('train', all_train_smiles),
            ('val', val_smiles),
            ('test', test_smiles)
        ]:
            for smi in tqdm(smiles_list,
                            desc=f'{model_label} {split_name}',
                            leave=False):
                intermediate.clear()
                idx, mask = model_obj.tokenize(smi)
                with torch.no_grad():
                    model_obj.encoder(idx, mask)

                mask_cpu = mask.cpu().float()
                for l in LAYERS_TO_ANALYZE:
                    if l in intermediate:
                        hs = intermediate[l]
                        if hs.dim() == 3:
                            hs = hs[0]
                        pooled = (
                            hs * mask_cpu[0].unsqueeze(-1)
                        ).sum(0) / mask_cpu[0].sum()
                        layer_embeddings[l][split_name].append(
                            pooled.numpy())

        remove_hooks(hooks)

        for l in LAYERS_TO_ANALYZE:
            for split in ['train', 'val', 'test']:
                layer_embeddings[l][split] = np.array(
                    layer_embeddings[l][split])

        # Ridge regression at each layer
        print(f"\n  {model_label} — Ridge regression:")
        for l in LAYERS_TO_ANALYZE:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(
                layer_embeddings[l]['train'])
            X_te = scaler.transform(
                layer_embeddings[l]['test'])

            best_rmse = float('inf')
            best_alpha = 1.0
            for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_tr, train_y)
                X_val = scaler.transform(
                    layer_embeddings[l]['val'])
                val_pred = ridge.predict(X_val)
                val_rmse = np.sqrt(
                    mean_squared_error(val_y, val_pred))
                if val_rmse < best_rmse:
                    best_rmse = val_rmse
                    best_alpha = alpha

            ridge = Ridge(alpha=best_alpha)
            ridge.fit(X_tr, train_y)
            test_pred = ridge.predict(X_te)
            test_rmse = np.sqrt(
                mean_squared_error(test_y, test_pred))
            r2 = r2_score(test_y, test_pred)

            print(f"    Layer {l}: RMSE={test_rmse:.4f}, "
                  f"R²={r2:.4f}")
            results[f'{model_label}_ridge_layer{l}'] = {
                'rmse': float(test_rmse),
                'r2': float(r2),
                'alpha': best_alpha
            }

        # FC head at layer 12
        print(f"\n  {model_label} — FC Head (3 seeds):")
        fc_rmses = []
        for seed in range(3):
            set_seed(SEED + seed)
            _, head = train_head(
                layer_embeddings[12]['train'], train_y,
                layer_embeddings[12]['val'], val_y,
                n_epochs=200, lr=1e-3, patience=20
            )

            # Bug fix: evaluate on test set, not val set
            head.eval()
            X_te = torch.tensor(
                layer_embeddings[12]['test'],
                dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                test_pred = head(X_te).cpu().numpy()
            rmse = np.sqrt(
                mean_squared_error(test_y, test_pred))
            fc_rmses.append(rmse)
            print(f"    Seed {seed}: Test RMSE={rmse:.4f}")

        results[f'{model_label}_fc_frozen'] = {
            'mean_rmse': float(np.mean(fc_rmses)),
            'std_rmse': float(np.std(fc_rmses)),
        }

    # Figure
    fig, ax = plt.subplots(figsize=(10, 5))
    categories = []
    rmses = []
    colors_list = []

    for model_label, color in [
        ('pretrained', 'steelblue'), ('random', 'coral')
    ]:
        for l in LAYERS_TO_ANALYZE:
            key = f'{model_label}_ridge_layer{l}'
            if key in results:
                categories.append(
                    f'{model_label[:4]}\nRidge L{l}')
                rmses.append(results[key]['rmse'])
                colors_list.append(color)
        key = f'{model_label}_fc_frozen'
        if key in results:
            categories.append(f'{model_label[:4]}\nFC Head')
            rmses.append(results[key]['mean_rmse'])
            colors_list.append(color)

    x = np.arange(len(categories))
    ax.bar(x, rmses, color=colors_list,
           edgecolor='k', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=7,
                       rotation=45, ha='right')
    ax.set_ylabel('Test RMSE')
    ax.set_title(
        'SMI-TED ESOL: Frozen Embeddings Performance by Layer')
    ax.grid(True, alpha=0.3, axis='y')

    for i, v in enumerate(rmses):
        ax.text(i, v + 0.01, f'{v:.3f}',
                ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'l4a_frozen_performance.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    with open(RESULTS_DIR / 'l4a_frozen_performance.json',
              'w') as f:
        json.dump(results, f, indent=2)

    return results


# ─────────────────────────────────────────────────────────────────────
# L4b: CKA Between Pretrained and Random
# ─────────────────────────────────────────────────────────────────────

def linear_CKA(X, Y):
    """Centered Kernel Alignment between two representation matrices."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2
    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)


def run_experiment_l4b(model, random_model=None):
    """CKA between pretrained and random model at each layer."""
    print("\n" + "=" * 70)
    print("L4b: CKA Between Pretrained and Random Model")
    print("=" * 70)

    if random_model is None:
        print("  Skipping — no random model")
        return {}

    # Use ESOL test molecules
    df = pd.read_csv(ESOL_PATH)
    smiles_col = 'smiles' if 'smiles' in df.columns \
        else df.columns[0]
    smiles_list = df[smiles_col].tolist()[:100]

    layers = list(range(13))

    def extract_all_layers(mdl, smi_list):
        hooks, intermediate = register_hooks(mdl)
        layer_embs = {l: [] for l in layers}

        for smi in tqdm(smi_list, desc="Extracting",
                        leave=False):
            intermediate.clear()
            idx, mask = mdl.tokenize(smi)
            with torch.no_grad():
                mdl.encoder(idx, mask)

            mask_cpu = mask.cpu().float()
            for l in layers:
                if l in intermediate:
                    hs = intermediate[l]
                    if hs.dim() == 3:
                        hs = hs[0]
                    pooled = (
                        hs * mask_cpu[0].unsqueeze(-1)
                    ).sum(0) / mask_cpu[0].sum()
                    layer_embs[l].append(pooled.numpy())

        remove_hooks(hooks)
        return {l: np.array(v)
                for l, v in layer_embs.items() if v}

    print("  Extracting pretrained embeddings...")
    pre_layers = extract_all_layers(model, smiles_list)
    print("  Extracting random embeddings...")
    rand_layers = extract_all_layers(random_model, smiles_list)

    cka_values = []
    available_layers = sorted(
        set(pre_layers.keys()) & set(rand_layers.keys()))

    for l in available_layers:
        cka = linear_CKA(pre_layers[l], rand_layers[l])
        cka_values.append(cka)
        print(f"  Layer {l}: CKA = {cka:.4f}")

    # Figure
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(available_layers, cka_values, '-o',
            color='steelblue', markersize=6)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Linear CKA')
    ax.set_title(
        'SMI-TED CKA: Pretrained vs Random (per layer)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'l4b_cka_analysis.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    results = {
        'layers': available_layers,
        'cka_pretrained_vs_random': [
            float(x) for x in cka_values],
    }

    with open(RESULTS_DIR / 'l4b_cka_analysis.json',
              'w') as f:
        json.dump(results, f, indent=2)

    return results


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Phase LCE: SMI-TED Pretraining and "
          "Battery Electrolyte Performance")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}")

    # L1: Statistical significance (no model needed)
    l1_results = run_experiment_l1()

    # Load models
    model = load_pretrained_model()
    random_model = load_random_model(model)

    # L2a: Electrolyte embeddings
    l2a_results = run_experiment_l2a(model, random_model)

    # L3b: Embedding additivity
    l3b_results = run_experiment_l3b(model)

    # L3c: Permutation sensitivity
    l3c_results = run_experiment_l3c(model)

    # L3a: Cross-molecule patterns
    l3a_results = run_experiment_l3a(model, random_model)

    # L4b: CKA analysis
    l4b_results = run_experiment_l4b(model, random_model)

    # Free random model before heavy experiments
    del random_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    random_model = load_random_model(model)

    # L4a: Frozen embeddings performance
    l4a_results = run_experiment_l4a(model, random_model)

    # L2b: Sample efficiency
    l2b_results = run_experiment_l2b(model, random_model)

    print("\n" + "=" * 70)
    print("ALL LCE EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"Results: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == '__main__':
    main()
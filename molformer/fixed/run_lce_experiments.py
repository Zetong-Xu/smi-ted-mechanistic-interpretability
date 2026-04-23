"""
Phase LCE: Understanding Why MOLFormer Pretraining Helps Predict Battery
Electrolyte Performance (LCE = Logarithmic Coulombic Efficiency).

Experiments:
  L1: Statistical Significance of the Paper's Claims
  L2a: Electrolyte Molecule Embedding Analysis (pretrained vs random)
  L2b: Sample Efficiency on ESOL (pretrained vs random at varying N)
  L3a: Cross-Molecule Attention Patterns
  L3b: Embedding Additivity Test
  L3c: Permutation Sensitivity
  L4a: Frozen vs Fine-Tuned Performance on ESOL
  L4b: CKA Between Pretrained and Fine-Tuned

Critical checks:
  - Always compare pretrained vs random baseline
  - Always compare against simple baselines
  - Report confidence intervals — especially critical with N=13 test set
  - Do not claim "the model learns electrolyte chemistry" if the signal is
    explainable by token embeddings + regularization
"""

import os
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import wilcoxon
from scipy.spatial.distance import cosine as cosine_dist
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.disable(logging.WARNING)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

SEED = 42
DEVICE = 'cuda:0'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE_DIR / 'results'
FIGURES_DIR = PHASE_DIR / 'figures'
for d in [RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(exist_ok=True)

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
# Electrolyte molecules from the paper
# ─────────────────────────────────────────────────────────────────────

ELECTROLYTE_MOLECULES = {
    # Solvents
    'EC':   'C1COC(=O)O1',
    'DMC':  'COC(=O)OC',
    'DEC':  'CCOC(=O)OCC',
    'EMC':  'CCOC(=O)OC',
    'PC':   'CC1COC(=O)O1',
    'FEC':  'C1OC(=O)O[C@@H]1F',
    'VC':   'C1=COC(=O)O1',
    # Salts
    'LiPF6':  '[Li+].F[P-](F)(F)(F)(F)F',
    'LiBF4':  '[Li+].F[B-](F)(F)F',
    'LiTFSI': '[Li+].C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F',
}

# Chemical groupings for validation
ELECTROLYTE_GROUPS = {
    'linear_carbonates': ['DMC', 'DEC', 'EMC'],
    'cyclic_carbonates': ['EC', 'PC', 'FEC', 'VC'],
    'li_salts': ['LiPF6', 'LiBF4', 'LiTFSI'],
}


# ─────────────────────────────────────────────────────────────────────
# Model Loading & Hooks (reused from phase_geometry)
# ─────────────────────────────────────────────────────────────────────

def load_pretrained_model():
    from transformers import AutoModel, AutoTokenizer
    print("Loading pretrained MOLFormer...")
    tokenizer = AutoTokenizer.from_pretrained(
        'ibm/MoLFormer-XL-both-10pct', trust_remote_code=True)
    model = AutoModel.from_pretrained(
        'ibm/MoLFormer-XL-both-10pct', trust_remote_code=True)
    model = model.to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}")
    return model, tokenizer


def load_random_model(pretrained_model):
    from transformers import AutoModel
    print("Creating randomly initialized MOLFormer...")
    random_model = AutoModel.from_config(
        pretrained_model.config, trust_remote_code=True)
    random_model = random_model.to(DEVICE)
    random_model.eval()
    return random_model


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

    if hasattr(model, 'embeddings'):
        hooks.append(model.embeddings.register_forward_hook(make_hook(0)))
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        for i, layer in enumerate(model.encoder.layer):
            hooks.append(layer.register_forward_hook(make_hook(i + 1)))
    return hooks, intermediate


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ─────────────────────────────────────────────────────────────────────
# Embedding Extraction Utilities
# ─────────────────────────────────────────────────────────────────────

def get_mean_pooled_embedding(model, tokenizer, smiles, hooks_intermediate=None):
    """Get mean-pooled embedding for a SMILES string.

    If hooks_intermediate is provided, also returns layer-wise embeddings.
    """
    inputs = tokenizer(smiles, return_tensors='pt', padding=False,
                       truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pool over non-padding tokens
    hidden = outputs.last_hidden_state[0]  # (seq_len, 768)
    mask = inputs['attention_mask'][0].float()  # (seq_len,)
    pooled = (hidden * mask.unsqueeze(-1)).sum(0) / mask.sum()

    result = {'final': pooled.cpu().numpy()}

    if hooks_intermediate is not None:
        for layer_idx, hs in hooks_intermediate.items():
            layer_hidden = hs[0]  # (seq_len, 768)
            layer_pooled = (layer_hidden * mask.cpu().unsqueeze(-1)).sum(0) / mask.cpu().sum()
            result[layer_idx] = layer_pooled.numpy()

    return result


def get_embeddings_for_molecules(model, tokenizer, molecules_dict, layers=None):
    """Extract embeddings for a dict of {name: SMILES} at multiple layers."""
    if layers is None:
        layers = LAYERS_TO_ANALYZE

    hooks, intermediate = register_hooks(model)
    embeddings = {l: {} for l in layers}
    embeddings['final'] = {}

    for name, smi in molecules_dict.items():
        intermediate.clear()
        result = get_mean_pooled_embedding(model, tokenizer, smi, intermediate)
        embeddings['final'][name] = result['final']
        for l in layers:
            if l in result:
                embeddings[l][name] = result[l]

    remove_hooks(hooks)
    return embeddings


# ═════════════════════════════════════════════════════════════════════
# L1: Statistical Significance of the Paper's Claims
# ═════════════════════════════════════════════════════════════════════

def run_experiment_l1():
    """Paired bootstrap CI on RMSE difference + Wilcoxon test + LOO analysis."""
    print("\n" + "=" * 70)
    print("L1: Statistical Significance of Paper's Claims")
    print("=" * 70)

    exp = TABLE1_EXPERIMENTAL
    mol = TABLE1_MOLFORMER
    mm = TABLE1_MULTIMODAL
    n = len(exp)

    # Compute errors
    err_mol = np.abs(exp - mol)
    err_mm = np.abs(exp - mm)
    sq_err_mol = (exp - mol) ** 2
    sq_err_mm = (exp - mm) ** 2

    rmse_mol = np.sqrt(np.mean(sq_err_mol))
    rmse_mm = np.sqrt(np.mean(sq_err_mm))
    rmse_diff = rmse_mol - rmse_mm

    print(f"  MoLFormer RMSE:    {rmse_mol:.4f}")
    print(f"  MultiModal RMSE:   {rmse_mm:.4f}")
    print(f"  RMSE difference:   {rmse_diff:.4f}")

    # --- Paired bootstrap CI on RMSE difference ---
    n_boot = 10000
    set_seed()
    boot_diffs = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        boot_rmse_mol = np.sqrt(np.mean(sq_err_mol[idx]))
        boot_rmse_mm = np.sqrt(np.mean(sq_err_mm[idx]))
        boot_diffs.append(boot_rmse_mol - boot_rmse_mm)

    boot_diffs = np.array(boot_diffs)
    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)
    p_boot = np.mean(boot_diffs <= 0)  # fraction where MultiModal is worse

    print(f"\n  Bootstrap (10,000 resamples):")
    print(f"    RMSE diff 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"    P(MultiModal worse): {p_boot:.4f}")

    # --- Wilcoxon signed-rank test on absolute errors ---
    try:
        stat_w, p_wilcoxon = wilcoxon(err_mol, err_mm, alternative='greater')
    except Exception:
        stat_w, p_wilcoxon = np.nan, np.nan
    print(f"\n  Wilcoxon signed-rank (H1: MoLFormer errors > MultiModal errors):")
    print(f"    statistic={stat_w:.1f}, p={p_wilcoxon:.4f}")

    # --- Leave-one-out: which method wins per point ---
    mol_wins = np.sum(err_mol < err_mm)
    mm_wins = np.sum(err_mm < err_mol)
    ties = np.sum(err_mol == err_mm)
    print(f"\n  Point-by-point wins:")
    print(f"    MoLFormer wins: {mol_wins}/{n}")
    print(f"    MultiModal wins: {mm_wins}/{n}")
    print(f"    Ties: {ties}/{n}")

    # --- Bootstrap CI for individual RMSEs ---
    boot_rmse_mol_vals = []
    boot_rmse_mm_vals = []
    set_seed()
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        boot_rmse_mol_vals.append(np.sqrt(np.mean(sq_err_mol[idx])))
        boot_rmse_mm_vals.append(np.sqrt(np.mean(sq_err_mm[idx])))

    ci_mol = (np.percentile(boot_rmse_mol_vals, 2.5),
              np.percentile(boot_rmse_mol_vals, 97.5))
    ci_mm = (np.percentile(boot_rmse_mm_vals, 2.5),
             np.percentile(boot_rmse_mm_vals, 97.5))

    print(f"\n  Individual RMSE 95% CIs:")
    print(f"    MoLFormer: [{ci_mol[0]:.4f}, {ci_mol[1]:.4f}]")
    print(f"    MultiModal: [{ci_mm[0]:.4f}, {ci_mm[1]:.4f}]")

    overlap = ci_mol[0] < ci_mm[1] and ci_mm[0] < ci_mol[1]
    print(f"    CIs overlap: {overlap}")

    # --- Figure: parity plot + error comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Parity plots
    for ax, pred, name, rmse in [
        (axes[0], mol, 'MoLFormer', rmse_mol),
        (axes[1], mm, 'MultiModal-MoLFormer', rmse_mm)
    ]:
        ax.scatter(exp, pred, c='steelblue', s=50, edgecolors='k', linewidths=0.5)
        lims = [min(exp.min(), pred.min()) - 0.1, max(exp.max(), pred.max()) + 0.1]
        ax.plot(lims, lims, 'k--', alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Experimental LCE')
        ax.set_ylabel('Predicted LCE')
        ax.set_title(f'{name}\nRMSE = {rmse:.3f}')
        ax.set_aspect('equal')

    # Bootstrap distribution
    axes[2].hist(boot_diffs, bins=50, color='steelblue', edgecolor='k',
                 alpha=0.7, density=True)
    axes[2].axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
    axes[2].axvline(rmse_diff, color='darkblue', linestyle='-', linewidth=2,
                    label=f'Observed: {rmse_diff:.3f}')
    axes[2].axvline(ci_lower, color='gray', linestyle=':', linewidth=1)
    axes[2].axvline(ci_upper, color='gray', linestyle=':', linewidth=1)
    axes[2].set_xlabel('RMSE(MoLFormer) - RMSE(MultiModal)')
    axes[2].set_ylabel('Density')
    axes[2].set_title(f'Bootstrap RMSE Diff\n95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'l1_statistical_significance.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Figure saved: l1_statistical_significance.png")

    results = {
        'rmse_molformer': float(rmse_mol),
        'rmse_multimodal': float(rmse_mm),
        'rmse_difference': float(rmse_diff),
        'bootstrap_ci_95': [float(ci_lower), float(ci_upper)],
        'bootstrap_p_multimodal_worse': float(p_boot),
        'wilcoxon_statistic': float(stat_w) if not np.isnan(stat_w) else None,
        'wilcoxon_p': float(p_wilcoxon) if not np.isnan(p_wilcoxon) else None,
        'point_wins_molformer': int(mol_wins),
        'point_wins_multimodal': int(mm_wins),
        'rmse_ci_molformer': [float(ci_mol[0]), float(ci_mol[1])],
        'rmse_ci_multimodal': [float(ci_mm[0]), float(ci_mm[1])],
        'cis_overlap': bool(overlap),
        'n_test_points': n,
    }

    with open(RESULTS_DIR / 'l1_statistical_significance.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ═════════════════════════════════════════════════════════════════════
# L2a: Electrolyte Molecule Embedding Analysis
# ═════════════════════════════════════════════════════════════════════

def run_experiment_l2a(model, tokenizer, random_model=None):
    """Compare embeddings of electrolyte molecules: pretrained vs random."""
    print("\n" + "=" * 70)
    print("L2a: Electrolyte Molecule Embedding Analysis")
    print("=" * 70)

    # Extract embeddings
    print("  Extracting pretrained embeddings...")
    pre_emb = get_embeddings_for_molecules(model, tokenizer, ELECTROLYTE_MOLECULES)

    rand_emb = None
    if random_model is not None:
        print("  Extracting random embeddings...")
        rand_emb = get_embeddings_for_molecules(
            random_model, tokenizer, ELECTROLYTE_MOLECULES)

    names = list(ELECTROLYTE_MOLECULES.keys())
    n_mol = len(names)
    results = {}

    # --- Cosine similarity heatmaps at each layer ---
    layers_to_plot = [0, 6, 12, 'final']
    n_layers = len(layers_to_plot)
    n_rows = 2 if rand_emb else 1

    fig, axes = plt.subplots(n_rows, n_layers, figsize=(4 * n_layers, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for col, layer in enumerate(layers_to_plot):
        for row, (emb_dict, label) in enumerate([
            (pre_emb, 'Pretrained'),
            (rand_emb, 'Random'),
        ]):
            if emb_dict is None:
                continue
            vecs = np.array([emb_dict[layer][n] for n in names])
            # Cosine similarity matrix
            from sklearn.metrics.pairwise import cosine_similarity
            sim = cosine_similarity(vecs)

            ax = axes[row, col]
            im = ax.imshow(sim, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_xticks(range(n_mol))
            ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
            ax.set_yticks(range(n_mol))
            ax.set_yticklabels(names, fontsize=7)
            layer_label = f'Layer {layer}' if isinstance(layer, int) else 'Final'
            ax.set_title(f'{label} — {layer_label}', fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'l2a_cosine_similarity_heatmaps.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: l2a_cosine_similarity_heatmaps.png")

    # --- PCA of electrolyte embeddings ---
    fig, axes = plt.subplots(1, 2 if rand_emb else 1, figsize=(6 * (2 if rand_emb else 1), 5))
    if rand_emb is None:
        axes = [axes]

    group_colors = {}
    colors = ['tab:blue', 'tab:green', 'tab:red']
    for i, (gname, members) in enumerate(ELECTROLYTE_GROUPS.items()):
        for m in members:
            group_colors[m] = colors[i]

    for ax_idx, (emb_dict, label) in enumerate([
        (pre_emb, 'Pretrained'), (rand_emb, 'Random')
    ]):
        if emb_dict is None:
            continue
        vecs = np.array([emb_dict['final'][n] for n in names])
        pca = PCA(n_components=2)
        coords = pca.fit_transform(vecs)

        ax = axes[ax_idx]
        for i, name in enumerate(names):
            ax.scatter(coords[i, 0], coords[i, 1], c=group_colors[name],
                       s=80, edgecolors='k', linewidths=0.5, zorder=5)
            ax.annotate(name, (coords[i, 0], coords[i, 1]),
                        textcoords='offset points', xytext=(5, 5), fontsize=8)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors[i], label=gname)
            for i, gname in enumerate(ELECTROLYTE_GROUPS.keys())
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc='best')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title(f'{label} — PCA of Electrolyte Embeddings')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'l2a_pca_electrolyte_embeddings.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: l2a_pca_electrolyte_embeddings.png")

    # --- Quantitative: within-group vs between-group similarity ---
    for emb_dict, label in [(pre_emb, 'pretrained'), (rand_emb, 'random')]:
        if emb_dict is None:
            continue
        vecs = np.array([emb_dict['final'][n] for n in names])
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity(vecs)

        within_sims = []
        between_sims = []
        name_to_group = {}
        for gname, members in ELECTROLYTE_GROUPS.items():
            for m in members:
                name_to_group[m] = gname

        for i in range(n_mol):
            for j in range(i + 1, n_mol):
                if name_to_group[names[i]] == name_to_group[names[j]]:
                    within_sims.append(sim[i, j])
                else:
                    between_sims.append(sim[i, j])

        within_mean = np.mean(within_sims) if within_sims else 0
        between_mean = np.mean(between_sims) if between_sims else 0
        separation = within_mean - between_mean

        print(f"\n  {label.capitalize()} — within-group sim: {within_mean:.4f}, "
              f"between-group sim: {between_mean:.4f}, separation: {separation:.4f}")

        results[f'{label}_within_group_sim'] = float(within_mean)
        results[f'{label}_between_group_sim'] = float(between_mean)
        results[f'{label}_group_separation'] = float(separation)

    with open(RESULTS_DIR / 'l2a_electrolyte_embeddings.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ═════════════════════════════════════════════════════════════════════
# L2b: Sample Efficiency on ESOL
# ═════════════════════════════════════════════════════════════════════

class FrozenEncoderHead(nn.Module):
    """2-layer FC head matching the paper's architecture."""
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
        h = h + x  # skip connection
        z = self.fc2(h)
        z = self.dropout2(z)
        z = self.relu2(z)
        z = self.final(z + h)  # skip connection
        return z.squeeze(-1)


def extract_frozen_embeddings(model, tokenizer, smiles_list, batch_size=32):
    """Extract mean-pooled final-layer embeddings for a list of SMILES."""
    model.eval()
    all_embs = []

    for i in range(0, len(smiles_list), batch_size):
        batch_smi = smiles_list[i:i + batch_size]
        inputs = tokenizer(batch_smi, return_tensors='pt', padding=True,
                           truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        hidden = outputs.last_hidden_state  # (B, seq_len, 768)
        mask = inputs['attention_mask'].float().unsqueeze(-1)  # (B, seq_len, 1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        all_embs.append(pooled.cpu().numpy())

    return np.concatenate(all_embs, axis=0)


def train_head(X_train, y_train, X_val, y_val, n_epochs=100, lr=1e-3,
               batch_size=32, patience=15):
    """Train a 2-layer FC head on frozen embeddings."""
    head = FrozenEncoderHead(input_dim=X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_tr = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_v = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_v = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(n_epochs):
        head.train()
        perm = torch.randperm(len(X_tr))
        epoch_loss = 0
        n_batches = 0
        for j in range(0, len(X_tr), batch_size):
            idx = perm[j:j + batch_size]
            pred = head(X_tr[idx])
            loss = nn.functional.mse_loss(pred, y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        head.eval()
        with torch.no_grad():
            val_pred = head(X_v)
            val_loss = nn.functional.mse_loss(val_pred, y_v).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
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


def run_experiment_l2b(model, tokenizer, random_model=None):
    """Sample efficiency: pretrained vs random on ESOL at varying N."""
    print("\n" + "=" * 70)
    print("L2b: Sample Efficiency on ESOL")
    print("=" * 70)

    # Load ESOL data
    train_df = pd.read_csv(PROJECT_ROOT / 'datasets' / 'esol' / 'esol_train.csv')
    valid_df = pd.read_csv(PROJECT_ROOT / 'datasets' / 'esol' / 'esol_valid.csv')
    test_df = pd.read_csv(PROJECT_ROOT / 'datasets' / 'esol' / 'esol_test.csv')

    measure_col = train_df.columns[1]  # 'measured log solubility in mols per litre'
    all_train_smiles = train_df['smiles'].tolist()
    all_train_y = train_df[measure_col].values
    val_smiles = valid_df['smiles'].tolist()
    val_y = valid_df[measure_col].values
    test_smiles = test_df['smiles'].tolist()
    test_y = test_df[measure_col].values

    # Extract embeddings once for all splits
    print("  Extracting pretrained embeddings for all ESOL molecules...")
    all_smiles = all_train_smiles + val_smiles + test_smiles
    pre_emb = extract_frozen_embeddings(model, tokenizer, all_smiles)
    n_train = len(all_train_smiles)
    n_val = len(val_smiles)

    pre_train_emb = pre_emb[:n_train]
    pre_val_emb = pre_emb[n_train:n_train + n_val]
    pre_test_emb = pre_emb[n_train + n_val:]

    rand_train_emb = rand_val_emb = rand_test_emb = None
    if random_model is not None:
        print("  Extracting random embeddings for all ESOL molecules...")
        rand_emb = extract_frozen_embeddings(random_model, tokenizer, all_smiles)
        rand_train_emb = rand_emb[:n_train]
        rand_val_emb = rand_emb[n_train:n_train + n_val]
        rand_test_emb = rand_emb[n_train + n_val:]

    # Sample sizes to test
    sample_sizes = [10, 25, 50, 100, 200, 500, n_train]
    n_seeds = 10
    results = {'sample_sizes': sample_sizes, 'pretrained': {}, 'random': {}}

    for model_type, train_emb, val_emb_arr, test_emb_arr in [
        ('pretrained', pre_train_emb, pre_val_emb, pre_test_emb),
        ('random', rand_train_emb, rand_val_emb, rand_test_emb),
    ]:
        if train_emb is None:
            continue

        means = []
        stds = []
        all_rmses = []

        for N in sample_sizes:
            seed_rmses = []
            for seed in range(n_seeds):
                set_seed(SEED + seed)
                if N < n_train:
                    idx = np.random.choice(n_train, size=N, replace=False)
                else:
                    idx = np.arange(n_train)

                _, head = train_head(
                    train_emb[idx], all_train_y[idx],
                    val_emb_arr, val_y,
                    n_epochs=150, lr=1e-3, patience=20
                )
                # Evaluate on test (val was used for early stopping)
                head.eval()
                X_te = torch.tensor(test_emb_arr, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    test_pred = head(X_te).cpu().numpy()
                test_rmse = np.sqrt(mean_squared_error(test_y, test_pred))
                seed_rmses.append(test_rmse)

            means.append(np.mean(seed_rmses))
            stds.append(np.std(seed_rmses))
            all_rmses.append(seed_rmses)
            print(f"  {model_type} N={N}: RMSE={np.mean(seed_rmses):.4f} "
                  f"± {np.std(seed_rmses):.4f}")

        results[model_type] = {
            'means': [float(x) for x in means],
            'stds': [float(x) for x in stds],
            'all_rmses': [[float(x) for x in r] for r in all_rmses],
        }

    # Also run Ridge regression baseline
    print("\n  Running Ridge regression baseline...")
    ridge_means = []
    ridge_stds = []
    for N in sample_sizes:
        seed_rmses = []
        for seed in range(n_seeds):
            set_seed(SEED + seed)
            if N < n_train:
                idx = np.random.choice(n_train, size=N, replace=False)
            else:
                idx = np.arange(n_train)

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(pre_train_emb[idx])
            X_te = scaler.transform(pre_test_emb)

            ridge = Ridge(alpha=1.0)
            ridge.fit(X_tr, all_train_y[idx])
            pred = ridge.predict(X_te)
            rmse = np.sqrt(mean_squared_error(test_y, pred))
            seed_rmses.append(rmse)

        ridge_means.append(np.mean(seed_rmses))
        ridge_stds.append(np.std(seed_rmses))
        print(f"  Ridge (pretrained) N={N}: RMSE={np.mean(seed_rmses):.4f} "
              f"± {np.std(seed_rmses):.4f}")

    results['ridge_pretrained'] = {
        'means': [float(x) for x in ridge_means],
        'stds': [float(x) for x in ridge_stds],
    }

    # --- Figure: learning curves ---
    fig, ax = plt.subplots(figsize=(8, 5))

    for key, label, color, marker in [
        ('pretrained', 'Pretrained + FC Head', 'steelblue', 'o'),
        ('random', 'Random + FC Head', 'coral', 's'),
        ('ridge_pretrained', 'Pretrained + Ridge', 'seagreen', '^'),
    ]:
        if key not in results or not results[key]:
            continue
        m = np.array(results[key]['means'])
        s = np.array(results[key]['stds'])
        ax.plot(sample_sizes, m, f'-{marker}', color=color, label=label,
                markersize=6)
        ax.fill_between(sample_sizes, m - s, m + s, alpha=0.15, color=color)

    # Mark N≈147 (LCE dataset size)
    ax.axvline(147, color='gray', linestyle='--', alpha=0.5, label='LCE dataset size (N≈147)')

    ax.set_xscale('log')
    ax.set_xlabel('Training Set Size (N)')
    ax.set_ylabel('Test RMSE')
    ax.set_title('Sample Efficiency: Pretrained vs Random on ESOL')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'l2b_sample_efficiency.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Figure saved: l2b_sample_efficiency.png")

    with open(RESULTS_DIR / 'l2b_sample_efficiency.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ═════════════════════════════════════════════════════════════════════
# L3a: Cross-Molecule Attention Patterns
# ═════════════════════════════════════════════════════════════════════

def run_experiment_l3a(model, tokenizer, random_model=None):
    """Analyze attention patterns on concatenated multi-SMILES input."""
    print("\n" + "=" * 70)
    print("L3a: Cross-Molecule Attention Patterns")
    print("=" * 70)

    # Test formulation: EC + DMC + LiPF6 (common electrolyte)
    formulations = [
        ('EC + DMC + LiPF6',
         ['C1COC(=O)O1', 'COC(=O)OC', '[Li+].F[P-](F)(F)(F)(F)F']),
        ('EC + EMC + LiBF4',
         ['C1COC(=O)O1', 'CCOC(=O)OC', '[Li+].F[B-](F)(F)F']),
    ]

    # Use '.' as separator — it's a single SMILES token (ID 34)
    # and represents disconnected fragments, natural for multi-molecule input
    sep_token = '.'
    print(f"  Using '.' as separator (single token in SMILES vocabulary)")

    results = {}

    for form_name, smiles_list in formulations:
        print(f"\n  Formulation: {form_name}")

        # Concatenate with '.' separator
        concat_smiles = sep_token.join(smiles_list)
        print(f"    Input: {concat_smiles}")

        # Find molecule boundaries by tokenizing each molecule individually
        individual_token_lengths = []
        for smi in smiles_list:
            toks = tokenizer.tokenize(smi)
            individual_token_lengths.append(len(toks))
            print(f"    {smi}: {len(toks)} tokens")

        # Build boundaries: account for CLS token at position 0
        # and '.' separator tokens between molecules
        # Full sequence: [CLS] mol1_tokens [.] mol2_tokens [.] mol3_tokens [EOS]
        boundaries = []
        pos = 1  # skip CLS
        for i, n_tok in enumerate(individual_token_lengths):
            boundaries.append((pos, pos + n_tok))
            pos += n_tok
            if i < len(individual_token_lengths) - 1:
                pos += 1  # skip '.' separator token

        print(f"    Molecule boundaries (in attention matrix): {boundaries}")

        for mdl, label in [(model, 'pretrained'), (random_model, 'random')]:
            if mdl is None:
                continue

            # Forward pass with attention output
            inputs = tokenizer(concat_smiles, return_tensors='pt', padding=False,
                               truncation=True, max_length=512)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = mdl(**inputs, output_attentions=True)

            if outputs.attentions is None or len(outputs.attentions) == 0:
                print(f"    {label}: No attention weights returned")
                continue

            # Analyze cross-molecule attention at each layer
            seq_len = inputs['input_ids'].shape[1]
            n_layers = len(outputs.attentions)
            n_heads = outputs.attentions[0].shape[1]

            # Boundaries already account for CLS offset
            cross_attn_fractions = np.zeros((n_layers, n_heads))
            within_attn_fractions = np.zeros((n_layers, n_heads))

            for layer_idx in range(n_layers):
                attn = outputs.attentions[layer_idx][0].cpu().numpy()
                # attn shape: (n_heads, seq_len, seq_len)

                for head_idx in range(n_heads):
                    head_attn = attn[head_idx]  # (seq_len, seq_len)

                    total_attn = 0
                    cross_attn = 0
                    within_attn = 0

                    for mol_i, (si, ei) in enumerate(boundaries):
                        if si >= seq_len or ei > seq_len:
                            continue
                        for mol_j, (sj, ej) in enumerate(boundaries):
                            if sj >= seq_len or ej > seq_len:
                                continue
                            block_sum = head_attn[si:ei, sj:ej].sum()
                            total_attn += block_sum
                            if mol_i == mol_j:
                                within_attn += block_sum
                            else:
                                cross_attn += block_sum

                    if total_attn > 0:
                        cross_attn_fractions[layer_idx, head_idx] = cross_attn / total_attn
                        within_attn_fractions[layer_idx, head_idx] = within_attn / total_attn

            results[f'{form_name}_{label}_cross_attn_fraction'] = {
                'mean_per_layer': cross_attn_fractions.mean(axis=1).tolist(),
                'max_per_layer': cross_attn_fractions.max(axis=1).tolist(),
                'overall_mean': float(cross_attn_fractions.mean()),
            }

            print(f"    {label} — cross-molecule attention fraction per layer:")
            for l_idx in range(n_layers):
                mean_cross = cross_attn_fractions[l_idx].mean()
                max_cross = cross_attn_fractions[l_idx].max()
                print(f"      Layer {l_idx}: mean={mean_cross:.4f}, max={max_cross:.4f}")

    # --- Figure: cross-attention heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (form_name, _) in enumerate(formulations):
        for label, color in [('pretrained', 'steelblue'), ('random', 'coral')]:
            key = f'{form_name}_{label}_cross_attn_fraction'
            if key in results:
                vals = results[key]['mean_per_layer']
                axes[ax_idx].plot(range(len(vals)), vals, '-o', color=color,
                                 label=label, markersize=4)

        axes[ax_idx].set_xlabel('Layer')
        axes[ax_idx].set_ylabel('Cross-Molecule Attention Fraction')
        axes[ax_idx].set_title(form_name)
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'l3a_cross_molecule_attention.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Figure saved: l3a_cross_molecule_attention.png")

    # Save serializable results
    serializable_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable_results[k] = v
        else:
            serializable_results[k] = str(v)

    with open(RESULTS_DIR / 'l3a_cross_attention.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)

    return results


# ═════════════════════════════════════════════════════════════════════
# L3b: Embedding Additivity Test
# ═════════════════════════════════════════════════════════════════════

def run_experiment_l3b(model, tokenizer):
    """Test whether concatenated embedding ≈ weighted average of individual embeddings."""
    print("\n" + "=" * 70)
    print("L3b: Embedding Additivity Test")
    print("=" * 70)

    # Test formulations with typical molar percentages
    formulations = [
        {
            'name': 'EC:DMC:LiPF6 (3:7 + 1M)',
            'components': ['C1COC(=O)O1', 'COC(=O)OC', '[Li+].F[P-](F)(F)(F)(F)F'],
            'weights': [0.3, 0.5, 0.2],
        },
        {
            'name': 'EC:EMC:LiBF4 (1:1 + 1M)',
            'components': ['C1COC(=O)O1', 'CCOC(=O)OC', '[Li+].F[B-](F)(F)F'],
            'weights': [0.35, 0.35, 0.3],
        },
        {
            'name': 'EC:DEC:FEC:LiPF6',
            'components': ['C1COC(=O)O1', 'CCOC(=O)OCC',
                           'C1OC(=O)O[C@@H]1F', '[Li+].F[P-](F)(F)(F)(F)F'],
            'weights': [0.3, 0.3, 0.2, 0.2],
        },
    ]

    # Use '.' as separator — same format as L3a (no spaces)
    sep_token = '.'

    results = []

    for form in formulations:
        print(f"\n  Formulation: {form['name']}")

        # 1. Individual embeddings
        individual_embs = []
        for smi in form['components']:
            emb = get_mean_pooled_embedding(model, tokenizer, smi)
            individual_embs.append(emb['final'])

        # 2. Weighted average
        weights = np.array(form['weights'])
        weighted_avg = sum(w * e for w, e in zip(weights, individual_embs))

        # 3. Simple average
        simple_avg = np.mean(individual_embs, axis=0)

        # 4. Concatenated input embedding (same format as L3a)
        concat_smiles = sep_token.join(form['components'])
        concat_emb = get_mean_pooled_embedding(model, tokenizer, concat_smiles)['final']

        # Compare
        cos_concat_weighted = 1 - cosine_dist(concat_emb, weighted_avg)
        cos_concat_simple = 1 - cosine_dist(concat_emb, simple_avg)
        l2_concat_weighted = np.linalg.norm(concat_emb - weighted_avg)
        l2_concat_simple = np.linalg.norm(concat_emb - simple_avg)

        # Baseline: random direction similarity
        rng = np.random.RandomState(SEED)
        random_vec = rng.randn(len(concat_emb))
        random_vec = random_vec / np.linalg.norm(random_vec) * np.linalg.norm(concat_emb)
        cos_concat_random = 1 - cosine_dist(concat_emb, random_vec)

        # Cross-component similarities for context
        cross_sims = []
        for i in range(len(individual_embs)):
            for j in range(i + 1, len(individual_embs)):
                cross_sims.append(1 - cosine_dist(individual_embs[i], individual_embs[j]))
        mean_cross_sim = np.mean(cross_sims)

        print(f"    cos(concat, weighted_avg) = {cos_concat_weighted:.4f}")
        print(f"    cos(concat, simple_avg)   = {cos_concat_simple:.4f}")
        print(f"    cos(concat, random)       = {cos_concat_random:.4f}")
        print(f"    mean cross-component sim  = {mean_cross_sim:.4f}")
        print(f"    L2(concat, weighted_avg)  = {l2_concat_weighted:.4f}")

        results.append({
            'name': form['name'],
            'cos_concat_weighted_avg': float(cos_concat_weighted),
            'cos_concat_simple_avg': float(cos_concat_simple),
            'cos_concat_random': float(cos_concat_random),
            'l2_concat_weighted_avg': float(l2_concat_weighted),
            'l2_concat_simple_avg': float(l2_concat_simple),
            'mean_cross_component_sim': float(mean_cross_sim),
        })

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(8, 5))

    form_names = [r['name'] for r in results]
    x = np.arange(len(form_names))
    width = 0.25

    bars1 = [r['cos_concat_weighted_avg'] for r in results]
    bars2 = [r['cos_concat_simple_avg'] for r in results]
    bars3 = [r['cos_concat_random'] for r in results]

    ax.bar(x - width, bars1, width, label='vs Weighted Avg', color='steelblue')
    ax.bar(x, bars2, width, label='vs Simple Avg', color='seagreen')
    ax.bar(x + width, bars3, width, label='vs Random', color='coral')

    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Concatenated Embedding vs Averages')
    ax.set_xticks(x)
    ax.set_xticklabels([r['name'] for r in results], fontsize=8, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'l3b_embedding_additivity.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Figure saved: l3b_embedding_additivity.png")

    with open(RESULTS_DIR / 'l3b_embedding_additivity.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ═════════════════════════════════════════════════════════════════════
# L3c: Permutation Sensitivity
# ═════════════════════════════════════════════════════════════════════

def run_experiment_l3c(model, tokenizer):
    """Test whether molecule ordering affects the concatenated embedding."""
    print("\n" + "=" * 70)
    print("L3c: Permutation Sensitivity")
    print("=" * 70)

    test_sets = [
        ('EC + DMC + LiPF6',
         ['C1COC(=O)O1', 'COC(=O)OC', '[Li+].F[P-](F)(F)(F)(F)F']),
        ('EC + EMC + FEC + LiPF6',
         ['C1COC(=O)O1', 'CCOC(=O)OC', 'C1OC(=O)O[C@@H]1F',
          '[Li+].F[P-](F)(F)(F)(F)F']),
    ]

    # Use '.' as separator — same format as L3a and L3b (no spaces)
    sep_token = '.'

    results = []

    for name, smiles_list in test_sets:
        print(f"\n  Formulation: {name}")

        # Generate all permutations (or sample if too many)
        n_components = len(smiles_list)
        all_perms = list(permutations(range(n_components)))
        if len(all_perms) > 120:  # cap at 5! permutations
            set_seed()
            all_perms = random.sample(all_perms, 120)

        print(f"    Testing {len(all_perms)} permutations...")

        perm_embeddings = []
        for perm in all_perms:
            ordered = [smiles_list[i] for i in perm]
            concat = sep_token.join(ordered)
            emb = get_mean_pooled_embedding(model, tokenizer, concat)['final']
            perm_embeddings.append(emb)

        perm_embeddings = np.array(perm_embeddings)

        # Compute pairwise cosine similarities between permutations
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(perm_embeddings)
        # Extract upper triangle (excluding diagonal)
        triu_idx = np.triu_indices(len(all_perms), k=1)
        pairwise_sims = sim_matrix[triu_idx]

        # Compute L2 distances
        from scipy.spatial.distance import pdist
        l2_dists = pdist(perm_embeddings, metric='euclidean')

        # Embedding variance across permutations
        embedding_std = perm_embeddings.std(axis=0).mean()

        # Reference: similarity between different formulations
        ref_emb = get_mean_pooled_embedding(
            model, tokenizer, 'CCCCCCCCCC')['final']  # decane (unrelated)
        ref_sim = cosine_similarity(
            perm_embeddings[:1], ref_emb.reshape(1, -1))[0, 0]

        print(f"    Pairwise cosine sim: mean={pairwise_sims.mean():.6f}, "
              f"min={pairwise_sims.min():.6f}, max={pairwise_sims.max():.6f}")
        print(f"    L2 distance: mean={l2_dists.mean():.4f}, "
              f"max={l2_dists.max():.4f}")
        print(f"    Embedding std (per dim): {embedding_std:.6f}")
        print(f"    Similarity to unrelated molecule: {ref_sim:.4f}")

        results.append({
            'name': name,
            'n_permutations': len(all_perms),
            'n_components': n_components,
            'cosine_sim_mean': float(pairwise_sims.mean()),
            'cosine_sim_min': float(pairwise_sims.min()),
            'cosine_sim_max': float(pairwise_sims.max()),
            'cosine_sim_std': float(pairwise_sims.std()),
            'l2_dist_mean': float(l2_dists.mean()),
            'l2_dist_max': float(l2_dists.max()),
            'embedding_std_per_dim': float(embedding_std),
            'sim_to_unrelated': float(ref_sim),
        })

    # --- Figure ---
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4))
    if len(results) == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        data = [r['cosine_sim_mean'], r['cosine_sim_min'],
                r['sim_to_unrelated']]
        labels = ['Mean perm\nsimilarity', 'Min perm\nsimilarity',
                  'vs Unrelated\nmolecule']
        colors = ['steelblue', 'lightblue', 'coral']
        bars = ax.bar(labels, data, color=colors, edgecolor='k')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title(f'{r["name"]}\n({r["n_permutations"]} permutations)')
        ax.set_ylim([-0.5, 1.05])
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars, data):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'l3c_permutation_sensitivity.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Figure saved: l3c_permutation_sensitivity.png")

    with open(RESULTS_DIR / 'l3c_permutation_sensitivity.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ═════════════════════════════════════════════════════════════════════
# L4a: Frozen vs Fine-Tuned Performance on ESOL
# ═════════════════════════════════════════════════════════════════════

def run_experiment_l4a(model, tokenizer, random_model=None):
    """Compare frozen embeddings at each layer: Ridge and FC head, pretrained vs random."""
    print("\n" + "=" * 70)
    print("L4a: Frozen Embeddings Performance on ESOL")
    print("=" * 70)

    # Load ESOL
    train_df = pd.read_csv(PROJECT_ROOT / 'datasets' / 'esol' / 'esol_train.csv')
    valid_df = pd.read_csv(PROJECT_ROOT / 'datasets' / 'esol' / 'esol_valid.csv')
    test_df = pd.read_csv(PROJECT_ROOT / 'datasets' / 'esol' / 'esol_test.csv')

    measure_col = train_df.columns[1]

    all_train_smiles = train_df['smiles'].tolist()
    train_y = train_df[measure_col].values
    val_smiles = valid_df['smiles'].tolist()
    val_y = valid_df[measure_col].values
    test_smiles = test_df['smiles'].tolist()
    test_y = test_df[measure_col].values

    results = {}

    # --- Extract embeddings at multiple layers ---
    for model_obj, model_label in [(model, 'pretrained'), (random_model, 'random')]:
        if model_obj is None:
            continue

        hooks, intermediate = register_hooks(model_obj)

        # Extract layer-wise embeddings
        layer_embeddings = {l: {'train': [], 'val': [], 'test': []} for l in LAYERS_TO_ANALYZE}

        for split_name, smiles_list in [('train', all_train_smiles),
                                         ('val', val_smiles),
                                         ('test', test_smiles)]:
            for smi in tqdm(smiles_list, desc=f'{model_label} {split_name}',
                            leave=False):
                inputs = tokenizer(smi, return_tensors='pt', padding=False,
                                   truncation=True, max_length=512)
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                intermediate.clear()
                with torch.no_grad():
                    _ = model_obj(**inputs)

                mask = inputs['attention_mask'][0].cpu().float()
                for l in LAYERS_TO_ANALYZE:
                    if l in intermediate:
                        hs = intermediate[l][0]
                        pooled = (hs * mask.unsqueeze(-1)).sum(0) / mask.sum()
                        layer_embeddings[l][split_name].append(pooled.numpy())

        remove_hooks(hooks)

        for l in LAYERS_TO_ANALYZE:
            for split in ['train', 'val', 'test']:
                layer_embeddings[l][split] = np.array(layer_embeddings[l][split])

        # --- 1. Frozen embeddings + Ridge ---
        print(f"\n  {model_label} — Frozen + Ridge:")
        for l in LAYERS_TO_ANALYZE:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(layer_embeddings[l]['train'])
            X_te = scaler.transform(layer_embeddings[l]['test'])

            # Tune alpha
            best_rmse = float('inf')
            for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_tr, train_y)
                X_val = scaler.transform(layer_embeddings[l]['val'])
                val_pred = ridge.predict(X_val)
                val_rmse = np.sqrt(mean_squared_error(val_y, val_pred))
                if val_rmse < best_rmse:
                    best_rmse = val_rmse
                    best_alpha = alpha

            ridge = Ridge(alpha=best_alpha)
            ridge.fit(X_tr, train_y)
            test_pred = ridge.predict(X_te)
            test_rmse = np.sqrt(mean_squared_error(test_y, test_pred))
            r2 = r2_score(test_y, test_pred)

            print(f"    Layer {l}: RMSE={test_rmse:.4f}, R²={r2:.4f} (α={best_alpha})")
            results[f'{model_label}_ridge_layer{l}'] = {
                'rmse': float(test_rmse), 'r2': float(r2), 'alpha': best_alpha}

        # --- 2. Frozen embeddings + FC head (final layer only) ---
        print(f"\n  {model_label} — Frozen + FC Head (3 seeds):")
        fc_rmses = []
        for seed in range(3):
            set_seed(SEED + seed)
            # Use layer 12 embeddings; val for early stopping, test for eval
            _, head = train_head(
                layer_embeddings[12]['train'], train_y,
                layer_embeddings[12]['val'], val_y,
                n_epochs=200, lr=1e-3, patience=20
            )
            head.eval()
            X_te = torch.tensor(layer_embeddings[12]['test'],
                                dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                test_pred = head(X_te).cpu().numpy()
            rmse = np.sqrt(mean_squared_error(test_y, test_pred))
            fc_rmses.append(rmse)
            print(f"    Seed {seed}: Test RMSE={rmse:.4f}")

        results[f'{model_label}_fc_frozen'] = {
            'mean_rmse': float(np.mean(fc_rmses)),
            'std_rmse': float(np.std(fc_rmses)),
        }

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(10, 5))

    # Collect results for bar chart
    categories = []
    rmses = []
    colors_list = []

    for model_label, color in [('pretrained', 'steelblue'), ('random', 'coral')]:
        for l in LAYERS_TO_ANALYZE:
            key = f'{model_label}_ridge_layer{l}'
            if key in results:
                categories.append(f'{model_label[:4]}\nRidge L{l}')
                rmses.append(results[key]['rmse'])
                colors_list.append(color)

        key = f'{model_label}_fc_frozen'
        if key in results:
            categories.append(f'{model_label[:4]}\nFC Head')
            rmses.append(results[key]['mean_rmse'])
            colors_list.append(color)

    x = np.arange(len(categories))
    ax.bar(x, rmses, color=colors_list, edgecolor='k', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('Test RMSE')
    ax.set_title('ESOL: Frozen Embeddings Performance by Layer')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, v in enumerate(rmses):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'l4a_frozen_vs_finetuned.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Figure saved: l4a_frozen_vs_finetuned.png")

    with open(RESULTS_DIR / 'l4a_frozen_vs_finetuned.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ═════════════════════════════════════════════════════════════════════
# L4b: CKA Between Pretrained and Fine-Tuned
# ═════════════════════════════════════════════════════════════════════

def linear_CKA(X, Y):
    """Compute linear Centered Kernel Alignment between two representation matrices.

    X, Y: (n_samples, n_features) — can have different n_features.
    Returns CKA ∈ [0, 1].
    """
    # Center
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Gram matrices
    XtX = X @ X.T
    YtY = Y @ Y.T

    # HSIC
    hsic_xy = np.sum(XtX * YtY)
    hsic_xx = np.sum(XtX * XtX)
    hsic_yy = np.sum(YtY * YtY)

    cka = hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)
    return cka


def run_experiment_l4b(model, tokenizer, random_model=None):
    """CKA between pretrained and random model at each layer."""
    print("\n" + "=" * 70)
    print("L4b: CKA Between Pretrained and Random Model")
    print("=" * 70)

    if random_model is None:
        print("  Skipping — no random model provided")
        return {}

    # Use a subset of ESOL molecules
    test_df = pd.read_csv(PROJECT_ROOT / 'datasets' / 'esol' / 'esol_test.csv')
    smiles_list = test_df['smiles'].tolist()[:100]

    layers = list(range(13))  # all layers 0-12

    # Extract layer-wise embeddings for both models
    def extract_all_layers(mdl, smi_list):
        hooks, intermediate = register_hooks(mdl)
        layer_embs = {l: [] for l in layers}

        for smi in tqdm(smi_list, desc="Extracting", leave=False):
            inputs = tokenizer(smi, return_tensors='pt', padding=False,
                               truncation=True, max_length=512)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            intermediate.clear()
            with torch.no_grad():
                _ = mdl(**inputs)

            mask = inputs['attention_mask'][0].cpu().float()
            for l in layers:
                if l in intermediate:
                    hs = intermediate[l][0]
                    pooled = (hs * mask.unsqueeze(-1)).sum(0) / mask.sum()
                    layer_embs[l].append(pooled.numpy())

        remove_hooks(hooks)
        return {l: np.array(v) for l, v in layer_embs.items() if v}

    print("  Extracting pretrained embeddings...")
    pre_layers = extract_all_layers(model, smiles_list)
    print("  Extracting random embeddings...")
    rand_layers = extract_all_layers(random_model, smiles_list)

    # Compute CKA at each layer
    cka_values = []
    available_layers = sorted(set(pre_layers.keys()) & set(rand_layers.keys()))

    for l in available_layers:
        cka = linear_CKA(pre_layers[l], rand_layers[l])
        cka_values.append(cka)
        print(f"  Layer {l}: CKA = {cka:.4f}")

    # Also compute self-CKA across layers (pretrained)
    self_cka_matrix = np.zeros((len(available_layers), len(available_layers)))
    for i, li in enumerate(available_layers):
        for j, lj in enumerate(available_layers):
            self_cka_matrix[i, j] = linear_CKA(pre_layers[li], pre_layers[lj])

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # CKA pretrained vs random per layer
    axes[0].plot(available_layers, cka_values, '-o', color='steelblue', markersize=6)
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Linear CKA')
    axes[0].set_title('CKA: Pretrained vs Random (per layer)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    # Self-CKA heatmap (pretrained)
    im = axes[1].imshow(self_cka_matrix, cmap='viridis', vmin=0, vmax=1)
    axes[1].set_xticks(range(len(available_layers)))
    axes[1].set_xticklabels(available_layers, fontsize=8)
    axes[1].set_yticks(range(len(available_layers)))
    axes[1].set_yticklabels(available_layers, fontsize=8)
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Layer')
    axes[1].set_title('Self-CKA (Pretrained)')
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'l4b_cka_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Figure saved: l4b_cka_analysis.png")

    results = {
        'layers': available_layers,
        'cka_pretrained_vs_random': [float(x) for x in cka_values],
        'self_cka_pretrained': self_cka_matrix.tolist(),
    }

    with open(RESULTS_DIR / 'l4b_cka_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Phase LCE: Understanding Why MOLFormer Pretraining Helps")
    print("Predict Battery Electrolyte Performance")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")

    # ── L1: Statistical Significance ──
    l1_results = run_experiment_l1()

    # ── Load models ──
    model, tokenizer = load_pretrained_model()
    random_model = load_random_model(model)

    # ── L2a: Electrolyte Molecule Embeddings ──
    l2a_results = run_experiment_l2a(model, tokenizer, random_model)

    # ── L3b: Embedding Additivity ──
    l3b_results = run_experiment_l3b(model, tokenizer)

    # ── L3c: Permutation Sensitivity ──
    l3c_results = run_experiment_l3c(model, tokenizer)

    # ── L3a: Cross-Molecule Attention ──
    l3a_results = run_experiment_l3a(model, tokenizer, random_model)

    # ── L4b: CKA Analysis ──
    l4b_results = run_experiment_l4b(model, tokenizer, random_model)

    # Free random model memory before heavy fine-tuning experiments
    del random_model
    torch.cuda.empty_cache()

    # Reload random model for L4a (needs it)
    random_model = load_random_model(model)

    # ── L4a: Frozen vs Fine-Tuned ──
    l4a_results = run_experiment_l4a(model, tokenizer, random_model)

    del random_model
    torch.cuda.empty_cache()

    # Reload random model for L2b
    random_model = load_random_model(model)

    # ── L2b: Sample Efficiency ──
    l2b_results = run_experiment_l2b(model, tokenizer, random_model)

    print("\n" + "=" * 70)
    print(f"All experiments complete. End time: {datetime.now()}")
    print(f"Results in: {RESULTS_DIR}")
    print(f"Figures in: {FIGURES_DIR}")
    print("=" * 70)

    # ── Summary ──
    print("\n── KEY FINDINGS ──")

    if l1_results:
        ci = l1_results['bootstrap_ci_95']
        print(f"\nL1: RMSE diff CI = [{ci[0]:.3f}, {ci[1]:.3f}]")
        print(f"    Wilcoxon p = {l1_results.get('wilcoxon_p', 'N/A')}")
        if l1_results['cis_overlap']:
            print("    ⚠ Individual RMSE CIs overlap — difference may not be significant")

    if l2a_results:
        for key in ['pretrained', 'random']:
            sep = l2a_results.get(f'{key}_group_separation')
            if sep is not None:
                print(f"\nL2a: {key} group separation = {sep:.4f}")


if __name__ == '__main__':
    main()
"""
Phase 0: PCA Visualization of MOLFormer Representations.

Experiments:
  0A: Molecule-level PCA across layers
  0B: Atom-level PCA with chemical property coloring
  0C: Pretrained vs. randomly initialized model PCA
  0D: Layer trajectory visualization for selected molecules
"""

import os
import sys
import json
import random
import warnings
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform, cosine as cosine_dist
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.disable(logging.WARNING)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import Descriptors, rdmolops

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

SEED = 42
DEVICE = 'cuda:0'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PHASE_DIR / 'results'
FIGURES_DIR = PHASE_DIR / 'figures'
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

N_MOL_SAMPLES = 2000   # molecule-level experiments (0A, 0C, 0D)
N_ATOM_SAMPLES = 500   # atom-level experiment (0B)
LAYERS_TO_ANALYZE = [0, 4, 6, 8, 12]  # for 0A
ATOM_LAYERS = [0, 4, 6]               # for 0B

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()


# ─────────────────────────────────────────────────────────────────────
# Token-to-Atom Mapping (from src/run_experiments.py)
# ─────────────────────────────────────────────────────────────────────

def get_atom_indices_from_smiles(smiles, tokenizer):
    """Map SMILES tokens to atom indices. Returns (atom_map, mol) or (None, None)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    tokens = tokenizer.tokenize(smiles)
    atom_map = []
    current_atom = 0
    num_atoms = mol.GetNumAtoms()

    for tok in tokens:
        if current_atom >= num_atoms:
            atom_map.append(-1)
            continue
        if tok.startswith('['):
            atom_map.append(current_atom)
            current_atom += 1
        elif len(tok) == 1 and tok in 'BCNOPSFIcnops':
            atom_map.append(current_atom)
            current_atom += 1
        elif len(tok) == 2 and tok[0].isupper() and tok[1].islower() and tok in ('Cl', 'Br', 'Si', 'Se', 'se'):
            atom_map.append(current_atom)
            current_atom += 1
        else:
            atom_map.append(-1)

    return atom_map, mol


def get_atom_token_indices(smiles, tokenizer):
    """Get token position indices (with <bos> offset) for atoms.
    Returns (atom_token_indices, mol) or (None, None)."""
    atom_map, mol = get_atom_indices_from_smiles(smiles, tokenizer)
    if atom_map is None:
        return None, None

    # +1 offset for <bos> token
    full_map = [-1] + atom_map + [-1]  # <bos> + tokens + <eos>
    atom_token_indices = [i for i, a in enumerate(full_map) if a >= 0]

    if len(atom_token_indices) != mol.GetNumAtoms():
        return None, None

    return atom_token_indices, mol


# ─────────────────────────────────────────────────────────────────────
# Model Loading & Hidden State Extraction
# ─────────────────────────────────────────────────────────────────────

def load_pretrained_model():
    from transformers import AutoModel, AutoTokenizer
    print("Loading pretrained MOLFormer...")
    tokenizer = AutoTokenizer.from_pretrained('ibm/MoLFormer-XL-both-10pct', trust_remote_code=True)
    model = AutoModel.from_pretrained('ibm/MoLFormer-XL-both-10pct', trust_remote_code=True)
    model = model.to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}")
    return model, tokenizer


def load_random_model(pretrained_model):
    """Create a randomly initialized model with the same architecture."""
    from transformers import AutoModel
    print("Creating randomly initialized MOLFormer...")
    random_model = AutoModel.from_config(pretrained_model.config)
    random_model = random_model.to(DEVICE)
    random_model.eval()
    return random_model


def register_hooks(model):
    """Register forward hooks to capture all 13 layer outputs."""
    hooks = []
    intermediate = {}

    def make_hook(idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                intermediate[idx] = output[0].detach().cpu()
            else:
                intermediate[idx] = output.detach().cpu()
        return hook_fn

    # Embedding layer
    if hasattr(model, 'embeddings'):
        hooks.append(model.embeddings.register_forward_hook(make_hook(0)))

    # Encoder layers
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        for i, layer in enumerate(model.encoder.layer):
            hooks.append(layer.register_forward_hook(make_hook(i + 1)))

    return hooks, intermediate


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


def extract_molecule_embeddings(model, tokenizer, smiles_list, layers=None):
    """Extract molecule-level embeddings (atom-only mean pool) at specified layers.

    Returns:
        mol_embeddings: dict[layer_idx] -> np.array of shape (n_valid, 768)
        valid_indices: list of indices into smiles_list that were successfully processed
    """
    if layers is None:
        layers = list(range(13))

    hooks, intermediate = register_hooks(model)

    mol_embeddings = {l: [] for l in layers}
    valid_indices = []

    for idx, smi in enumerate(tqdm(smiles_list, desc="Extracting mol embeddings")):
        atom_token_indices, mol = get_atom_token_indices(smi, tokenizer)
        if atom_token_indices is None:
            continue

        inputs = tokenizer(smi, return_tensors='pt', padding=False, truncation=True, max_length=200)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        intermediate.clear()
        with torch.no_grad():
            _ = model(**inputs)

        # Mean-pool over atom token positions only
        success = True
        for l in layers:
            if l not in intermediate:
                success = False
                break
            hs = intermediate[l][0]  # (seq_len, 768)
            if max(atom_token_indices) >= hs.shape[0]:
                success = False
                break
            atom_hs = hs[atom_token_indices]  # (n_atoms, 768)
            mol_emb = atom_hs.mean(dim=0).numpy()  # (768,)
            mol_embeddings[l].append(mol_emb)

        if success:
            valid_indices.append(idx)

    remove_hooks(hooks)

    for l in layers:
        mol_embeddings[l] = np.array(mol_embeddings[l])

    print(f"Successfully extracted embeddings for {len(valid_indices)}/{len(smiles_list)} molecules")
    return mol_embeddings, valid_indices


def extract_atom_embeddings(model, tokenizer, smiles_list, layers=None):
    """Extract per-atom hidden states at specified layers.

    Returns:
        atom_embeddings: dict[layer_idx] -> np.array of shape (n_atoms_total, 768)
        atom_properties: list of dicts with atom properties
    """
    if layers is None:
        layers = [0, 4, 6]

    hooks, intermediate = register_hooks(model)

    atom_embeddings = {l: [] for l in layers}
    atom_properties = []

    processed = 0
    for smi in tqdm(smiles_list, desc="Extracting atom embeddings"):
        atom_token_indices, mol = get_atom_token_indices(smi, tokenizer)
        if atom_token_indices is None:
            continue

        inputs = tokenizer(smi, return_tensors='pt', padding=False, truncation=True, max_length=200)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        intermediate.clear()
        with torch.no_grad():
            _ = model(**inputs)

        success = True
        for l in layers:
            if l not in intermediate:
                success = False
                break
            hs = intermediate[l][0]
            if max(atom_token_indices) >= hs.shape[0]:
                success = False
                break
            atom_hs = hs[atom_token_indices].numpy()  # (n_atoms, 768)
            atom_embeddings[l].append(atom_hs)

        if not success:
            continue

        # Collect atom properties
        for atom in mol.GetAtoms():
            atom_properties.append({
                'atom_type': atom.GetSymbol(),
                'is_aromatic': atom.GetIsAromatic(),
                'is_in_ring': atom.IsInRing(),
                'degree': atom.GetDegree(),
                'hybridization': str(atom.GetHybridization()),
            })

        processed += 1

    remove_hooks(hooks)

    for l in layers:
        atom_embeddings[l] = np.concatenate(atom_embeddings[l], axis=0)

    print(f"Processed {processed} molecules, {len(atom_properties)} atoms total")
    return atom_embeddings, atom_properties


# ─────────────────────────────────────────────────────────────────────
# Data Loading & Molecule Properties
# ─────────────────────────────────────────────────────────────────────

def load_qm9_data(n_samples):
    df = pd.read_csv(PROJECT_ROOT / 'datasets' / 'qm9' / 'qm9_test.csv')
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=SEED).reset_index(drop=True)
    print(f"Loaded {len(df)} QM9 molecules")
    return df


def compute_molecule_properties(smiles_list, qm9_df, valid_indices):
    """Compute RDKit properties + extract QM9 properties for valid molecules."""
    props = {
        'mol_weight': [],
        'n_rings': [],
        'n_aromatic_rings': [],
        'n_atoms': [],
        'gap': [],
    }

    for idx in valid_indices:
        smi = smiles_list[idx]
        mol = Chem.MolFromSmiles(smi)

        props['mol_weight'].append(Descriptors.MolWt(mol))
        ring_info = mol.GetRingInfo()
        n_rings = ring_info.NumRings()
        props['n_rings'].append(n_rings)

        # Count aromatic rings
        n_arom = 0
        for ring in ring_info.AtomRings():
            if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
                n_arom += 1
        props['n_aromatic_rings'].append(n_arom)
        props['n_atoms'].append(mol.GetNumAtoms())
        props['gap'].append(qm9_df.iloc[idx]['gap'])

    return {k: np.array(v) for k, v in props.items()}


# ─────────────────────────────────────────────────────────────────────
# Experiment 0A: Molecule-Level PCA
# ─────────────────────────────────────────────────────────────────────

def run_experiment_0a(mol_embeddings, mol_props):
    print("\n" + "=" * 60)
    print("EXPERIMENT 0A: Molecule-Level PCA")
    print("=" * 60)

    # Fit PCA at each layer
    pca_results = {}
    transformed = {}
    for layer in LAYERS_TO_ANALYZE:
        X = mol_embeddings[layer]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=min(50, X.shape[0], X.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        pca_results[layer] = {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'pc1_var': float(pca.explained_variance_ratio_[0]),
            'pc2_var': float(pca.explained_variance_ratio_[1]),
        }
        transformed[layer] = X_pca
        print(f"  Layer {layer}: PC1={pca.explained_variance_ratio_[0]:.3f}, "
              f"PC2={pca.explained_variance_ratio_[1]:.3f}, "
              f"PC1+2={sum(pca.explained_variance_ratio_[:2]):.3f}")

    # Save variance results
    with open(RESULTS_DIR / 'exp0a_variance_explained.json', 'w') as f:
        json.dump({str(k): v for k, v in pca_results.items()}, f, indent=2)

    # ── Figure 0A-1: PCA across layers, colored by HOMO-LUMO gap ──
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for ax, layer in zip(axes, LAYERS_TO_ANALYZE):
        X_pca = transformed[layer]
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=mol_props['gap'],
                       cmap='viridis', s=5, alpha=0.6)
        var_sum = pca_results[layer]['pc1_var'] + pca_results[layer]['pc2_var']
        ax.set_title(f"Layer {layer}\nPC1+2 var: {var_sum:.1%}", fontsize=11)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
    fig.colorbar(sc, ax=axes[-1], label='HOMO-LUMO Gap')
    fig.suptitle('Molecule-Level PCA Across Layers (colored by HOMO-LUMO gap)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exp0a_pca_across_layers.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 0A-2: PCA at layer 6, colored by different properties ──
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    X_pca = transformed[6]
    prop_names = ['mol_weight', 'n_rings', 'n_aromatic_rings', 'gap']
    prop_labels = ['Molecular Weight', 'Number of Rings', 'Aromatic Rings', 'HOMO-LUMO Gap']
    cmaps = ['plasma', 'tab10', 'tab10', 'viridis']

    for ax, pname, plabel, cmap in zip(axes, prop_names, prop_labels, cmaps):
        vals = mol_props[pname]
        if pname in ('n_rings', 'n_aromatic_rings'):
            # Discrete coloring
            unique_vals = sorted(set(vals.astype(int)))
            colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_vals), 2)))
            for i, v in enumerate(unique_vals):
                mask = vals.astype(int) == v
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[colors[i % 10]],
                          s=5, alpha=0.6, label=str(v))
            ax.legend(title=plabel, fontsize=7, markerscale=3)
        else:
            sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=vals, cmap=cmap, s=5, alpha=0.6)
            fig.colorbar(sc, ax=ax, label=plabel)
        ax.set_title(plabel, fontsize=11)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

    fig.suptitle('Layer 6 PCA — Colored by Molecule Properties', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exp0a_pca_layer6_properties.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 0A-3: UMAP at layer 6 ──
    print("  Running UMAP at layer 6...")
    import umap
    X_scaled = StandardScaler().fit_transform(mol_embeddings[6])
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='cosine', random_state=SEED)
    X_umap = reducer.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, pname, plabel, cmap in zip(axes, prop_names, prop_labels, cmaps):
        vals = mol_props[pname]
        if pname in ('n_rings', 'n_aromatic_rings'):
            unique_vals = sorted(set(vals.astype(int)))
            colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_vals), 2)))
            for i, v in enumerate(unique_vals):
                mask = vals.astype(int) == v
                ax.scatter(X_umap[mask, 0], X_umap[mask, 1], c=[colors[i % 10]],
                          s=5, alpha=0.6, label=str(v))
            ax.legend(title=plabel, fontsize=7, markerscale=3)
        else:
            sc = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=vals, cmap=cmap, s=5, alpha=0.6)
            fig.colorbar(sc, ax=ax, label=plabel)
        ax.set_title(plabel, fontsize=11)
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')

    fig.suptitle('Layer 6 UMAP — Colored by Molecule Properties', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exp0a_umap_layer6.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 0A-4: Variance explained curves ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for layer in LAYERS_TO_ANALYZE:
        cumvar = pca_results[layer]['cumulative_variance']
        ax.plot(range(1, len(cumvar) + 1), cumvar, marker='.', label=f'Layer {layer}', linewidth=2)
    ax.set_xlabel('Number of Principal Components', fontsize=12)
    ax.set_ylabel('Cumulative Variance Explained', fontsize=12)
    ax.set_title('Cumulative Variance Explained by Layer', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 50)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exp0a_variance_explained.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Experiment 0A plots saved.")
    return pca_results


# ─────────────────────────────────────────────────────────────────────
# Experiment 0B: Atom-Level PCA
# ─────────────────────────────────────────────────────────────────────

def run_experiment_0b(model, tokenizer, smiles_list_0b):
    print("\n" + "=" * 60)
    print("EXPERIMENT 0B: Atom-Level PCA")
    print("=" * 60)

    atom_embeddings, atom_properties = extract_atom_embeddings(
        model, tokenizer, smiles_list_0b, layers=ATOM_LAYERS)

    df_atoms = pd.DataFrame(atom_properties)
    print(f"  Atom type distribution:\n{df_atoms['atom_type'].value_counts().head(10)}")

    # Balance sampling: cap carbon at 2× second most common
    type_counts = df_atoms['atom_type'].value_counts()
    if 'C' in type_counts.index and len(type_counts) > 1:
        second_most = type_counts.iloc[1]  # second most common
        carbon_cap = 2 * second_most
        carbon_indices = df_atoms[df_atoms['atom_type'] == 'C'].index.tolist()
        non_carbon_indices = df_atoms[df_atoms['atom_type'] != 'C'].index.tolist()

        if len(carbon_indices) > carbon_cap:
            np.random.seed(SEED)
            sampled_carbons = np.random.choice(carbon_indices, carbon_cap, replace=False).tolist()
        else:
            sampled_carbons = carbon_indices

        balanced_indices = sorted(sampled_carbons + non_carbon_indices)
    else:
        balanced_indices = list(range(len(df_atoms)))

    df_balanced = df_atoms.iloc[balanced_indices].reset_index(drop=True)
    print(f"  After balancing: {len(df_balanced)} atoms")
    print(f"  Balanced distribution:\n{df_balanced['atom_type'].value_counts().head(10)}")

    # Save atom counts
    atom_counts = {
        'before_balancing': df_atoms['atom_type'].value_counts().to_dict(),
        'after_balancing': df_balanced['atom_type'].value_counts().to_dict(),
        'total_before': len(df_atoms),
        'total_after': len(df_balanced),
    }
    with open(RESULTS_DIR / 'exp0b_atom_counts.json', 'w') as f:
        json.dump(atom_counts, f, indent=2)

    # PCA on balanced atoms
    balanced_embeddings = {}
    for l in ATOM_LAYERS:
        balanced_embeddings[l] = atom_embeddings[l][balanced_indices]

    # ── Figure 0B-1: Atom PCA grid (3 layers × 4 properties) ──
    fig, axes = plt.subplots(3, 4, figsize=(22, 15))
    prop_cols = ['atom_type', 'is_aromatic', 'is_in_ring', 'degree']
    prop_titles = ['Atom Type', 'Aromatic', 'Ring Member', 'Degree']

    for row, layer in enumerate(ATOM_LAYERS):
        X = balanced_embeddings[layer]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        var_explained = sum(pca.explained_variance_ratio_[:2])

        for col, (prop, title) in enumerate(zip(prop_cols, prop_titles)):
            ax = axes[row, col]
            vals = df_balanced[prop].values

            if prop == 'atom_type':
                # Color by atom type (categorical)
                unique_types = sorted(df_balanced['atom_type'].unique())
                type_colors = {t: plt.cm.tab10(i % 10) for i, t in enumerate(unique_types)}
                for atype in unique_types:
                    mask = vals == atype
                    if mask.sum() > 0:
                        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                                  c=[type_colors[atype]], s=3, alpha=0.5, label=atype)
                ax.legend(fontsize=6, markerscale=3, loc='best')
            elif prop in ('is_aromatic', 'is_in_ring'):
                # Boolean coloring
                for val, color, label in [(True, 'red', 'Yes'), (False, 'blue', 'No')]:
                    mask = vals == val
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                              c=color, s=3, alpha=0.4, label=label)
                ax.legend(fontsize=8, markerscale=3)
            else:
                # Degree (discrete numeric)
                unique_vals = sorted(set(vals.astype(int)))
                colors_list = plt.cm.viridis(np.linspace(0, 1, max(len(unique_vals), 2)))
                for i, v in enumerate(unique_vals):
                    mask = vals.astype(int) == v
                    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                              c=[colors_list[i]], s=3, alpha=0.5, label=str(v))
                ax.legend(title='Degree', fontsize=6, markerscale=3)

            if row == 0:
                ax.set_title(title, fontsize=12)
            if col == 0:
                ax.set_ylabel(f'Layer {layer}\nPC2', fontsize=11)
            else:
                ax.set_ylabel('PC2')
            ax.set_xlabel('PC1')

            if col == 0:
                ax.annotate(f'var: {var_explained:.1%}', xy=(0.02, 0.98),
                           xycoords='axes fraction', fontsize=8, va='top')

    fig.suptitle('Atom-Level PCA: Layers × Properties (balanced sampling)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exp0b_atom_pca_grid.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 0B-2: Carbon-only PCA at layer 6, colored by hybridization ──
    carbon_mask = df_balanced['atom_type'] == 'C'
    if carbon_mask.sum() > 50:
        X_carbon = balanced_embeddings[6][carbon_mask.values]
        carbon_hyb = df_balanced.loc[carbon_mask, 'hybridization'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_carbon)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(8, 6))
        unique_hyb = sorted(set(carbon_hyb))
        hyb_colors = {h: plt.cm.Set1(i % 9) for i, h in enumerate(unique_hyb)}
        for hyb in unique_hyb:
            mask = carbon_hyb == hyb
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=[hyb_colors[hyb]], s=8, alpha=0.5, label=hyb)
        ax.legend(title='Hybridization', fontsize=10, markerscale=3)
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        var_str = f"{sum(pca.explained_variance_ratio_[:2]):.1%}"
        ax.set_title(f'Carbon Atoms at Layer 6 — Colored by Hybridization\n(var explained: {var_str})',
                    fontsize=13)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'exp0b_carbon_hybridization.png', dpi=150, bbox_inches='tight')
        plt.close()

    print("  Experiment 0B plots saved.")


# ─────────────────────────────────────────────────────────────────────
# Experiment 0C: Pretrained vs Random PCA
# ─────────────────────────────────────────────────────────────────────

def run_experiment_0c(pretrained_model, tokenizer, smiles_list, mol_props):
    print("\n" + "=" * 60)
    print("EXPERIMENT 0C: Pretrained vs Random PCA")
    print("=" * 60)

    # We already have pretrained embeddings at layer 6 from 0A
    # Extract random model embeddings
    random_model = load_random_model(pretrained_model)

    print("  Extracting random model embeddings at layer 6...")
    rand_embeddings, rand_valid = extract_molecule_embeddings(
        random_model, tokenizer, smiles_list, layers=[6])

    # Use only molecules valid in both
    pretrained_embeddings, pretrained_valid = extract_molecule_embeddings(
        pretrained_model, tokenizer, smiles_list, layers=[6])

    # Find common valid indices
    common_valid = sorted(set(pretrained_valid) & set(rand_valid))
    print(f"  Common valid molecules: {len(common_valid)}")

    # Recompute properties for common molecules
    common_props = compute_molecule_properties(smiles_list, qm9_df_global, common_valid)

    # Map to array indices
    pre_idx_map = {v: i for i, v in enumerate(pretrained_valid)}
    rand_idx_map = {v: i for i, v in enumerate(rand_valid)}
    pre_indices = [pre_idx_map[v] for v in common_valid]
    rand_indices = [rand_idx_map[v] for v in common_valid]

    X_pre = pretrained_embeddings[6][pre_indices]
    X_rand = rand_embeddings[6][rand_indices]

    # Separate PCA for each
    scaler_pre = StandardScaler()
    pca_pre = PCA(n_components=50)
    X_pre_pca = pca_pre.fit_transform(scaler_pre.fit_transform(X_pre))

    scaler_rand = StandardScaler()
    pca_rand = PCA(n_components=50)
    X_rand_pca = pca_rand.fit_transform(scaler_rand.fit_transform(X_rand))

    # Save variance
    var_results = {
        'pretrained': {
            'pc1_var': float(pca_pre.explained_variance_ratio_[0]),
            'pc2_var': float(pca_pre.explained_variance_ratio_[1]),
            'cumulative_10': float(np.cumsum(pca_pre.explained_variance_ratio_)[:10][-1]),
        },
        'random': {
            'pc1_var': float(pca_rand.explained_variance_ratio_[0]),
            'pc2_var': float(pca_rand.explained_variance_ratio_[1]),
            'cumulative_10': float(np.cumsum(pca_rand.explained_variance_ratio_)[:10][-1]),
        }
    }
    with open(RESULTS_DIR / 'exp0c_variance_explained.json', 'w') as f:
        json.dump(var_results, f, indent=2)

    # ── Figure 0C-1: Colored by HOMO-LUMO gap ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    gap = common_props['gap']
    vmin, vmax = np.percentile(gap, [2, 98])

    for ax, X_pca, pca_obj, title in [
        (axes[0], X_pre_pca, pca_pre, 'Pretrained'),
        (axes[1], X_rand_pca, pca_rand, 'Random Init'),
    ]:
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=gap, cmap='viridis',
                       s=5, alpha=0.6, vmin=vmin, vmax=vmax)
        var_sum = pca_obj.explained_variance_ratio_[0] + pca_obj.explained_variance_ratio_[1]
        ax.set_title(f'{title} (Layer 6)\nPC1+2 var: {var_sum:.1%}', fontsize=12)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
    fig.colorbar(sc, ax=axes, label='HOMO-LUMO Gap')
    fig.suptitle('Pretrained vs Random — HOMO-LUMO Gap', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exp0c_pretrained_vs_random_gap.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 0C-2: Colored by ring count ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rings = common_props['n_rings'].astype(int)
    unique_rings = sorted(set(rings))
    ring_colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_rings), 2)))

    for ax, X_pca, pca_obj, title in [
        (axes[0], X_pre_pca, pca_pre, 'Pretrained'),
        (axes[1], X_rand_pca, pca_rand, 'Random Init'),
    ]:
        for i, r in enumerate(unique_rings):
            mask = rings == r
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[ring_colors[i % 10]],
                      s=5, alpha=0.6, label=str(r))
        var_sum = pca_obj.explained_variance_ratio_[0] + pca_obj.explained_variance_ratio_[1]
        ax.set_title(f'{title} (Layer 6)\nPC1+2 var: {var_sum:.1%}', fontsize=12)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend(title='Rings', fontsize=7, markerscale=3)

    fig.suptitle('Pretrained vs Random — Ring Count', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exp0c_pretrained_vs_random_rings.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Pretrained PC1+2 var: {var_results['pretrained']['pc1_var'] + var_results['pretrained']['pc2_var']:.1%}")
    print(f"  Random PC1+2 var: {var_results['random']['pc1_var'] + var_results['random']['pc2_var']:.1%}")
    print("  Experiment 0C plots saved.")

    # Clean up random model
    del random_model
    torch.cuda.empty_cache()

    return var_results


# ─────────────────────────────────────────────────────────────────────
# Experiment 0D: Trajectory Visualization
# ─────────────────────────────────────────────────────────────────────

TRAJECTORY_MOLECULES = {
    'aliphatic': {
        'Methane': 'C',
        'Ethanol': 'CCO',
        'Propane': 'CCC',
        'Butanoic acid': 'CCCC(=O)O',
        'Hexane': 'CCCCCC',
    },
    'aromatic': {
        'Benzene': 'c1ccccc1',
        'Toluene': 'Cc1ccccc1',
        'Aniline': 'Nc1ccccc1',
        'Phenol': 'Oc1ccccc1',
        'Pyridine': 'c1ccncc1',
    },
    'multi_ring': {
        'Naphthalene': 'c1ccc2ccccc2c1',
        'Indole': 'c1ccc2[nH]ccc2c1',
        'Quinoline': 'c1ccc2ncccc2c1',
        'Biphenyl': 'c1ccc(-c2ccccc2)cc1',
        'Anthracene': 'c1ccc2cc3ccccc3cc2c1',
    },
    'heteroatom': {
        'Thiophene': 'c1ccsc1',
        'Furan': 'c1ccoc1',
        'Imidazole': 'c1cnc[nH]1',
        'Morpholine': 'C1COCCN1',
        'Piperazine': 'C1CNCCN1',
    },
}

CATEGORY_COLORS = {
    'aliphatic': '#3498db',
    'aromatic': '#e74c3c',
    'multi_ring': '#2ecc71',
    'heteroatom': '#f39c12',
}


def run_experiment_0d(model, tokenizer):
    print("\n" + "=" * 60)
    print("EXPERIMENT 0D: Trajectory Visualization")
    print("=" * 60)

    # Validate all SMILES
    valid_molecules = {}
    for category, mols in TRAJECTORY_MOLECULES.items():
        valid_molecules[category] = {}
        for name, smi in mols.items():
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"  WARNING: Invalid SMILES for {name}: {smi}")
                continue
            atom_indices, _ = get_atom_token_indices(smi, tokenizer)
            if atom_indices is None:
                print(f"  WARNING: Token mapping failed for {name}: {smi}")
                continue
            valid_molecules[category][name] = smi

    total_mols = sum(len(v) for v in valid_molecules.values())
    print(f"  Valid molecules: {total_mols}")

    # Extract embeddings at all 13 layers
    hooks, intermediate = register_hooks(model)

    all_trajectories = []  # list of (name, category, (13, 768) array)
    all_names = []
    all_categories = []

    for category, mols in valid_molecules.items():
        for name, smi in mols.items():
            atom_indices, _ = get_atom_token_indices(smi, tokenizer)
            inputs = tokenizer(smi, return_tensors='pt', padding=False, truncation=True, max_length=200)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            intermediate.clear()
            with torch.no_grad():
                _ = model(**inputs)

            trajectory = []
            for layer in range(13):
                hs = intermediate[layer][0]  # (seq_len, 768)
                atom_hs = hs[atom_indices]
                mol_emb = atom_hs.mean(dim=0).numpy()
                trajectory.append(mol_emb)
            trajectory = np.stack(trajectory)  # (13, 768)

            all_trajectories.append(trajectory)
            all_names.append(name)
            all_categories.append(category)

    remove_hooks(hooks)

    # Stack all: (n_mols × 13, 768)
    all_points = np.concatenate(all_trajectories, axis=0)  # (n_mols*13, 768)
    print(f"  Total trajectory points: {all_points.shape}")

    # Shared PCA
    scaler = StandardScaler()
    all_scaled = scaler.fit_transform(all_points)
    pca = PCA(n_components=2)
    all_pca = pca.fit_transform(all_scaled)

    # Reshape back: (n_mols, 13, 2)
    n_mols = len(all_trajectories)
    trajectories_pca = all_pca.reshape(n_mols, 13, 2)

    # Save trajectory data
    traj_data = {}
    for i, (name, cat) in enumerate(zip(all_names, all_categories)):
        traj_data[name] = {
            'category': cat,
            'pc_coords': trajectories_pca[i].tolist(),
        }
    with open(RESULTS_DIR / 'exp0d_trajectory_data.json', 'w') as f:
        json.dump(traj_data, f, indent=2)

    # ── Figure 0D-1: Trajectories colored by category ──
    fig, ax = plt.subplots(figsize=(12, 9))

    for i, (name, cat) in enumerate(zip(all_names, all_categories)):
        coords = trajectories_pca[i]  # (13, 2)
        color = CATEGORY_COLORS[cat]

        # Draw trajectory line
        ax.plot(coords[:, 0], coords[:, 1], '-', color=color, alpha=0.6, linewidth=1.5)

        # Draw arrows between consecutive layers
        for l in range(12):
            dx = coords[l+1, 0] - coords[l, 0]
            dy = coords[l+1, 1] - coords[l, 1]
            ax.annotate('', xy=(coords[l+1, 0], coords[l+1, 1]),
                       xytext=(coords[l, 0], coords[l, 1]),
                       arrowprops=dict(arrowstyle='->', color=color, alpha=0.4, lw=1))

        # Mark start (layer 0) and end (layer 12)
        ax.scatter(coords[0, 0], coords[0, 1], marker='o', c=color, s=60, zorder=5,
                  edgecolors='black', linewidths=0.5)
        ax.scatter(coords[-1, 0], coords[-1, 1], marker='*', c=color, s=120, zorder=5,
                  edgecolors='black', linewidths=0.5)

        # Label endpoint
        ax.annotate(name, (coords[-1, 0], coords[-1, 1]), fontsize=6,
                   textcoords='offset points', xytext=(5, 5), alpha=0.8)

    # Legend
    legend_elements = [
        Line2D([0], [0], color=c, label=cat.replace('_', ' ').title(), linewidth=2)
        for cat, c in CATEGORY_COLORS.items()
    ]
    legend_elements.append(Line2D([0], [0], marker='o', color='gray', label='Layer 0',
                                  markerfacecolor='gray', markersize=8, linewidth=0))
    legend_elements.append(Line2D([0], [0], marker='*', color='gray', label='Layer 12',
                                  markerfacecolor='gray', markersize=12, linewidth=0))
    ax.legend(handles=legend_elements, fontsize=9, loc='best')

    var_str = f"PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%}"
    ax.set_title(f'Molecule Representation Trajectories Through Layers\n({var_str})', fontsize=13)
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exp0d_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ── Figure 0D-2: Layer-wise inter-molecule distances ──
    # Compute pairwise cosine distances at each layer
    layer_distances = {'overall': [], 'within_group': [], 'between_group': []}

    for layer in range(13):
        # Get embeddings at this layer for all molecules
        layer_embs = np.array([all_trajectories[i][layer] for i in range(n_mols)])

        # Overall pairwise cosine distance
        dists = pdist(layer_embs, metric='cosine')
        layer_distances['overall'].append(float(np.mean(dists)))

        # Within-group and between-group
        within = []
        between = []
        for i in range(n_mols):
            for j in range(i+1, n_mols):
                d = cosine_dist(layer_embs[i], layer_embs[j])
                if all_categories[i] == all_categories[j]:
                    within.append(d)
                else:
                    between.append(d)

        layer_distances['within_group'].append(float(np.mean(within)) if within else 0)
        layer_distances['between_group'].append(float(np.mean(between)) if between else 0)

    with open(RESULTS_DIR / 'exp0d_layer_distances.json', 'w') as f:
        json.dump(layer_distances, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    layers = list(range(13))
    ax.plot(layers, layer_distances['overall'], 'k-o', label='Overall', linewidth=2)
    ax.plot(layers, layer_distances['within_group'], 'b--o', label='Within group', linewidth=2)
    ax.plot(layers, layer_distances['between_group'], 'r--o', label='Between groups', linewidth=2)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Cosine Distance', fontsize=12)
    ax.set_title('Pairwise Molecule Distance by Layer', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exp0d_layer_distances.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  Experiment 0D plots saved.")
    return layer_distances


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

# Global reference for qm9 dataframe (used in 0C property computation)
qm9_df_global = None

def main():
    global qm9_df_global

    print("=" * 60)
    print("Phase 0: PCA Visualization of MOLFormer Representations")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")

    # Load model
    model, tokenizer = load_pretrained_model()

    # Load data
    qm9_df = load_qm9_data(N_MOL_SAMPLES)
    qm9_df_global = qm9_df
    smiles_list = qm9_df['smiles'].tolist()

    # ── Extract molecule embeddings for 0A (reused in 0C) ──
    print("\nExtracting molecule embeddings at layers", LAYERS_TO_ANALYZE, "...")
    mol_embeddings, valid_indices = extract_molecule_embeddings(
        model, tokenizer, smiles_list, layers=LAYERS_TO_ANALYZE)

    mol_props = compute_molecule_properties(smiles_list, qm9_df, valid_indices)
    print(f"Valid molecules with properties: {len(valid_indices)}")

    # ── Experiment 0A ──
    pca_results = run_experiment_0a(mol_embeddings, mol_props)

    # ── Experiment 0B ──
    smiles_0b = qm9_df['smiles'].tolist()[:N_ATOM_SAMPLES]
    run_experiment_0b(model, tokenizer, smiles_0b)

    # ── Experiment 0C ──
    run_experiment_0c(model, tokenizer, smiles_list, mol_props)

    # ── Experiment 0D ──
    layer_distances = run_experiment_0d(model, tokenizer)

    print("\n" + "=" * 60)
    print("ALL PHASE 0 EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"Results: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == '__main__':
    main()
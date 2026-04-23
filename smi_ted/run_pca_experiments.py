"""
Phase 0: PCA Visualization of SMI-TED Representations.

Experiments:
  0A: Molecule-level PCA across layers
  0B: Atom-level PCA with chemical property coloring
  0C: Pretrained vs randomly initialized model PCA
  0D: Layer trajectory visualization for selected molecules

Adapted from:
    https://github.com/ChicagoHAI/interp_Molmformer

SMI-TED migration changes:
    1. Model loading: load_smi_ted() instead of HuggingFace AutoModel
    2. Token-to-atom mapping: SMI-TED regex pattern + canonicalization
    3. Hook paths: model.encoder.tok_emb and
       model.encoder.blocks.layers[i]
    4. Forward pass: model.tokenize() + model.encoder()
    5. Molecule embedding: autoencoder latent z instead of mean pooling
    6. Random model: created from same config via Smi_ted(tokenizer)

Bug fixes from original MOLFormer code:
    1. Removed global variable qm9_df_global, pass qm9_df directly
    2. Added try/except for UMAP (already installed but good practice)
    3. Added length check for carbon sampling edge case
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
import regex as re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, cosine as cosine_dist
from tqdm import tqdm

warnings.filterwarnings('ignore')
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
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

SMI_TED_PATH = ('/Users/xuzetong/projects/materials/models'
                '/smi_ted/inference')
CKPT_FILENAME = 'smi-ted-Light_40.pt'

QM9_PATH = ('/Users/xuzetong/projects/materials/models/smi_ted'
            '/finetune/moleculenet/qm9/qm9_small_test.csv')

RESULTS_DIR = Path('./results/smi_ted/pca')
FIGURES_DIR = Path('./results/smi_ted/pca/figures')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_MOL_SAMPLES = 2000
N_ATOM_SAMPLES = 500
LAYERS_TO_ANALYZE = [0, 4, 6, 8, 12]
ATOM_LAYERS = [0, 4, 6]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()


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
    print(f"Model loaded on {DEVICE}")
    return model


def load_random_model(pretrained_model):
    """
    Create a randomly initialized SMI-TED with same architecture.

    Migration change:
    - MOLFormer: AutoModel.from_config(pretrained_model.config)
    - SMI-TED: create new Smi_ted instance with same tokenizer
      and load_checkpoint resets weights randomly
    """
    sys.path.insert(0, SMI_TED_PATH)
    from smi_ted_light.load import Smi_ted

    print("Creating randomly initialized SMI-TED...")
    random_model = Smi_ted(pretrained_model.tokenizer)
    random_model.config = pretrained_model.config
    random_model.max_len = pretrained_model.max_len
    random_model.n_embd = pretrained_model.n_embd

    # Build encoder and decoder with same config
    from smi_ted_light.load import MoLEncoder, MoLDecoder
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
    print("Random model created")
    return random_model


# ─────────────────────────────────────────────────────────────────────
# Token-to-Atom Mapping
# Migration: SMI-TED regex pattern + canonicalization
# ─────────────────────────────────────────────────────────────────────

SMILES_PATTERN = (
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p"
    r"|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)
ATOM_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p)"
)


def get_atom_indices_from_smiles(smiles):
    """Map SMILES tokens to RDKit atom indices."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    canonical_smi = Chem.MolToSmiles(
        mol, canonical=True, isomericSmiles=False)
    mol_canonical = Chem.MolFromSmiles(canonical_smi)
    if mol_canonical is None:
        return None, None

    tokens = re.findall(SMILES_PATTERN, canonical_smi)
    atom_map = []
    current_atom = 0
    num_atoms = mol_canonical.GetNumAtoms()

    for tok in tokens:
        if current_atom >= num_atoms:
            atom_map.append(-1)
            continue
        if ATOM_PATTERN.fullmatch(tok):
            atom_map.append(current_atom)
            current_atom += 1
        else:
            atom_map.append(-1)

    return atom_map, mol_canonical


def get_atom_token_indices(smiles):
    """
    Get token position indices for atoms with <bos> offset.

    Returns (atom_token_indices, mol) or (None, None)
    """
    atom_map, mol = get_atom_indices_from_smiles(smiles)
    if atom_map is None:
        return None, None

    # +1 offset for <bos> token
    full_map = [-1] + atom_map + [-1]
    atom_token_indices = [i for i, a in enumerate(full_map)
                          if a >= 0]

    if len(atom_token_indices) != mol.GetNumAtoms():
        return None, None

    return atom_token_indices, mol


# ─────────────────────────────────────────────────────────────────────
# Hook Mechanism
# Migration: hook paths updated for SMI-TED structure
# ─────────────────────────────────────────────────────────────────────

def register_hooks(model):
    """
    Register forward hooks on SMI-TED encoder layers.

    Migration change:
    - MOLFormer: model.embeddings + model.encoder.layer[i]
    - SMI-TED:   model.encoder.tok_emb +
                 model.encoder.blocks.layers[i]
    """
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
# Migration: model.tokenize() + model.encoder()
#            + autoencoder latent z for molecule embedding
# ─────────────────────────────────────────────────────────────────────

def extract_molecule_embeddings(model, smiles_list, layers=None):
    """
    Extract molecule-level embeddings at specified layers.

    Migration change:
    - Forward pass uses model.tokenize() and model.encoder()
    - Mean pools over atom token positions only (same as MOLFormer)
    - Note: molecule-level embedding for downstream tasks should
      use autoencoder latent z, but for PCA visualization we use
      per-layer atom mean pooling to analyze layer-wise structure
    """
    if layers is None:
        layers = list(range(13))

    hooks, intermediate = register_hooks(model)
    mol_embeddings = {l: [] for l in layers}
    valid_indices = []

    for idx, smi in enumerate(
            tqdm(smiles_list, desc="Extracting mol embeddings")):
        atom_token_indices, mol = get_atom_token_indices(smi)
        if atom_token_indices is None:
            continue

        # Migration: model.tokenize() returns (idx, mask)
        tok_idx, mask = model.tokenize(smi)

        intermediate.clear()
        with torch.no_grad():
            _ = model.encoder(tok_idx, mask)

        success = True
        for l in layers:
            if l not in intermediate:
                success = False
                break
            hs = intermediate[l]
            if hs.dim() == 3:
                hs = hs[0]
            if max(atom_token_indices) >= hs.shape[0]:
                success = False
                break
            atom_hs = hs[atom_token_indices]
            mol_emb = atom_hs.mean(dim=0).numpy()
            mol_embeddings[l].append(mol_emb)

        if success:
            valid_indices.append(idx)

    remove_hooks(hooks)

    for l in layers:
        mol_embeddings[l] = np.array(mol_embeddings[l])

    print(f"Extracted embeddings for "
          f"{len(valid_indices)}/{len(smiles_list)} molecules")
    return mol_embeddings, valid_indices


def extract_atom_embeddings(model, smiles_list, layers=None):
    """Extract per-atom hidden states at specified layers."""
    if layers is None:
        layers = [0, 4, 6]

    hooks, intermediate = register_hooks(model)
    atom_embeddings = {l: [] for l in layers}
    atom_properties = []
    processed = 0

    for smi in tqdm(smiles_list, desc="Extracting atom embeddings"):
        atom_token_indices, mol = get_atom_token_indices(smi)
        if atom_token_indices is None:
            continue

        tok_idx, mask = model.tokenize(smi)

        intermediate.clear()
        with torch.no_grad():
            _ = model.encoder(tok_idx, mask)

        success = True
        for l in layers:
            if l not in intermediate:
                success = False
                break
            hs = intermediate[l]
            if hs.dim() == 3:
                hs = hs[0]
            if max(atom_token_indices) >= hs.shape[0]:
                success = False
                break
            atom_hs = hs[atom_token_indices].numpy()
            atom_embeddings[l].append(atom_hs)

        if not success:
            continue

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
        atom_embeddings[l] = np.concatenate(
            atom_embeddings[l], axis=0)

    print(f"Processed {processed} molecules, "
          f"{len(atom_properties)} atoms")
    return atom_embeddings, atom_properties


# ─────────────────────────────────────────────────────────────────────
# Data Loading & Molecule Properties
# ─────────────────────────────────────────────────────────────────────

def load_qm9_data(n_samples):
    df = pd.read_csv(QM9_PATH)
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=SEED).reset_index(
            drop=True)
    print(f"Loaded {len(df)} QM9 molecules")
    return df


def compute_molecule_properties(smiles_list, qm9_df, valid_indices):
    """Compute RDKit properties for valid molecules."""
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

        n_arom = 0
        for ring in ring_info.AtomRings():
            if all(mol.GetAtomWithIdx(a).GetIsAromatic()
                   for a in ring):
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

    pca_results = {}
    transformed = {}

    for layer in LAYERS_TO_ANALYZE:
        X = mol_embeddings[layer]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=min(50, X.shape[0], X.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        pca_results[layer] = {
            'explained_variance_ratio':
                pca.explained_variance_ratio_.tolist(),
            'cumulative_variance':
                np.cumsum(pca.explained_variance_ratio_).tolist(),
            'pc1_var': float(pca.explained_variance_ratio_[0]),
            'pc2_var': float(pca.explained_variance_ratio_[1]),
        }
        transformed[layer] = X_pca
        print(f"  Layer {layer}: "
              f"PC1={pca.explained_variance_ratio_[0]:.3f}, "
              f"PC2={pca.explained_variance_ratio_[1]:.3f}, "
              f"PC1+2="
              f"{sum(pca.explained_variance_ratio_[:2]):.3f}")

    with open(RESULTS_DIR / 'exp0a_variance_explained.json',
              'w') as f:
        json.dump({str(k): v for k, v in pca_results.items()},
                  f, indent=2)

    # Figure 0A-1: PCA across layers colored by HOMO-LUMO gap
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for ax, layer in zip(axes, LAYERS_TO_ANALYZE):
        X_pca = transformed[layer]
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                        c=mol_props['gap'],
                        cmap='viridis', s=5, alpha=0.6)
        var_sum = (pca_results[layer]['pc1_var'] +
                   pca_results[layer]['pc2_var'])
        ax.set_title(f"Layer {layer}\n"
                     f"PC1+2 var: {var_sum:.1%}", fontsize=11)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
    fig.colorbar(sc, ax=axes[-1], label='HOMO-LUMO Gap')
    fig.suptitle('SMI-TED Molecule PCA Across Layers '
                 '(colored by HOMO-LUMO gap)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exp0a_pca_across_layers.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 0A-2: PCA at layer 6 by different properties
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    X_pca = transformed[6]
    prop_names = ['mol_weight', 'n_rings', 'n_aromatic_rings', 'gap']
    prop_labels = ['Molecular Weight', 'Number of Rings',
                   'Aromatic Rings', 'HOMO-LUMO Gap']
    cmaps = ['plasma', 'tab10', 'tab10', 'viridis']

    for ax, pname, plabel, cmap in zip(
            axes, prop_names, prop_labels, cmaps):
        vals = mol_props[pname]
        if pname in ('n_rings', 'n_aromatic_rings'):
            unique_vals = sorted(set(vals.astype(int)))
            colors = plt.cm.tab10(
                np.linspace(0, 1, max(len(unique_vals), 2)))
            for i, v in enumerate(unique_vals):
                mask = vals.astype(int) == v
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                           c=[colors[i % 10]],
                           s=5, alpha=0.6, label=str(v))
            ax.legend(title=plabel, fontsize=7, markerscale=3)
        else:
            sc = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                            c=vals, cmap=cmap, s=5, alpha=0.6)
            fig.colorbar(sc, ax=ax, label=plabel)
        ax.set_title(plabel, fontsize=11)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

    fig.suptitle('SMI-TED Layer 6 PCA — Colored by Properties',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exp0a_pca_layer6_properties.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 0A-3: UMAP at layer 6
    try:
        import umap
        print("  Running UMAP at layer 6...")
        X_scaled = StandardScaler().fit_transform(
            mol_embeddings[6])
        reducer = umap.UMAP(
            n_neighbors=30, min_dist=0.3,
            metric='cosine', random_state=SEED)
        X_umap = reducer.fit_transform(X_scaled)

        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        for ax, pname, plabel, cmap in zip(
                axes, prop_names, prop_labels, cmaps):
            vals = mol_props[pname]
            if pname in ('n_rings', 'n_aromatic_rings'):
                unique_vals = sorted(set(vals.astype(int)))
                colors = plt.cm.tab10(
                    np.linspace(0, 1, max(len(unique_vals), 2)))
                for i, v in enumerate(unique_vals):
                    mask = vals.astype(int) == v
                    ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                               c=[colors[i % 10]],
                               s=5, alpha=0.6, label=str(v))
                ax.legend(title=plabel, fontsize=7, markerscale=3)
            else:
                sc = ax.scatter(X_umap[:, 0], X_umap[:, 1],
                                c=vals, cmap=cmap, s=5, alpha=0.6)
                fig.colorbar(sc, ax=ax, label=plabel)
            ax.set_title(plabel, fontsize=11)
            ax.set_xlabel('UMAP1')
            ax.set_ylabel('UMAP2')

        fig.suptitle('SMI-TED Layer 6 UMAP — '
                     'Colored by Properties',
                     fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'exp0a_umap_layer6.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
    except ImportError:
        print("  umap-learn not installed, skipping UMAP")

    # Figure 0A-4: Variance explained curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for layer in LAYERS_TO_ANALYZE:
        cumvar = pca_results[layer]['cumulative_variance']
        ax.plot(range(1, len(cumvar) + 1), cumvar,
                marker='.', label=f'Layer {layer}', linewidth=2)
    ax.set_xlabel('Number of Principal Components', fontsize=12)
    ax.set_ylabel('Cumulative Variance Explained', fontsize=12)
    ax.set_title('SMI-TED Cumulative Variance by Layer',
                 fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 50)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exp0a_variance_explained.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    print("  Experiment 0A plots saved.")
    return pca_results


# ─────────────────────────────────────────────────────────────────────
# Experiment 0B: Atom-Level PCA
# ─────────────────────────────────────────────────────────────────────

def run_experiment_0b(model, smiles_list_0b):
    print("\n" + "=" * 60)
    print("EXPERIMENT 0B: Atom-Level PCA")
    print("=" * 60)

    atom_embeddings, atom_properties = extract_atom_embeddings(
        model, smiles_list_0b, layers=ATOM_LAYERS)

    df_atoms = pd.DataFrame(atom_properties)
    print(f"  Atom distribution:\n"
          f"{df_atoms['atom_type'].value_counts().head(10)}")

    # Balance carbon sampling
    type_counts = df_atoms['atom_type'].value_counts()
    if 'C' in type_counts.index and len(type_counts) > 1:
        # Bug fix: added length check for edge case
        second_most = type_counts.iloc[1]
        carbon_cap = 2 * second_most
        carbon_indices = df_atoms[
            df_atoms['atom_type'] == 'C'].index.tolist()
        non_carbon_indices = df_atoms[
            df_atoms['atom_type'] != 'C'].index.tolist()

        if len(carbon_indices) > carbon_cap:
            np.random.seed(SEED)
            sampled_carbons = np.random.choice(
                carbon_indices, carbon_cap,
                replace=False).tolist()
        else:
            sampled_carbons = carbon_indices

        balanced_indices = sorted(
            sampled_carbons + non_carbon_indices)
    else:
        balanced_indices = list(range(len(df_atoms)))

    df_balanced = df_atoms.iloc[balanced_indices].reset_index(
        drop=True)
    print(f"  After balancing: {len(df_balanced)} atoms")

    balanced_embeddings = {}
    for l in ATOM_LAYERS:
        balanced_embeddings[l] = atom_embeddings[l][balanced_indices]

    # Figure 0B-1: Atom PCA grid (3 layers x 4 properties)
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

        for col, (prop, title) in enumerate(
                zip(prop_cols, prop_titles)):
            ax = axes[row, col]
            vals = df_balanced[prop].values

            if prop == 'atom_type':
                unique_types = sorted(
                    df_balanced['atom_type'].unique())
                type_colors = {
                    t: plt.cm.tab10(i % 10)
                    for i, t in enumerate(unique_types)}
                for atype in unique_types:
                    mask = vals == atype
                    if mask.sum() > 0:
                        ax.scatter(
                            X_pca[mask, 0], X_pca[mask, 1],
                            c=[type_colors[atype]],
                            s=3, alpha=0.5, label=atype)
                ax.legend(fontsize=6, markerscale=3, loc='best')
            elif prop in ('is_aromatic', 'is_in_ring'):
                for val, color, label in [
                    (True, 'red', 'Yes'),
                    (False, 'blue', 'No')
                ]:
                    mask = vals == val
                    ax.scatter(
                        X_pca[mask, 0], X_pca[mask, 1],
                        c=color, s=3, alpha=0.4, label=label)
                ax.legend(fontsize=8, markerscale=3)
            else:
                unique_vals = sorted(set(vals.astype(int)))
                colors_list = plt.cm.viridis(
                    np.linspace(0, 1, max(len(unique_vals), 2)))
                for i, v in enumerate(unique_vals):
                    mask = vals.astype(int) == v
                    ax.scatter(
                        X_pca[mask, 0], X_pca[mask, 1],
                        c=[colors_list[i]],
                        s=3, alpha=0.5, label=str(v))
                ax.legend(title='Degree', fontsize=6,
                          markerscale=3)

            if row == 0:
                ax.set_title(title, fontsize=12)
            if col == 0:
                ax.set_ylabel(f'Layer {layer}\nPC2', fontsize=11)
            else:
                ax.set_ylabel('PC2')
            ax.set_xlabel('PC1')

            if col == 0:
                ax.annotate(
                    f'var: {var_explained:.1%}',
                    xy=(0.02, 0.98),
                    xycoords='axes fraction',
                    fontsize=8, va='top')

    fig.suptitle('SMI-TED Atom-Level PCA: Layers x Properties',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exp0b_atom_pca_grid.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 0B-2: Carbon hybridization at layer 6
    carbon_mask = df_balanced['atom_type'] == 'C'
    if carbon_mask.sum() > 50:
        X_carbon = balanced_embeddings[6][carbon_mask.values]
        carbon_hyb = df_balanced.loc[
            carbon_mask, 'hybridization'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_carbon)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(8, 6))
        unique_hyb = sorted(set(carbon_hyb))
        hyb_colors = {
            h: plt.cm.Set1(i % 9)
            for i, h in enumerate(unique_hyb)}
        for hyb in unique_hyb:
            mask = carbon_hyb == hyb
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=[hyb_colors[hyb]],
                       s=8, alpha=0.5, label=hyb)
        ax.legend(title='Hybridization', fontsize=10,
                  markerscale=3)
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        var_str = (f"{sum(pca.explained_variance_ratio_[:2]):.1%}")
        ax.set_title(
            f'SMI-TED Carbon Atoms at Layer 6 — Hybridization\n'
            f'(var explained: {var_str})', fontsize=13)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'exp0b_carbon_hybridization.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

    print("  Experiment 0B plots saved.")


# ─────────────────────────────────────────────────────────────────────
# Experiment 0C: Pretrained vs Random PCA
# Bug fix: pass qm9_df directly instead of using global variable
# ─────────────────────────────────────────────────────────────────────

def run_experiment_0c(pretrained_model, random_model,
                      smiles_list, qm9_df):
    """
    Compare pretrained vs randomly initialized SMI-TED.

    Bug fix from original MOLFormer code:
    - Removed global variable qm9_df_global
    - qm9_df is now passed directly as a parameter
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 0C: Pretrained vs Random PCA")
    print("=" * 60)

    print("  Extracting pretrained embeddings at layer 6...")
    pre_embeddings, pre_valid = extract_molecule_embeddings(
        pretrained_model, smiles_list, layers=[6])

    print("  Extracting random embeddings at layer 6...")
    rand_embeddings, rand_valid = extract_molecule_embeddings(
        random_model, smiles_list, layers=[6])

    common_valid = sorted(set(pre_valid) & set(rand_valid))
    print(f"  Common valid molecules: {len(common_valid)}")

    # Bug fix: compute properties using passed qm9_df
    common_props = compute_molecule_properties(
        smiles_list, qm9_df, common_valid)

    pre_idx_map = {v: i for i, v in enumerate(pre_valid)}
    rand_idx_map = {v: i for i, v in enumerate(rand_valid)}
    pre_indices = [pre_idx_map[v] for v in common_valid]
    rand_indices = [rand_idx_map[v] for v in common_valid]

    X_pre = pre_embeddings[6][pre_indices]
    X_rand = rand_embeddings[6][rand_indices]

    scaler_pre = StandardScaler()
    pca_pre = PCA(n_components=50)
    X_pre_pca = pca_pre.fit_transform(
        scaler_pre.fit_transform(X_pre))

    scaler_rand = StandardScaler()
    pca_rand = PCA(n_components=50)
    X_rand_pca = pca_rand.fit_transform(
        scaler_rand.fit_transform(X_rand))

    var_results = {
        'pretrained': {
            'pc1_var': float(pca_pre.explained_variance_ratio_[0]),
            'pc2_var': float(pca_pre.explained_variance_ratio_[1]),
            'cumulative_10': float(
                np.cumsum(pca_pre.explained_variance_ratio_)[:10][-1]),
        },
        'random': {
            'pc1_var': float(pca_rand.explained_variance_ratio_[0]),
            'pc2_var': float(pca_rand.explained_variance_ratio_[1]),
            'cumulative_10': float(
                np.cumsum(
                    pca_rand.explained_variance_ratio_)[:10][-1]),
        }
    }

    with open(RESULTS_DIR / 'exp0c_variance_explained.json',
              'w') as f:
        json.dump(var_results, f, indent=2)

    gap = common_props['gap']
    vmin, vmax = np.percentile(gap, [2, 98])

    # Figure 0C-1: colored by HOMO-LUMO gap
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, X_pca, pca_obj, title in [
        (axes[0], X_pre_pca, pca_pre, 'Pretrained'),
        (axes[1], X_rand_pca, pca_rand, 'Random Init'),
    ]:
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                        c=gap, cmap='viridis',
                        s=5, alpha=0.6,
                        vmin=vmin, vmax=vmax)
        var_sum = (pca_obj.explained_variance_ratio_[0] +
                   pca_obj.explained_variance_ratio_[1])
        ax.set_title(f'SMI-TED {title} (Layer 6)\n'
                     f'PC1+2 var: {var_sum:.1%}', fontsize=12)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
    fig.colorbar(sc, ax=axes, label='HOMO-LUMO Gap')
    fig.suptitle('Pretrained vs Random — HOMO-LUMO Gap',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / 'exp0c_pretrained_vs_random_gap.png',
        dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 0C-2: colored by ring count
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rings = common_props['n_rings'].astype(int)
    unique_rings = sorted(set(rings))
    ring_colors = plt.cm.tab10(
        np.linspace(0, 1, max(len(unique_rings), 2)))

    for ax, X_pca, pca_obj, title in [
        (axes[0], X_pre_pca, pca_pre, 'Pretrained'),
        (axes[1], X_rand_pca, pca_rand, 'Random Init'),
    ]:
        for i, r in enumerate(unique_rings):
            mask = rings == r
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=[ring_colors[i % 10]],
                       s=5, alpha=0.6, label=str(r))
        var_sum = (pca_obj.explained_variance_ratio_[0] +
                   pca_obj.explained_variance_ratio_[1])
        ax.set_title(f'SMI-TED {title} (Layer 6)\n'
                     f'PC1+2 var: {var_sum:.1%}', fontsize=12)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend(title='Rings', fontsize=7, markerscale=3)

    fig.suptitle('Pretrained vs Random — Ring Count',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / 'exp0c_pretrained_vs_random_rings.png',
        dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Pretrained PC1+2 var: "
          f"{var_results['pretrained']['pc1_var'] + var_results['pretrained']['pc2_var']:.1%}")
    print(f"  Random PC1+2 var: "
          f"{var_results['random']['pc1_var'] + var_results['random']['pc2_var']:.1%}")
    print("  Experiment 0C plots saved.")

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


def run_experiment_0d(model):
    print("\n" + "=" * 60)
    print("EXPERIMENT 0D: Trajectory Visualization")
    print("=" * 60)

    hooks, intermediate = register_hooks(model)

    all_trajectories = []
    all_names = []
    all_categories = []

    for category, mols in TRAJECTORY_MOLECULES.items():
        for name, smi in mols.items():
            atom_indices, _ = get_atom_token_indices(smi)
            if atom_indices is None:
                print(f"  Skipping {name}: token mapping failed")
                continue

            tok_idx, mask = model.tokenize(smi)
            intermediate.clear()
            with torch.no_grad():
                _ = model.encoder(tok_idx, mask)

            trajectory = []
            for layer in range(13):
                if layer not in intermediate:
                    break
                hs = intermediate[layer]
                if hs.dim() == 3:
                    hs = hs[0]
                if max(atom_indices) >= hs.shape[0]:
                    break
                atom_hs = hs[atom_indices]
                mol_emb = atom_hs.mean(dim=0).numpy()
                trajectory.append(mol_emb)

            if len(trajectory) == 13:
                all_trajectories.append(np.stack(trajectory))
                all_names.append(name)
                all_categories.append(category)

    remove_hooks(hooks)

    if not all_trajectories:
        print("  No valid molecules for trajectory visualization")
        return {}

    # Shared PCA
    all_points = np.concatenate(all_trajectories, axis=0)
    scaler = StandardScaler()
    all_scaled = scaler.fit_transform(all_points)
    pca = PCA(n_components=2)
    all_pca = pca.fit_transform(all_scaled)

    n_mols = len(all_trajectories)
    trajectories_pca = all_pca.reshape(n_mols, 13, 2)

    # Figure 0D-1: Trajectories
    fig, ax = plt.subplots(figsize=(12, 9))

    for i, (name, cat) in enumerate(
            zip(all_names, all_categories)):
        coords = trajectories_pca[i]
        color = CATEGORY_COLORS[cat]

        ax.plot(coords[:, 0], coords[:, 1], '-',
                color=color, alpha=0.6, linewidth=1.5)

        ax.scatter(coords[0, 0], coords[0, 1],
                   marker='o', c=color, s=60, zorder=5,
                   edgecolors='black', linewidths=0.5)
        ax.scatter(coords[-1, 0], coords[-1, 1],
                   marker='*', c=color, s=120, zorder=5,
                   edgecolors='black', linewidths=0.5)

        ax.annotate(name, (coords[-1, 0], coords[-1, 1]),
                    fontsize=6, textcoords='offset points',
                    xytext=(5, 5), alpha=0.8)

    legend_elements = [
        Line2D([0], [0], color=c,
               label=cat.replace('_', ' ').title(),
               linewidth=2)
        for cat, c in CATEGORY_COLORS.items()
    ]
    legend_elements.append(
        Line2D([0], [0], marker='o', color='gray',
               label='Layer 0', markerfacecolor='gray',
               markersize=8, linewidth=0))
    legend_elements.append(
        Line2D([0], [0], marker='*', color='gray',
               label='Layer 12', markerfacecolor='gray',
               markersize=12, linewidth=0))
    ax.legend(handles=legend_elements, fontsize=9, loc='best')

    var_str = (f"PC1: {pca.explained_variance_ratio_[0]:.1%}, "
               f"PC2: {pca.explained_variance_ratio_[1]:.1%}")
    ax.set_title(
        f'SMI-TED Molecule Trajectories Through Layers\n'
        f'({var_str})', fontsize=13)
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exp0d_trajectories.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 0D-2: Layer-wise distances
    layer_distances = {
        'overall': [],
        'within_group': [],
        'between_group': []
    }

    for layer in range(13):
        layer_embs = np.array(
            [all_trajectories[i][layer] for i in range(n_mols)])
        dists = pdist(layer_embs, metric='cosine')
        layer_distances['overall'].append(float(np.mean(dists)))

        within = []
        between = []
        for i in range(n_mols):
            for j in range(i + 1, n_mols):
                d = cosine_dist(layer_embs[i], layer_embs[j])
                if all_categories[i] == all_categories[j]:
                    within.append(d)
                else:
                    between.append(d)

        layer_distances['within_group'].append(
            float(np.mean(within)) if within else 0)
        layer_distances['between_group'].append(
            float(np.mean(between)) if between else 0)

    with open(RESULTS_DIR / 'exp0d_layer_distances.json',
              'w') as f:
        json.dump(layer_distances, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    layers = list(range(13))
    ax.plot(layers, layer_distances['overall'],
            'k-o', label='Overall', linewidth=2)
    ax.plot(layers, layer_distances['within_group'],
            'b--o', label='Within group', linewidth=2)
    ax.plot(layers, layer_distances['between_group'],
            'r--o', label='Between groups', linewidth=2)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Cosine Distance', fontsize=12)
    ax.set_title('SMI-TED Pairwise Molecule Distance by Layer',
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exp0d_layer_distances.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    print("  Experiment 0D plots saved.")
    return layer_distances


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 0: PCA Visualization of SMI-TED Representations")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")

    # Load pretrained model
    model = load_pretrained_model()

    # Load data
    qm9_df = load_qm9_data(N_MOL_SAMPLES)
    smiles_list = qm9_df['smiles'].tolist()

    # Extract molecule embeddings for 0A
    print("\nExtracting molecule embeddings...")
    mol_embeddings, valid_indices = extract_molecule_embeddings(
        model, smiles_list, layers=LAYERS_TO_ANALYZE)

    mol_props = compute_molecule_properties(
        smiles_list, qm9_df, valid_indices)
    print(f"Valid molecules: {len(valid_indices)}")

    # Experiment 0A
    pca_results = run_experiment_0a(mol_embeddings, mol_props)

    # Experiment 0B
    smiles_0b = smiles_list[:N_ATOM_SAMPLES]
    run_experiment_0b(model, smiles_0b)

    # Experiment 0C
    # Bug fix: pass qm9_df directly instead of global variable
    random_model = load_random_model(model)
    run_experiment_0c(model, random_model, smiles_list, qm9_df)
    del random_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Experiment 0D
    run_experiment_0d(model)

    print("\n" + "=" * 60)
    print("ALL PHASE 0 EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"Results: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == '__main__':
    main()
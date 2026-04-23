"""
Mechanistic Interpretability of MOLFormer's SMILES Pretraining.

This script performs three experiments:
1. Attention-distance correlation across all 12 layers × 12 heads
2. Linear probing for chemical properties at each layer
3. Attention head ablation study

Requires: transformers==4.34.0, torch, rdkit, numpy, pandas, sklearn, matplotlib, seaborn
"""

import os
import sys
import json
import random
import warnings
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Suppress noisy warnings
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torch_cache'
warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.disable(logging.WARNING)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdmolops

SEED = 42
DEVICE = 'cuda:0'
WORKSPACE = Path('/workspaces/understanding_how_pretraining__20260311_040651_298c3487_claude_2026-03-11_04-06-54')
RESULTS_DIR = WORKSPACE / 'results'
FIGURES_DIR = WORKSPACE / 'figures'
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()


# ─────────────────────────────────────────────────────────────────────
# Utility: Token-to-Atom Mapping
# ─────────────────────────────────────────────────────────────────────

def smiles_token_to_atom_indices(smiles, tokens):
    """Map SMILES tokens (from MOLFormer tokenizer) to atom indices in the RDKit molecule.

    Returns a list of atom_idx for each token (or -1 if the token doesn't correspond to an atom).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Get canonical SMILES with atom mapping
    # We parse the SMILES character by character to match tokens to atoms
    atom_idx = 0
    token_to_atom = []

    for token in tokens:
        if token in ('<bos>', '<eos>', '<pad>'):
            token_to_atom.append(-1)
            continue

        # Check if this token starts with an atom symbol
        # Atom symbols: C, N, O, S, P, F, Cl, Br, I, B, Si, etc.
        # Also in brackets: [NH], [O-], etc.
        is_atom = False

        if token.startswith('['):
            # Bracket atom - always an atom
            is_atom = True
        elif len(token) >= 1 and token[0].isalpha():
            # Simple atom: C, N, O, c, n, o, etc.
            # But not ring closure digits or bond symbols
            first_char = token[0]
            if first_char in 'cnops':  # aromatic atoms (lowercase)
                is_atom = True
            elif first_char.isupper():
                # Could be atom (C, N, O, S, P, F, B, I) or Cl, Br
                is_atom = True

        if is_atom and atom_idx < mol.GetNumAtoms():
            token_to_atom.append(atom_idx)
            atom_idx += 1
        else:
            token_to_atom.append(-1)

    return token_to_atom


def get_atom_indices_from_smiles(smiles, tokenizer):
    """More robust token-to-atom mapping using RDKit's atom map."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    # Tokenize
    tokens = tokenizer.tokenize(smiles)

    # Use a simple approach: iterate through tokens and track which ones are atoms
    # SMILES tokens that represent atoms: single uppercase letter, uppercase+lowercase, bracket expressions
    atom_map = []
    current_atom = 0
    num_atoms = mol.GetNumAtoms()

    for tok in tokens:
        if current_atom >= num_atoms:
            atom_map.append(-1)
            continue

        # Check if token represents an atom
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


def get_3d_distance_matrix(mol):
    """Generate 3D conformer and return distance matrix."""
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)
    if result != 0:
        params2 = AllChem.ETKDGv3()
        params2.useRandomCoords = True
        params2.randomSeed = 42
        result = AllChem.EmbedMolecule(mol, params2)
        if result != 0:
            return None

    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except:
        pass

    # Get positions for heavy atoms only
    conf = mol.GetConformer()
    heavy_atom_indices = [i for i in range(mol.GetNumAtoms())
                          if mol.GetAtomWithIdx(i).GetAtomicNum() != 1]

    positions = np.array([list(conf.GetAtomPosition(idx)) for idx in heavy_atom_indices])

    # Compute pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    return dist_matrix


def get_atom_properties(mol):
    """Extract chemical properties for each atom in the molecule."""
    props = []
    for atom in mol.GetAtoms():
        props.append({
            'atom_type': atom.GetSymbol(),
            'atomic_num': atom.GetAtomicNum(),
            'is_aromatic': atom.GetIsAromatic(),
            'is_in_ring': atom.IsInRing(),
            'degree': atom.GetDegree(),
            'formal_charge': atom.GetFormalCharge(),
            'num_hs': atom.GetTotalNumHs(),
            'hybridization': str(atom.GetHybridization()),
        })
    return props


# ─────────────────────────────────────────────────────────────────────
# Load Model and Data
# ─────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer():
    from transformers import AutoModel, AutoTokenizer
    print("Loading MOLFormer model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('ibm/MoLFormer-XL-both-10pct', trust_remote_code=True)
    model = AutoModel.from_pretrained('ibm/MoLFormer-XL-both-10pct', trust_remote_code=True)
    model = model.to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}")
    return model, tokenizer


def load_qm9_data(n_samples=1000):
    """Load QM9 SMILES from pre-downloaded dataset."""
    df = pd.read_csv(WORKSPACE / 'datasets' / 'qm9' / 'qm9_test.csv')
    # Use test set for analysis (not contaminated by any training)
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=SEED)
    print(f"Loaded {len(df)} QM9 molecules")
    return df


def load_esol_data():
    """Load ESOL dataset for downstream task evaluation."""
    df = pd.read_csv(WORKSPACE / 'datasets' / 'esol' / 'esol.csv')
    print(f"Loaded {len(df)} ESOL molecules")
    return df


# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT 1: Attention-Distance Correlation
# ─────────────────────────────────────────────────────────────────────

def extract_attention_and_distances(model, tokenizer, smiles_list, max_molecules=500):
    """Extract attention matrices and 3D distance matrices for molecules."""
    results = []

    for smi in tqdm(smiles_list, desc="Extracting attention"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() < 3:
            continue

        # Get 3D distances
        dist_matrix = get_3d_distance_matrix(mol)
        if dist_matrix is None:
            continue

        # Get atom mapping
        atom_map, _ = get_atom_indices_from_smiles(smi, tokenizer)
        if atom_map is None:
            continue

        # Tokenize and get attention
        inputs = tokenizer(smi, return_tensors='pt', padding=False, truncation=True, max_length=200)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # attentions: tuple of (batch, heads, seq_len, seq_len) for each layer
        attentions = [att[0].cpu().numpy() for att in outputs.attentions]  # Remove batch dim
        hidden_states = outputs.last_hidden_state[0].cpu().numpy()

        # Get atom-only attention submatrix
        # Include +1 offset for <bos> token
        full_atom_map = [-1] + atom_map + [-1]  # <bos> + tokens + <eos>
        atom_indices = [i for i, a in enumerate(full_atom_map) if a >= 0]

        num_atoms = mol.GetNumAtoms()
        if len(atom_indices) != num_atoms:
            continue  # Token-atom mapping mismatch

        if num_atoms > dist_matrix.shape[0]:
            continue

        # Extract atom-to-atom attention for each layer and head
        atom_attentions = []  # shape: (n_layers, n_heads, n_atoms, n_atoms)
        for layer_att in attentions:
            layer_atom_att = layer_att[:, atom_indices, :][:, :, atom_indices]  # (heads, atoms, atoms)
            atom_attentions.append(layer_atom_att)
        atom_attentions = np.array(atom_attentions)  # (12, 12, n_atoms, n_atoms)

        # Get hidden states at atom positions for each layer
        # We only have last hidden state from HF model, but we need per-layer
        # For now store what we have

        results.append({
            'smiles': smi,
            'dist_matrix': dist_matrix[:num_atoms, :num_atoms],
            'atom_attentions': atom_attentions,
            'num_atoms': num_atoms,
        })

        if len(results) >= max_molecules:
            break

    print(f"Successfully processed {len(results)} molecules")
    return results


def compute_attention_distance_correlation(results):
    """Compute correlation between attention and 3D distance for each layer and head."""
    n_layers = 12
    n_heads = 12

    # Storage for correlations
    cosine_sims = np.zeros((n_layers, n_heads))
    pearson_corrs = np.zeros((n_layers, n_heads))
    spearman_corrs = np.zeros((n_layers, n_heads))

    # Also stratify by distance range
    short_corrs = np.zeros((n_layers, n_heads))  # <=2A
    medium_corrs = np.zeros((n_layers, n_heads))  # 2-4A
    long_corrs = np.zeros((n_layers, n_heads))    # >4A

    counts = np.zeros((n_layers, n_heads))

    for res in tqdm(results, desc="Computing correlations"):
        dist = res['dist_matrix']
        att = res['atom_attentions']
        n = res['num_atoms']

        if n < 3:
            continue

        # Convert distance to similarity (inverse distance)
        # Avoid division by zero on diagonal
        dist_sim = np.zeros_like(dist)
        mask = dist > 0
        dist_sim[mask] = 1.0 / dist[mask]

        # Get upper triangle (excluding diagonal) for correlation
        triu_idx = np.triu_indices(n, k=1)
        dist_flat = dist[triu_idx]
        dist_sim_flat = dist_sim[triu_idx]

        if len(dist_flat) < 3:
            continue

        # Distance ranges
        short_mask = dist_flat <= 2.0
        medium_mask = (dist_flat > 2.0) & (dist_flat <= 4.0)
        long_mask = dist_flat > 4.0

        for layer in range(n_layers):
            for head in range(n_heads):
                att_matrix = att[layer, head, :n, :n]
                att_flat = att_matrix[triu_idx]

                if att_flat.std() < 1e-10 or dist_sim_flat.std() < 1e-10:
                    continue

                # Cosine similarity between attention and distance similarity
                try:
                    cs = 1 - cosine_dist(att_flat, dist_sim_flat)
                    cosine_sims[layer, head] += cs
                except:
                    continue

                # Pearson correlation (attention vs inverse distance)
                try:
                    pc, _ = pearsonr(att_flat, dist_sim_flat)
                    pearson_corrs[layer, head] += pc
                except:
                    pass

                # Spearman correlation
                try:
                    sc, _ = spearmanr(att_flat, dist_sim_flat)
                    spearman_corrs[layer, head] += sc
                except:
                    pass

                # Stratified correlations
                for mask_name, mask_arr, corr_arr in [
                    ('short', short_mask, short_corrs),
                    ('medium', medium_mask, medium_corrs),
                    ('long', long_mask, long_corrs)
                ]:
                    if mask_arr.sum() >= 3:
                        try:
                            sc2, _ = spearmanr(att_flat[mask_arr], dist_sim_flat[mask_arr])
                            corr_arr[layer, head] += sc2
                        except:
                            pass

                counts[layer, head] += 1

    # Average
    valid = counts > 0
    for arr in [cosine_sims, pearson_corrs, spearman_corrs, short_corrs, medium_corrs, long_corrs]:
        arr[valid] /= counts[valid]

    return {
        'cosine_similarity': cosine_sims,
        'pearson_correlation': pearson_corrs,
        'spearman_correlation': spearman_corrs,
        'short_range_correlation': short_corrs,
        'medium_range_correlation': medium_corrs,
        'long_range_correlation': long_corrs,
        'sample_counts': counts,
    }


def plot_attention_distance_results(corr_results):
    """Create visualizations for Experiment 1."""

    # 1. Heatmap of cosine similarity (layer × head)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, key, title in zip(axes,
                               ['cosine_similarity', 'pearson_correlation', 'spearman_correlation'],
                               ['Cosine Similarity', 'Pearson Correlation', 'Spearman Correlation']):
        data = corr_results[key]
        sns.heatmap(data, ax=ax, cmap='RdBu_r', center=0,
                    xticklabels=range(1, 13), yticklabels=range(1, 13),
                    annot=True, fmt='.2f', annot_kws={'size': 7})
        ax.set_xlabel('Head')
        ax.set_ylabel('Layer')
        ax.set_title(f'{title}\n(Attention vs 1/Distance)')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exp1_attention_distance_heatmaps.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Distance-stratified correlation
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, key, title in zip(axes,
                               ['short_range_correlation', 'medium_range_correlation', 'long_range_correlation'],
                               ['Short Range (≤2Å)', 'Medium Range (2-4Å)', 'Long Range (>4Å)']):
        data = corr_results[key]
        sns.heatmap(data, ax=ax, cmap='RdBu_r', center=0,
                    xticklabels=range(1, 13), yticklabels=range(1, 13),
                    annot=True, fmt='.2f', annot_kws={'size': 7})
        ax.set_xlabel('Head')
        ax.set_ylabel('Layer')
        ax.set_title(f'Spearman Correlation\n{title}')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exp1_distance_stratified.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Layer-wise average correlation
    fig, ax = plt.subplots(figsize=(8, 5))
    layer_means = corr_results['spearman_correlation'].mean(axis=1)
    layer_stds = corr_results['spearman_correlation'].std(axis=1)
    layers = np.arange(1, 13)

    ax.errorbar(layers, layer_means, yerr=layer_stds, marker='o', capsize=3, linewidth=2)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Spearman Correlation', fontsize=12)
    ax.set_title('Layer-wise Attention-Distance Correlation\n(Mean ± Std across 12 heads)', fontsize=13)
    ax.set_xticks(layers)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exp1_layer_wise_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Experiment 1 plots saved.")


# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT 2: Linear Probing
# ─────────────────────────────────────────────────────────────────────

def extract_hidden_states_per_layer(model, tokenizer, smiles_list, max_molecules=2000):
    """Extract hidden states from all layers for probing.

    We hook into the model to get intermediate layer outputs.
    """
    layer_outputs = {i: [] for i in range(13)}  # 0 = embedding, 1-12 = layers
    atom_labels = []

    # Register hooks to capture intermediate hidden states
    hooks = []
    intermediate_outputs = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is a tuple; first element is hidden states
            if isinstance(output, tuple):
                intermediate_outputs[layer_idx] = output[0].detach().cpu()
            else:
                intermediate_outputs[layer_idx] = output.detach().cpu()
        return hook_fn

    # Hook into embedding layer output
    if hasattr(model, 'embeddings'):
        hooks.append(model.embeddings.register_forward_hook(make_hook(0)))

    # Hook into each encoder layer
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        for i, layer in enumerate(model.encoder.layer):
            hooks.append(layer.register_forward_hook(make_hook(i + 1)))

    processed = 0
    for smi in tqdm(smiles_list, desc="Extracting hidden states"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() < 2:
            continue

        # Get atom mapping
        atom_map, mol_obj = get_atom_indices_from_smiles(smi, tokenizer)
        if atom_map is None:
            continue

        # Get atom properties
        props = get_atom_properties(mol_obj)

        # Tokenize
        inputs = tokenizer(smi, return_tensors='pt', padding=False, truncation=True, max_length=200)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        intermediate_outputs.clear()

        with torch.no_grad():
            _ = model(**inputs, output_attentions=False)

        # Map tokens to atoms using atom_map
        full_atom_map = [-1] + atom_map + [-1]  # <bos> + tokens + <eos>
        atom_token_indices = [i for i, a in enumerate(full_atom_map) if a >= 0]

        if len(atom_token_indices) != mol.GetNumAtoms():
            continue

        # Extract hidden states at atom positions for each layer
        for layer_idx in intermediate_outputs:
            hs = intermediate_outputs[layer_idx][0]  # Remove batch dim: (seq_len, hidden)
            atom_hs = hs[atom_token_indices].numpy()  # (n_atoms, hidden)
            layer_outputs[layer_idx].append(atom_hs)

        # Store labels
        for prop in props:
            atom_labels.append(prop)

        processed += 1
        if processed >= max_molecules:
            break

    # Remove hooks
    for h in hooks:
        h.remove()

    # Concatenate
    layer_embeddings = {}
    for layer_idx in layer_outputs:
        if layer_outputs[layer_idx]:
            layer_embeddings[layer_idx] = np.concatenate(layer_outputs[layer_idx], axis=0)

    print(f"Extracted hidden states for {processed} molecules, {len(atom_labels)} atoms")
    return layer_embeddings, atom_labels


def run_linear_probing(layer_embeddings, atom_labels):
    """Train linear probes for chemical properties at each layer."""
    # Prepare labels
    df_labels = pd.DataFrame(atom_labels)

    # Properties to probe
    probing_tasks = {
        'atom_type': df_labels['atom_type'].values,
        'is_aromatic': df_labels['is_aromatic'].astype(int).values,
        'is_in_ring': df_labels['is_in_ring'].astype(int).values,
        'degree': df_labels['degree'].values,
    }

    results = {}

    for task_name, labels in probing_tasks.items():
        print(f"\nProbing for: {task_name}")

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(labels)
        n_classes = len(le.classes_)
        print(f"  Classes: {le.classes_} ({n_classes} classes)")

        # Train/test split
        n = len(y)
        indices = np.random.permutation(n)
        split = int(0.8 * n)
        train_idx, test_idx = indices[:split], indices[split:]

        task_results = {}

        for layer_idx in sorted(layer_embeddings.keys()):
            X = layer_embeddings[layer_idx]
            if len(X) != len(y):
                continue

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Linear probe
            clf = LogisticRegression(max_iter=1000, random_state=SEED, n_jobs=-1, C=1.0)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            task_results[layer_idx] = {'accuracy': acc, 'f1': f1}
            print(f"  Layer {layer_idx}: Acc={acc:.4f}, F1={f1:.4f}")

        results[task_name] = task_results

    return results


def plot_probing_results(probing_results):
    """Visualize linear probing results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    for task_name, task_res in probing_results.items():
        layers = sorted(task_res.keys())
        accs = [task_res[l]['accuracy'] for l in layers]
        axes[0].plot(layers, accs, marker='o', label=task_name, linewidth=2)

    axes[0].set_xlabel('Layer', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Linear Probe Accuracy by Layer', fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(range(13))

    # F1 plot
    for task_name, task_res in probing_results.items():
        layers = sorted(task_res.keys())
        f1s = [task_res[l]['f1'] for l in layers]
        axes[1].plot(layers, f1s, marker='o', label=task_name, linewidth=2)

    axes[1].set_xlabel('Layer', fontsize=12)
    axes[1].set_ylabel('F1 Score (weighted)', fontsize=12)
    axes[1].set_title('Linear Probe F1 Score by Layer', fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(range(13))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exp2_probing_results.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Experiment 2 plots saved.")


# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT 3: Attention Head Ablation
# ─────────────────────────────────────────────────────────────────────

def extract_embeddings_with_ablation(model, tokenizer, smiles_list, labels,
                                      ablate_layer=None, ablate_head=None):
    """Extract molecule embeddings, optionally ablating a specific attention head."""
    embeddings = []
    hooks = []

    if ablate_layer is not None and ablate_head is not None:
        # Hook to zero out a specific head's output
        def ablation_hook(module, input, output):
            # output is tuple (hidden_states, attention_weights)
            hidden = output[0] if isinstance(output, tuple) else output
            # hidden shape: (batch, seq_len, hidden_dim)
            head_dim = hidden.shape[-1] // 12  # 768 / 12 = 64
            start = ablate_head * head_dim
            end = start + head_dim
            hidden[:, :, start:end] = 0
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            target_layer = model.encoder.layer[ablate_layer]
            hooks.append(target_layer.register_forward_hook(ablation_hook))

    for smi in smiles_list:
        inputs = tokenizer(smi, return_tensors='pt', padding=False, truncation=True, max_length=200)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=False)

        # Mean pooling of last hidden state
        hidden = outputs.last_hidden_state[0].cpu().numpy()
        emb = hidden.mean(axis=0)
        embeddings.append(emb)

    for h in hooks:
        h.remove()

    return np.array(embeddings)


def run_ablation_experiment(model, tokenizer, corr_results):
    """Ablate attention heads and measure impact on downstream ESOL prediction."""
    print("\n=== Experiment 3: Ablation Study ===")

    # Load ESOL data
    esol_file = WORKSPACE / 'datasets' / 'esol' / 'esol.csv'
    if not esol_file.exists():
        # Try alternate name
        esol_files = list((WORKSPACE / 'datasets' / 'esol').glob('*.csv'))
        if esol_files:
            esol_file = esol_files[0]
        else:
            print("ESOL dataset not found, skipping ablation")
            return None

    df = pd.read_csv(esol_file)
    smiles_col = 'smiles' if 'smiles' in df.columns else df.columns[0]

    # Find the target column (measured log solubility)
    target_col = None
    for col in df.columns:
        if col != smiles_col and df[col].dtype in [np.float64, np.float32, float]:
            target_col = col
            break

    if target_col is None:
        print("No numeric target found in ESOL")
        return None

    smiles_list = df[smiles_col].tolist()
    labels = df[target_col].values

    # Filter valid SMILES
    valid_idx = [i for i, s in enumerate(smiles_list) if Chem.MolFromSmiles(s) is not None]
    smiles_list = [smiles_list[i] for i in valid_idx]
    labels = labels[valid_idx]

    # Use a subset for speed
    if len(smiles_list) > 500:
        idx = np.random.choice(len(smiles_list), 500, replace=False)
        smiles_list = [smiles_list[i] for i in idx]
        labels = labels[idx]

    # Train/test split
    n = len(smiles_list)
    split = int(0.8 * n)
    perm = np.random.permutation(n)
    train_idx, test_idx = perm[:split], perm[split:]

    train_smiles = [smiles_list[i] for i in train_idx]
    test_smiles = [smiles_list[i] for i in test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, r2_score

    # Baseline: no ablation
    print("Computing baseline embeddings...")
    train_emb = extract_embeddings_with_ablation(model, tokenizer, train_smiles, train_labels)
    test_emb = extract_embeddings_with_ablation(model, tokenizer, test_smiles, test_labels)

    ridge = Ridge(alpha=1.0)
    ridge.fit(train_emb, train_labels)
    pred = ridge.predict(test_emb)
    baseline_mae = mean_absolute_error(test_labels, pred)
    baseline_r2 = r2_score(test_labels, pred)
    print(f"Baseline: MAE={baseline_mae:.4f}, R²={baseline_r2:.4f}")

    # Identify top-5 spatial heads (highest Spearman correlation)
    spearman = corr_results['spearman_correlation']
    flat_idx = np.argsort(spearman.ravel())[::-1]
    top_heads = [(idx // 12, idx % 12) for idx in flat_idx[:5]]
    bottom_heads = [(idx // 12, idx % 12) for idx in flat_idx[-5:]]

    # Random heads for comparison
    random_heads = [(np.random.randint(12), np.random.randint(12)) for _ in range(5)]

    ablation_results = {
        'baseline': {'mae': baseline_mae, 'r2': baseline_r2},
        'top_spatial_heads': {},
        'bottom_spatial_heads': {},
        'random_heads': {},
    }

    for group_name, heads in [('top_spatial_heads', top_heads),
                               ('bottom_spatial_heads', bottom_heads),
                               ('random_heads', random_heads)]:
        print(f"\nAblating {group_name}: {heads}")
        group_maes = []
        group_r2s = []

        for layer, head in heads:
            train_emb_abl = extract_embeddings_with_ablation(
                model, tokenizer, train_smiles, train_labels,
                ablate_layer=layer, ablate_head=head)
            test_emb_abl = extract_embeddings_with_ablation(
                model, tokenizer, test_smiles, test_labels,
                ablate_layer=layer, ablate_head=head)

            ridge_abl = Ridge(alpha=1.0)
            ridge_abl.fit(train_emb_abl, train_labels)
            pred_abl = ridge_abl.predict(test_emb_abl)
            mae = mean_absolute_error(test_labels, pred_abl)
            r2 = r2_score(test_labels, pred_abl)

            ablation_results[group_name][(layer, head)] = {'mae': mae, 'r2': r2}
            group_maes.append(mae)
            group_r2s.append(r2)
            print(f"  Layer {layer}, Head {head}: MAE={mae:.4f}, R²={r2:.4f}")

        ablation_results[group_name]['mean_mae'] = np.mean(group_maes)
        ablation_results[group_name]['mean_r2'] = np.mean(group_r2s)

    return ablation_results


def plot_ablation_results(ablation_results):
    """Visualize ablation results."""
    if ablation_results is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    groups = ['top_spatial_heads', 'bottom_spatial_heads', 'random_heads']
    labels = ['Top Spatial\nHeads', 'Bottom Spatial\nHeads', 'Random\nHeads']
    colors = ['#e74c3c', '#3498db', '#95a5a6']

    baseline_mae = ablation_results['baseline']['mae']
    baseline_r2 = ablation_results['baseline']['r2']

    # MAE plot
    maes = [ablation_results[g]['mean_mae'] for g in groups]
    bars = axes[0].bar(labels, maes, color=colors, alpha=0.8, edgecolor='black')
    axes[0].axhline(y=baseline_mae, color='green', linestyle='--', linewidth=2, label=f'Baseline (MAE={baseline_mae:.3f})')
    axes[0].set_ylabel('MAE (ESOL)', fontsize=12)
    axes[0].set_title('Effect of Head Ablation on ESOL Prediction', fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # R² plot
    r2s = [ablation_results[g]['mean_r2'] for g in groups]
    bars = axes[1].bar(labels, r2s, color=colors, alpha=0.8, edgecolor='black')
    axes[1].axhline(y=baseline_r2, color='green', linestyle='--', linewidth=2, label=f'Baseline (R²={baseline_r2:.3f})')
    axes[1].set_ylabel('R² (ESOL)', fontsize=12)
    axes[1].set_title('Effect of Head Ablation on ESOL Prediction', fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exp3_ablation_results.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Experiment 3 plots saved.")


# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT 4: Attention Pattern Visualization
# ─────────────────────────────────────────────────────────────────────

def visualize_attention_example(model, tokenizer):
    """Visualize attention patterns for a specific molecule."""
    # Aspirin as an example
    smi = 'CC(=O)Oc1ccccc1C(=O)O'

    inputs = tokenizer(smi, return_tensors='pt', padding=False, truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Plot attention for selected layers
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    for layer_idx in range(12):
        ax = axes[layer_idx // 4, layer_idx % 4]
        att = outputs.attentions[layer_idx][0].cpu().numpy().mean(axis=0)  # Average over heads

        n = min(len(tokens), att.shape[0])
        sns.heatmap(att[:n, :n], ax=ax, cmap='Blues',
                    xticklabels=tokens[:n], yticklabels=tokens[:n],
                    square=True, cbar_kws={'shrink': 0.5})
        ax.set_title(f'Layer {layer_idx + 1}', fontsize=10)
        ax.tick_params(labelsize=6)

    plt.suptitle(f'Attention Patterns: {smi}\n(Averaged across 12 heads)', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exp4_attention_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Attention visualization saved.")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Mechanistic Interpretability of MOLFormer's SMILES Pretraining")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")

    # Save config
    config = {
        'seed': SEED,
        'device': DEVICE,
        'timestamp': datetime.now().isoformat(),
        'model': 'ibm/MoLFormer-XL-both-10pct',
    }

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Load data
    qm9_df = load_qm9_data(n_samples=1000)
    smiles_list = qm9_df['smiles'].tolist()

    # ── Experiment 1: Attention-Distance Correlation ──
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Attention-Distance Correlation")
    print("=" * 70)

    results_exp1 = extract_attention_and_distances(model, tokenizer, smiles_list, max_molecules=500)
    corr_results = compute_attention_distance_correlation(results_exp1)
    plot_attention_distance_results(corr_results)

    # Save results
    corr_saveable = {k: v.tolist() for k, v in corr_results.items()}
    with open(RESULTS_DIR / 'exp1_correlations.json', 'w') as f:
        json.dump(corr_saveable, f, indent=2)

    # Print summary
    print("\n--- Experiment 1 Summary ---")
    spearman = corr_results['spearman_correlation']
    print(f"Mean Spearman correlation: {spearman.mean():.4f} (±{spearman.std():.4f})")
    print(f"Max correlation: {spearman.max():.4f} at Layer {spearman.argmax()//12+1}, Head {spearman.argmax()%12+1}")
    print(f"Layer-wise means: {['%.3f' % x for x in spearman.mean(axis=1)]}")

    # ── Experiment 2: Linear Probing ──
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Linear Probing for Chemical Properties")
    print("=" * 70)

    layer_embeddings, atom_labels = extract_hidden_states_per_layer(
        model, tokenizer, smiles_list, max_molecules=1000)
    probing_results = run_linear_probing(layer_embeddings, atom_labels)
    plot_probing_results(probing_results)

    # Save results
    probing_saveable = {}
    for task, res in probing_results.items():
        probing_saveable[task] = {str(k): v for k, v in res.items()}
    with open(RESULTS_DIR / 'exp2_probing.json', 'w') as f:
        json.dump(probing_saveable, f, indent=2)

    # ── Experiment 3: Ablation Study ──
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Attention Head Ablation")
    print("=" * 70)

    ablation_results = run_ablation_experiment(model, tokenizer, corr_results)
    plot_ablation_results(ablation_results)

    # Save ablation results
    if ablation_results:
        abl_saveable = {}
        abl_saveable['baseline'] = ablation_results['baseline']
        for group in ['top_spatial_heads', 'bottom_spatial_heads', 'random_heads']:
            abl_saveable[group] = {
                'mean_mae': ablation_results[group].get('mean_mae'),
                'mean_r2': ablation_results[group].get('mean_r2'),
                'heads': {f"L{k[0]}H{k[1]}": v for k, v in ablation_results[group].items()
                          if isinstance(k, tuple)}
            }
        with open(RESULTS_DIR / 'exp3_ablation.json', 'w') as f:
            json.dump(abl_saveable, f, indent=2)

    # ── Experiment 4: Attention Visualization ──
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Attention Pattern Visualization")
    print("=" * 70)

    visualize_attention_example(model, tokenizer)

    # Save config
    config['n_molecules_exp1'] = len(results_exp1)
    config['n_molecules_exp2'] = len(atom_labels) if atom_labels else 0
    with open(RESULTS_DIR / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == '__main__':
    main()
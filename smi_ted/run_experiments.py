"""
Mechanistic Interpretability of SMI-TED's SMILES Pretraining.

Experiments:
    Experiment 2: Linear probing for chemical properties at each layer
    Experiment 3: Attention head ablation study

Note: Experiment 1 (Attention-Distance Correlation) is not applicable
to SMI-TED because it uses linear attention (FAVOR+) which does not
produce explicit attention matrices.

Adapted from:
    https://github.com/ChicagoHAI/interp_Molmformer

Bugs fixed from original MOLFormer code:
    1. Ablation hook location: moved to inner_attention (before
       output projection) so individual heads can be truly ablated
    2. Data leakage in linear probing: changed to molecule-level
       train/test split

SMI-TED migration changes:
    1. Model loading: load_smi_ted() instead of HuggingFace AutoModel
    2. Token-to-atom mapping: SMI-TED regex pattern + canonicalization
    3. Hook paths: model.encoder.tok_emb and
       model.encoder.blocks.layers[i] instead of
       model.embeddings and model.encoder.layer[i]
    4. Forward pass: model.tokenize() + model.encoder()
       instead of tokenizer() + model()
    5. Molecule embedding: autoencoder latent z instead of mean pooling
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
import regex as re
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score,
                             mean_absolute_error, r2_score)
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

SEED = 42
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

SMI_TED_PATH = '/Users/xuzetong/projects/materials/models/smi_ted/inference'
CKPT_FILENAME = 'smi-ted-Light_40.pt'

QM9_PATH = ('/Users/xuzetong/projects/materials/models/smi_ted'
            '/finetune/moleculenet/qm9/qm9_small_test.csv')
ESOL_PATH = '/Users/xuzetong/projects/materials/models/smi_ted/data/esol.csv'

RESULTS_DIR = Path('./results/smi_ted')
FIGURES_DIR = Path('./results/smi_ted/figures')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


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

def load_model_and_tokenizer():
    """
    Load SMI-TED model and tokenizer.

    Migration change:
    - MOLFormer: HuggingFace AutoModel + AutoTokenizer
    - SMI-TED: load_smi_ted() with local checkpoint,
      tokenizer accessed via model.tokenizer
    """
    sys.path.insert(0, SMI_TED_PATH)
    from smi_ted_light.load import load_smi_ted

    print("Loading SMI-TED model...")
    model = load_smi_ted(
        folder=os.path.join(SMI_TED_PATH, 'smi_ted_light'),
        ckpt_filename=CKPT_FILENAME
    )
    model.encoder.eval()
    if torch.cuda.is_available():
        model.encoder.cuda()

    tokenizer = model.tokenizer
    print(f"Model loaded. Device: {DEVICE}")
    print(f"Vocab size: {len(tokenizer.vocab)}")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────
# Token-to-Atom Mapping
# Migration: SMI-TED regex pattern + canonicalization
# ─────────────────────────────────────────────────────────────────────

# SMI-TED's official tokenization pattern (copied from load.py)
SMILES_PATTERN = (
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p"
    r"|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)
ATOM_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p)"
)


def get_atom_indices_from_smiles(smiles):
    """
    Map SMILES tokens to RDKit atom indices.

    Migration changes:
    1. Uses SMI-TED's official regex pattern instead of
       MOLFormer's hand-written rules (BCNOPSFIcnops etc.)
    2. Canonicalizes SMILES before tokenizing to match
       SMI-TED's internal normalize_smiles() call in
       model.tokenize(). Without this, token order and
       RDKit atom numbering may not match.

    Returns:
        atom_map: list of atom indices (-1 for non-atom tokens)
        mol: RDKit molecule object (canonical)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    # Canonicalize to match SMI-TED's normalize_smiles()
    # SMI-TED uses isomericSmiles=False internally
    canonical_smi = Chem.MolToSmiles(
        mol, canonical=True, isomericSmiles=False)
    mol_canonical = Chem.MolFromSmiles(canonical_smi)
    if mol_canonical is None:
        return None, None

    # Tokenize using SMI-TED's official regex
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


# ─────────────────────────────────────────────────────────────────────
# Utility Functions (model-independent, unchanged from MOLFormer)
# ─────────────────────────────────────────────────────────────────────

def get_atom_properties(mol):
    """Extract chemical properties for each atom."""
    return [{
        'atom_type': atom.GetSymbol(),
        'is_aromatic': atom.GetIsAromatic(),
        'is_in_ring': atom.IsInRing(),
        'degree': atom.GetDegree(),
    } for atom in mol.GetAtoms()]


def load_qm9_data(n_samples=1000):
    """Load QM9 SMILES."""
    df = pd.read_csv(QM9_PATH)
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=SEED)
    print(f"Loaded {len(df)} QM9 molecules")
    return df


def load_esol_data():
    """Load ESOL dataset."""
    df = pd.read_csv(ESOL_PATH)
    print(f"Loaded {len(df)} ESOL molecules")
    return df


# ─────────────────────────────────────────────────────────────────────
# Hook Mechanism
# Migration: hook paths updated for SMI-TED structure
# MOLFormer: model.embeddings, model.encoder.layer[i]
# SMI-TED:   model.encoder.tok_emb, model.encoder.blocks.layers[i]
# ─────────────────────────────────────────────────────────────────────

def register_hooks(model):
    """
    Register forward hooks on SMI-TED encoder layers.

    Migration change:
    - MOLFormer: model.embeddings (layer 0)
                 model.encoder.layer[i] (layers 1-12)
    - SMI-TED:   model.encoder.tok_emb (layer 0)
                 model.encoder.blocks.layers[i] (layers 1-12)
    """
    hooks = []
    intermediate = {}

    def make_hook(idx):
        def hook_fn(module, input, output):
            # fast_transformers outputs tensors directly
            # HuggingFace outputs tuples
            if isinstance(output, tuple):
                intermediate[idx] = output[0].detach().cpu()
            else:
                intermediate[idx] = output.detach().cpu()
        return hook_fn

    # Layer 0: embedding layer
    hooks.append(
        model.encoder.tok_emb.register_forward_hook(make_hook(0))
    )

    # Layers 1-12: encoder layers
    for i, layer in enumerate(model.encoder.blocks.layers):
        hooks.append(
            layer.register_forward_hook(make_hook(i + 1))
        )

    return hooks, intermediate


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT 2: Linear Probing
# Migration changes:
#   1. Forward pass: model.tokenize() + model.encoder()
#   2. Hook paths updated for SMI-TED
# Bug fix:
#   Molecule-level train/test split (was atom-level in original)
# ─────────────────────────────────────────────────────────────────────

def extract_hidden_states_per_layer(model, smiles_list,
                                     max_molecules=800):
    """
    Extract hidden states from all 13 layers for linear probing.

    Migration changes:
    1. Forward pass uses model.tokenize() and model.encoder()
    2. Hook paths updated for SMI-TED structure
    3. Handles SMI-TED's padded encoder output correctly

    Bug fix (from original MOLFormer code):
    - Records molecule_idx for molecule-level train/test split
    """
    hooks, intermediate = register_hooks(model)

    layer_outputs = {i: [] for i in range(13)}
    atom_labels = []
    processed = 0

    for smi in tqdm(smiles_list, desc="Extracting hidden states"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() < 2:
            continue

        # Migration: use SMI-TED regex + canonicalization
        atom_map, mol_obj = get_atom_indices_from_smiles(smi)
        if atom_map is None:
            continue

        props = get_atom_properties(mol_obj)

        # Migration: model.tokenize() returns (idx, mask)
        idx, mask = model.tokenize(smi)

        intermediate.clear()
        with torch.no_grad():
            _ = model.encoder(idx, mask)

        # +1 offset for <bos> token (same as MOLFormer)
        full_atom_map = [-1] + atom_map + [-1]
        atom_token_indices = [i for i, a in enumerate(full_atom_map)
                              if a >= 0]

        if len(atom_token_indices) != mol_obj.GetNumAtoms():
            continue

        # Extract hidden states at atom positions for each layer
        success = True
        for layer_idx in range(13):
            if layer_idx not in intermediate:
                success = False
                break
            hs = intermediate[layer_idx]
            # Remove batch dimension if present
            if hs.dim() == 3:
                hs = hs[0]  # (seq_len, 768)
            if max(atom_token_indices) >= hs.shape[0]:
                success = False
                break
            atom_hs = hs[atom_token_indices].numpy()
            layer_outputs[layer_idx].append(atom_hs)

        if not success:
            continue

        # Bug fix: record molecule_idx for molecule-level split
        # Original code did not track this, causing data leakage
        for prop in props:
            prop['molecule_idx'] = processed
            atom_labels.append(prop)

        processed += 1
        if processed >= max_molecules:
            break

    remove_hooks(hooks)

    layer_embeddings = {}
    for layer_idx in layer_outputs:
        if layer_outputs[layer_idx]:
            layer_embeddings[layer_idx] = np.concatenate(
                layer_outputs[layer_idx], axis=0)

    print(f"Extracted hidden states: {processed} molecules, "
          f"{len(atom_labels)} atoms")
    return layer_embeddings, atom_labels


def run_linear_probing(layer_embeddings, atom_labels):
    """
    Train linear probes for chemical properties at each layer.

    Bug fix from original MOLFormer code:
    - Original: atom-level split, atoms from same molecule
      could appear in both train and test (data leakage)
    - Fixed: molecule-level split, all atoms from a molecule
      go to either train or test, never both
    """
    df_labels = pd.DataFrame(atom_labels)

    probing_tasks = {
        'atom_type': df_labels['atom_type'].values,
        'is_aromatic': df_labels['is_aromatic'].astype(int).values,
        'is_in_ring': df_labels['is_in_ring'].astype(int).values,
        'degree': df_labels['degree'].values,
    }

    results = {}

    for task_name, labels in probing_tasks.items():
        print(f"\nProbing for: {task_name}")

        le = LabelEncoder()
        y = le.fit_transform(labels)
        print(f"  Classes: {le.classes_}")

        # Bug fix: molecule-level train/test split
        molecule_indices = df_labels['molecule_idx'].values
        unique_mols = np.unique(molecule_indices)
        np.random.shuffle(unique_mols)
        mol_split = int(0.8 * len(unique_mols))
        train_mols = set(unique_mols[:mol_split])
        test_mols = set(unique_mols[mol_split:])

        train_idx = np.where(
            [m in train_mols for m in molecule_indices])[0]
        test_idx = np.where(
            [m in test_mols for m in molecule_indices])[0]

        task_results = {}

        for layer_idx in sorted(layer_embeddings.keys()):
            X = layer_embeddings[layer_idx]
            if len(X) != len(y):
                continue

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if len(np.unique(y_train)) < 2:
                continue

            clf = LogisticRegression(
                max_iter=1000, random_state=SEED,
                n_jobs=-1, C=1.0)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            task_results[layer_idx] = {'accuracy': acc, 'f1': f1}
            print(f"  Layer {layer_idx}: "
                  f"Acc={acc:.4f}, F1={f1:.4f}")

        results[task_name] = task_results

    return results


def plot_probing_results(probing_results):
    """Visualize linear probing results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for task_name, task_res in probing_results.items():
        layers = sorted(task_res.keys())
        accs = [task_res[l]['accuracy'] for l in layers]
        f1s = [task_res[l]['f1'] for l in layers]
        axes[0].plot(layers, accs, marker='o',
                     label=task_name, linewidth=2)
        axes[1].plot(layers, f1s, marker='o',
                     label=task_name, linewidth=2)

    axes[0].set_xlabel('Layer', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title(
        'SMI-TED Linear Probe Accuracy by Layer', fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(range(13))

    axes[1].set_xlabel('Layer', fontsize=12)
    axes[1].set_ylabel('F1 Score (weighted)', fontsize=12)
    axes[1].set_title(
        'SMI-TED Linear Probe F1 Score by Layer', fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(range(13))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exp2_probing_results.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Experiment 2 plots saved.")


# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT 3: Attention Head Ablation
# Migration changes:
#   1. Hook path: inner_attention instead of encoder layer
#   2. Molecule embedding: autoencoder latent z
# Bug fix:
#   Hook on inner_attention (before out_projection) so
#   individual heads can be truly ablated
# Note:
#   Since Experiment 1 is skipped, heads are selected by
#   scanning all 144 heads directly on ESOL performance
# ─────────────────────────────────────────────────────────────────────

def get_molecule_embeddings_batch(model, smiles_list,
                                   ablate_layer=None,
                                   ablate_head=None):
    """
    Get molecule embeddings for a list of SMILES,
    optionally ablating a specific attention head.

    Migration changes:
    1. Molecule embedding: autoencoder latent z (768-dim)
       instead of mean pooling over last_hidden_state
       This is SMI-TED's native molecule representation
    2. Ablation hook: model.encoder.blocks.layers[i].attention
       .inner_attention instead of model.encoder.layer[i]

    Bug fix:
    - Hook on inner_attention (output shape: batch, seq_len,
      n_heads, head_dim = 1, seq_len, 12, 64)
    - Zeroing output[:, :, target_head, :] truly ablates
      one head before out_projection mixes them
    - Original MOLFormer code hooked after out_projection
      where heads are already mixed
    """
    hooks = []

    if ablate_layer is not None and ablate_head is not None:

        def make_ablation_hook(target_head):
            def hook_fn(module, input, output):
                # inner_attention output shape:
                # (batch, seq_len, n_heads, head_dim)
                # confirmed: (1, seq_len, 12, 64)
                # Zeroing target_head dimension truly ablates
                # this head before out_projection
                output[:, :, target_head, :] = 0
                return output
            return hook_fn

        # Bug fix: hook on inner_attention, not on the full
        # attention layer or encoder layer
        target = (model.encoder.blocks
                  .layers[ablate_layer]
                  .attention.inner_attention)
        hooks.append(
            target.register_forward_hook(
                make_ablation_hook(ablate_head)
            )
        )

    embeddings = []
    max_len = model.max_len
    n_embd = model.n_embd

    for smi in smiles_list:
        idx, mask = model.tokenize(smi)
        with torch.no_grad():
            token_embeddings = model.encoder(idx, mask)
            # Migration: autoencoder latent z
            # (batch, max_len*768) -> (batch, 768)
            smiles_embedding = model.decoder.autoencoder.encoder(
                token_embeddings.view(-1, max_len * n_embd)
            )
        embeddings.append(smiles_embedding.cpu().numpy()[0])

    for h in hooks:
        h.remove()

    return np.array(embeddings)


def run_ablation_experiment(model, esol_df):
    """
    Ablate attention heads and measure impact on ESOL prediction.

    Note: In MOLFormer, top/bottom heads were selected using
    Experiment 1 (attention-distance correlation). Since
    Experiment 1 is not applicable to SMI-TED (linear attention
    has no explicit attention matrix), we scan all 144 heads
    directly and rank by their impact on ESOL prediction.
    """
    print("\n=== Experiment 3: Attention Head Ablation ===")

    # Prepare ESOL data
    smiles_col = 'smiles' if 'smiles' in esol_df.columns \
        else esol_df.columns[0]
    target_col = None
    for col in esol_df.columns:
        if col != smiles_col and esol_df[col].dtype in \
                [np.float64, np.float32, float]:
            target_col = col
            break

    if target_col is None:
        print("No numeric target found in ESOL, skipping")
        return None

    # Filter valid SMILES
    valid = [(s, y) for s, y in
             zip(esol_df[smiles_col], esol_df[target_col])
             if Chem.MolFromSmiles(s) is not None]
    smiles_list = [v[0] for v in valid]
    labels = np.array([v[1] for v in valid])

    # Subsample for speed
    if len(smiles_list) > 500:
        idx = np.random.choice(
            len(smiles_list), 500, replace=False)
        smiles_list = [smiles_list[i] for i in idx]
        labels = labels[idx]

    # Train/test split
    n = len(smiles_list)
    perm = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = perm[:split], perm[split:]
    train_smi = [smiles_list[i] for i in train_idx]
    test_smi = [smiles_list[i] for i in test_idx]
    train_y = labels[train_idx]
    test_y = labels[test_idx]

    # Baseline: no ablation
    print("Computing baseline embeddings...")
    train_emb = get_molecule_embeddings_batch(model, train_smi)
    test_emb = get_molecule_embeddings_batch(model, test_smi)

    ridge = Ridge(alpha=1.0)
    ridge.fit(train_emb, train_y)
    pred = ridge.predict(test_emb)
    base_mae = mean_absolute_error(test_y, pred)
    base_r2 = r2_score(test_y, pred)
    print(f"Baseline: MAE={base_mae:.4f}, R²={base_r2:.4f}")

    # Scan all 144 heads
    print("\nScanning all 144 heads...")
    head_impacts = {}

    for layer in range(12):
        for head in range(12):
            tr = get_molecule_embeddings_batch(
                model, train_smi, layer, head)
            te = get_molecule_embeddings_batch(
                model, test_smi, layer, head)
            r = Ridge(alpha=1.0)
            r.fit(tr, train_y)
            p = r.predict(te)
            mae = mean_absolute_error(test_y, p)
            delta = mae - base_mae
            head_impacts[(layer, head)] = {
                'mae': float(mae),
                'delta_mae': float(delta)
            }
            print(f"  L{layer}H{head}: "
                  f"MAE={mae:.4f} (delta={delta:+.4f})")

    # Sort by impact
    sorted_heads = sorted(
        head_impacts.items(),
        key=lambda x: x[1]['delta_mae'],
        reverse=True
    )

    print("\nTop 5 most impactful heads "
          "(ablation hurts prediction most):")
    for (layer, head), data in sorted_heads[:5]:
        print(f"  L{layer}H{head}: "
              f"MAE={data['mae']:.4f} "
              f"(delta={data['delta_mae']:+.4f})")

    print("\nBottom 5 least impactful heads:")
    for (layer, head), data in sorted_heads[-5:]:
        print(f"  L{layer}H{head}: "
              f"MAE={data['mae']:.4f} "
              f"(delta={data['delta_mae']:+.4f})")

    ablation_results = {
        'baseline': {
            'mae': float(base_mae),
            'r2': float(base_r2)
        },
        'head_impacts': {
            f"L{k[0]}H{k[1]}": v
            for k, v in head_impacts.items()
        },
        'top_5_heads': [
            f"L{k[0]}H{k[1]}"
            for k, _ in sorted_heads[:5]
        ],
        'bottom_5_heads': [
            f"L{k[0]}H{k[1]}"
            for k, _ in sorted_heads[-5:]
        ],
    }

    return ablation_results


def plot_ablation_results(ablation_results):
    """Visualize ablation results."""
    if ablation_results is None:
        return

    head_impacts = ablation_results['head_impacts']
    base_mae = ablation_results['baseline']['mae']

    # Reshape into 12x12 matrix
    delta_matrix = np.zeros((12, 12))
    for key, data in head_impacts.items():
        # key format: 'L{layer}H{head}'
        parts = key.split('H')
        layer = int(parts[0][1:])
        head = int(parts[1])
        delta_matrix[layer, head] = data['delta_mae']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap
    sns.heatmap(delta_matrix, ax=axes[0],
                cmap='RdBu_r', center=0,
                annot=True, fmt='.2f',
                annot_kws={'size': 7},
                xticklabels=range(12),
                yticklabels=range(12))
    axes[0].set_xlabel('Head')
    axes[0].set_ylabel('Layer')
    axes[0].set_title(
        'SMI-TED Head Ablation: Delta MAE\n'
        '(red = hurts prediction, blue = helps)')

    # Bar chart: top 5 vs bottom 5
    top_5 = ablation_results['top_5_heads']
    bottom_5 = ablation_results['bottom_5_heads']
    selected = top_5 + bottom_5
    selected_maes = [head_impacts[h]['mae'] for h in selected]
    colors = ['#e74c3c'] * 5 + ['#3498db'] * 5

    axes[1].bar(range(len(selected)), selected_maes,
                color=colors, alpha=0.8, edgecolor='black')
    axes[1].axhline(
        y=base_mae, color='green', linestyle='--',
        linewidth=2, label=f'Baseline (MAE={base_mae:.3f})')
    axes[1].set_xticks(range(len(selected)))
    axes[1].set_xticklabels(selected, rotation=45,
                             ha='right', fontsize=8)
    axes[1].set_ylabel('MAE (ESOL)')
    axes[1].set_title(
        'Top 5 (red) vs Bottom 5 (blue) Impactful Heads')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exp3_ablation_results.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Experiment 3 plots saved.")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Mechanistic Interpretability of SMI-TED")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")
    print()
    print("Note: Experiment 1 (Attention-Distance Correlation)")
    print("is skipped. SMI-TED uses linear attention (FAVOR+)")
    print("which does not produce explicit attention matrices.")
    print("=" * 70)

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Load data
    qm9_df = load_qm9_data(n_samples=1000)
    smiles_list = qm9_df['smiles'].tolist()

    # ── Experiment 2: Linear Probing ──
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Linear Probing for Chemical Properties")
    print("=" * 70)

    layer_embeddings, atom_labels = extract_hidden_states_per_layer(
        model, smiles_list, max_molecules=800)
    probing_results = run_linear_probing(
        layer_embeddings, atom_labels)
    plot_probing_results(probing_results)

    probing_saveable = {
        task: {str(k): v for k, v in res.items()}
        for task, res in probing_results.items()
    }
    with open(RESULTS_DIR / 'exp2_probing.json', 'w') as f:
        json.dump(probing_saveable, f, indent=2)
    print(f"Results saved to {RESULTS_DIR / 'exp2_probing.json'}")

    # ── Experiment 3: Ablation Study ──
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Attention Head Ablation")
    print("=" * 70)

    esol_df = load_esol_data()
    ablation_results = run_ablation_experiment(model, esol_df)
    plot_ablation_results(ablation_results)

    if ablation_results:
        with open(RESULTS_DIR / 'exp3_ablation.json', 'w') as f:
            json.dump(ablation_results, f, indent=2)
        print(f"Results saved to "
              f"{RESULTS_DIR / 'exp3_ablation.json'}")

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"Results: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == '__main__':
    main()
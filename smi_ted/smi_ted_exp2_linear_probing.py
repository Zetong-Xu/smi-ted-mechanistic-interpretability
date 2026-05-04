"""
Experiment 2: Linear Probing for Chemical Properties — SMI-TED
==============================================================
Standalone script. Changes from previous version:
  1. 8 atom properties (added hybridization, chiral_tag,
     formal_charge, total_valence)
  2. Frequency baseline (DummyClassifier, most_frequent)
  3. Random-input activation baseline (both retained)
  4. One figure per property, unified y-axis (0–1)
  5. Molecule-level 80/20 train/test split (no data leakage)

Note: Experiment 1 (Attention-Distance Correlation) is skipped.
SMI-TED uses linear attention (FAVOR+) — no explicit attention matrices.

Requires: torch, rdkit, numpy, pandas, sklearn, matplotlib, regex,
          smi_ted_light (local)
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
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rdkit import Chem

# ─────────────────────────────────────────────────────────────────────
# Config  — adjust paths as needed
# ─────────────────────────────────────────────────────────────────────

SEED   = 42
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

SMI_TED_PATH  = '/home/zetong/smi_ted_inference/inference'
CKPT_FILENAME = 'smi-ted-Light_40.pt'
QM9_PATH      = ('/home/zetong/smi-ted-mechanistic-interpretability'
                 '/datasets/qm9/qm9_test.csv')

RESULTS_DIR = Path('./results/smi_ted')
FIGURES_DIR = RESULTS_DIR / 'figures'
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
# SMI-TED tokenization constants
# ─────────────────────────────────────────────────────────────────────

SMILES_PATTERN = (
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p"
    r"|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)
ATOM_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p)"
)


# ─────────────────────────────────────────────────────────────────────
# Model & Data Loading
# ─────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer():
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


def load_qm9_data(n_samples=1000):
    df = pd.read_csv(QM9_PATH)
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=SEED)
    print(f"Loaded {len(df)} QM9 molecules")
    return df


# ─────────────────────────────────────────────────────────────────────
# Token-to-Atom Mapping
# ─────────────────────────────────────────────────────────────────────

def get_atom_indices_from_smiles(smiles):
    """
    Map SMILES tokens to RDKit atom indices using SMI-TED's official
    regex pattern. Canonicalizes SMILES to match SMI-TED's internal
    normalize_smiles() call.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    canonical_smi = Chem.MolToSmiles(
        mol, canonical=True, isomericSmiles=False)
    mol_canonical = Chem.MolFromSmiles(canonical_smi)
    if mol_canonical is None:
        return None, None

    tokens = re.findall(SMILES_PATTERN, canonical_smi)

    atom_map    = []
    current_atom = 0
    num_atoms   = mol_canonical.GetNumAtoms()

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
# Atom Properties — 8 properties
# ─────────────────────────────────────────────────────────────────────

def get_atom_properties(mol):
    """Extract 8 chemical properties for each atom."""
    return [{
        'atom_type':     atom.GetSymbol(),
        'hybridization': str(atom.GetHybridization()),
        'is_aromatic':   atom.GetIsAromatic(),
        'is_in_ring':    atom.IsInRing(),
        'chiral_tag':    str(atom.GetChiralTag()),
        'degree':        atom.GetDegree(),
        'formal_charge': atom.GetFormalCharge(),
        'total_valence': atom.GetTotalValence(),
    } for atom in mol.GetAtoms()]


# ─────────────────────────────────────────────────────────────────────
# Hook Registration
# ─────────────────────────────────────────────────────────────────────

def register_hooks(model):
    """
    Register forward hooks on SMI-TED encoder layers.
    Layer 0  : model.encoder.tok_emb
    Layers 1-12: model.encoder.blocks.layers[i]
    """
    hooks        = []
    intermediate = {}

    def make_hook(idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                intermediate[idx] = output[0].detach().cpu()
            else:
                intermediate[idx] = output.detach().cpu()
        return hook_fn

    hooks.append(
        model.encoder.tok_emb.register_forward_hook(make_hook(0)))

    for i, layer in enumerate(model.encoder.blocks.layers):
        hooks.append(layer.register_forward_hook(make_hook(i + 1)))

    return hooks, intermediate


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ─────────────────────────────────────────────────────────────────────
# Hidden State Extraction
# ─────────────────────────────────────────────────────────────────────

def extract_hidden_states_per_layer(model, smiles_list,
                                     max_molecules=800,
                                     use_random_input=False):
    """
    Extract hidden states from all 13 layers at atom token positions.

    If use_random_input=True, the tok_emb output is replaced with
    random Gaussian noise (random-input activation baseline).
    """
    hooks, intermediate = register_hooks(model)

    random_hooks = []
    if use_random_input:
        def randomize_hook(module, input, output):
            if isinstance(output, tuple):
                return (torch.randn_like(output[0]),) + output[1:]
            return torch.randn_like(output)
        random_hooks.append(
            model.encoder.tok_emb.register_forward_hook(randomize_hook))

    layer_outputs = {i: [] for i in range(13)}
    atom_labels   = []
    processed     = 0
    desc = ("Extracting hidden states"
            + (" (random)" if use_random_input else ""))

    for smi in tqdm(smiles_list, desc=desc):
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() < 2:
            continue

        atom_map, mol_obj = get_atom_indices_from_smiles(smi)
        if atom_map is None:
            continue

        props = get_atom_properties(mol_obj)

        idx, mask = model.tokenize(smi)
        intermediate.clear()
        with torch.no_grad():
            _ = model.encoder(idx, mask)

        full_atom_map      = [-1] + atom_map + [-1]
        atom_token_indices = [i for i, a in enumerate(full_atom_map)
                              if a >= 0]

        if len(atom_token_indices) != mol_obj.GetNumAtoms():
            continue

        success = True
        for layer_idx in range(13):
            if layer_idx not in intermediate:
                success = False
                break
            hs = intermediate[layer_idx]
            if hs.dim() == 3:
                hs = hs[0]
            if max(atom_token_indices) >= hs.shape[0]:
                success = False
                break
            atom_hs = hs[atom_token_indices].numpy()
            layer_outputs[layer_idx].append(atom_hs)

        if not success:
            continue

        for prop in props:
            prop['molecule_idx'] = processed
            atom_labels.append(prop)

        processed += 1
        if processed >= max_molecules:
            break

    remove_hooks(hooks)
    for h in random_hooks:
        h.remove()

    layer_embeddings = {}
    for layer_idx in layer_outputs:
        if layer_outputs[layer_idx]:
            layer_embeddings[layer_idx] = np.concatenate(
                layer_outputs[layer_idx], axis=0)

    print(f"Extracted hidden states: {processed} molecules, "
          f"{len(atom_labels)} atoms"
          + (" [random input]" if use_random_input else ""))
    return layer_embeddings, atom_labels


# ─────────────────────────────────────────────────────────────────────
# Linear Probing
# ─────────────────────────────────────────────────────────────────────

def run_linear_probing(layer_embeddings, atom_labels,
                        layer_embeddings_random=None):
    """
    Train logistic regression probes for 8 chemical properties.

    Three baselines recorded per property:
      - frequency_baseline : DummyClassifier(most_frequent)
      - accuracy_random    : probe on random-input activations
    """
    df_labels = pd.DataFrame(atom_labels)

    probing_tasks = {
        'atom_type':     df_labels['atom_type'].values,
        'hybridization': df_labels['hybridization'].values,
        'is_aromatic':   df_labels['is_aromatic'].astype(int).values,
        'is_in_ring':    df_labels['is_in_ring'].astype(int).values,
        'chiral_tag':    df_labels['chiral_tag'].values,
        'degree':        df_labels['degree'].values,
        'formal_charge': df_labels['formal_charge'].values,
        'total_valence': df_labels['total_valence'].values,
    }

    # Molecule-level 80/20 split — shared across all tasks
    molecule_indices = df_labels['molecule_idx'].values
    unique_mols      = np.unique(molecule_indices)
    rng              = np.random.default_rng(SEED)
    rng.shuffle(unique_mols)
    mol_split  = int(0.8 * len(unique_mols))
    train_mols = set(unique_mols[:mol_split])
    test_mols  = set(unique_mols[mol_split:])
    train_idx  = np.where([m in train_mols for m in molecule_indices])[0]
    test_idx   = np.where([m in test_mols  for m in molecule_indices])[0]

    results = {}

    for task_name, labels in probing_tasks.items():
        print(f"\nProbing for: {task_name}")

        le = LabelEncoder()
        y  = le.fit_transform(labels)
        print(f"  Classes: {le.classes_}")

        if len(np.unique(y[train_idx])) < 2:
            print("  Skipped (fewer than 2 classes in training set)")
            continue

        y_train, y_test = y[train_idx], y[test_idx]

        # Frequency baseline
        dummy = DummyClassifier(strategy='most_frequent')
        dummy.fit(np.zeros((len(train_idx), 1)), y_train)
        freq_acc = accuracy_score(
            y_test, dummy.predict(np.zeros((len(test_idx), 1))))
        print(f"  Frequency baseline: {freq_acc:.4f}")

        task_results = {}

        for layer_idx in sorted(layer_embeddings.keys()):
            X = layer_embeddings[layer_idx]
            if len(X) != len(y):
                continue

            X_train, X_test = X[train_idx], X[test_idx]

            clf = LogisticRegression(max_iter=1000, random_state=SEED,
                                     n_jobs=-1, C=1.0)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1  = f1_score(y_test, y_pred, average='weighted')
            entry = {
                'accuracy':           acc,
                'f1':                 f1,
                'frequency_baseline': freq_acc,
            }

            # Random-input baseline probe
            if layer_embeddings_random is not None:
                X_r = layer_embeddings_random.get(layer_idx)
                if X_r is not None and len(X_r) == len(y):
                    clf_r = LogisticRegression(max_iter=1000,
                                               random_state=SEED,
                                               n_jobs=-1, C=1.0)
                    clf_r.fit(X_r[train_idx], y_train)
                    entry['accuracy_random'] = accuracy_score(
                        y_test, clf_r.predict(X_r[test_idx]))

            task_results[layer_idx] = entry

            rand_str = (f", RandomAcc={entry['accuracy_random']:.4f}"
                        if 'accuracy_random' in entry else "")
            print(f"  Layer {layer_idx}: "
                  f"Acc={acc:.4f}, F1={f1:.4f}{rand_str}")

        results[task_name] = task_results

    return results


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────

def plot_probing_results(probing_results):
    """
    One figure per property. Each figure shows:
      - Model activations        (solid blue)
      - Random-input baseline    (dashed gray)
      - Frequency baseline       (dotted red horizontal line)
    All figures share unified y-axis (0 to 1).
    """
    for task_name, task_res in probing_results.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        layers = sorted(task_res.keys())
        accs   = [task_res[l]['accuracy'] for l in layers]

        ax.plot(layers, accs, marker='o', linewidth=2,
                color='steelblue', label='Model activations')

        if 'accuracy_random' in task_res[layers[0]]:
            accs_r = [task_res[l]['accuracy_random'] for l in layers]
            ax.plot(layers, accs_r, marker='s', linewidth=2,
                    linestyle='--', color='gray',
                    label='Random input baseline')

        freq_acc = task_res[layers[0]]['frequency_baseline']
        ax.axhline(freq_acc, linestyle=':', linewidth=2,
                   color='tomato',
                   label=f'Frequency baseline ({freq_acc:.2f})')

        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(
            f'SMI-TED Linear Probe Accuracy: {task_name}',
            fontsize=13)
        ax.set_xticks(range(13))
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = FIGURES_DIR / f'exp2_probing_{task_name}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SMI-TED — Experiment 2: Linear Probing")
    print("=" * 70)
    print(f"Timestamp : {datetime.now().isoformat()}")
    print(f"Device    : {DEVICE}")
    print(f"Seed      : {SEED}")
    print()
    print("Note: Experiment 1 (Attention-Distance Correlation) is")
    print("skipped — SMI-TED uses linear attention (FAVOR+) which")
    print("does not produce explicit attention matrices.")
    print("=" * 70)

    # Add fast_transformers to path
    sys.path.insert(
        0, os.path.join(SMI_TED_PATH, 'smi_ted_light'))

    model, tokenizer = load_model_and_tokenizer()

    qm9_df      = load_qm9_data(n_samples=1000)
    smiles_list = qm9_df['smiles'].tolist()

    # Normal activations
    layer_embeddings, atom_labels = extract_hidden_states_per_layer(
        model, smiles_list, max_molecules=800)

    # Random-input baseline
    layer_embeddings_random, _ = extract_hidden_states_per_layer(
        model, smiles_list, max_molecules=800,
        use_random_input=True)

    probing_results = run_linear_probing(
        layer_embeddings, atom_labels, layer_embeddings_random)

    plot_probing_results(probing_results)

    # Save numerical results
    probing_saveable = {
        task: {str(k): v for k, v in res.items()}
        for task, res in probing_results.items()
    }
    out_json = RESULTS_DIR / 'exp2_probing.json'
    with open(out_json, 'w') as f:
        json.dump(probing_saveable, f, indent=2)
    print(f"\nNumerical results saved to: {out_json}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 2 COMPLETE")
    print(f"Figures : {FIGURES_DIR}")
    print(f"Results : {RESULTS_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()

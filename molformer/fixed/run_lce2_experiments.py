"""
Phase LCE-2: What Internal Features Help LCE Prediction?

Experiments:
  E1: Parse and prepare the LCE dataset from Kim et al. PNAS 2023
  E2: Interpretable baselines (elemental features, RDKit, ECFP4, MOLFormer)
  E3: Feature attribution (dimension importance, descriptor correlation, layer-wise probing)
  E4: Fine-tuning analysis (CKA before/after, embedding movement)
  E5: Attention head intervention (ablation, cross-molecule attention)

Lessons from Phase LCE code review:
  - Always evaluate on held-out test set, never report validation as test
  - Name experiments accurately to match what's actually implemented
  - Account for confounds before interpreting
  - Use consistent input formats across related experiments
  - Check that extracted-but-unused variables are a bug
"""

import os
import re
import json
import random
import warnings
import logging
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.disable(logging.WARNING)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
DATA_DIR = PROJECT_ROOT / 'datasets' / 'lce'
XLSX_PATH = DATA_DIR / 'pnas.2214357120.sd01.xlsx'

for d in [RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()

# Table 1 experimental LCE values from Soares et al. (used to identify test set)
TABLE1_EXPERIMENTAL = np.array([
    1.094, 1.384, 1.468, 1.710, 1.832,
    2.104, 2.274, 1.071, 1.166, 1.335,
    1.129, 1.501, 1.663
])

# ─────────────────────────────────────────────────────────────────────
# Model Loading (reused from Phase LCE)
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
        hidden = outputs.last_hidden_state
        mask = inputs['attention_mask'].float().unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        all_embs.append(pooled.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def extract_layerwise_embeddings(model, tokenizer, smiles_list, layers=None):
    """Extract mean-pooled embeddings at each layer for a list of SMILES."""
    if layers is None:
        layers = list(range(13))  # 0 (embedding) + 1-12 (transformer layers)
    hooks, intermediate = register_hooks(model)
    layer_embs = {l: [] for l in layers}

    for smi in smiles_list:
        intermediate.clear()
        inputs = tokenizer(smi, return_tensors='pt', padding=False,
                           truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)
        mask = inputs['attention_mask'][0].float().cpu()
        for l in layers:
            if l in intermediate:
                h = intermediate[l][0]  # (seq_len, 768)
                pooled = (h * mask.unsqueeze(-1)).sum(0) / mask.sum()
                layer_embs[l].append(pooled.numpy())

    remove_hooks(hooks)
    return {l: np.array(v) for l, v in layer_embs.items() if len(v) > 0}


# ─────────────────────────────────────────────────────────────────────
# FC Head (reused from Phase LCE)
# ─────────────────────────────────────────────────────────────────────

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


def train_head(X_train, y_train, X_val, y_val, n_epochs=200, lr=1e-3,
               batch_size=32, patience=20, input_dim=None):
    """Train a 2-layer FC head on frozen embeddings.

    Returns (val_rmse, head). Caller must separately evaluate on test set.
    """
    if input_dim is None:
        input_dim = X_train.shape[1]
    head = FrozenEncoderHead(input_dim=input_dim).to(DEVICE)
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


def evaluate_head_on_test(head, X_test, y_test):
    """Evaluate a trained FC head on test set. Always call this separately."""
    X_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    head.eval()
    with torch.no_grad():
        pred = head(X_t).cpu().numpy()
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    return rmse, r2, pred


# ═════════════════════════════════════════════════════════════════════
# E1: Parse and Prepare LCE Dataset
# ═════════════════════════════════════════════════════════════════════

def build_molecule_lookup():
    """Build molecule name → SMILES lookup from the Molecular Database sheet."""
    df = pd.read_excel(XLSX_PATH, sheet_name='Molecular Database', header=None)

    lookup = {}  # short_name -> SMILES
    roles = {}   # short_name -> 'solvent' or 'salt'

    current_role = None
    for i in range(len(df)):
        name_raw = df.iloc[i, 1]
        if pd.isna(name_raw):
            continue
        name_raw = str(name_raw).strip()

        if name_raw == 'Solvent':
            current_role = 'solvent'
            continue
        elif name_raw == 'Salt':
            current_role = 'salt'
            continue

        # Get SMILES - solvents have it in col 14, salts in col 12
        smiles = df.iloc[i, 14] if current_role == 'solvent' else df.iloc[i, 12]
        if pd.isna(smiles):
            continue
        smiles = str(smiles).strip()

        # Extract short name: take everything before the first '('
        short_name = name_raw.split('(')[0].strip()
        # Also store the full name for matching
        lookup[short_name] = smiles
        roles[short_name] = current_role

        # Store with common aliases
        if '(' in name_raw:
            lookup[name_raw] = smiles
            roles[name_raw] = current_role

    # Manual aliases for common names used in formulation strings
    aliases = {
        'MeTHF': 'MTHF',
        'tetraglyme': 'Tetraglyme',
        'TEGDME': 'Tetraglyme',
        '12-Crown-4': '12-crown-4',
        '18-crown-6': '18-crown-6',
        'DEE': 'Diethyl ether',
        'LiFNSI': 'LiFNFSI',
    }
    for alias, canonical in aliases.items():
        if canonical in lookup:
            lookup[alias] = lookup[canonical]
            roles[alias] = roles[canonical]

    return lookup, roles


def parse_formulation(name, mol_lookup):
    """Parse a formulation name string into lists of (molecule, amount_info).

    Returns dict with:
      'solvents': [(name, smiles), ...]
      'salts': [(name, smiles), ...]
    """
    solvents = []
    salts = []

    # Known salt names (sorted by length, longest first to match greedily)
    salt_names = sorted(
        [n for n, r in mol_lookup.items()
         if r == 'salt' or n.startswith('Li') or n.startswith('Rb')
         or n == 'TMS-FNFSI'],
        key=len, reverse=True
    )
    # Actually, mol_lookup stores name->smiles, we need roles
    # This function needs both lookup and roles
    # Let me restructure - we'll pass roles too

    # For now, known salts from the database
    known_salts = [
        'LiFSI', 'LiPF6', 'LiTFSI', 'LiBOB', 'LiDFP', 'LiNO3', 'LiDFOB',
        'LiTFPFB', 'LiBF4', 'LiFNFSI', 'LiFNSI', 'LiPO2F2', 'LiAsF6',
        'LiClO4', 'Li2S5', 'TMS-FNFSI', 'RbNO3', 'LiBETI'
    ]

    known_solvents = sorted(
        [n for n in mol_lookup if n not in known_salts
         and not n.startswith('Li') and n != 'RbNO3'
         and n != 'TMS-FNFSI'],
        key=len, reverse=True
    )

    # Strategy: find all molecule names that appear in the formulation string
    found_salts = []
    found_solvents = []
    remaining = name

    # First find salts (they usually come before solvents in the name)
    for salt in sorted(known_salts, key=len, reverse=True):
        if salt in remaining:
            smiles = mol_lookup.get(salt)
            if smiles:
                found_salts.append((salt, smiles))
                remaining = remaining.replace(salt, ' ', 1)

    # Then find solvents
    for solv in known_solvents:
        if solv in remaining:
            smiles = mol_lookup.get(solv)
            if smiles:
                found_solvents.append((solv, smiles))
                remaining = remaining.replace(solv, ' ', 1)

    return {'solvents': found_solvents, 'salts': found_salts}


def run_experiment_e1():
    """Parse and prepare the LCE dataset from Kim et al. PNAS 2023 XLSX."""
    print("\n" + "=" * 70)
    print("E1: Parse and Prepare LCE Dataset")
    print("=" * 70)

    # Build molecule lookup
    smiles_lookup, roles_lookup = build_molecule_lookup()
    print(f"  Molecular database: {len(smiles_lookup)} entries")
    print(f"    Solvents: {sum(1 for v in roles_lookup.values() if v == 'solvent')}")
    print(f"    Salts: {sum(1 for v in roles_lookup.values() if v == 'salt')}")

    # Combined lookup (name -> smiles, without role distinction for parsing)
    mol_lookup = {}
    for name, smiles in smiles_lookup.items():
        mol_lookup[name] = smiles

    # Read dataset
    df_raw = pd.read_excel(XLSX_PATH, sheet_name='Dataset', header=None)
    data = df_raw.iloc[2:, :].reset_index(drop=True)

    # Column indices (from inspection):
    # 1: Electrolyte name
    # 2-4: Solvent 1-3 Volume %
    # 5-7: Solvent 1-3 mol/L
    # 8-10: Salt 1-3 mol/L
    # 11-23: Elemental features (FC, OC, FO, InOr, F, sF, aF, O, sO, aO, C, sC, aC)
    # 24: CE (%)
    # 25: LCE

    ELEMENTAL_COLS = list(range(11, 24))  # 13 features
    ELEMENTAL_NAMES = ['FC', 'OC', 'FO', 'InOr', 'F', 'sF', 'aF',
                       'O', 'sO', 'aO', 'C', 'sC', 'aC']

    n_formulations = len(data)
    print(f"\n  Total formulations: {n_formulations}")

    # Parse each formulation
    formulations = []
    parse_failures = []

    for i in range(n_formulations):
        name = str(data.iloc[i, 1])
        ce = pd.to_numeric(data.iloc[i, 24], errors='coerce')
        lce = pd.to_numeric(data.iloc[i, 25], errors='coerce')
        elemental = [pd.to_numeric(data.iloc[i, c], errors='coerce')
                     for c in ELEMENTAL_COLS]

        # Solvent volumes and mol/L
        solv_vols = [pd.to_numeric(data.iloc[i, c], errors='coerce')
                     for c in [2, 3, 4]]
        solv_molL = [pd.to_numeric(data.iloc[i, c], errors='coerce')
                     for c in [5, 6, 7]]
        salt_molL = [pd.to_numeric(data.iloc[i, c], errors='coerce')
                     for c in [8, 9, 10]]

        # Parse molecule names from the formulation string
        parsed = parse_formulation(name, mol_lookup)

        # Collect SMILES list
        all_smiles = []
        all_names = []
        all_roles = []
        for sname, smi in parsed['salts']:
            all_smiles.append(smi)
            all_names.append(sname)
            all_roles.append('salt')
        for sname, smi in parsed['solvents']:
            all_smiles.append(smi)
            all_names.append(sname)
            all_roles.append('solvent')

        # Build composition percentage vector (mol/L values)
        comp_pcts = []
        for val in salt_molL + solv_molL:
            comp_pcts.append(float(val) if not pd.isna(val) else 0.0)

        if len(all_smiles) == 0:
            parse_failures.append((i, name))

        formulations.append({
            'idx': i,
            'name': name,
            'molecule_names': all_names,
            'molecule_smiles': all_smiles,
            'molecule_roles': all_roles,
            'composition_pcts': comp_pcts,
            'elemental_features': elemental,
            'ce': float(ce) if not pd.isna(ce) else None,
            'lce': float(lce) if not pd.isna(lce) else None,
        })

    # Report parsing quality
    n_parsed = sum(1 for f in formulations if len(f['molecule_smiles']) > 0)
    print(f"  Successfully parsed: {n_parsed}/{n_formulations}")
    if parse_failures:
        print(f"  Parse failures ({len(parse_failures)}):")
        for idx, name in parse_failures[:10]:
            print(f"    {idx}: {name}")

    # Identify train/test split by matching Table 1 LCE values
    # Use greedy one-to-one assignment with tolerance check
    MATCH_TOLERANCE = 0.02  # max allowed LCE difference
    lce_values = np.array([f['lce'] for f in formulations])
    test_indices = []
    used_dataset_indices = set()

    print(f"\n  Test set matching (tolerance={MATCH_TOLERANCE}):")
    for t1_val in TABLE1_EXPERIMENTAL:
        diffs = np.abs(lce_values - t1_val)
        # Sort by distance, pick first unused
        sorted_idx = np.argsort(diffs)
        matched = False
        for candidate in sorted_idx:
            if int(candidate) not in used_dataset_indices:
                diff = diffs[candidate]
                if diff > MATCH_TOLERANCE:
                    print(f"    WARNING: LCE={t1_val:.3f} best match diff={diff:.4f} "
                          f"exceeds tolerance (heuristic)")
                test_indices.append(int(candidate))
                used_dataset_indices.add(int(candidate))
                matched = True
                break
        if not matched:
            print(f"    ERROR: No match for LCE={t1_val:.3f}")

    test_indices = sorted(test_indices)
    train_indices = [i for i in range(n_formulations) if i not in test_indices]

    if len(test_indices) != len(TABLE1_EXPERIMENTAL):
        print(f"  WARNING: Expected {len(TABLE1_EXPERIMENTAL)} test points, "
              f"matched {len(test_indices)}. Split is heuristically inferred.")

    print(f"\n  Train/test split: {len(train_indices)} train, {len(test_indices)} test")
    print(f"  NOTE: Split is heuristically inferred from Table 1 LCE values, "
          f"not verified against the Soares paper's exact split.")
    print(f"  Test formulations:")
    for ti in test_indices:
        f = formulations[ti]
        print(f"    [{ti}] LCE={f['lce']:.4f} {f['name']}")
        print(f"         Molecules: {f['molecule_names']}")

    # Mark split
    for f in formulations:
        f['split'] = 'test' if f['idx'] in test_indices else 'train'

    # Build concatenated SMILES (Soares approach: join with '.')
    for f in formulations:
        f['concat_smiles'] = '.'.join(f['molecule_smiles']) if f['molecule_smiles'] else ''

    # Summary statistics
    n_molecules_per = [len(f['molecule_smiles']) for f in formulations]
    print(f"\n  Molecules per formulation: "
          f"mean={np.mean(n_molecules_per):.1f}, "
          f"min={min(n_molecules_per)}, max={max(n_molecules_per)}")

    # Save parsed data
    results = {
        'n_formulations': n_formulations,
        'n_parsed': n_parsed,
        'n_train': len(train_indices),
        'n_test': len(test_indices),
        'test_indices': test_indices,
        'train_indices': train_indices,
        'elemental_feature_names': ELEMENTAL_NAMES,
        'parse_failures': [(idx, name) for idx, name in parse_failures],
        'formulations': formulations,
    }

    with open(RESULTS_DIR / 'e1_parsed_dataset.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to e1_parsed_dataset.json")
    return results


# ═════════════════════════════════════════════════════════════════════
# E2: Interpretable Baselines on LCE
# ═════════════════════════════════════════════════════════════════════

def compute_rdkit_descriptors(smiles):
    """Compute RDKit molecular descriptors for a SMILES string."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'MW': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'RotBonds': Descriptors.NumRotatableBonds(mol),
        'Rings': rdMolDescriptors.CalcNumRings(mol),
        'AromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'HeavyAtoms': mol.GetNumHeavyAtoms(),
        'FractionCSP3': Descriptors.FractionCSP3(mol),
        'NumF': sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'F'),
        'NumO': sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'O'),
        'NumN': sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'N'),
        'NumS': sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'S'),
        'NumP': sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'P'),
        'F_ratio': sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'F') / max(mol.GetNumHeavyAtoms(), 1),
        'O_ratio': sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'O') / max(mol.GetNumHeavyAtoms(), 1),
    }


def compute_ecfp4(smiles, n_bits=2048):
    """Compute ECFP4 fingerprint for a SMILES string."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(fp)


def weighted_average_features(molecule_smiles, composition_pcts, feature_fn):
    """Compute weighted average of molecular features using composition percentages."""
    features = []
    weights = []

    # composition_pcts: [salt1_molL, salt2_molL, salt3_molL, solv1_molL, solv2_molL, solv3_molL]
    # We need to match these to the molecules
    # For simplicity, use mol/L as weights (they're already in the data)
    total_weight = sum(p for p in composition_pcts if p > 0)
    if total_weight == 0:
        total_weight = 1.0

    for smi, pct in zip(molecule_smiles, composition_pcts):
        if pct <= 0 or not smi:
            continue
        feat = feature_fn(smi)
        if feat is not None:
            features.append(feat)
            weights.append(pct / total_weight)

    if not features:
        return None

    # Normalize weights
    w = np.array(weights)
    w = w / w.sum()

    if isinstance(features[0], dict):
        result = {}
        for key in features[0]:
            result[key] = sum(f[key] * wi for f, wi in zip(features, w))
        return result
    else:
        return sum(f * wi for f, wi in zip(features, w))


def build_formulation_features(formulations, model=None, tokenizer=None, random_model=None):
    """Build feature matrices for all formulations.

    Returns dict of {feature_type: (X_train, X_test, y_train, y_test, feature_names)}.
    """
    train_forms = [f for f in formulations if f['split'] == 'train']
    test_forms = [f for f in formulations if f['split'] == 'test']

    y_train = np.array([f['lce'] for f in train_forms])
    y_test = np.array([f['lce'] for f in test_forms])

    features = {}

    # --- 1. Kim et al. elemental features (13 features, already in the data) ---
    print("  Building elemental features...")
    X_elem_train = np.array([f['elemental_features'] for f in train_forms], dtype=float)
    X_elem_test = np.array([f['elemental_features'] for f in test_forms], dtype=float)
    # Replace NaN with 0
    X_elem_train = np.nan_to_num(X_elem_train, 0)
    X_elem_test = np.nan_to_num(X_elem_test, 0)
    features['elemental_13'] = (X_elem_train, X_elem_test, y_train, y_test,
                                ['FC', 'OC', 'FO', 'InOr', 'F', 'sF', 'aF',
                                 'O', 'sO', 'aO', 'C', 'sC', 'aC'])

    # --- 2. RDKit descriptors (weighted avg per formulation) ---
    print("  Building RDKit descriptor features...")
    rdkit_train = []
    rdkit_test = []
    rdkit_names = None

    for forms_list, X_list in [(train_forms, rdkit_train), (test_forms, rdkit_test)]:
        for f in forms_list:
            if not f['molecule_smiles']:
                X_list.append(None)
                continue
            # Match SMILES to mol/L weights
            # The composition_pcts are [salt1, salt2, salt3, solv1, solv2, solv3]
            # molecule order is salts first, then solvents
            n_salts = len([r for r in f['molecule_roles'] if r == 'salt'])
            n_solvents = len([r for r in f['molecule_roles'] if r == 'solvent'])
            salt_weights = f['composition_pcts'][:3][:n_salts]
            solv_weights = f['composition_pcts'][3:][:n_solvents]
            mol_weights = salt_weights + solv_weights

            # Pad if needed
            while len(mol_weights) < len(f['molecule_smiles']):
                mol_weights.append(1.0)

            weighted_desc = weighted_average_features(
                f['molecule_smiles'], mol_weights, compute_rdkit_descriptors)
            if weighted_desc is not None:
                if rdkit_names is None:
                    rdkit_names = sorted(weighted_desc.keys())
                X_list.append([weighted_desc[k] for k in rdkit_names])
            else:
                X_list.append(None)

    # Handle failures
    rdkit_train_clean = [x for x in rdkit_train if x is not None]
    rdkit_test_clean = [x for x in rdkit_test if x is not None]
    if rdkit_train_clean and rdkit_test_clean:
        # For now, replace None with zeros (should be rare)
        default = [0.0] * len(rdkit_names)
        X_rdkit_train = np.array([x if x is not None else default for x in rdkit_train])
        X_rdkit_test = np.array([x if x is not None else default for x in rdkit_test])
        X_rdkit_train = np.nan_to_num(X_rdkit_train, 0)
        X_rdkit_test = np.nan_to_num(X_rdkit_test, 0)
        features['rdkit'] = (X_rdkit_train, X_rdkit_test, y_train, y_test, rdkit_names)
    else:
        print("    WARNING: RDKit features failed for too many formulations")

    # --- 3. ECFP4 fingerprints (weighted avg per formulation) ---
    print("  Building ECFP4 features...")
    ecfp_train = []
    ecfp_test = []
    n_bits = 2048

    for forms_list, X_list in [(train_forms, ecfp_train), (test_forms, ecfp_test)]:
        for f in forms_list:
            if not f['molecule_smiles']:
                X_list.append(np.zeros(n_bits))
                continue
            n_salts = len([r for r in f['molecule_roles'] if r == 'salt'])
            n_solvents = len([r for r in f['molecule_roles'] if r == 'solvent'])
            salt_weights = f['composition_pcts'][:3][:n_salts]
            solv_weights = f['composition_pcts'][3:][:n_solvents]
            mol_weights = salt_weights + solv_weights
            while len(mol_weights) < len(f['molecule_smiles']):
                mol_weights.append(1.0)

            result = weighted_average_features(
                f['molecule_smiles'], mol_weights,
                lambda s: compute_ecfp4(s, n_bits))
            X_list.append(result if result is not None else np.zeros(n_bits))

    X_ecfp_train = np.array(ecfp_train)
    X_ecfp_test = np.array(ecfp_test)
    features['ecfp4'] = (X_ecfp_train, X_ecfp_test, y_train, y_test,
                         [f'bit_{i}' for i in range(n_bits)])

    # --- 4. Composition percentages only ---
    print("  Building composition-only features...")
    X_comp_train = np.array([f['composition_pcts'] for f in train_forms])
    X_comp_test = np.array([f['composition_pcts'] for f in test_forms])
    X_comp_train = np.nan_to_num(X_comp_train, 0)
    X_comp_test = np.nan_to_num(X_comp_test, 0)
    features['composition_only'] = (X_comp_train, X_comp_test, y_train, y_test,
                                    ['salt1_molL', 'salt2_molL', 'salt3_molL',
                                     'solv1_molL', 'solv2_molL', 'solv3_molL'])

    # --- 5. Elemental + Composition ---
    X_elem_comp_train = np.hstack([X_elem_train, X_comp_train])
    X_elem_comp_test = np.hstack([X_elem_test, X_comp_test])
    features['elemental_plus_comp'] = (
        X_elem_comp_train, X_elem_comp_test, y_train, y_test,
        features['elemental_13'][4] + features['composition_only'][4])

    # --- 6. Frozen MOLFormer embeddings (if model provided) ---
    if model is not None and tokenizer is not None:
        # --- 6a. Concatenated SMILES approach (Soares paper's actual method) ---
        print("  Building MOLFormer concat-SMILES features...")
        train_concat_smiles = [f['concat_smiles'] for f in train_forms]
        test_concat_smiles = [f['concat_smiles'] for f in test_forms]

        valid_train_mask = [len(s) > 0 for s in train_concat_smiles]
        valid_test_mask = [len(s) > 0 for s in test_concat_smiles]

        if all(valid_train_mask) and all(valid_test_mask):
            X_molf_train = extract_frozen_embeddings(model, tokenizer, train_concat_smiles)
            X_molf_test = extract_frozen_embeddings(model, tokenizer, test_concat_smiles)
            features['molf_concat_pretrained'] = (
                X_molf_train, X_molf_test, y_train, y_test,
                [f'dim_{i}' for i in range(768)])

            X_mm_train = np.hstack([X_molf_train, X_comp_train])
            X_mm_test = np.hstack([X_molf_test, X_comp_test])
            features['molf_concat_multimodal'] = (
                X_mm_train, X_mm_test, y_train, y_test,
                [f'dim_{i}' for i in range(768)] + features['composition_only'][4])
        else:
            print("    WARNING: Some formulations have no SMILES, skipping concat MOLFormer")

        # --- 6b. Per-molecule weighted average (apples-to-apples with RDKit/ECFP) ---
        print("  Building MOLFormer per-molecule weighted-avg features...")

        def molformer_weighted_emb(forms_list, mdl):
            """Weighted average of per-molecule embeddings for each formulation."""
            # First, collect all unique molecule SMILES and embed once
            unique_smi = set()
            for f in forms_list:
                for s in f['molecule_smiles']:
                    unique_smi.add(s)
            unique_smi = sorted(unique_smi)
            if not unique_smi:
                return None
            emb_map = {}
            embs = extract_frozen_embeddings(mdl, tokenizer, unique_smi)
            for s, e in zip(unique_smi, embs):
                emb_map[s] = e

            X = []
            for f in forms_list:
                if not f['molecule_smiles']:
                    X.append(np.zeros(768))
                    continue
                n_salts = len([r for r in f['molecule_roles'] if r == 'salt'])
                n_solvents = len([r for r in f['molecule_roles'] if r == 'solvent'])
                salt_w = f['composition_pcts'][:3][:n_salts]
                solv_w = f['composition_pcts'][3:][:n_solvents]
                mol_weights = salt_w + solv_w
                while len(mol_weights) < len(f['molecule_smiles']):
                    mol_weights.append(1.0)
                total = sum(w for w in mol_weights if w > 0)
                if total == 0:
                    total = 1.0
                weighted = np.zeros(768)
                for smi, w in zip(f['molecule_smiles'], mol_weights):
                    if w > 0 and smi in emb_map:
                        weighted += emb_map[smi] * (w / total)
                X.append(weighted)
            return np.array(X)

        X_wmolf_train = molformer_weighted_emb(train_forms, model)
        X_wmolf_test = molformer_weighted_emb(test_forms, model)
        if X_wmolf_train is not None and X_wmolf_test is not None:
            features['molf_weighted_pretrained'] = (
                X_wmolf_train, X_wmolf_test, y_train, y_test,
                [f'dim_{i}' for i in range(768)])

            X_wmm_train = np.hstack([X_wmolf_train, X_comp_train])
            X_wmm_test = np.hstack([X_wmolf_test, X_comp_test])
            features['molf_weighted_multimodal'] = (
                X_wmm_train, X_wmm_test, y_train, y_test,
                [f'dim_{i}' for i in range(768)] + features['composition_only'][4])

        # --- 6c. Random MOLFormer (both concat and weighted) ---
        if random_model is not None:
            print("  Building random MOLFormer embedding features...")
            X_rand_train = extract_frozen_embeddings(random_model, tokenizer, train_concat_smiles)
            X_rand_test = extract_frozen_embeddings(random_model, tokenizer, test_concat_smiles)
            features['molf_concat_random'] = (
                X_rand_train, X_rand_test, y_train, y_test,
                [f'dim_{i}' for i in range(768)])

            X_rand_mm_train = np.hstack([X_rand_train, X_comp_train])
            X_rand_mm_test = np.hstack([X_rand_test, X_comp_test])
            features['molf_concat_rand_multimodal'] = (
                X_rand_mm_train, X_rand_mm_test, y_train, y_test,
                [f'dim_{i}' for i in range(768)] + features['composition_only'][4])

            # Random per-molecule weighted
            X_wrand_train = molformer_weighted_emb(train_forms, random_model)
            X_wrand_test = molformer_weighted_emb(test_forms, random_model)
            if X_wrand_train is not None and X_wrand_test is not None:
                features['molf_weighted_random'] = (
                    X_wrand_train, X_wrand_test, y_train, y_test,
                    [f'dim_{i}' for i in range(768)])

    return features


def run_experiment_e2(e1_results, model=None, tokenizer=None, random_model=None):
    """Interpretable baselines: compare feature representations for LCE prediction."""
    print("\n" + "=" * 70)
    print("E2: Interpretable Baselines on LCE")
    print("=" * 70)

    formulations = e1_results['formulations']

    # Build all feature representations
    features = build_formulation_features(formulations, model, tokenizer, random_model)

    # Evaluate each representation with multiple models
    results = {}

    for feat_name, (X_train, X_test, y_train, y_test, feat_names) in features.items():
        print(f"\n  --- {feat_name} (dim={X_train.shape[1]}) ---")

        feat_results = {'n_features': X_train.shape[1]}

        # Standardize
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Ridge regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_s, y_train)
        pred_train = ridge.predict(X_train_s)
        pred_test = ridge.predict(X_test_s)
        rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
        r2_test = r2_score(y_test, pred_test)
        print(f"    Ridge: train RMSE={rmse_train:.4f}, test RMSE={rmse_test:.4f}, R2={r2_test:.4f}")
        feat_results['ridge'] = {
            'train_rmse': float(rmse_train),
            'test_rmse': float(rmse_test),
            'test_r2': float(r2_test),
            'predictions': pred_test.tolist(),
        }

        # Ridge with CV to find best alpha
        best_alpha = 1.0
        best_cv_score = -999
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            cv_scores = cross_val_score(
                Ridge(alpha=alpha), X_train_s, y_train,
                cv=min(5, len(y_train)), scoring='neg_mean_squared_error')
            mean_cv = cv_scores.mean()
            if mean_cv > best_cv_score:
                best_cv_score = mean_cv
                best_alpha = alpha

        ridge_cv = Ridge(alpha=best_alpha)
        ridge_cv.fit(X_train_s, y_train)
        pred_test_cv = ridge_cv.predict(X_test_s)
        rmse_test_cv = np.sqrt(mean_squared_error(y_test, pred_test_cv))
        r2_test_cv = r2_score(y_test, pred_test_cv)
        print(f"    Ridge (best alpha={best_alpha}): test RMSE={rmse_test_cv:.4f}, R2={r2_test_cv:.4f}")
        feat_results['ridge_cv'] = {
            'best_alpha': float(best_alpha),
            'test_rmse': float(rmse_test_cv),
            'test_r2': float(r2_test_cv),
            'predictions': pred_test_cv.tolist(),
        }

        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=SEED)
        rf.fit(X_train, y_train)
        pred_test_rf = rf.predict(X_test)
        rmse_test_rf = np.sqrt(mean_squared_error(y_test, pred_test_rf))
        r2_test_rf = r2_score(y_test, pred_test_rf)
        print(f"    RF: test RMSE={rmse_test_rf:.4f}, R2={r2_test_rf:.4f}")
        feat_results['random_forest'] = {
            'test_rmse': float(rmse_test_rf),
            'test_r2': float(r2_test_rf),
            'predictions': pred_test_rf.tolist(),
        }

        # Gradient Boosting
        gb = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=SEED)
        gb.fit(X_train, y_train)
        pred_test_gb = gb.predict(X_test)
        rmse_test_gb = np.sqrt(mean_squared_error(y_test, pred_test_gb))
        r2_test_gb = r2_score(y_test, pred_test_gb)
        print(f"    GBR: test RMSE={rmse_test_gb:.4f}, R2={r2_test_gb:.4f}")
        feat_results['gradient_boosting'] = {
            'test_rmse': float(rmse_test_gb),
            'test_r2': float(r2_test_gb),
            'predictions': pred_test_gb.tolist(),
        }

        # FC Head (for MOLFormer features, to match paper's setup)
        if X_train.shape[1] >= 100:  # Only for high-dim features
            # Split train into train/val for FC head
            n_val = max(10, len(y_train) // 5)
            perm = np.random.permutation(len(y_train))
            val_idx = perm[:n_val]
            tr_idx = perm[n_val:]
            val_rmse, head = train_head(
                X_train_s[tr_idx], y_train[tr_idx],
                X_train_s[val_idx], y_train[val_idx],
                input_dim=X_train_s.shape[1])
            # Evaluate on TEST set (lesson from Phase LCE)
            test_rmse_fc, test_r2_fc, pred_test_fc = evaluate_head_on_test(
                head, X_test_s, y_test)
            print(f"    FC Head: test RMSE={test_rmse_fc:.4f}, R2={test_r2_fc:.4f}")
            feat_results['fc_head'] = {
                'test_rmse': float(test_rmse_fc),
                'test_r2': float(test_r2_fc),
                'predictions': pred_test_fc.tolist(),
            }

        feat_results['y_test'] = y_test.tolist()
        results[feat_name] = feat_results

    # --- Summary table ---
    print("\n\n  ═══════════════════════════════════════════════════")
    print("  SUMMARY: Test RMSE by feature representation × model")
    print("  ═══════════════════════════════════════════════════")
    model_types = ['ridge_cv', 'random_forest', 'gradient_boosting', 'fc_head']
    header = f"  {'Features':<25s}"
    for mt in model_types:
        header += f" {mt:>18s}"
    print(header)
    print("  " + "-" * (25 + 19 * len(model_types)))

    for feat_name in features:
        row = f"  {feat_name:<25s}"
        for mt in model_types:
            if mt in results[feat_name]:
                rmse = results[feat_name][mt]['test_rmse']
                row += f" {rmse:>18.4f}"
            else:
                row += f" {'—':>18s}"
        print(row)

    # Reference: Soares paper reports RMSE 0.195 for MultiModal-MoLFormer
    print(f"\n  Reference: Soares et al. MultiModal-MoLFormer RMSE = 0.195")

    # --- Figure: bar chart of test RMSEs ---
    fig, ax = plt.subplots(figsize=(12, 6))
    feat_order = list(features.keys())
    x = np.arange(len(feat_order))
    width = 0.2

    for i, mt in enumerate(model_types):
        rmses = []
        for fn in feat_order:
            if mt in results[fn]:
                rmses.append(results[fn][mt]['test_rmse'])
            else:
                rmses.append(0)
        bars = ax.bar(x + i * width, rmses, width, label=mt, alpha=0.8)

    ax.axhline(y=0.195, color='red', linestyle='--', linewidth=1.5,
               label='Soares MultiModal (0.195)')
    ax.set_xlabel('Feature Representation')
    ax.set_ylabel('Test RMSE')
    ax.set_title('E2: Interpretable Baselines — LCE Test RMSE')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(feat_order, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(1.0, max(r.get('ridge_cv', {}).get('test_rmse', 0)
                                for r in results.values()) * 1.2))

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'e2_baselines_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Figure saved: e2_baselines_comparison.png")

    # --- Figure: parity plots for best model per representation ---
    n_feats = len(feat_order)
    n_cols = min(4, n_feats)
    n_rows = (n_feats + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :] if n_cols > 1 else np.array([[axes]])
    axes = axes.flatten()

    for idx, feat_name in enumerate(feat_order):
        ax = axes[idx]
        # Find best model for this feature
        best_mt = None
        best_rmse = 999
        for mt in model_types:
            if mt in results[feat_name]:
                if results[feat_name][mt]['test_rmse'] < best_rmse:
                    best_rmse = results[feat_name][mt]['test_rmse']
                    best_mt = mt

        if best_mt is not None:
            pred = np.array(results[feat_name][best_mt]['predictions'])
            y_t = np.array(results[feat_name]['y_test'])
            ax.scatter(y_t, pred, c='steelblue', s=50, edgecolors='k', linewidths=0.5)
            lims = [min(y_t.min(), pred.min()) - 0.1, max(y_t.max(), pred.max()) + 0.1]
            ax.plot(lims, lims, 'k--', alpha=0.5)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel('True LCE')
            ax.set_ylabel('Predicted LCE')
            ax.set_title(f'{feat_name}\n{best_mt} RMSE={best_rmse:.3f}', fontsize=9)
            ax.set_aspect('equal')

    for idx in range(len(feat_order), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'e2_parity_plots.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: e2_parity_plots.png")

    with open(RESULTS_DIR / 'e2_baselines.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ═════════════════════════════════════════════════════════════════════
# E3: Feature Attribution on the LCE Task
# ═════════════════════════════════════════════════════════════════════

def run_experiment_e3(e1_results, model, tokenizer, random_model=None):
    """Feature attribution: what embedding dimensions drive LCE prediction?"""
    print("\n" + "=" * 70)
    print("E3: Feature Attribution on LCE Task")
    print("=" * 70)

    formulations = e1_results['formulations']
    train_forms = [f for f in formulations if f['split'] == 'train']
    test_forms = [f for f in formulations if f['split'] == 'test']

    train_smiles = [f['concat_smiles'] for f in train_forms]
    test_smiles = [f['concat_smiles'] for f in test_forms]
    y_train = np.array([f['lce'] for f in train_forms])
    y_test = np.array([f['lce'] for f in test_forms])

    results = {}

    # ── E3.1: Embedding dimension importance via Ridge coefficients ──
    print("\n  E3.1: Embedding dimension importance...")
    X_train = extract_frozen_embeddings(model, tokenizer, train_smiles)
    X_test = extract_frozen_embeddings(model, tokenizer, test_smiles)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Ridge with CV
    best_alpha = 1.0
    best_cv = -999
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        cv = cross_val_score(Ridge(alpha=alpha), X_train_s, y_train,
                             cv=5, scoring='neg_mean_squared_error').mean()
        if cv > best_cv:
            best_cv = cv
            best_alpha = alpha

    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X_train_s, y_train)
    pred_test = ridge.predict(X_test_s)
    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
    print(f"    Ridge (alpha={best_alpha}) test RMSE={rmse_test:.4f}")

    # Dimension importance = |coefficient| (on standardized features)
    dim_importance = np.abs(ridge.coef_)
    top_k = 30
    top_dims = np.argsort(dim_importance)[::-1][:top_k]
    print(f"    Top {top_k} important dimensions: {top_dims.tolist()}")

    results['dimension_importance'] = {
        'ridge_alpha': float(best_alpha),
        'test_rmse': float(rmse_test),
        'top_30_dims': top_dims.tolist(),
        'top_30_importance': dim_importance[top_dims].tolist(),
        'all_importance': dim_importance.tolist(),
        'ridge_coefficients': ridge.coef_.tolist(),
    }

    # ── E3.2: Dimension-to-descriptor translation table ──
    print("\n  E3.2: Dimension-to-descriptor correlation...")

    # Get individual molecule embeddings and descriptors
    smiles_lookup, roles_lookup = build_molecule_lookup()

    # Collect unique molecules from formulations
    unique_mols = {}
    for f in formulations:
        for name, smi in zip(f['molecule_names'], f['molecule_smiles']):
            if smi not in unique_mols:
                unique_mols[smi] = name

    print(f"    Unique molecules in formulations: {len(unique_mols)}")

    # Get embeddings for each unique molecule
    mol_smiles_list = list(unique_mols.keys())
    mol_names_list = [unique_mols[s] for s in mol_smiles_list]

    mol_embeddings = extract_frozen_embeddings(model, tokenizer, mol_smiles_list)

    # Get RDKit descriptors for each molecule
    mol_descriptors = []
    for smi in mol_smiles_list:
        desc = compute_rdkit_descriptors(smi)
        mol_descriptors.append(desc if desc is not None else {})

    descriptor_names = sorted(mol_descriptors[0].keys()) if mol_descriptors[0] else []

    # Also get Kim et al. elemental features for each molecule from Molecular Database
    df_moldb = pd.read_excel(XLSX_PATH, sheet_name='Molecular Database', header=None)
    # Solvents: rows 3-55, elem in cols 2-9 (C,H,O,F,N,S,P,other), MW col 11, SMILES col 14
    # Salts: rows 58+, elem in cols 2-9 (C,H,O,F,N,S,P,Li,B/Cl), SMILES col 12
    kim_elem_names = ['C_count', 'H_count', 'O_count', 'F_count', 'N_count', 'S_count', 'P_count']

    # Build SMILES → elemental feature lookup from the XLSX
    mol_elem_lookup = {}
    for i in range(3, 57):  # solvents
        smiles = df_moldb.iloc[i, 14]
        if pd.isna(smiles):
            continue
        smiles = str(smiles).strip()
        elem = [pd.to_numeric(df_moldb.iloc[i, c], errors='coerce') for c in range(2, 9)]
        elem = [0.0 if pd.isna(v) else float(v) for v in elem]
        mol_elem_lookup[smiles] = elem
    for i in range(58, len(df_moldb)):  # salts
        smiles = df_moldb.iloc[i, 12]
        if pd.isna(smiles):
            continue
        smiles = str(smiles).strip()
        elem = [pd.to_numeric(df_moldb.iloc[i, c], errors='coerce') for c in range(2, 9)]
        elem = [0.0 if pd.isna(v) else float(v) for v in elem]
        mol_elem_lookup[smiles] = elem

    # Add Kim elemental features to the descriptor list for each molecule
    for i, smi in enumerate(mol_smiles_list):
        kim_elem = mol_elem_lookup.get(smi, [0.0] * 7)
        for j, name in enumerate(kim_elem_names):
            mol_descriptors[i][name] = kim_elem[j]

    # Update descriptor_names to include Kim features
    descriptor_names = sorted(mol_descriptors[0].keys()) if mol_descriptors[0] else []
    print(f"    Descriptors per molecule: {len(descriptor_names)} "
          f"(RDKit + Kim elemental: {kim_elem_names})")

    # Build correlation matrix: each top dimension vs each descriptor
    desc_matrix = np.array([[d.get(dn, 0) for dn in descriptor_names]
                            for d in mol_descriptors])

    correlation_table = {}
    for dim_idx in top_dims[:20]:  # Top 20 most important dims
        dim_values = mol_embeddings[:, dim_idx]
        correlations = {}
        for j, dn in enumerate(descriptor_names):
            desc_values = desc_matrix[:, j]
            if np.std(desc_values) > 1e-10 and np.std(dim_values) > 1e-10:
                r, p = pearsonr(dim_values, desc_values)
                correlations[dn] = {'pearson_r': float(r), 'p_value': float(p)}

        # Find best correlated descriptor
        if correlations:
            best_desc = max(correlations, key=lambda k: abs(correlations[k]['pearson_r']))
            best_r = correlations[best_desc]['pearson_r']
            print(f"    Dim {dim_idx:3d} (importance={dim_importance[dim_idx]:.4f}): "
                  f"best corr with {best_desc} (r={best_r:.3f})")
        correlation_table[int(dim_idx)] = correlations

    results['dimension_descriptor_correlation'] = {
        'n_molecules': len(mol_smiles_list),
        'descriptor_names': descriptor_names,
        'top_20_correlations': correlation_table,
    }

    # ── E3.3: Layer-wise LCE probing ──
    print("\n  E3.3: Layer-wise LCE probing...")
    layers = list(range(13))  # 0 (embedding) + 1-12 (transformer layers)

    layer_embs_train = extract_layerwise_embeddings(model, tokenizer, train_smiles, layers)
    layer_embs_test = extract_layerwise_embeddings(model, tokenizer, test_smiles, layers)

    layer_probe_results = {}
    for l in layers:
        if l not in layer_embs_train or l not in layer_embs_test:
            continue
        Xl_train = layer_embs_train[l]
        Xl_test = layer_embs_test[l]

        sc = StandardScaler()
        Xl_train_s = sc.fit_transform(Xl_train)
        Xl_test_s = sc.transform(Xl_test)

        ridge_l = Ridge(alpha=best_alpha)
        ridge_l.fit(Xl_train_s, y_train)
        pred_l = ridge_l.predict(Xl_test_s)
        rmse_l = np.sqrt(mean_squared_error(y_test, pred_l))
        r2_l = r2_score(y_test, pred_l)

        layer_probe_results[l] = {'test_rmse': float(rmse_l), 'test_r2': float(r2_l)}
        print(f"    Layer {l:2d}: RMSE={rmse_l:.4f}, R2={r2_l:.4f}")

    results['layerwise_lce_probing'] = layer_probe_results

    # Also do random model layerwise
    if random_model is not None:
        print("\n  E3.3b: Layer-wise LCE probing (random model)...")
        rand_layer_embs_train = extract_layerwise_embeddings(
            random_model, tokenizer, train_smiles, layers)
        rand_layer_embs_test = extract_layerwise_embeddings(
            random_model, tokenizer, test_smiles, layers)

        rand_layer_results = {}
        for l in layers:
            if l not in rand_layer_embs_train or l not in rand_layer_embs_test:
                continue
            Xl_train = rand_layer_embs_train[l]
            Xl_test = rand_layer_embs_test[l]

            sc = StandardScaler()
            Xl_train_s = sc.fit_transform(Xl_train)
            Xl_test_s = sc.transform(Xl_test)

            ridge_l = Ridge(alpha=best_alpha)
            ridge_l.fit(Xl_train_s, y_train)
            pred_l = ridge_l.predict(Xl_test_s)
            rmse_l = np.sqrt(mean_squared_error(y_test, pred_l))
            r2_l = r2_score(y_test, pred_l)
            rand_layer_results[l] = {'test_rmse': float(rmse_l), 'test_r2': float(r2_l)}
            print(f"    Layer {l:2d}: RMSE={rmse_l:.4f}, R2={r2_l:.4f}")

        results['layerwise_lce_probing_random'] = rand_layer_results

    # ── E3.4: Molecule role probing ──
    print("\n  E3.4: Molecule role probing...")
    # Classify molecules as solvent vs salt using embeddings at each layer

    mol_roles = []
    for smi in mol_smiles_list:
        name = unique_mols[smi]
        role = roles_lookup.get(name, None)
        if role is None:
            # Try matching by short name
            for rname, rrole in roles_lookup.items():
                if rname in name or name in rname:
                    role = rrole
                    break
        mol_roles.append(role if role else 'unknown')

    # Filter to known roles
    known_mask = [r in ('solvent', 'salt') for r in mol_roles]
    mol_smiles_known = [s for s, k in zip(mol_smiles_list, known_mask) if k]
    mol_roles_known = [r for r, k in zip(mol_roles, known_mask) if k]
    role_labels = np.array([0 if r == 'solvent' else 1 for r in mol_roles_known])

    print(f"    Molecules with known roles: {len(mol_smiles_known)} "
          f"(solvents={sum(role_labels == 0)}, salts={sum(role_labels == 1)})")

    if len(mol_smiles_known) >= 10:
        mol_layer_embs = extract_layerwise_embeddings(model, tokenizer, mol_smiles_known, layers)

        role_probe_results = {}
        for l in layers:
            if l not in mol_layer_embs:
                continue
            Xl = mol_layer_embs[l]
            sc = StandardScaler()
            Xl_s = sc.fit_transform(Xl)

            # Use LOO cross-validation (small dataset)
            correct = 0
            for i in range(len(Xl_s)):
                X_tr = np.delete(Xl_s, i, axis=0)
                y_tr = np.delete(role_labels, i)
                X_te = Xl_s[i:i+1]
                clf = RidgeClassifier(alpha=1.0)
                clf.fit(X_tr, y_tr)
                pred = clf.predict(X_te)
                if pred[0] == role_labels[i]:
                    correct += 1

            accuracy = correct / len(Xl_s)
            role_probe_results[l] = {'accuracy': float(accuracy)}
            print(f"    Layer {l:2d}: LOO accuracy={accuracy:.3f}")

        results['role_probing'] = role_probe_results
    else:
        print("    Not enough molecules with known roles for probing")

    # ── Figures ──

    # Figure: Layer-wise LCE probing
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # LCE probing by layer
    ax = axes[0]
    layers_plotted = sorted(layer_probe_results.keys())
    rmses = [layer_probe_results[l]['test_rmse'] for l in layers_plotted]
    ax.plot(layers_plotted, rmses, 'o-', color='steelblue', label='Pretrained', linewidth=2)

    if 'layerwise_lce_probing_random' in results:
        rand_rmses = [results['layerwise_lce_probing_random'].get(l, {}).get('test_rmse', np.nan)
                      for l in layers_plotted]
        ax.plot(layers_plotted, rand_rmses, 's--', color='coral', label='Random', linewidth=2)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Test RMSE')
    ax.set_title('E3.3: Layer-wise LCE Probing (Ridge)')
    ax.legend()
    ax.set_xticks(layers_plotted)

    # Role probing by layer
    ax = axes[1]
    if 'role_probing' in results:
        rp = results['role_probing']
        layers_rp = sorted(rp.keys())
        accs = [rp[l]['accuracy'] for l in layers_rp]
        ax.plot(layers_rp, accs, 'o-', color='darkgreen', linewidth=2)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.set_xlabel('Layer')
        ax.set_ylabel('LOO Accuracy')
        ax.set_title('E3.4: Molecule Role Probing (Solvent vs Salt)')
        ax.legend()
        ax.set_xticks(layers_rp)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'e3_layerwise_probing.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Figure saved: e3_layerwise_probing.png")

    # Figure: Dimension importance + correlation heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Dimension importance bar
    ax = axes[0]
    top20_dims = top_dims[:20]
    top20_imp = dim_importance[top20_dims]
    ax.barh(range(len(top20_dims)), top20_imp[::-1], color='steelblue')
    ax.set_yticks(range(len(top20_dims)))
    ax.set_yticklabels([f'dim_{d}' for d in top20_dims[::-1]], fontsize=8)
    ax.set_xlabel('|Ridge Coefficient|')
    ax.set_title('E3.1: Top 20 Most Important Embedding Dimensions')

    # Correlation heatmap (top 10 dims × descriptors)
    ax = axes[1]
    top10_dims = top_dims[:10]
    corr_matrix = []
    for d in top10_dims:
        row = []
        for dn in descriptor_names:
            if int(d) in correlation_table and dn in correlation_table[int(d)]:
                row.append(correlation_table[int(d)][dn]['pearson_r'])
            else:
                row.append(0.0)
        corr_matrix.append(row)

    corr_matrix = np.array(corr_matrix)
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(descriptor_names)))
    ax.set_xticklabels(descriptor_names, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(top10_dims)))
    ax.set_yticklabels([f'dim_{d}' for d in top10_dims], fontsize=8)
    ax.set_title('E3.2: Dim-Descriptor Correlation (top 10 dims)')
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'e3_dimension_attribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: e3_dimension_attribution.png")

    with open(RESULTS_DIR / 'e3_feature_attribution.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ═════════════════════════════════════════════════════════════════════
# E4: What Changes When Fine-Tuning on LCE?
# ═════════════════════════════════════════════════════════════════════

def linear_CKA(X, Y):
    """Centered Kernel Alignment between two representation matrices."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2
    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)


class FineTunableModel(nn.Module):
    """MOLFormer + FC head for end-to-end fine-tuning."""
    def __init__(self, encoder, comp_dim=6, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        hidden_dim = 768 + comp_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_ids, attention_mask, composition):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.float().unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        combined = torch.cat([pooled, composition], dim=-1)
        return self.head(combined).squeeze(-1)


def run_experiment_e4(e1_results, model, tokenizer, random_model=None):
    """Fine-tuning analysis: what changes in the representation when fine-tuned on LCE?"""
    print("\n" + "=" * 70)
    print("E4: Fine-Tuning Analysis")
    print("=" * 70)

    formulations = e1_results['formulations']
    train_forms = [f for f in formulations if f['split'] == 'train']
    test_forms = [f for f in formulations if f['split'] == 'test']

    train_smiles = [f['concat_smiles'] for f in train_forms]
    test_smiles = [f['concat_smiles'] for f in test_forms]
    y_train = np.array([f['lce'] for f in train_forms])
    y_test = np.array([f['lce'] for f in test_forms])

    train_comp = np.array([f['composition_pcts'] for f in train_forms], dtype=np.float32)
    test_comp = np.array([f['composition_pcts'] for f in test_forms], dtype=np.float32)
    train_comp = np.nan_to_num(train_comp, 0)
    test_comp = np.nan_to_num(test_comp, 0)

    results = {}
    layers = list(range(13))

    # ── Pre-fine-tuning embeddings (all molecules in formulations) ──
    print("\n  Extracting pre-fine-tuning embeddings...")
    # Use all unique concat SMILES (formulations)
    all_smiles = train_smiles + test_smiles
    pre_layer_embs = extract_layerwise_embeddings(model, tokenizer, all_smiles, layers)

    # ── Fine-tune ──
    print("\n  Fine-tuning MOLFormer on LCE...")
    ft_model = FineTunableModel(deepcopy(model), comp_dim=6).to(DEVICE)

    optimizer = torch.optim.AdamW(ft_model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Tokenize training data
    train_tokens = tokenizer(train_smiles, return_tensors='pt', padding=True,
                             truncation=True, max_length=512)
    train_tokens = {k: v.to(DEVICE) for k, v in train_tokens.items()}
    train_comp_t = torch.tensor(train_comp, dtype=torch.float32).to(DEVICE)
    train_y_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)

    # Simple training loop (no validation split to maximize training data)
    n_epochs = 50
    batch_size = 16
    best_loss = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        ft_model.train()
        perm = torch.randperm(len(y_train))
        epoch_loss = 0
        n_batches = 0

        for j in range(0, len(y_train), batch_size):
            idx = perm[j:j + batch_size]
            batch_ids = train_tokens['input_ids'][idx]
            batch_mask = train_tokens['attention_mask'][idx]
            batch_comp = train_comp_t[idx]
            batch_y = train_y_t[idx]

            pred = ft_model(batch_ids, batch_mask, batch_comp)
            loss = nn.functional.mse_loss(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ft_model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in ft_model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: loss={avg_loss:.4f}")

    if best_state:
        ft_model.load_state_dict(best_state)
    ft_model.eval()

    # Evaluate on test
    test_tokens = tokenizer(test_smiles, return_tensors='pt', padding=True,
                            truncation=True, max_length=512)
    test_tokens = {k: v.to(DEVICE) for k, v in test_tokens.items()}
    test_comp_t = torch.tensor(test_comp, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        test_pred = ft_model(test_tokens['input_ids'], test_tokens['attention_mask'],
                             test_comp_t).cpu().numpy()

    ft_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    ft_r2 = r2_score(y_test, test_pred)
    print(f"\n  Fine-tuned model: test RMSE={ft_rmse:.4f}, R2={ft_r2:.4f}")
    results['fine_tuned_performance'] = {
        'test_rmse': float(ft_rmse),
        'test_r2': float(ft_r2),
        'predictions': test_pred.tolist(),
    }

    # ── Post-fine-tuning embeddings ──
    print("\n  Extracting post-fine-tuning embeddings...")
    ft_encoder = ft_model.encoder
    ft_encoder.eval()
    post_layer_embs = extract_layerwise_embeddings(ft_encoder, tokenizer, all_smiles, layers)

    # ── CKA: pre vs post at each layer ──
    print("\n  CKA: pretrained vs fine-tuned at each layer...")
    cka_results = {}
    for l in layers:
        if l in pre_layer_embs and l in post_layer_embs:
            cka = linear_CKA(pre_layer_embs[l], post_layer_embs[l])
            cka_results[l] = float(cka)
            print(f"    Layer {l:2d}: CKA={cka:.4f}")

    results['cka_pre_vs_post'] = cka_results

    # ── CKA: pretrained vs random at each layer (for comparison) ──
    if random_model is not None:
        print("\n  CKA: pretrained vs random at each layer...")
        rand_layer_embs = extract_layerwise_embeddings(
            random_model, tokenizer, all_smiles, layers)
        cka_rand = {}
        for l in layers:
            if l in pre_layer_embs and l in rand_layer_embs:
                cka = linear_CKA(pre_layer_embs[l], rand_layer_embs[l])
                cka_rand[l] = float(cka)
                print(f"    Layer {l:2d}: CKA={cka:.4f}")
        results['cka_pretrained_vs_random'] = cka_rand

    # ── Embedding movement analysis (per-molecule, as planned) ──
    print("\n  Embedding movement analysis (per individual molecule)...")
    # Collect unique molecule SMILES from all formulations
    unique_mol_smiles = {}
    for f in formulations:
        for name, smi in zip(f['molecule_names'], f['molecule_smiles']):
            if smi not in unique_mol_smiles:
                unique_mol_smiles[smi] = name
    mol_smi_list = sorted(unique_mol_smiles.keys())
    mol_name_list = [unique_mol_smiles[s] for s in mol_smi_list]
    print(f"    Unique molecules: {len(mol_smi_list)}")

    # Get pre- and post-fine-tuning embeddings for individual molecules
    pre_mol_embs = extract_frozen_embeddings(model, tokenizer, mol_smi_list)
    post_mol_embs = extract_frozen_embeddings(ft_encoder, tokenizer, mol_smi_list)

    # L2 and cosine distances
    mol_dists = np.linalg.norm(post_mol_embs - pre_mol_embs, axis=1)
    from scipy.spatial.distance import cosine as cosine_dist
    mol_cos_dists = [cosine_dist(pre_mol_embs[i], post_mol_embs[i])
                     for i in range(len(pre_mol_embs))]

    results['embedding_movement'] = {
        'n_molecules': len(mol_smi_list),
        'l2_mean': float(np.mean(mol_dists)),
        'l2_std': float(np.std(mol_dists)),
        'cosine_mean': float(np.mean(mol_cos_dists)),
        'cosine_std': float(np.std(mol_cos_dists)),
        'per_molecule': [
            {'name': mol_name_list[i], 'l2': float(mol_dists[i]),
             'cosine': float(mol_cos_dists[i])}
            for i in range(len(mol_smi_list))
        ],
    }
    print(f"    L2 distance: mean={np.mean(mol_dists):.4f} +/- {np.std(mol_dists):.4f}")
    print(f"    Cosine distance: mean={np.mean(mol_cos_dists):.4f} +/- "
          f"{np.std(mol_cos_dists):.4f}")

    # Top 10 most-moved molecules
    top_moved = np.argsort(mol_dists)[::-1][:10]
    print(f"    Top 10 most-moved molecules:")
    for idx in top_moved:
        print(f"      {mol_name_list[idx]:20s} L2={mol_dists[idx]:.4f} "
              f"cos={mol_cos_dists[idx]:.4f}")

    # Check if molecules in high-CE formulations move more
    # Compute mean LCE for formulations each molecule appears in
    mol_mean_lce = {}
    for f in formulations:
        for smi in f['molecule_smiles']:
            if smi not in mol_mean_lce:
                mol_mean_lce[smi] = []
            mol_mean_lce[smi].append(f['lce'])
    mol_avg_lce = np.array([np.mean(mol_mean_lce.get(s, [0])) for s in mol_smi_list])
    r_l2, p_l2 = pearsonr(mol_dists, mol_avg_lce)
    print(f"    Correlation (molecule L2 movement vs mean formulation LCE): "
          f"r={r_l2:.3f}, p={p_l2:.4f}")
    results['embedding_movement']['movement_lce_correlation'] = {
        'pearson_r': float(r_l2), 'p_value': float(p_l2)
    }
    # Store for plotting
    dists = mol_dists
    all_lce = mol_avg_lce

    # ── Figures ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # CKA comparison
    ax = axes[0]
    cka_layers = sorted(cka_results.keys())
    cka_vals = [cka_results[l] for l in cka_layers]
    ax.plot(cka_layers, cka_vals, 'o-', color='steelblue', linewidth=2,
            label='Pre vs Post fine-tuning')
    if 'cka_pretrained_vs_random' in results:
        cka_rand_vals = [results['cka_pretrained_vs_random'].get(l, np.nan)
                         for l in cka_layers]
        ax.plot(cka_layers, cka_rand_vals, 's--', color='coral', linewidth=2,
                label='Pretrained vs Random')
    ax.set_xlabel('Layer')
    ax.set_ylabel('CKA')
    ax.set_title('E4: CKA — How Much Does Fine-Tuning Change Representations?')
    ax.legend()
    ax.set_xticks(cka_layers)
    ax.set_ylim(0, 1.05)

    # Embedding movement histogram
    ax = axes[1]
    if 'embedding_movement' in results:
        ax.hist(dists, bins=20, color='steelblue', edgecolor='k', alpha=0.7)
        ax.set_xlabel('L2 Distance (pre vs post, per molecule)')
        ax.set_ylabel('Count')
        ax.set_title('E4: Per-Molecule Embedding Movement')

    # Movement vs mean-formulation-LCE scatter
    ax = axes[2]
    if 'embedding_movement' in results:
        ax.scatter(all_lce, dists, c='steelblue', s=30, alpha=0.6, edgecolors='k', linewidths=0.3)
        ax.set_xlabel('Mean Formulation LCE')
        ax.set_ylabel('L2 Movement (per molecule)')
        ax.set_title(f'E4: Molecule Movement vs LCE (r={r_l2:.3f}, p={p_l2:.3f})')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'e4_finetuning_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Figure saved: e4_finetuning_analysis.png")

    with open(RESULTS_DIR / 'e4_finetuning.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results, ft_model


# ═════════════════════════════════════════════════════════════════════
# E5: Attention Head Intervention
# ═════════════════════════════════════════════════════════════════════

def get_attention_patterns(model, tokenizer, smiles, layer_idx=None):
    """Extract attention patterns from MOLFormer.

    Note: MOLFormer uses linear attention (FAVOR+), so these are
    pseudo-attention matrices, not standard softmax attention.
    """
    inputs = tokenizer(smiles, return_tensors='pt', padding=False,
                       truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # outputs.attentions is tuple of (batch, n_heads, seq_len, seq_len)
    attentions = [a[0].cpu().numpy() for a in outputs.attentions]
    return attentions  # list of 12 arrays, each (12, seq_len, seq_len)


def run_experiment_e5(e1_results, ft_model, tokenizer, pretrained_model):
    """Attention head intervention: which heads matter for LCE?"""
    print("\n" + "=" * 70)
    print("E5: Attention Head Intervention")
    print("=" * 70)

    formulations = e1_results['formulations']
    test_forms = [f for f in formulations if f['split'] == 'test']
    train_forms = [f for f in formulations if f['split'] == 'train']

    test_smiles = [f['concat_smiles'] for f in test_forms]
    y_test = np.array([f['lce'] for f in test_forms])
    test_comp = np.array([f['composition_pcts'] for f in test_forms], dtype=np.float32)
    test_comp = np.nan_to_num(test_comp, 0)

    results = {}

    # ── E5.1: Cross-molecule attention before vs after fine-tuning ──
    print("\n  E5.1: Cross-molecule attention analysis...")

    # Select a few test formulations with multiple molecules
    multi_mol_forms = [f for f in test_forms if len(f['molecule_smiles']) >= 2]
    if not multi_mol_forms:
        multi_mol_forms = [f for f in train_forms if len(f['molecule_smiles']) >= 2][:5]

    cross_attn_results = {'pretrained': [], 'finetuned': []}

    for f in multi_mol_forms[:5]:
        concat_smi = f['concat_smiles']
        if not concat_smi:
            continue

        # Get token boundaries for each molecule
        mol_boundaries = []
        offset = 1  # skip [CLS]
        for smi in f['molecule_smiles']:
            tokens = tokenizer.tokenize(smi)
            n_tokens = len(tokens)
            mol_boundaries.append((offset, offset + n_tokens))
            offset += n_tokens + 1  # +1 for separator '.'

        # Get attention from pretrained model
        try:
            pre_attns = get_attention_patterns(pretrained_model, tokenizer, concat_smi)
        except Exception as e:
            print(f"    Attention extraction failed for pretrained: {e}")
            continue

        # Get attention from fine-tuned encoder
        try:
            ft_attns = get_attention_patterns(ft_model.encoder, tokenizer, concat_smi)
        except Exception as e:
            print(f"    Attention extraction failed for fine-tuned: {e}")
            continue

        # Compute cross-molecule attention fraction per layer per head
        for model_name, attns in [('pretrained', pre_attns), ('finetuned', ft_attns)]:
            form_cross = []
            for layer_attns in attns:
                # layer_attns shape: (n_heads, seq_len, seq_len)
                n_heads = layer_attns.shape[0]
                head_cross = []
                for h in range(n_heads):
                    attn = layer_attns[h]
                    total_attn = 0
                    cross_attn = 0
                    for src_start, src_end in mol_boundaries:
                        for tgt_start, tgt_end in mol_boundaries:
                            if src_start >= attn.shape[0] or tgt_start >= attn.shape[1]:
                                continue
                            src_e = min(src_end, attn.shape[0])
                            tgt_e = min(tgt_end, attn.shape[1])
                            block = attn[src_start:src_e, tgt_start:tgt_e]
                            total_attn += block.sum()
                            if (src_start, src_end) != (tgt_start, tgt_end):
                                cross_attn += block.sum()
                    frac = cross_attn / max(total_attn, 1e-10)
                    head_cross.append(float(frac))
                form_cross.append(head_cross)
            cross_attn_results[model_name].append(form_cross)

    # Average cross-molecule attention across formulations
    if cross_attn_results['pretrained']:
        n_layers = len(cross_attn_results['pretrained'][0])
        n_heads = len(cross_attn_results['pretrained'][0][0])

        avg_cross = {}
        for model_name in ['pretrained', 'finetuned']:
            data = cross_attn_results[model_name]
            avg = np.zeros((n_layers, n_heads))
            for form_data in data:
                for l in range(n_layers):
                    for h in range(n_heads):
                        avg[l, h] += form_data[l][h]
            avg /= len(data)
            avg_cross[model_name] = avg

        # Overall cross-molecule attention
        for model_name in ['pretrained', 'finetuned']:
            mean_cross = avg_cross[model_name].mean()
            print(f"    {model_name}: mean cross-molecule attention = {mean_cross:.4f}")

        results['cross_molecule_attention'] = {
            'pretrained_mean': float(avg_cross['pretrained'].mean()),
            'finetuned_mean': float(avg_cross['finetuned'].mean()),
            'pretrained_by_layer': avg_cross['pretrained'].mean(axis=1).tolist(),
            'finetuned_by_layer': avg_cross['finetuned'].mean(axis=1).tolist(),
            'n_formulations_analyzed': len(cross_attn_results['pretrained']),
        }

        # ── E5.2: Head ablation ──
        print("\n  E5.2: Head ablation...")

        # Baseline test RMSE
        ft_model.eval()
        test_tokens = tokenizer(test_smiles, return_tensors='pt', padding=True,
                                truncation=True, max_length=512)
        test_tokens_dev = {k: v.to(DEVICE) for k, v in test_tokens.items()}
        test_comp_t = torch.tensor(test_comp, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            baseline_pred = ft_model(
                test_tokens_dev['input_ids'],
                test_tokens_dev['attention_mask'],
                test_comp_t).cpu().numpy()
        baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
        print(f"    Baseline RMSE: {baseline_rmse:.4f}")

        # Ablate each head by zeroing its contribution in the self-attention output.
        # Hook into layer.attention.self (MolformerSelfAttention), whose output is
        # context_layer of shape (batch, seq_len, 768) — the concatenation of 12
        # heads each of size 64. We zero the target head's 64-dim slice there,
        # before it goes through the attention output dense layer.
        ablation_results = {}
        encoder = ft_model.encoder

        for layer_idx in range(12):
            for head_idx in range(12):
                def make_ablation_hook(target_head):
                    def hook_fn(module, input, output):
                        # output is tuple: (context_layer,) or (context_layer, attn_probs)
                        context = output[0].clone()
                        head_dim = 64
                        start = target_head * head_dim
                        end = start + head_dim
                        context[:, :, start:end] = 0
                        return (context,) + output[1:]
                    return hook_fn

                attn_self = encoder.encoder.layer[layer_idx].attention.self
                hook = attn_self.register_forward_hook(make_ablation_hook(head_idx))

                try:
                    with torch.no_grad():
                        ablated_pred = ft_model(
                            test_tokens_dev['input_ids'],
                            test_tokens_dev['attention_mask'],
                            test_comp_t).cpu().numpy()
                    ablated_rmse = np.sqrt(mean_squared_error(y_test, ablated_pred))
                    delta = ablated_rmse - baseline_rmse
                    ablation_results[f'L{layer_idx}_H{head_idx}'] = {
                        'rmse': float(ablated_rmse),
                        'delta_rmse': float(delta),
                    }
                except Exception as e:
                    print(f"    Ablation L{layer_idx}_H{head_idx} failed: {e}")
                finally:
                    hook.remove()

        # Find most impactful heads
        sorted_heads = sorted(ablation_results.items(),
                              key=lambda x: x[1]['delta_rmse'], reverse=True)
        print(f"\n    Top 10 most impactful heads (ablation increases RMSE most):")
        for name, data in sorted_heads[:10]:
            print(f"      {name}: RMSE={data['rmse']:.4f} (delta={data['delta_rmse']:+.4f})")

        results['head_ablation'] = {
            'baseline_rmse': float(baseline_rmse),
            'ablation_results': ablation_results,
            'top_10_heads': [(name, data) for name, data in sorted_heads[:10]],
        }

        # ── Figures ──
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Cross-molecule attention heatmaps
        for ax_idx, (model_name, title) in enumerate([
            ('pretrained', 'Pretrained'), ('finetuned', 'Fine-tuned')
        ]):
            ax = axes[ax_idx]
            im = ax.imshow(avg_cross[model_name], cmap='viridis', aspect='auto')
            ax.set_xlabel('Head')
            ax.set_ylabel('Layer')
            ax.set_title(f'{title}\nCross-Molecule Attention')
            plt.colorbar(im, ax=ax, fraction=0.046)

        # Head ablation heatmap
        ax = axes[2]
        delta_matrix = np.zeros((12, 12))
        for l in range(12):
            for h in range(12):
                key = f'L{l}_H{h}'
                if key in ablation_results:
                    delta_matrix[l, h] = ablation_results[key]['delta_rmse']

        im = ax.imshow(delta_matrix, cmap='RdBu_r', aspect='auto',
                       vmin=-np.abs(delta_matrix).max(),
                       vmax=np.abs(delta_matrix).max())
        ax.set_xlabel('Head')
        ax.set_ylabel('Layer')
        ax.set_title('Head Ablation: Delta RMSE\n(red = hurts prediction)')
        plt.colorbar(im, ax=ax, fraction=0.046)

        plt.tight_layout()
        fig.savefig(FIGURES_DIR / 'e5_attention_intervention.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\n  Figure saved: e5_attention_intervention.png")

    with open(RESULTS_DIR / 'e5_attention.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Phase LCE-2: What Internal Features Help LCE Prediction?")
    print("=" * 70)

    # E1: Parse dataset
    e1_results = run_experiment_e1()

    # Load models
    model, tokenizer = load_pretrained_model()
    random_model = load_random_model(model)

    # E2: Interpretable baselines
    e2_results = run_experiment_e2(e1_results, model, tokenizer, random_model)

    # E3: Feature attribution
    e3_results = run_experiment_e3(e1_results, model, tokenizer, random_model)

    # E4: Fine-tuning analysis
    e4_results, ft_model = run_experiment_e4(e1_results, model, tokenizer, random_model)

    # E5: Attention head intervention
    e5_results = run_experiment_e5(e1_results, ft_model, tokenizer, model)

    print("\n" + "=" * 70)
    print("Phase LCE-2: All experiments complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
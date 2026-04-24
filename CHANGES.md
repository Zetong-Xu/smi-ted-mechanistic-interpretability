# Changes from Original MOLFormer Code

This document records bugs identified in the original MOLFormer
interpretability code and the changes made when migrating experiments
to SMI-TED. The code review was conducted by one person and may not
be exhaustive.

Original code source:
https://github.com/ChicagoHAI/interp_Molmformer

---

## Part 1: Bugs in Original MOLFormer Code

### run_experiments.py

#### Bug 1: Ablation Hook Location (Critical)

**Location**: `extract_embeddings_with_ablation()`

**Problem**:
The hook was attached to the full encoder layer output, which is
after the output projection matrix. At this point, all 12 attention
heads have already been mixed together. Zeroing out a 64-dimensional
slice of the 768-dimensional output does not equal zeroing out one
head's contribution, because the output projection has already
combined information from all heads into every dimension.

**Original code**:
```python
target_layer = model.encoder.layer[ablate_layer]
hooks.append(target_layer.register_forward_hook(ablation_hook))
```

**Fixed code**:
```python
# Hook before output projection, where head outputs are still separate
target_layer = model.encoder.layer[ablate_layer].attention.self
hooks.append(target_layer.register_forward_hook(ablation_hook))
```

**Impact**: Experiment 3 ablation results are unreliable in the
original code. The ablation did not actually isolate individual
head contributions.

---

#### Bug 2: Data Leakage in Linear Probing (Critical)

**Location**: `run_linear_probing()`

**Problem**:
Train/test split was performed at the atom level. Since all atoms
from a molecule are concatenated into a single matrix, atoms from
the same molecule could appear in both train and test sets. The
probe would then see atoms from test molecules during training,
inflating test accuracy.

**Original code**:
```python
n = len(y)
indices = np.random.permutation(n)
split = int(0.8 * n)
train_idx, test_idx = indices[:split], indices[split:]
```

**Fixed code**:
```python
# Split at molecule level to prevent data leakage
molecule_indices = df_labels['molecule_idx'].values
unique_mols = np.unique(molecule_indices)
np.random.shuffle(unique_mols)
mol_split = int(0.8 * len(unique_mols))
train_mols = set(unique_mols[:mol_split])
test_mols = set(unique_mols[mol_split:])
train_idx = np.where([m in train_mols for m in molecule_indices])[0]
test_idx = np.where([m in test_mols for m in molecule_indices])[0]
```

**Impact**: Linear probing accuracy in Experiment 2 was overestimated
in the original code.

---

#### Bug 3: Redundant Code (Minor)

The following code was defined but never called or used:
- `smiles_token_to_atom_indices()`: defined but never called
- `load_esol_data()`: defined but never called
- `hidden_states` variable in `extract_attention_and_distances()`:
  assigned but never used
- `labels` parameter in `extract_embeddings_with_ablation()`:
  accepted but never used

**Impact**: No effect on results. Causes confusion when reading code.

---

### run_pca_experiments.py

#### Bug 1: Global Variable Usage (Medium)

**Location**: `run_experiment_0c()`

**Problem**:
`qm9_df_global` is a global variable assigned in `main()`. If the
function is called in a different order or from a different context,
the variable may still be None and cause a runtime error.

**Fix**: Pass `qm9_df` as a direct parameter to `run_experiment_0c()`
instead of relying on a global variable.

---

#### Bug 2: Missing UMAP Dependency Guard (Medium)

**Location**: `run_experiment_0a()`

**Problem**:
`import umap` is called without a try/except block. If `umap-learn`
is not installed, the entire experiment crashes rather than gracefully
skipping the UMAP visualization.

**Fix**: Wrap UMAP section in try/except ImportError.

---

#### Bug 3: Carbon Sampling Edge Case (Low)

**Location**: `run_experiment_0b()`

**Problem**:
`type_counts.iloc[1]` assumes at least two atom types exist in the
dataset. On single-atom-type datasets this raises an IndexError.

**Fix**: Add a length check before accessing `iloc[1]`.

---

### run_lce_experiments.py

#### Bug 1: Validation RMSE Reported as Test RMSE (Critical)

**Location**: `run_experiment_l4a()`

**Problem**:
The original code used validation set RMSE as the reported test
performance. The validation set is used during training for early
stopping, so the model has indirectly seen this data during training.
Reporting validation RMSE overestimates true generalization
performance.

**Fix**: After training completes, evaluate the best model checkpoint
on a held-out test set that was not used during training or model
selection.

**Impact**: L4a FC head performance numbers were overestimated in the
original code.

---

#### Bug 2: Inconsistent Concatenation Separator (Medium)

**Location**: `run_experiment_l3b()` and `run_experiment_l3c()`

**Problem**:
L3a uses `'.'` as the molecule separator (no spaces), but original
L3b and L3c used `' . '` (with spaces). Different separators produce
different token sequences, making results across L3 sub-experiments
incomparable.

**Fix**: All L3 experiments now use `'.'` without spaces for
consistency.

---

#### Bug 3: Over-attribution of Embedding Differences (Medium)

**Location**: `run_experiment_l3b()` result interpretation

**Problem**:
The deviation between the concatenated embedding and the weighted
average of individual embeddings was attributed entirely to
cross-molecule interaction effects. However, at least three confounds
contribute to this deviation:
- The separator token `'.'` is present in concatenated input but
  absent in individual molecule inputs
- Special tokens (BOS/EOS) contribute differently at different
  sequence lengths
- Token-count weighting differs between concatenated and individual
  forward passes

**Fix**: Added comments clarifying these confounds. The deviation
cannot be solely attributed to cross-molecule interaction effects.

---

#### Bug 4: Function Name Mismatch (Low)

**Location**: `run_experiment_l4a()`

**Problem**:
The function description referred to "Frozen vs Fine-Tuned" but the
implementation only included frozen encoder experiments with no
fine-tuning comparison.

**Fix**: Updated description to "Frozen Embeddings Performance" to
match actual implementation.

---

### run_lce2_experiments.py

#### Bug 1: Heuristic Train/Test Split (Design Limitation)

**Location**: `run_experiment_e1()`

**Problem**:
The test set is identified by matching LCE values from Table 1 in
Soares et al. using nearest-neighbor matching with a tolerance of
0.02. This is an approximation, not the exact split used in the
paper. If the matching is incorrect, all downstream results cannot
be directly compared to the paper's reported numbers.

**Status**: No code fix possible without access to the original split
indices. Added explicit warning in output. All LCE-2 results should
be interpreted with this limitation in mind.

---

#### Bug 2: Ablation Hook Path (Verified Correct)

**Location**: `run_experiment_e5()`

**Initial concern**:
It was initially unclear whether
`encoder.encoder.layer[layer_idx]` contained a double-nested path
error.

**Verification**: The path is correct. `encoder` refers to
`ft_model.encoder` (the MOLFormer encoder object), and
`encoder.encoder` refers to the internal transformer encoder within
that object. No fix needed.

---

#### Bug 3: Fine-Tuning Overfitting (Design Limitation)

**Location**: `run_experiment_e4()`

**Problem**:
End-to-end fine-tuning of a 289M parameter model on approximately
137 training samples causes severe overfitting. Test RMSE after
fine-tuning is worse than using frozen Ridge regression, confirming
overfitting rather than generalization.

**Status**: This is a dataset size limitation, not a code bug. Results
are reported as-is with this limitation noted.

---

#### Bug 4: Typo in Excel Data (Data Quality Issue)

**Location**: `build_molecule_lookup()`

**Problem**:
The original Excel file (pnas.2214357120.sd01.xlsx) contains `[LI+]`
(uppercase) instead of `[Li+]` (correct) for LiTFPFB and LiBF4.
`[LI+]` is not valid SMILES and causes RDKit to fail parsing these
molecules.

**Fix**:
```python
lookup = {k: v.replace('[LI+]', '[Li+]')
          for k, v in lookup.items()}
```

---

## Part 2: SMI-TED Migration Changes

### Change 1: Model Loading

**MOLFormer**:
```python
from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    'ibm/MoLFormer-XL-both-10pct', trust_remote_code=True)
model = AutoModel.from_pretrained(
    'ibm/MoLFormer-XL-both-10pct', trust_remote_code=True)
```

**SMI-TED**:
```python
from smi_ted_light.load import load_smi_ted
model = load_smi_ted(
    folder='./smi_ted_light',
    ckpt_filename='smi-ted-Light_40.pt'
)
tokenizer = model.tokenizer
```

---

### Change 2: Token-to-Atom Mapping

**MOLFormer**:
Hand-written character-level rules to identify atom tokens
(e.g., checking if token is in `'BCNOPSFIcnops'`).

**SMI-TED**:
Uses SMI-TED's official regex pattern from `load.py`, combined with
canonicalization via `Chem.MolToSmiles(mol, canonical=True,
isomericSmiles=False)` to match SMI-TED's internal
`normalize_smiles()` call. Without canonicalization, the token order
and RDKit atom numbering may not match.

---

### Change 3: Hook Paths

**MOLFormer**:
```python
model.embeddings          # embedding layer
model.encoder.layer[i]   # encoder layer i
```

**SMI-TED**:
```python
model.encoder.tok_emb              # embedding layer
model.encoder.blocks.layers[i]    # encoder layer i
```

---

### Change 4: Forward Pass

**MOLFormer**:
```python
inputs = tokenizer(smi, return_tensors='pt', ...)
outputs = model(**inputs, output_attentions=False)
```

**SMI-TED**:
```python
idx, mask = model.tokenize(smi)
token_embeddings = model.encoder(idx, mask)
```

---

### Change 5: Attention Extraction (Experiment 1 Skipped)

**MOLFormer**:
```python
outputs = model(**inputs, output_attentions=True)
attentions = outputs.attentions  # explicit (L, H, N, N) matrices
```

**SMI-TED**:
SMI-TED uses linear attention (FAVOR+) via the
`fast_transformers` library. Linear attention computes
`φ(Q)(φ(K)ᵀV) / φ(Q)(φ(K)ᵀ1)` and does not produce explicit
`(seq_len × seq_len)` attention matrices. Experiment 1
(attention-distance correlation) is therefore not applicable to
SMI-TED and is skipped.

---

### Change 6: Molecule-Level Embedding

**MOLFormer**:
```python
# Mean pooling over last_hidden_state
emb = outputs.last_hidden_state[0].mean(axis=0)
```

**SMI-TED**:
```python
# Autoencoder latent vector z (SMI-TED's native representation)
token_embeddings = model.encoder(idx, mask)  # (1, 202, 768)
z = model.decoder.autoencoder.encoder(
    token_embeddings.view(-1, max_len * n_embd)
)  # (1, 768)
```

SMI-TED's encoder produces token embeddings padded to `max_len=202`.
The autoencoder encoder compresses this into a 768-dimensional latent
vector, which is SMI-TED's native molecule representation used in
downstream tasks.

---

### Change 7: Ablation Hook (Experiment 3)

**MOLFormer (fixed version)**:
```python
# Hook on attention.self (before output projection)
target = model.encoder.layer[ablate_layer].attention.self
# output shape: (batch, seq_len, 768) — heads concatenated
# zero out: hidden[:, :, head*64:(head+1)*64] = 0
```

**SMI-TED**:
```python
# Hook on inner_attention (before output projection)
target = (model.encoder.blocks.layers[ablate_layer]
          .attention.inner_attention)
# output shape: (batch, seq_len, n_heads, head_dim)
#             = (1, seq_len, 12, 64) — confirmed experimentally
# zero out: output[:, :, target_head, :] = 0
```

The `inner_attention` output preserves the per-head structure before
the `out_projection` mixes heads together. This is the correct
location to truly ablate individual head contributions.

---

### Change 8: Random Model Creation

**MOLFormer**:
```python
random_model = AutoModel.from_config(pretrained_model.config)
```

**SMI-TED**:
```python
from smi_ted_light.load import Smi_ted, MoLEncoder, MoLDecoder
random_model = Smi_ted(pretrained_model.tokenizer)
random_model.encoder = MoLEncoder(
    pretrained_model.config,
    len(pretrained_model.tokenizer.vocab)
)
random_model.decoder = MoLDecoder(
    len(pretrained_model.tokenizer.vocab),
    pretrained_model.max_len,
    pretrained_model.n_embd
)
```

`MoLEncoder` and `MoLDecoder` initialize with random weights by
default, producing a randomly initialized model with the same
architecture as the pretrained SMI-TED.
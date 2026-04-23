# Mechanistic Interpretability of SMI-TED

This repository applies mechanistic interpretability methods
to SMI-TED, a large encoder-decoder chemical foundation model
pretrained on 91 million SMILES from PubChem.

## Background

The experimental framework is adapted from
[ChicagoHAI/interp_Molformer](https://github.com/ChicagoHAI/interp_Molmformer),
which applies mechanistic interpretability methods to MOLFormer.
I conducted an initial review of the original code, identified
potential issues, and applied fixes before migrating the
experiments to SMI-TED. 

## Repository Structure

```
smi-ted-mechanistic-interpretability/
├── molformer/
│   ├── original/          # Original MOLFormer code (unmodified)
│   └── fixed/             # Version with identified bugs fixed
├── smi_ted/               # SMI-TED experiments (main work)
├── results/
│   ├── molformer/
│   └── smi_ted/
├── CHANGES.md             # Documented bugs and migration changes
└── README.md
```

## Experiments

### Phase Main: Mechanistic Interpretability
- **Experiment 1**: Attention-distance correlation
  - Does the model attend to spatially close atoms?
- **Experiment 2**: Linear probing for chemical properties
  - Are aromaticity, ring membership, and atom type linearly
    decodable from hidden states?
- **Experiment 3**: Attention head ablation
  - Which attention heads are causally important for
    downstream property prediction?

### Phase 0: PCA Visualization
- Molecule-level PCA across layers
- Atom-level PCA with chemical property coloring
- Pretrained vs randomly initialized model comparison
- Layer trajectory visualization

### Phase LCE: Battery Electrolyte Performance
- Statistical significance of pretraining claims
- Electrolyte molecule embedding analysis
- Cross-molecule attention patterns
- Sample efficiency on downstream tasks

### Phase LCE-2: Internal Feature Analysis
- Interpretable baselines vs MOLFormer embeddings
- Feature attribution: which embedding dimensions matter
- Fine-tuning analysis
- Attention head intervention

## Bugs Identified in Original Code

See CHANGES.md for full details. Two potentially critical bugs
were identified in the original MOLFormer experiments during
our initial review:

**Bug 1: Ablation hook location**
The hook was attached after output projection, where attention
heads are already mixed. Fixed by moving the hook to before
output projection.

**Bug 2: Data leakage in linear probing**
Train/test split was done at atom level, allowing atoms from
the same molecule to appear in both train and test sets.
Fixed by splitting at molecule level.

## SMI-TED vs MOLFormer

| | MOLFormer | SMI-TED |
|--|-----------|---------|
| Architecture | Encoder-only | Encoder-Decoder |
| Parameters | 47M | 289M |
| Pretraining data | 1.1B SMILES | 91M curated SMILES |
| Attention type | Linear (FAVOR+) | Linear (FAVOR+) |
| Molecule embedding | Mean pooling | Autoencoder latent z |
| Framework | HuggingFace | fast_transformers |

## Key Migration Changes

1. **Model loading**: `load_smi_ted()` instead of HuggingFace
   `AutoModel`
2. **Tokenizer**: SMI-TED's regex-based `MolTranBertTokenizer`
3. **Hook paths**: `model.encoder.blocks.layers[i]` instead of
   `model.encoder.layer[i]`
4. **Attention extraction**: Hook-based instead of
   `output_attentions=True`
5. **Molecule embedding**: Autoencoder latent z instead of
   mean pooling

## Setup

### Environment
```bash
conda activate fm4m
```

### Run Experiments
```bash
cd smi_ted
python run_experiments.py
```

## References

1. Ross, J., et al. (2022). Large-Scale Chemical Language
   Representations Capture Molecular Structure and Properties.
   *Nature Machine Intelligence*.

2. Soares, E., et al. (2024). A Large Encoder-Decoder Family
   of Foundation Models for Chemical Language.
   *arXiv:2407.20267*.

3. Kim, S. C., et al. (2023). Data-driven electrolyte design
   for lithium metal anodes. *PNAS*, 120(10), e2214357120.

4. ChicagoHAI. interp_Molformer. GitHub repository.
   https://github.com/ChicagoHAI/interp_Molmformer
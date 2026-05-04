"""
Experiment 2: MOLFormer vs SMI-TED Comparison Plots
====================================================
Reads results/molformer/exp2_probing.json and
        results/smi_ted/exp2_probing.json
and produces one comparison figure per property showing
both models on the same axes with unified y-axis (0–1).

Run this after both exp2 scripts have completed.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

MOLFORMER_JSON = Path('results/molformer/exp2_probing.json')
SMITED_JSON    = Path('results/smi_ted/exp2_probing.json')
FIGURES_DIR    = Path('results/comparison/figures')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Properties to compare (intersection of what both models have)
PROPERTIES = [
    'atom_type',
    'hybridization',
    'is_aromatic',
    'is_in_ring',
    'chiral_tag',
    'degree',
    'formal_charge',
    'total_valence',
]

# ─────────────────────────────────────────────────────────────────────
# Load Results
# ─────────────────────────────────────────────────────────────────────

def load_results(json_path):
    with open(json_path) as f:
        raw = json.load(f)
    # Keys are stored as strings — convert layer keys to int
    return {
        task: {int(k): v for k, v in layer_res.items()}
        for task, layer_res in raw.items()
    }

# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────

def plot_comparison(molformer_results, smited_results):
    """
    One figure per property. Each figure shows:
      - MOLFormer model activations   (solid blue)
      - SMI-TED model activations     (solid orange)
      - MOLFormer random baseline     (dashed blue)
      - SMI-TED random baseline       (dashed orange)
      - Frequency baseline            (dotted red, from MOLFormer;
                                       both models use same dataset
                                       so frequency is identical)
    All figures share unified y-axis (0 to 1).
    """
    for prop in PROPERTIES:
        if prop not in molformer_results or prop not in smited_results:
            print(f"Skipping {prop} — not found in both result files")
            continue

        mol_res = molformer_results[prop]
        smi_res = smited_results[prop]

        mol_layers = sorted(mol_res.keys())
        smi_layers = sorted(smi_res.keys())

        fig, ax = plt.subplots(figsize=(9, 5))

        # MOLFormer model activations
        mol_accs = [mol_res[l]['accuracy'] for l in mol_layers]
        ax.plot(mol_layers, mol_accs, marker='o', linewidth=2,
                color='steelblue', label='MOLFormer')

        # SMI-TED model activations
        smi_accs = [smi_res[l]['accuracy'] for l in smi_layers]
        ax.plot(smi_layers, smi_accs, marker='o', linewidth=2,
                color='darkorange', label='SMI-TED')

        # MOLFormer random baseline
        if 'accuracy_random' in mol_res[mol_layers[0]]:
            mol_rand = [mol_res[l]['accuracy_random'] for l in mol_layers]
            ax.plot(mol_layers, mol_rand, marker='s', linewidth=1.5,
                    linestyle='--', color='steelblue', alpha=0.6,
                    label='MOLFormer (random input)')

        # SMI-TED random baseline
        if 'accuracy_random' in smi_res[smi_layers[0]]:
            smi_rand = [smi_res[l]['accuracy_random'] for l in smi_layers]
            ax.plot(smi_layers, smi_rand, marker='s', linewidth=1.5,
                    linestyle='--', color='darkorange', alpha=0.6,
                    label='SMI-TED (random input)')

        # Frequency baseline (same dataset → same value for both models)
        if 'frequency_baseline' in mol_res[mol_layers[0]]:
            freq_acc = mol_res[mol_layers[0]]['frequency_baseline']
            ax.axhline(freq_acc, linestyle=':', linewidth=2,
                       color='tomato',
                       label=f'Frequency baseline ({freq_acc:.2f})')

        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(
            f'MOLFormer vs SMI-TED — Linear Probe Accuracy: {prop}',
            fontsize=13)
        ax.set_xticks(range(13))
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = FIGURES_DIR / f'exp2_comparison_{prop}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Experiment 2: MOLFormer vs SMI-TED Comparison")
    print("=" * 70)

    if not MOLFORMER_JSON.exists():
        raise FileNotFoundError(
            f"MOLFormer results not found: {MOLFORMER_JSON}\n"
            "Run molformer_exp2_linear_probing.py first.")

    if not SMITED_JSON.exists():
        raise FileNotFoundError(
            f"SMI-TED results not found: {SMITED_JSON}\n"
            "Run smi_ted_exp2_linear_probing.py first.")

    molformer_results = load_results(MOLFORMER_JSON)
    smited_results    = load_results(SMITED_JSON)

    print(f"MOLFormer properties: {list(molformer_results.keys())}")
    print(f"SMI-TED properties  : {list(smited_results.keys())}")

    plot_comparison(molformer_results, smited_results)

    print("\n" + "=" * 70)
    print("COMPARISON PLOTS COMPLETE")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()

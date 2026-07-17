# BridgeDTA

BridgeDTA: Lightweight Cross-Attention Bridges LargePre-trained Models for DrugTarget Affinity Prediction

## Overview

BridgeDTA is a parameter-efficient framework for Drug-Target Affinity (DTA) prediction. It bridges frozen Large Pre-trained Models (LPMs) with trainable Expert encoders through a Cross-Attention Bridge module. This design achieves strong performance with minimal trainable parameters.

**Key Components:**
- **KPGT** (frozen): Pre-trained molecular representation
- **ESM-2** (frozen): Pre-trained protein representation
- **Expert Encoders** (trainable): GIN-based Drug Encoder + CNN-based Protein Encoder
- **Cross-Attention Bridge** (trainable): Fuses multi-view features via cross-attention
- **CI-Loss**: Auxiliary ranking loss for concordance index optimization

## Project Structure

```
BridgeDTA/
├── data/                           # Datasets
│   ├── davis/                      # Davis dataset (kinase inhibitors)
│   │   ├── proteins.txt            # Protein FASTA sequences
│   │   ├── ligands_can.txt         # Ligand SMILES strings
│   │   ├── Y                       # Affinity matrix
│   │   ├── folds/                  # Cross-validation splits
│   │   └── processed/              # Train/test splits
│   └── kiba/                       # KIBA dataset (multi-assay bioactivity)
│       ├── proteins.txt            
│       ├── ligands_can.txt         
│       ├── Y                       
│       ├── folds/                  
│       └── processed/              
│
├── models/                         # Model definitions
│   ├── Bridge_DTA/                 # BridgeDTA (proposed model)
│   │   ├── bridge_dta_full.py      # Full model architecture
│   │   ├── feature_esm/            # Pre-computed ESM-2 protein features (.npz)
│   │   └── feature_kpgt/           # Pre-computed KPGT molecule features (.npz)
│   ├── DeepDTA/                    # Baseline: 1D CNN encoder
│   │   ├── deepdta.py
│   │   └── prepare_data.py
│   ├── GraphDTA/                   # Baseline: GIN graph neural network
│   │   ├── graphdta.py
│   │   └── prepare_data.py
│   ├── TEFDTA/                     # Baseline: Transformer encoder
│   │   ├── tefdta.py
│   │   └── process_data.py
│   ├── GS_DTA/                     # Baseline: GATv2 + GCN + BiLSTM + Transformer
│   │   ├── gs_dta.py
│   │   └── create_data.py
│   └── MultiKD_DTA/                # Baseline: GIN + BiLSTM + LinkAttention
│       ├── multikd_dta.py
│       └── create_data.py
│
├── experiments/                    # Single-run experiment scripts
│   ├── 1_tefdta.py                 # TEFDTA baseline
│   ├── 2_gs_dta.py                 # GS-DTA baseline
│   ├── 3_multikd_dta.py            # MultiKD-DTA baseline
│   ├── 4_bridge_dta_full.py        # BridgeDTA (proposed)
│   ├── 5_graphdta.py               # GraphDTA baseline
│   └── 6_deepdta.py                # DeepDTA baseline
│
├── exp_fold/                       # 5-fold cross-validation scripts
│   ├── bridgedta.py                # BridgeDTA 5-fold
│   ├── deepdta.py                  # DeepDTA 5-fold
│   ├── graphdta.py                 # GraphDTA 5-fold
│   ├── gsdta.py                    # GS-DTA 5-fold
│   ├── multikd_dta.py              # MultiKD-DTA 5-fold
│   └── tefdta.py                   # TEFDTA 5-fold
│
├── datahelper/                     # Data utilities
│   ├── bridge_dataset.py           # BridgeDTA-specific dataset
│   ├── dataset.py                  # Abstract DTADataset base class
│   └── split_train_test_data.py    # Train/test split utility
│
├── utils/
│   └── metrics.py                  # Evaluation metrics (MSE, CI, r²_m, PCC, Spearman)
│
└── trainer.py                      # General DTA trainer
```

## Installation

```bash
# 1. Create conda environment
conda create -n bridgedta python=3.9
conda activate bridgedta

# 2. Install PyTorch (adjust CUDA version as needed)
pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu116

# 3. Install PyTorch Geometric
pip install torch_geometric==2.3.1

# 4. Install all other dependencies
pip install -r requirements.txt
```

**Core dependencies:** Python 3.9+, PyTorch 1.13.1, PyG 2.3.1, RDKit, NumPy, SciPy, scikit-learn. See [`requirements.txt`](requirements.txt) for full list.

> **CUDA wheel note:** The PyG C++ backends (`torch_scatter`, `torch_sparse`, `torch_cluster`) must match the PyTorch build. After installing PyTorch, install them from the matching wheel index, e.g. for CPU:
> ```bash
> pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
> ```
> For CUDA 11.6, use `torch-1.13.1+cu116.html` instead.

> **Optional (feature extraction only):** If you intend to (re)generate the
> frozen `.npz` features yourself rather than downloading them, you also need
> `pip install fair-esm` (see *Regenerating the pre-computed features*).
> Running the published model and reproducing the paper's results does **not**
> require this — the Figshare `.npz` bundle is enough.

## ⚠️ Required: pre-computed KPGT & ESM-2 features

BridgeDTA (the proposed model) loads **frozen** pre-computed features from disk
at the start of every run. They are **not** included in the repository because
of their size (≈ several hundred MB). Without them, `experiments/4_bridge_dta_full.py`
and `exp_fold/bridgedta.py` will refuse to start with a `FileNotFoundError`.

**Download once before the first run:**

📥 **Download link:** [https://doi.org/10.6084/m9.figshare.32294592](https://doi.org/10.6084/m9.figshare.32294592)

After downloading, unzip and place the `.npz` files exactly as:

```
models/Bridge_DTA/
├── feature_esm/
│   ├── davis.npz
│   └── kiba.npz
└── feature_kpgt/
    ├── davis.npz           # graph-level (used when --kpgt_feature_level graph)
    ├── davis_node.npz      # token-level   (used when --kpgt_feature_level node, default)
    ├── kiba.npz
    └── kiba_node.npz
```

The four `_node.npz` files store: `smiles` (canonical SMILES array), `atom_feats`
(concatenated per-atom features `[total_atoms, 768]`), `offsets` (per-molecule
slice indices into `atom_feats`), and two virtual-node tensors `fp_vn`/`md_vn`
(`[N, 768]`).

If you only need to reproduce the **baselines** (DeepDTA, GraphDTA, TEFDTA,
GS-DTA), the pre-computed features are NOT required — those baselines use only
the `data/` text files that ship with this repository. The only baseline that
also needs an ESM `.npz` is MultiKD-DTA (it expects
`models/MultiKD_DTA/features/kiba.npz`, also obtainable from the Figshare link).

## Datasets

| Dataset | Compounds | Proteins | Interactions | Train | Test |
|---------|-----------|----------|-------------|-------|------|
| Davis   | 68        | 442      | 30,056      | 25,046| 5,010|
| KIBA    | 2,111     | 229      | 118,254     | 98,545| 19,709|

### Data Source

All data used in this paper are publicly available and can be accessed at [DeepDTA/data](https://github.com/hkmztrk/DeepDTA/tree/master/data).

The processed data splits are already included in the `data/` directory of this repository.

## Running Experiments

All entry-point scripts accept **command-line overrides** for the most
important hyper-parameters: `--dataset`, `--num_epochs`, `--batch_size`,
`--seed`, `--log_steps`, `--cold_start`. Anything not provided falls back to
the script's own defaults (documented in the `TrainingArguments` class at the
top of each script). This means you can run a fast smoke test without editing
any source file.

### Quick smoke test (no pre-computed features needed — baseline only)

```bash
# Run DeepDTA for 1 epoch on the Davis dataset on CPU
python experiments/6_deepdta.py --dataset davis --num_epochs 1 --batch_size 64 --log_steps 1
```

A successful run will print a `Training started at: ...` banner, log every
`TrainingArguments` value, then start epoch 1. The output directory and log
file are created under `results/` (see *Outputs* below). Expect dozens of
minutes per epoch on CPU; on GPU it finishes within a minute.

### Single-run training

```bash
python experiments/4_bridge_dta_full.py --dataset davis --num_epochs 500   # BridgeDTA (proposed)
python experiments/1_tefdta.py    --dataset davis --num_epochs 200          # TEFDTA baseline
python experiments/2_gs_dta.py    --dataset davis --num_epochs 500          # GS-DTA baseline
python experiments/3_multikd_dta.py --dataset kiba --num_epochs 500          # MultiKD-DTA baseline (needs features/kiba.npz)
python experiments/5_graphdta.py  --dataset davis --num_epochs 500          # GraphDTA baseline
python experiments/6_deepdta.py    --dataset davis --num_epochs 500          # DeepDTA baseline
```

### 5-fold cross-validation (formal evaluation)

```bash
python exp_fold/bridgedta.py 0   # fold index 0..4; BridgeDTA proposed model
python exp_fold/deepdta.py   0   # baselines (same fold index convention)
python exp_fold/graphdta.py  0
python exp_fold/gsdta.py     0
python exp_fold/multikd_dta.py 0   # needs models/MultiKD_DTA/features/kiba.npz
python exp_fold/tefdta.py    0
```

### Outputs

Results are saved under `results/<model_name>/` inside the repository:

```
results/<model_name>/
├── <MMDD>_<model>_<dataset>.log          # full training log (per-epoch loss, validation metrics, test results)
└── checkpoints_<dataset>/
    ├── best_model.pth                    # weights with the lowest validation MSE
    ├── final_model.pth                    # final-epoch weights (saved by trainer.save_model())
    └── plots/<model>_<dataset>_training_curves.png   # loss/CI/rm2 curves
```

The trainer writes a `Test Results:` block to the log at the end of training,
listing every metric in the *Evaluation Metrics* table below, e.g.:

```
Test Results:
mse: 0.1903
ci: 0.9041
rm2: 0.6720
pcc: 0.8512
spearman: 0.6347
loss: 0.1903
```

### Inputs

Per dataset, the `data/<dataset>/` directory contains the raw inputs used by
all models:

| File | Format | Meaning |
|------|--------|---------|
| `ligands_can.txt` | JSON object `{id: canonical_SMILES}` | unique compound set |
| `proteins.txt`    | JSON object `{id: FASTA}`            | unique protein set |
| `Y`               | pickled 2D NumPy array `[len(ligands), len(proteins)]` | affinity matrix; Davis stores `pK_d = -log10(K_d/1e9)`, KIBA stores the KIBA score |
| `folds/train_fold_setting1.txt` / `test_fold_setting1.txt` | JSON lists of row-indices | 5 cross-validation folds |
| `processed/{train,test}{_fold,N}.txt` | whitespace-separated rows `compound_idx protein_idx label` | pre-computed splits (produced by `python datahelper/split_train_test_data.py`) |

The BridgeDTA model additionally reads the pre-computed `.npz` features listed
in *⚠️ Required: pre-computed KPGT & ESM-2 features* above.

## (Optional) Regenerating the pre-computed features

The frozen features can be re-extracted from the original KPGT and ESM-2 models
if you cannot (or prefer not to) download the Figshare bundle. This is
not required for the paper's results — the Figshare files are sufficient.

### Pre-trained Model Weights

#### KPGT

- **Repository**: https://github.com/lihan97/KPGT
- **Feature extraction**: We use the `LiGhTPredictor` model from KPGT to extract both graph-level (2304-d) and node-level (768-d per atom) molecular features.

**Node-level feature extraction** (core logic from KPGT):

```python
# In LiGhTPredictor.generate_node_fps():
def generate_node_fps(self, g, fp, md, n_virtual_nodes=2):
    indicators = g.ndata['vavn']
    node_h = self.node_emb(g.ndata['begin_end'], indicators)
    edge_h = self.edge_emb(g.ndata['edge'], indicators)
    triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
    triplet_h = self.model(g, triplet_h)  # LiGhT Transformer layers
    # Filter out virtual nodes, keep only real atom embeddings
    atom_mask = indicators <= 0
    atom_h = triplet_h[atom_mask]  # [N_atoms, 768]
    atoms_per_graph = (g.batch_num_nodes() - n_virtual_nodes).tolist()
    fp_vn = triplet_h[indicators == 1]  # ECFP virtual-node [B, 768]
    md_vn = triplet_h[indicators == 2]  # MD virtual-node [B, 768]
    return atom_h, atoms_per_graph, fp_vn, md_vn
```

Extraction command:

```bash
cd KPGT/scripts
python extract_features.py \
    --config base \
    --model_path ../pretrained/base/base.pth \
    --data_path ../data \
    --dataset davis \
    --feature_level node   # or "graph" for 2304-d molecule-level features
```

#### ESM-2

- **Model**: `esm2_t30_150M_UR50D`
- **Repository**: https://github.com/facebookresearch/esm
- **Installation**: `pip install fair-esm`

We extract per-residue embeddings from the last hidden layer of ESM-2 for each protein sequence, then store them as `.npz` files.

```python
import torch
import esm

model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

# Example: extract features for a protein sequence
data = [("protein_1", "MKTLLILAVL...")]
_, _, batch_tokens = batch_converter(data)
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[30])
    # Per-residue embeddings: [1, seq_len, 640]
    embeddings = results["representations"][30]
```

## Evaluation Metrics

| Metric   | Direction | Description |
|----------|-----------|-------------|
| CI       | ↑         | Concordance Index for ranking accuracy |
| MSE      | ↓         | Mean Squared Error |
| r²_m     | ↑         | Modified coefficient of determination |
| PCC      | ↑         | Pearson Correlation Coefficient |
| Spearman | ↑         | Spearman Rank Correlation |

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

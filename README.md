# BindHack

**InSilico Protein-Protein Interaction (PPI) Hackathon**

A machine learning pipeline for predicting antibody-antigen binding affinity using sequence-based features and protein language model embeddings.

## Project Overview

This project tackles the challenge of predicting binding affinity between antibodies and antigens. Using the AbiBench dataset, we explore various feature engineering approaches:

- **Hand-crafted features**: Amino acid composition, physicochemical properties, and structural interactions
- **Deep learning embeddings**: ESM2 (Evolutionary Scale Modeling) protein language model
- **ML models**: Random Forest, SVM, with hyperparameter optimization via Optuna

## Team

- Andrew Aiginin
- Simon Konnov
- Ilia Bushmakin

## Project Structure

```
bindhack/
├── data/                    # Dataset files (train_data.csv, test sets)
├── notebooks/              # Jupyter notebooks for exploration and modeling
│   ├── 00-intro.ipynb     # Data loading, feature engineering, ESM embeddings
│   ├── 01-feature-engineering.ipynb
│   └── 02-training-and-validation.ipynb
├── src/bindhack/          # Python package for reusable code
└── pyproject.toml         # Project dependencies
```

## Key Dependencies

- **Data**: `polars`, `numpy`
- **Biology**: `biopython`, `biotite` (for PDB parsing and structure analysis)
- **ML**: `scikit-learn`, `xgboost`, `optuna`
- **Deep Learning**: `torch`, `transformers` (ESM2 embeddings)
- **Visualization**: `matplotlib`, `seaborn`

## How to...

### Use this repo

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Insilico-org/bindhack.git
   cd bindhack
   ```

2. **Install dependencies** (using `uv`):
   ```bash
   uv sync
   ```

3. **Activate the environment**:
   ```bash
   source .venv/bin/activate
   ```

4. **Start exploring**:
   - Open [notebooks/00-intro.ipynb](notebooks/00-intro.ipynb) for the complete walkthrough
   - Follow the pipeline: Data → Features → Models → Validation

### Connect to the remote server

If you're working with a remote GPU server:

```bash
ssh -P <PORT> root@<YOUR_SERVER_IP> -i <PATH_TO_PROVIDED_SSH_KEY> 
cd /workdir/bindhack
source .venv/bin/activate
```

## Tips for the Hackathon

- Start with the [00-intro.ipynb](notebooks/00-intro.ipynb) notebook - it covers the entire pipeline
- Different tasks (easy vs hard splits) may need different validation strategies
- Feature engineering is iterative - keep experimenting!
- Document your experiments and track what works
- Consider working in parallel: data prep, feature engineering, and modeling can be split among team members

## Resources

- **ESM2 Model**: [facebook/esm2_t6_8M_UR50D](https://huggingface.co/facebook/esm2_t6_8M_UR50D)
- **Biotite docs**: [biotite-python.org](https://www.biotite-python.org/)
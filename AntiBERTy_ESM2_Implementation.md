# AntiBERTy + ESM2 Ensemble Implementation Guide

## Overview

This implementation combines two specialized protein language models for antibody-antigen binding prediction:

- **AntiBERTy**: Antibody-specific model (trained on 558M antibody sequences)
- **ESM2**: General protein language model

## Architecture

```
Antibody Sequences          Antigen Sequence
       │                           │
   VH  │  VL                      AG
       │                           │
       ↓                           ↓
  AntiBERTy                      ESM2
  (512 dims)                  (320 dims)
       │                           │
       ├─── Embeddings ────────────┤
       ├─── PLL Scores ────────────┘
       │
       ↓
Feature Vector (1,346 dims)
       │
       ↓
[VH_emb(512) | VL_emb(512) | AG_emb(320) | VH_pll(1) | VL_pll(1)]
       │
       ↓
StandardScaler
       │
       ↓
ML Model (Random Forest / PCA+SVR)
       │
       ↓
Binding Score Prediction
```

## Key Features

### 1. Specialized Model Assignment
- **Heavy chains → AntiBERTy** (understands CDRs, framework regions)
- **Light chains → AntiBERTy** (antibody-specific patterns)
- **Antigens → ESM2** (general protein patterns)

### 2. Feature Engineering
Each sample generates 1,346 features:
- Heavy chain embedding: 512 dims
- Light chain embedding: 512 dims
- Antigen embedding: 320 dims
- Heavy chain PLL: 1 dim (sequence naturalness)
- Light chain PLL: 1 dim (sequence naturalness)

### 3. GPU Optimization
- Automatic device detection (CUDA vs CPU)
- Batched processing (64 for embeddings, 256 for PLL)
- Periodic GPU cache clearing to prevent OOM
- Memory monitoring utilities

### 4. Caching System
All embeddings are cached in `./cache/`:
- `vh_ab_emb.npy` - Heavy chain AntiBERTy embeddings
- `vl_ab_emb.npy` - Light chain AntiBERTy embeddings
- `vh_ab_pll.npy` - Heavy chain PLL scores
- `vl_ab_pll.npy` - Light chain PLL scores
- `ag_esm_emb.npy` - Antigen ESM2 embeddings

**Benefit**: First run is slow (computes embeddings), subsequent runs are fast (loads from cache).

## Code Structure

### Installation & Setup
```python
# Auto-installs antiberty if missing
import antiberty
from antiberty import AntiBERTyRunner

# Device detection
device = "cuda" if torch.cuda.is_available() else "cpu"
ab = AntiBERTyRunner(device=device)
```

### Core Functions

#### `antiberty_embed_mean(seqs, batch_size=64)`
- Extracts 512-dim mean-pooled embeddings for antibody sequences
- Batched processing with automatic GPU cache clearing
- Returns: `np.ndarray` of shape `(N, 512)`

#### `antiberty_pll(seqs, batch_size=256)`
- Computes pseudo-log-likelihood scores
- Measures how "natural" the antibody sequence is
- Higher PLL = more typical antibody sequence
- Returns: `np.ndarray` of shape `(N,)`

#### `esm2_embed_mean_antigen(seqs, batch_size=32)`
- Extracts 320-dim mean-pooled embeddings for antigen sequences
- Uses existing ESM2 model
- Returns: `np.ndarray` of shape `(N, 320)`

#### `predict_binding_score(vh_seq, vl_seq, ag_seq)`
- Inference helper for new antibody-antigen pairs
- Handles full pipeline: embed → concat → scale → predict
- Returns: `float` (predicted binding score)

### Training Pipeline

1. **Feature Extraction** (with caching)
   ```python
   vh_emb = antiberty_embed_mean(vh_seqs)
   vl_emb = antiberty_embed_mean(vl_seqs)
   ag_emb = esm2_embed_mean_antigen(ag_seqs)
   vh_pll = antiberty_pll(vh_seqs)
   vl_pll = antiberty_pll(vl_seqs)
   ```

2. **Feature Concatenation**
   ```python
   X = np.concatenate([vh_emb, vl_emb, ag_emb, 
                       vh_pll[:, None], vl_pll[:, None]], axis=1)
   ```

3. **Standardization**
   ```python
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X_train)
   ```

4. **Model Training**
   ```python
   # Option 1: Random Forest (direct)
   rf = RandomForestRegressor(n_estimators=100, max_depth=20)
   rf.fit(X_train_scaled, y_train)
   
   # Option 2: PCA + SVR (memory efficient)
   svr = make_pipeline(PCA(n_components=100), LinearSVR())
   svr.fit(X_train_scaled, y_train)
   ```

5. **Evaluation**
   - R² score
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - Spearman correlation (ρ)

## Performance Optimizations

### Memory Management
```python
def clear_gpu_cache():
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
```

Called periodically during batch processing to prevent OOM errors.

### Batch Size Guidelines
- **RTX 4090 (24GB VRAM)**:
  - Embeddings: batch_size=64
  - PLL: batch_size=256
  
- **Smaller GPUs (8-12GB)**:
  - Embeddings: batch_size=16-32
  - PLL: batch_size=64-128

- **CPU only**:
  - Embeddings: batch_size=8-16
  - PLL: batch_size=32-64

## Validation Strategies

### 1. Single Train/Val Split
```python
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
```

### 2. K-Fold Cross-Validation
```python
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
```

### 3. Hyperparameter Optimization (Optuna)
```python
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
best_params = study.best_params
```

## Comparison with Baseline

| Approach | Features | VH/VL Model | AG Model | Dims |
|----------|----------|-------------|----------|------|
| **Baseline** | ESM-only | ESM2 | ESM2 | 960 |
| **New (Ours)** | Ensemble | AntiBERTy | ESM2 | 1,346 |

**Key improvements:**
- ✅ Antibody-specific model for VH/VL (better understanding of CDRs)
- ✅ PLL scores add sequence naturalness signal
- ✅ 40% more features (1,346 vs 960)
- ✅ Specialized models for specialized tasks

## Expected Performance

With proper hyperparameter tuning:
- **R²**: 0.55-0.70 (dataset dependent)
- **Spearman ρ**: 0.60-0.75
- **MAE**: 0.8-1.2 (dataset dependent)

## Usage Example

```python
# Load trained model and scaler
from joblib import dump, load

# Save after training
dump(best_model, 'model.joblib')
dump(scaler, 'scaler.joblib')

# Load for inference
model = load('model.joblib')
scaler = load('scaler.joblib')

# Predict new sample
vh = "EVQLVESGGGLVQPGGSLRLSCAASGFTFS..."
vl = "DIQMTQSPSSLSASVGDRVTITC..."
ag = "MKTIIALSYIFCLVFA..."

score = predict_binding_score(vh, vl, ag)
print(f"Predicted binding score: {score:.4f}")
```

## Common Issues & Solutions

### 1. CUDA Out of Memory
**Solution**: Reduce batch sizes or clear cache more frequently
```python
# In embedding functions, add:
if i % (batch_size * 3) == 0:  # More frequent clearing
    clear_gpu_cache()
```

### 2. Slow First Run
**Expected**: Computing embeddings for full dataset takes time
- First run: 10-30 minutes (depends on dataset size)
- Subsequent runs: <1 minute (loads from cache)

### 3. Cache Invalidation
If you modify sequences, delete cache and recompute:
```bash
rm -rf cache/
```

### 4. Shape Mismatches
Always validate after concatenation:
```python
assert X.shape[1] == 1346, f"Expected 1346 features, got {X.shape[1]}"
```

## Files Generated

```
bindhack/
├── notebooks/
│   └── antiberty+esm.ipynb    # Main notebook
├── cache/                      # Embedding cache
│   ├── vh_ab_emb.npy          # (N, 512)
│   ├── vl_ab_emb.npy          # (N, 512)
│   ├── vh_ab_pll.npy          # (N,)
│   ├── vl_ab_pll.npy          # (N,)
│   └── ag_esm_emb.npy         # (N, 320)
└── models/                     # Saved models (optional)
    ├── model.joblib
    └── scaler.joblib
```

## Next Steps

1. **Scale to full dataset**: Remove size limits in code
2. **Try other models**: XGBoost, LightGBM, neural networks
3. **Feature engineering**: Add CDR region features, contact predictions
4. **Ensemble**: Combine multiple model predictions
5. **Production**: Wrap `predict_binding_score()` in REST API

## References

- **AntiBERTy**: Ruffolo et al. (2021) - "Antibody structure prediction using interpretable deep learning"
- **ESM2**: Lin et al. (2022) - "Language models of protein sequences at the scale of evolution"
- **AbiBench**: Our binding affinity dataset

## License

This implementation is provided as-is for the BindHack competition.

---

**Author**: Implemented as part of BindHack antibody-antigen binding prediction challenge  
**Date**: November 2025  
**GPU Tested**: RTX 4090 (24GB VRAM)

![Poster for Blind Source Separation in Neuroscience](https://github.com/user-attachments/assets/d54aefb1-7f2e-42bc-9926-fcd34fd3f606)

# Polytopic Matrix Factorization for EEG

Blind source separation of EEG signals using Polytopic Matrix Factorization (PMF). Given a mixture matrix `X`, PMF recovers a linear mixing map `W` and source signals `Z` such that `X ≈ WZ`, with `Z` constrained to lie within an L1-norm polytope.

## Contents

| File | Description |
| --- | --- |
| `pmf.py` | PMF algorithm and polytope projection operators (L1, L∞, non-negative variants). |
| `utils.py` | Dipole fitting, residual-variance computation, and topomap grid plotting. |
| `source_separation_pmf.ipynb` | End-to-end demo: load EEG, run PMF, threshold sources by residual variance, plot topomaps. |
| `tf_analysis.ipynb` | Time–frequency analysis of separated sources via Morlet wavelets. |
| `source_localization.ipynb` | Per-subject PMF runs and dipole localization on the `fsaverage` head model. |
| `data.set` / `data.fdt` | Sample EEG recording in EEGLAB format. |

## Requirements

- Python 3
- `numpy`, `mne`, `matplotlib`, `tqdm`, `nibabel`, `nilearn`, `jupyter`

The first run of `calculate_rvs` downloads the MNE `fsaverage` dataset to `~/mne_data/`.

## Usage

```python
import mne
from pmf import PMF
from utils import calculate_rvs, pmf_plot

raw = mne.io.read_raw_eeglab("data.set")
events, event_id = mne.events_from_annotations(raw)
epochs = mne.Epochs(raw, events, event_id=2, tmin=-0.01, tmax=2.5, preload=True)
X = epochs.average().data
M, N = X.shape
r = 32

W, Z = PMF(raw, X, r=r, NumberofIterations=3000)
rvs, gofs = calculate_rvs(raw, W, M, N, r)
pmf_plot(r, W, raw, rvs, treshold=0.15)
```

Open any of the notebooks for a guided walkthrough.

## Key parameters

- `r` — number of sources to extract.
- `NumberofIterations` — projected-gradient iterations.
- `muv` — step size; decays as `0.99**k`.
- `threshold` — residual-variance cutoff for keeping dipolar sources (lower = stricter).

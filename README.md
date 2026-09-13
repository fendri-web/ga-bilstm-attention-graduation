# Supporting Code: Combining Genetic Algorithm and Attention in a BiLSTM Model for Student Graduation Prediction

Supporting code for the paper *"Combining Genetic Algorithm and Attention in the Best BiLSTM Model: An Ablation Study of Component Contributions in Student Graduation Prediction."*

## Code Structure

| File | Function | Corresponding Paper Section |
|---|---|---|
| `01_main_pipeline.py` | Main pipeline: data loading, preprocessing, feature engineering, GA-based feature selection, training and evaluation of 8 model variants, single-run 2×2 ablation analysis, confusion matrix, early prediction | Sections 3.1–3.5, 4.1–4.7 |
| `02_multirun_ablation.py` | 2×2 ablation evaluation using a **3-repeat × 5-fold** scheme (n=15 per variant) for stronger significance testing — produces Tables 10 and 11 | Section 4.8 |
| `03_skip_ga_utility.py` | Utility to reload GA feature-selection results from a saved checkpoint, without re-running the time-consuming GA process | Supports fast replication |

## How to Run

### Option A — Run the full pipeline from scratch
```bash
pip install tensorflow scikit-learn pandas numpy scipy matplotlib xlrd

python 01_main_pipeline.py
```
Requires 3 data files (not included in this repository due to institutional privacy policy):
- `Kelulusan_Data_Um_Ad_3.xls`
- `Kelulusan_Train.xls`
- `Kelulusan_Test.xls`

**Estimated runtime**: ~30–60 minutes with GPU, 5–7 hours without GPU (the GA process is the most time-consuming part).

### Option B — Skip the GA process (use checkpoint)
If you have already run `01_main_pipeline.py` once and saved the GA results (`ga_history.pkl`), use `03_skip_ga_utility.py` to proceed directly to model training without repeating the GA search.

### Running the multi-run analysis (Section 4.8)
```bash
python 02_multirun_ablation.py
```
This script retrains the four key variants (BiLSTM, GA-BiLSTM, BiLSTM-Attention, GA-BiLSTM-Attention) across 3 independent runs (different seeds) × 5-fold cross-validation, producing the raw data underlying Tables 10, 11, and Figure 9 (raincloud plot) in the paper.

## Reproducibility

- Fixed random seeds (42, and 123/456 for the 2nd/3rd repeat in the multi-run analysis) are used throughout all scripts to ensure reproducible results.
- Raw evaluation results (per-fold accuracy) for all models are available in **Appendix B** (Tables B1 and B2) of the paper, enabling independent verification of all reported statistical tests.

## Dependencies

```
tensorflow>=2.15
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
scipy>=1.11
matplotlib>=3.7
xlrd>=2.0
```

## Citation

If this code is useful for your research, please cite the associated paper (full citation details will be added upon publication).

## License

[Specify a license according to your policy, e.g., MIT License]

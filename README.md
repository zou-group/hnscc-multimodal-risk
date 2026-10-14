# Multimodal HNSCC risk models (imaging, gene expression, clinical)

Code for the imaging, molecular, and imaging+molecular PFS risk models and attention heatmaps in

Belete*, Thakkar* et al. *Integrated histopathology–transcriptomic biomarker enhances survival prediction in HNSCC patients treated with immunotherapy.* Translational Oncology 70 (2026) 102822. https://doi.org/10.1016/j.tranon.2026.102822

This code covers Figures 3 and 4 of the paper. All scripts are run from this folder and write results to `./outputs/`.

## Setup

```
conda env create -f environment.yml
conda activate hnscc_env
```

UNI, H-optimus-0 and Virchow2 are gated HuggingFace models, so set `HF_TOKEN` before extracting features with them. PLIP does not need a token.

## Pipeline

Run the steps in this order. Steps 2 and 6 need a GPU.

1. **Patch the WSIs with CLAM** (https://github.com/mahmoodlab/CLAM). Use 224 x 224 patch size.
   ```
   python create_patches_fp.py --source <Images>/HE_images --save_dir <Images>/CLAM_HE_images --patch_size 224 --seg --patch --stitch
   ```
   Note that CLAM has now been depricated in favor of [TRIDENT](https://github.com/mahmoodlab/TRIDENT). I would recommend using TRIDENT if possible as it works very well and has newer capabilities.

2. **Extract patch features** with a frozen foundation model. PLIP was used for the paper (UNI2, H-optimus-0 and Virchow2 were compared in Supp Fig S3).
   ```
   python extract_features.py --model plip
   ```
   Writes `<Images>/HE_features/<model>/pt_files/<slide_id>.pt` (num patches x feature dim).
3. **Find significant genes** per treatment (univariate Cox on train patients, FDR < 0.05).
   ```
   python significant_genes.py
   ```
4. **Build the molecular risk signature** (lasso Cox, alpha 0.1, on the significant genes). Prints train/test c-index and makes the Fig 3B KM plot. Also writes `outputs/molecular/lasso_results_alpha=0.1.csv`, which step 6 reads.
   ```
   python risk_score.py
   ```
5. **Clinical baseline** (CoxPH on age, gender, HPV, biopsy site). Prints the test c-index.
   ```
   python clinical_model.py
   ```
6. **Train the attention MIL model** with 5-fold CV on the train split, then score the held-out test split. Treatment is one-hot encoded and the molecular risk score is added by late fusion.
   ```
   python imaging_model.py # imaging only
   python imaging_model.py --use_molecular # imaging + molecular
   python imaging_model.py --use_clinical # imaging + clinical
   python imaging_model.py --use_clinical --use_molecular # imaging + clinical + molecular
   ```
   Writes to `outputs/<imaging|+clinical+molecular|...>/`. That includes `test_results.csv` (test c-index and log-rank p per treatment), the combined KM plots, high/low risk patient CSVs, and per-patient risk scores. It also writes the fold checkpoints to `outputs/checkpoints/` and the fold-averaged attention weights to `outputs/avg_attention/`.
7. **Attention heatmaps** (Fig 4). Red is high attention.
   ```
   python visualize_attention_maps.py
   ```
   Reads the imaging+molecular PLIP outputs from step 6 and writes `outputs/attention_heatmaps/`.

## Paths to set for your data

All input paths are marked with `# TODO` comments. Set `data_path` (left as `''`) to your data folder in `significant_genes.py`, `risk_score.py`, `clinical_model.py`, `imaging_model.py`, `extract_features.py` and `visualize_attention_maps.py` (two places in that file). Then update the input file and folder names next to the other `# TODO` comments to match your data. By default the image folders are expected under `<data_path>/Images/`
- `HE_images/` and `processed_HE_images/` hold the slides (`<slide_id>_HE.svs` or `.tiff`)
- `CLAM_HE_images/patches/` holds the CLAM patch coordinates, and `CLAM_HE_images/process_list_autogen_no_ext.csv` lists the slides
- `HE_features/<model>/` holds the extracted features (`feat_dir` in `imaging_model.py`)

Slide IDs start with the patient ID followed by an underscore (`<patient_id>_..._HE`). The code uses that to match slides to patients.

## Expected input tables

`preprocessing.py` (`preprocess_data_pfs`) is the place to adapt to a new cohort's format.

- `pfs_baseline_hpv_ecog_ihc_tnm_gender.tsv` has one row per patient sample, with columns `patient_id`, `isolate_analyte` (rows with `rna` are kept), `rna_analysis_id`, `regimen_` (`io`, `combo` or `chemo`), `gender`, `age`, `biopsy_site`, `LOT1_start_date`, `PFS_Ti_days`, `PFS_status`, `pdl1_cps`, `HPV_status`, `HPV_status_origin`
- `g_rna_gene_expression_hnscc.tsv` is long format, with columns `patient_id`, `analysis_id`, `gene_code` (Ensembl ID), `log2_gene_tpm_corrected`

HPV status is imputed from CDKN2A expression for patients without a lab result (`utils.hpv_imputation`). The result is cached in `outputs/hpv_imputed_pfs.csv`.

## Hyperparameters

Best settings from our sweep (Supp Table S1), set in `imaging_model.py`. The Optuna search is in `optimize_hyperparameters` (commented out in `train`).

| Model | Hidden dim | Dropout | Learning rate |
| --- | --- | --- | --- |
| Imaging | 224 | 0.2 | 8e-4 |
| Imaging + molecular | 160 | 0.3 | 5e-4 |

Other settings are Adam with weight decay 1e-3, gradient clipping at 1.0, batch size 8, 15 epochs, a 70/30 train/test split stratified by event, and a random seed of 42.

## License

MIT, see `LICENSE`. Pretrained foundation models (PLIP, UNI2, H-optimus-0, Virchow2) and CLAM are covered by their own licenses.

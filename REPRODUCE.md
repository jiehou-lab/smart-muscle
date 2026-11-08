# Reproducing the manuscript: a step-by-step guide

The histology images and the trained weights are not distributed. Two commands cover what
can be reproduced without them:

| command | checks | time |
|---|---|---|
| `python reproduce.py check` | every reported number and figure follows from the saved per-patch predictions | ~2 min |
| `python reproduce.py --synthetic all` | the complete pipeline runs end to end, on synthetic data | ~45 min with a GPU |

Run them from the repository folder after installing the environment (README §2). Times
are from the reference machine: Windows 11, Intel laptop CPU, 14 GB RAM, NVIDIA RTX 3070
Laptop GPU (8 GB).

Nothing writes into `models/` or `reference/`. Every output goes to `runs/`, which can be
deleted at any time.

---

## 1. The manuscript's numbers and figures

```bash
python reproduce.py check
```

1. **Environment.** Prints the package versions and the GPU, and checks the inputs exist.
2. **Dataset.** Skipped, because the images are not distributed. The dataset's counts are
   frozen in `reference/dataset_report.json` (notebook 01 prints them).
3. **Numbers.** Recomputes every metric from the per-patch `predictions.csv` of each model
   and compares it, at the three decimals the manuscript prints, with
   `reference/manuscript_numbers.json`. That file holds the values the text, Table 2
   and the figures were written from.
4. **Figures.** Redraws Figures 1–4 into `runs/figures/` and compares them pixel by pixel
   with `figures/manuscript/`.

Expected ending:

```
249 of 249 manuscript numbers reproduced to 3 decimals
figure1_patch_f1         pixel-identical to the manuscript figure
figure2_roc              pixel-identical to the manuscript figure
figure3_wsi_f1           pixel-identical to the manuscript figure
figure4_aclr_transfer    pixel-identical to the manuscript figure
CHECK PASSED (dataset step skipped: images not distributed)
```

Figures are pixel-identical only with the pinned matplotlib (3.10.7) and its default
fonts. With another version the figures still draw, but small rendering differences are
reported as "differs".

**What the 249 numbers are.** For each of the 21 rat runs (7 models × 3 tasks) and the 6
human runs (2 models × 3 comparisons), the code recomputes patch- and WSI-level F1,
macro-F1, accuracy and ROC-AUC, plus the patch and slide counts. The full list, with the
manuscript values, is in `reference/manuscript_numbers.json`. Notebook 06 shows them as
tables, including Table 2.

## 2. The whole pipeline on synthetic data

```bash
python reproduce.py --synthetic all
```

| step | what | time (GPU) |
|---|---|---|
| 1. data | `scripts/make_synthetic_data.py` draws synthetic laminin-stained sections and tiles them into patches; `scripts/build_dataset.py` balances them into `Dataset_balanced` and checks for leakage | ≈ 2 min |
| 2. ResNet-50 / MLP | `tools/train_eval_all.py`: ResNet-50, ResNet-50 + morph, morphometric MLP on the 3 tasks (seed 95, early stopping) | ≈ 11 min |
| 3. LR | `scripts/lr_tuning.py`: feature cache, animal-grouped 3-fold CV over C (and the Perfect weight), refit of the choice, test scoring (65 CV fits) | ≈ 27 min (CPU) |
| 4. XGBoost | `scripts/run_xgb.py`: fixed settings | ≈ 5 min (CPU) |
| 5. human | `scripts/human_eval.py`: self-test, then zero-shot scoring of the synthetic human comparisons | ≈ 1 min |
| 6. report | `scripts/manuscript_numbers.py`, `scripts/manuscript_figures.py`: every metric, Figures 1–4; `scripts/verify_models.py`: every saved model is reloaded and must reproduce its own predictions | ≈ 3 min |

Outputs, all in `runs/synthetic/`:

| file | contents |
|---|---|
| `rerun/` | the 21 trained models, laid out as `models/` (weights, configs, logs, `evaluation/predictions.csv`), plus `human/` |
| `synthetic_summary.csv`, `synthetic_numbers.json` | every metric |
| `figures/` | Figures 1–4 drawn from the synthetic run |
| `lr_tuning_cv.csv` | the LR cross-validation table |
| `verify_models/verification.csv` | the reload check |

Expected ending:

```
21 of 21 rat runs and 6 of 6 human runs scored; 4 of 4 figures drawn -> runs/synthetic/figures
...
27 of 27 saved models reproduce their saved predictions -> runs/synthetic/verify_models/verification.csv
==============================================================================
SYNTHETIC PIPELINE COMPLETE
```


**Options.**
* `--epochs 3` caps ResNet-50 / MLP training, for a CPU-only machine.
* Each step can be run on its own: `python reproduce.py --synthetic data`, then
  `train [--only cnn,lr,xgb,human]`, `models --human`, `report`. Finished runs are skipped,
  so an interrupted run resumes.
* `python scripts/make_synthetic_data.py --seed 7 --scale 2 --force` makes a different or
  larger synthetic dataset.

**What to expect from the scores.** The synthetic classes are designed to be learnable but
to overlap (Poor and Very poor share part of their severity range). Most models therefore
separate Control from VML well and the other tasks less well. The scores describe the
synthetic images only and say nothing about the manuscript.

## 3. With the original data

If the original images and workbooks are placed as in `data/README.md` (and the weights in
`models/`), the same commands without `--synthetic` re-run the manuscript itself:

| command | what | time (reference machine) |
|---|---|---|
| `python reproduce.py models --human` | reload the 21 saved models and re-score the test data; must reproduce the shipped predictions | ~15 min |
| `python reproduce.py all` | retrain everything into `runs/rerun/` and compare every number with the manuscript | ~3.5 h |
| `python reproduce.py train --lr-cv` | also redo the LR cross-validation instead of reusing `reference/lr_tuning_cv.csv` | +12 h |

What the authors saw on the reference machine:
* Reloading: 27 of 27 saved models reproduced their predictions exactly.
* Retraining: LR, XGBoost and the MLP reproduce exactly. ResNet-50 does not: patches load
  on 6 threads (`data.loader_threads: 6`, `vmlevaluator/data/loaders.py`), and the random
  flips draw from PyTorch's single global random generator, so which patch receives which
  flip depends on thread timing.
  * A retrain is therefore a new run of the same procedure: 189 of 249 numbers matched.
  * For run-to-run determinism in new work, use `--set data.loader_threads=0` (slower; not
    used for the manuscript).

---

## The same steps as notebooks

| step | notebook |
|---|---|
| manuscript numbers and Table 2 | `06_manuscript_numbers_and_tables` |
| Figures 1–4 | `07_figures_1_to_4` |
| synthetic pipeline, stage by stage | `00` (prepares the data), `01`, `02`, `03`, `04`, `05`, then `08` (reload check) |
| synthetic pipeline in one go | `09_reproduce_end_to_end` |

## If something differs

* **`MISSING` in `check`.** A path is wrong. All paths are in `scripts/paths.py` and can
  be overridden with environment variables (`VML_DATASETS`, `VML_DATA`,
  `VML_HUMAN_ROOT`, `VML_RESULTS`, `VML_WORK`).
* **Figures reported as "differs".** Use matplotlib 3.10.7; the numbers are unaffected.
* **ResNet-50 training cannot download weights.** timm fetches the ImageNet weights
  (`resnet50_a1_0-14fe96d1.pth`) on first use. Offline, place the file in
  `pretrained/torch/hub/checkpoints/`; `scripts/paths.py` then points `TORCH_HOME` there.
* **Out of GPU memory.** Lower the batch size:
  `python tools/train_eval_all.py ... --set data.batch_size=32` (default 64).
* **Out of RAM in the LR / XGBoost steps.** Close other programs, or run one step at a
  time: `python reproduce.py --synthetic train --only lr`.
* **Start the synthetic run over.** Delete `data_synthetic/` and `runs/synthetic/`.

## Where each manuscript item comes from

| manuscript | produced by | file |
|---|---|---|
| Methods / Table 1 counts | `scripts/build_dataset.py --check`, notebook 01 | `reference/dataset_report.json` |
| Figure 1 (patch F1), Figure 3 (WSI F1) | `scripts/manuscript_figures.py` | `runs/figures/` |
| Figure 2 (ROC) | `scripts/manuscript_figures.py`, from every model's `predictions.csv` | `runs/figures/` |
| Figure 4 (human transfer) | `scripts/manuscript_figures.py` | `runs/figures/` |
| Table 2 (ablation) | notebook 06 | `runs/table2_ablation.csv` |
| Every number in the text | `scripts/manuscript_numbers.py` | `runs/manuscript_numbers.json` |
| LR setting (C = 10⁻⁶; Perfect weight 4) | `scripts/lr_tuning.py` | `reference/lr_tuning_cv.csv`, `reference/lr_tuned_params.json` |

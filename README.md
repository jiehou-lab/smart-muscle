# smart-muscle

**Integrating Histology and Morphometrics for Deep Learning-Based Assessment of
Skeletal Muscle Injury**

Isaac Nkrumah, Amelia Ridolfo, Carlos Alejandro Formoso Godoy, Christine M. Latham,
Christopher S. Fry, Koyal Garg\*, Jie Hou\*

Code, notebooks and saved predictions for the manuscript of the same name.
**Under review** — this repository is the reproducibility package that
accompanies it.

## What this is

Assessing skeletal muscle injury from histology is manual, subjective and hard
to standardise — especially when the question is not *is this injured?* but
*how far has it recovered?* This project asks whether combining what a
convolutional network sees in a tissue patch with two slide-level morphometric
measurements — mean myofiber cross-sectional area and myofiber count — does
better than either on its own.

The pipeline classifies rat volumetric muscle loss (VML) histology on three
tasks of increasing difficulty, comparing ResNet-50 against logistic regression
and XGBoost, each with and without the morphometric features. The rat-trained
networks are then applied **without retraining** to human vastus lateralis
biopsies after ACL reconstruction, to see whether anything transfers across
species.

The short answer: image features alone separate control from injured tissue
easily, and morphometrics add almost nothing there. The harder the severity
distinction, the more the fusion matters — on the three-class task the
multimodal ResNet-50 reaches 0.938 patch-level macro-F1 against 0.601 for the
image-only model.

### Headline results

Patch level, held-out test split. Binary tasks report F1 for the more injured
class; the three-class task reports macro-F1.

| task | LR | XGBoost | XGBoost + morph | ResNet-50 | ResNet-50 + morph |
|---|---|---|---|---|---|
| Control vs VML | 0.760 | 0.972 | **1.000** | 0.965 | **1.000** |
| Poor vs Very poor | 0.591 | 0.721 | **0.989** | 0.844 | **0.998** |
| Three-class (macro-F1) | 0.474 | 0.470 | 0.826 | 0.601 | **0.938** |

Zero-shot on human vastus lateralis, ResNet-50, slide-level F1 for the
uninjured group:

| comparison | image only | + morphometrics |
|---|---|---|
| uninjured vs 1-month post-ACLR | 0.353 | **0.849** |
| uninjured vs 4-month post-ACLR | 0.353 | **0.778** |
| 4-month vs 1-month post-ACLR | 0.606 | 0.606 |

None of these is transcribed by hand. `python reproduce.py check` recomputes
every one of them from the per-patch predictions in `models/`.

---

## Quick start

```bash
git clone https://github.com/jiehou-lab/smart-muscle.git
cd smart-muscle

python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e .

python reproduce.py check
```

Expected ending:

```
249 of 249 manuscript numbers reproduced to 3 decimals
figure1_patch_f1         pixel-identical to the manuscript figure
figure2_roc              pixel-identical to the manuscript figure
figure3_wsi_f1           pixel-identical to the manuscript figure
figure4_aclr_transfer    pixel-identical to the manuscript figure
CHECK PASSED (dataset step skipped: images not distributed)
```

About two minutes, **no GPU and no histology images needed** — it recomputes
from the saved per-patch predictions and redraws the figures.

### Running the whole pipeline, with no data of ours

The histology is not distributable, so the repository ships a generator for a
synthetic stand-in with the same structure, slide counts and splits. This is
the supported way to exercise the full pipeline — preprocessing, training,
evaluation and figures — end to end:

```bash
python reproduce.py --synthetic all               # generate, build, train everything
python reproduce.py --synthetic all --epochs 3    # shorter ResNet-50 training on CPU
```

If you have the original images and weights in place (`data/`, `models/`), the
same commands without `--synthetic` re-run the manuscript itself:

```bash
python reproduce.py models [--human]    # re-score the saved weights on the test data
python reproduce.py train [--only cnn,lr,xgb,human] [--lr-cv]
python reproduce.py compare             # compare runs/rerun/ with the manuscript
python reproduce.py all                 # train, then compare
```

Nothing is ever written into `data/`, `models/` or `reference/`. Every output
goes to `runs/`, which can be deleted at any time. `REPRODUCE.md` is the
step-by-step guide.

---

## Requirements

Python 3.9 or newer. `pip install -e .` installs everything the **check** path
needs:

| | |
|---|---|
| numerical | `numpy`, `pandas`, `scipy`, `scikit-learn`, `scikit-image` |
| models | `torch`, `torchvision`, `timm`, `xgboost` |
| config and I/O | `omegaconf`, `joblib`, `Pillow`, `openpyxl`, `tqdm`, `tabulate` |
| figures | `matplotlib` |

Version floors are in `requirements.txt`; the exact conda environment of the
reference machine is `environment.yml` (`conda env create -f environment.yml`).
That machine ran **torch 1.13, scikit-learn 1.3.2 and xgboost 3.1** on Windows 11
with an RTX 3070 Laptop GPU (8 GB).

Optional extras, each `pip install -e ".[name]"`:

| extra | for |
|---|---|
| `notebooks` | JupyterLab, ipywidgets, seaborn — needed to open `notebooks/` |
| `preprocess` | OpenCV, for the human red-channel preprocessing chain |
| `wsi` | pyvips, for pyramidal slide formats (`.svs`, `.ndpi`, `.mrxs`) |

The install pulls PyTorch, so allow about 2 GB of disk even though the check
needs no GPU. ImageNet weights for ResNet-50 ship in `pretrained/`, so
retraining needs no internet.

`reproduce.py check` is the verification this repository offers: it recomputes
all 249 reported numbers from the saved predictions and redraws Figures 1–4,
comparing each against the submitted version pixel by pixel.

---

## What is here

```
reproduce.py           one entry point for everything below
REPRODUCE.md           step-by-step reproduction guide
RELEASING.md           maintainer notes for publishing this repository

vmlevaluator/          the library: data loading, training, evaluation, figures
configs/               task and model configurations (OmegaConf layers)
scripts/               the manuscript scripts, called by reproduce.py
notebooks/             ten notebooks, plus executed copies with outputs
reference/             frozen manuscript numbers, the LR CV table, the dataset report
models/                per-patch predictions and trained weights, by model family
figures/manuscript/    the figures as submitted, for pixel comparison
data/                  histology — not distributed
data_synthetic/        generated stand-in data, so the pipeline runs without it
pretrained/            ImageNet weights, so retraining needs no internet
runs/                  everything a re-run writes; safe to delete
tools/                 train_eval_all.py
tests/                 the library's test suite
```

### `models/`

Three folders, one per model family, exactly as the manuscript run produced
them.

```
models/
├── neural_networks/        ResNet-50 (cnn, cnn_morph) and the morphometric-only MLP
│   ├── cnn_morph__multiclass/
│   │   ├── config.yaml            the fully resolved configuration of this run
│   │   ├── norm_stats.json        morphometric mean/std, fitted on training slides only
│   │   ├── evaluation/predictions.csv
│   │   └── weights/f1/best_model.pth
│   └── human/                zero-shot scores on the human VL cohort
├── logistic_regression/    CV-tuned LR (lr, lr_morph)
└── xgboost/                XGBoost (xgb, xgb_morph)
```

**Every number and figure is recomputed from `predictions.csv`.** Nothing reads
a stored metric, so a figure cannot disagree with the predictions behind it.

### Loading a trained model

```python
import sys; sys.path.insert(0, "scripts")
from vmlevaluator.config import load_experiment
from vmlevaluator.evaluate.runner import evaluate_experiment

cfg = load_experiment("multiclass", "cnn_morph",
                      data_root="data/Dataset_balanced",
                      results_root="models/neural_networks")
evaluate_experiment(cfg)       # re-scores the test split from the saved weights
```

Or directly:

```python
import torch, joblib
state = torch.load("models/neural_networks/cnn_morph__multiclass/weights/f1/best_model.pth",
                   map_location="cpu")
model = joblib.load("models/xgboost/xgb_morph__multiclass/weights/acc/best_model.joblib")
```

Two files beside the checkpoint are needed to use it correctly, and neither is
optional. `config.yaml` records the architecture, input size and preprocessing
it was fitted under. `norm_stats.json` holds the morphometric mean and standard
deviation, **fitted on the training slides only** — applying a fusion model to
new slides means reusing those exact numbers, not refitting.

**The weights are not in this repository.** They are about 600 MB, and the
histology they were trained on is not distributable either, so neither is
committed here (see `.gitignore`). Nothing above is blocked by that:
`reproduce.py check` recomputes every number and figure from the saved
predictions, and `reproduce.py --synthetic all` runs the entire pipeline —
training included — on generated data. The weights are available from the
authors on request; drop them into `models/` at the paths shown above and the
commands in this section work unchanged.

---

## The data

| task | classes | train / test WSIs | train / test patches |
|---|---|---|---|
| `control_vml` | Control, VML | 20 / 8 | 3,428 / 1,466 |
| `poor_very_poor` | Poor, Very poor | 11 / 6 | 3,344 / 1,800 |
| `multiclass` | Perfect, Poor, Very poor | 15 / 11 | 2,985 / 1,329 |

The 41-slide rat dataset, made class-balanced within each of train, validation
and test. Class index increases with injury severity, so the positive class is
always the more injured one.

Human slides appear in the prediction tables under pseudonyms — `S01_V3L` is
subject 1, visit 3, left — rather than the study codes they were collected
under. The mapping is one-to-one, so the paired structure across visits is
intact and every slide-level number is computed exactly as it was.

Paths all default to locations inside the repository and can each be overridden
by an environment variable — see the table at the top of `scripts/paths.py`.

---

## Conventions

The choices behind every number, stated once so the tables are unambiguous.

- **Binary rat tasks** — F1 for the more injured class (VML; Very poor).
- **Three-class** — macro-F1.
- **Human comparisons** — F1 for the uninjured group (the 4-month group in
  4-month vs 1-month).
- **Slide level** — patch probabilities averaged per slide, then argmax.
- **ROC-AUC** — binary: the more injured class; three-class: macro one-vs-rest.
- **Human transfer** — the rat Control-vs-VML model is applied to uninjured vs
  1-month and uninjured vs 4-month; the rat Poor-vs-Very-poor model to 4-month
  vs 1-month. Human morphometrics are standardised with the rat training
  statistics.

ResNet-50 runs are seeded and use deterministic kernels; on a different GPU or
driver the last digits can still move. The classical models are deterministic on
CPU.

---

## License and citation

MIT — see `LICENSE`. Citation metadata is in `CITATION.cff`.

The manuscript is under review; please cite the preprint, or contact the
corresponding authors (koyal.garg@slu.edu, houjie@msu.edu) for the current
reference.

# smart-muscle

**Integrating Histology and Morphometrics for Deep Learning-Based Assessment of
Skeletal Muscle Injury**

Isaac Nkrumah, Amelia Ridolfo, Carlos Alejandro Formoso Godoy, Christine M. Latham,
Christopher S. Fry, Koyal Garg\*, Jie Hou\*  ·

## What it does

Grading muscle injury from histology is manual and subjective. This package
combines what a ResNet-50 sees in a tissue patch with two slide-level
morphometric measurements — mean myofiber cross-sectional area and myofiber
count — and compares that fusion against logistic regression and XGBoost on
three rat volumetric muscle loss tasks of increasing difficulty. The rat-trained
networks are then applied without retraining to human vastus lateralis biopsies
after ACL reconstruction.

## Run it

```bash
pip install -e .

python reproduce.py check               # ~2 min, no GPU, no image data
python reproduce.py --synthetic all     # the whole pipeline on synthetic data, ~45 min
```

`check` recomputes every number reported in the manuscript from the saved
per-patch predictions and redraws Figures 1–4, comparing each against the
submitted version pixel by pixel. It needs neither the histology nor the trained
weights.

`--synthetic all` generates stand-in data with the same structure and splits,
then runs preprocessing, training, evaluation and figures end to end.

With the images and weights in place, the same commands without `--synthetic`
re-run the manuscript itself — `python reproduce.py --help` lists them. The ten
notebooks in `notebooks/` walk through the same pipeline one stage at a time,
from the dataset through to the figures; every `RUN` flag in them is `False` as
shipped, so they read the manuscript run and check it rather than retraining.

## models/

```
models/
├── neural_networks/        ResNet-50 (cnn, cnn_morph) and the morphometric-only MLP
│   ├── cnn_morph__multiclass/
│   │   ├── config.yaml             architecture and preprocessing for this run
│   │   ├── norm_stats.json         morphometric mean/std, training slides only
│   │   └── evaluation/
│   │       ├── predictions.csv     per-patch labels and class probabilities
│   │       └── metrics.json
│   └── human/                 zero-shot scores on the human cohort
├── logistic_regression/    CV-tuned LR (lr, lr_morph)
└── xgboost/                XGBoost (xgb, xgb_morph)
```

Every number and figure is recomputed from `predictions.csv`; nothing reads a
stored metric, so a figure cannot disagree with the predictions behind it.

**The trained weights are included**, under `models/<family>/<run>/weights/`.
Alongside each one, `config.yaml` records the architecture, input size and
preprocessing it was fitted under, and `norm_stats.json` the morphometric mean
and standard deviation fitted on the training slides only — applying a fusion
model to new slides means reusing those exact statistics rather than refitting.

```bash
python scripts/verify_models.py      # reload every model and re-score the test split
```

That needs the histology, which is not distributable. `reproduce.py check` needs
neither the images nor the weights.

## License

MIT — see `LICENSE`. Citation metadata in `CITATION.cff`.

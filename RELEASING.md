# Publishing this repository (maintainers)

The repository holds code, notebooks, figures, the saved per-patch predictions and the
synthetic data generator. `.gitignore` keeps out everything else:
* the images and workbooks in `data/`;
* weights (`*.pth`, `*.joblib`, `pretrained/`);
* `data_synthetic/` and `runs/`.

Remote: **https://github.com/jiehou-lab/smart-muscle**

## Before the first push

1. **Licence.** Done — `LICENSE` is MIT, and names the authors. It also records that
   the predictions ship under the same terms while the histology does not.
2. **Authors.** Done — `CITATION.cff` lists all seven in manuscript order.
3. **Human study codes.** Done — `build_smart_muscle.py` runs
   `anonymise_human_ids.py` over the tree. The human predictions used slide codes
   naming the study and the subject; each subject is now a pseudonym (`S01`, `S02`,
   ...), with the visit and side kept so the paired design stays visible. The
   absolute paths from the author's drive are gone from every prediction table too.
   Nothing changed numerically: the slide-level grouping key is one-to-one with the
   old one, and `reproduce.py check` still gives 249 of 249.

   `human_slide_id_map.csv` is written **outside** the repository, beside it. Keep
   it somewhere safe — it is the only way back from a pseudonym to a study code —
   and never commit it.

## First push

```powershell
cd data/smart-muscle
git init -b main
git add .
git status        # must NOT list data/, *.pth, *.joblib, pretrained/, data_synthetic/, runs/
git commit -m "smart-muscle v1.0.0: code, notebooks and results for the manuscript"
git remote add origin https://github.com/jiehou-lab/smart-muscle.git
git push -u origin main
```

The `git status` line is the one to read carefully rather than skim. If any weight or
image file is listed, stop: `.gitignore` is not being applied, and a push would put
~600 MB of model weights and undistributable histology on a public remote.

## Test as a reviewer would

In a new folder, with nothing of yours on the path:

```powershell
git clone https://github.com/jiehou-lab/smart-muscle.git test_clone
cd test_clone
pip install -e .
python reproduce.py check                 # CHECK PASSED (dataset step skipped: images not distributed)
python reproduce.py --synthetic all       # SYNTHETIC PIPELINE COMPLETE
```

Both commands must pass on a clone that has no weights and no histology. That is the
whole point of shipping the saved predictions and the synthetic generator: a reviewer
who has neither can still confirm every number, redraw every figure, and run the entire
pipeline end to end.

## If the weights are published later

They are excluded on purpose, not by accident — about 600 MB, and the histology behind
them cannot be redistributed. If you later put them somewhere citable, add the link to
the weights section of `README.md`; do not commit them. Each ResNet-50 checkpoint is
~92 MiB, which trips GitHub's 50 MB warning and sits close to its 100 MiB hard limit.

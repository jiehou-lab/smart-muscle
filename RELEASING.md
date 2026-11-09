# Releasing (maintainers)

Remote: **https://github.com/jiehou-lab/smart-muscle**

## What is and is not committed

Committed: code, notebooks, configs, the saved per-patch predictions, the
synthetic data generator, and the trained weights (21 files, ~630 MB).

Excluded by `.gitignore`: `data/` (histology, not distributable), `pretrained/`
(timm downloads it), `data_synthetic/` and `runs/` (both regenerated).

## Before pushing

Human slide study codes are already pseudonymised — `MSTN36_V3L` → `S01_V3L`.
`human_slide_id_map.csv` sits **outside** the repository and must never be
committed: it is the only way back from a pseudonym to a study code.

```powershell
cd F:\Medical_imaging\Medical_imaging_codes\smart-muscle
git add -A
git status        # must NOT list anything under data/
git commit -m "..."
git push
```

Read the `git status` line rather than skim it. If anything under `data/` is
listed, stop — `.gitignore` is not being applied, and a push would put
undistributable histology on a public remote.

Three things to expect, none of them errors: GitHub warns that six files exceed
50 MB (the largest is 92.0 MiB against a 100 MiB hard limit, so no Git LFS); a
~640 MB HTTPS push can die partway with `RPC failed`, in which case
`git config http.postBuffer 524288000` and retry, the commit being unaffected;
and git keeps every blob forever, so the repository is now permanently ~640 MB.

## Test as a reviewer would

In a new folder, with nothing of yours on the path:

```powershell
git clone https://github.com/jiehou-lab/smart-muscle.git test_clone
cd test_clone
pip install -e .
python reproduce.py check                 # CHECK PASSED
python reproduce.py --synthetic all       # SYNTHETIC PIPELINE COMPLETE
```

Both must pass without the histology — that is the point of shipping the saved
predictions and the synthetic generator.

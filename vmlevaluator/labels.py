"""Class-label policy: severity-ordered classes, positive = most injured.

Why this module exists
----------------------
The question this study asks is "how injured is this muscle?", so the class the
metrics call *positive* should be the **more injured** one. In the original
layout that was not true, and it was not even consistent between tasks: the
folder integers on disk had grown up task by task, and two of the three tasks
turned out to be ordered the opposite way round from what the configs declared.

Rather than move any image files, this module applies a permutation to the
*label integers* as the dataset is read. Nothing about the data changes -- the
patches, the slides and the train/test split are byte-for-byte the same -- only
which integer each folder is called.

The policy
----------
**Class index increases with injury severity, and the positive class is the
most injured one.** Under that rule:

    control_vml       0 Control     1 VML
    poor_very_poor    0 Poor        1 Very poor
    multiclass        0 Perfect     1 Poor        2 Very poor

which is also, as it happens, exactly the class ordering the manuscript text
already describes.

What this changes, and what it does not
---------------------------------------
For a **binary** task, relabelling swaps which class F1, precision and recall
are reported for, and replaces ROC-AUC with ``1 - AUC``. It does **not** change
the fitted model: every learner here is trained with balanced class weights,
which are symmetric under a label swap, so the decision boundary, the confusion
matrix and accuracy are identical. For the **three-class** task the permutation
is invisible to every metric used -- macro-F1, accuracy and the confusion
matrix are all permutation-invariant -- and only the axis labels move.

So this is a reporting convention made explicit and testable, not a
re-experiment. ``tests/test_labels.py`` verifies the ordering directly against
the morphometric workbook rather than trusting the folder names.

The one modelling consequence: with severity-ordered labels the biologically
correct monotone direction for the morphometric features is the *same* for
every task -- P(class) must not increase with fibre calibre or fibre count, so
the constraint sign is always -1. Previously it differed per task and had to be
worked out case by case.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

__all__ = [
    "FOLDER_TO_CLASS",
    "SEVERITY_CLASS_NAMES",
    "POSITIVE_CLASS_INDEX",
    "MOST_INJURED_INDEX",
    "remap_labels",
    "describe_policy",
]

#: ``FOLDER_TO_CLASS[task][folder_integer] -> model class index``.
#:
#: Verified against the morphometric workbook over all slides, not assumed from
#: the folder names (see tests/test_labels.py):
#:
#:   poor_very_poor  folder 0 = Very poor (7/7 slides, mean 1,835 myofibres)
#:                   folder 1 = Poor      (9/9 slides, mean 6,717 myofibres)
#:   multiclass      folder 0 = Perfect   (mean 9,293 myofibres, CSA 1,849)
#:                   folder 1 = Very poor (mean 1,835 myofibres, CSA   924)
#:                   folder 2 = Poor      (mean 6,717 myofibres, CSA 1,635)
FOLDER_TO_CLASS: Dict[str, List[int]] = {
    "control_vml": [0, 1],        # already severity-ordered
    "poor_very_poor": [1, 0],     # folder 0 is Very poor -> class 1
    "multiclass": [0, 2, 1],      # folder 1 is Very poor -> class 2
}

#: Class names in the new, severity-increasing order.
SEVERITY_CLASS_NAMES: Dict[str, List[str]] = {
    "control_vml": ["Control", "VML"],
    "poor_very_poor": ["Poor", "Very poor"],
    "multiclass": ["Perfect", "Poor", "Very poor"],
}

#: Which class index the binary metrics treat as positive. Always the last one,
#: because the classes are severity-ordered.
POSITIVE_CLASS_INDEX: Dict[str, int] = {
    task: len(names) - 1 for task, names in SEVERITY_CLASS_NAMES.items()
}

#: The most injured class, for the one-vs-rest column reported on the
#: three-class task. Identical to POSITIVE_CLASS_INDEX by construction; named
#: separately because the two mean different things and could diverge if a
#: future task were not severity-ordered.
MOST_INJURED_INDEX: Dict[str, int] = dict(POSITIVE_CLASS_INDEX)


def remap_labels(labels: Sequence[int], task: str) -> List[int]:
    """Apply the task's folder-to-class permutation to a sequence of labels."""
    mapping = FOLDER_TO_CLASS.get(task)
    if mapping is None:
        raise KeyError(
            f"no label policy for task {task!r}; known tasks: {sorted(FOLDER_TO_CLASS)}")
    out = []
    for value in labels:
        value = int(value)
        if not 0 <= value < len(mapping):
            raise ValueError(
                f"folder label {value} is outside the {len(mapping)} classes "
                f"declared for task {task!r}")
        out.append(mapping[value])
    return out


def describe_policy(task: str) -> str:
    """One line per class, for logs and notebook output."""
    mapping = FOLDER_TO_CLASS[task]
    names = SEVERITY_CLASS_NAMES[task]
    lines = [f"{task}: severity-ordered, positive class = "
             f"{names[POSITIVE_CLASS_INDEX[task]]!r}"]
    for folder, cls in enumerate(mapping):
        lines.append(f"    folder {folder} -> class {cls} ({names[cls]})")
    return "\n".join(lines)

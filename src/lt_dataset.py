"""CIFAR-100-LT data pipeline.

Generates long-tailed training subsets from the balanced CIFAR-100 train split,
following the LDAM-DRW protocol (Cao et al., CVPR 2019):
    n_j = round(n_max * rho ** (-j / (C - 1))),  j = 0..C-1

A balanced k-shot support set is carved out per class BEFORE long-tail sampling;
support features serve as candidate prototypes (and, at test time, are the
mechanism by which new classes enter the candidate table with zero retraining).
"""

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config import CFG

DATA_ROOT = CFG.cifar_root

CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR_STD = (0.2673, 0.2564, 0.2762)

# LDAM-DRW bucket convention
def bucket_of(n):
    if n > 100:
        return "head"
    if n >= 20:
        return "medium"
    return "tail"


def _load_split(name):
    with open(DATA_ROOT / name, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    x = d["data"].reshape(-1, 3, 32, 32).astype(np.uint8)
    fine = np.array(d["fine_labels"], dtype=np.int64)
    coarse = np.array(d["coarse_labels"], dtype=np.int64)
    return x, fine, coarse


def class_counts(num_classes, rho, n_max):
    j = np.arange(num_classes)
    return np.round(n_max * rho ** (-j / (num_classes - 1))).astype(int)


class IndexedCifar(Dataset):
    """CIFAR split with per-sample index; transforms applied on the fly."""

    def __init__(self, x, fine, coarse, transform=None):
        self.x, self.fine, self.coarse = x, fine, coarse
        self.transform = transform

    def __len__(self):
        return len(self.fine)

    def __getitem__(self, i):
        img = self.x[i].transpose(1, 2, 0)  # HWC for torchvision
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.fine[i]), int(self.coarse[i]), i


def make_lt_datasets(rho=100, n_max=500, support_shots=5, seed=0,
                     train_tf=None, support_tf=None, test_tf=None,
                     num_classes=100):
    """Returns (train_lt, support, test, meta).

    train_lt : long-tailed subset of the non-support pool
    support  : balanced support_shots-way dataset (prototypes, new-class entry)
    meta     : dict with cls_num_list, buckets, prior, superclass grouping
    """
    rng = np.random.RandomState(seed)
    x, fine, coarse = _load_split("train")

    keep = fine < num_classes
    x, fine, coarse = x[keep], fine[keep], coarse[keep]

    support_idx, train_pool_idx = [], []
    for c in range(num_classes):
        idx = np.where(fine == c)[0]
        rng.shuffle(idx)
        support_idx.extend(idx[:support_shots])
        train_pool_idx.extend(idx[support_shots:])

    counts = class_counts(num_classes, rho, n_max)
    final_idx = []
    for c in range(num_classes):
        idx = np.where(fine[train_pool_idx] == c)[0]
        cls_pool = np.array(train_pool_idx)[idx]
        rng.shuffle(cls_pool)
        final_idx.extend(cls_pool[: counts[c]])
    final_idx = np.array(final_idx)
    rng.shuffle(final_idx)

    train_lt = IndexedCifar(x[final_idx], fine[final_idx], coarse[final_idx], train_tf)
    support = IndexedCifar(x[np.array(support_idx)], fine[np.array(support_idx)],
                           coarse[np.array(support_idx)], support_tf)
    support_plain = IndexedCifar(x[np.array(support_idx)], fine[np.array(support_idx)],
                                 coarse[np.array(support_idx)], support_tf)

    tx, tfine, tcoarse = _load_split("test")
    keep = tfine < num_classes
    test = IndexedCifar(tx[keep], tfine[keep], tcoarse[keep], test_tf)

    # superclass of each fine class (from first occurrence)
    sup_of = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        sup_of[c] = coarse[fine == c][0]

    meta = dict(
        cls_num_list=counts.tolist(),
        buckets=[bucket_of(n) for n in counts],
        prior=(counts / counts.sum()).astype(np.float64),
        superclass_of=sup_of,
        num_superclass=int(sup_of.max()) + 1,
        rho=rho,
    )
    return train_lt, support, support_plain, test, meta


def subset_by_class(ds, class_ids, seed=0):
    """Return an IndexedCifar over `class_ids` only (original labels kept),
    plus the boolean mask into ds arrays."""
    ids = np.asarray(class_ids)
    mask = np.isin(ds.fine, ids)
    sub = IndexedCifar(ds.x[mask], ds.fine[mask], ds.coarse[mask], ds.transform)
    return sub, mask


def k_shots_per_class(ds, class_ids, k, seed=7):
    """Balanced k-shot subset over `class_ids` (used to build few-shot
    prototypes for classes that never entered training)."""
    rng = np.random.RandomState(seed)
    ids = np.asarray(class_ids)
    sel = []
    for c in ids:
        idx = np.where(ds.fine == c)[0]
        rng.shuffle(idx)
        sel.extend(idx[:k])
    sel = np.array(sel)
    return IndexedCifar(ds.x[sel], ds.fine[sel], ds.coarse[sel], ds.transform)


def partition_classes(meta, n_train, n_neg, seed=0):
    """Three-way stratified split of classes for the expansion+abstention
    protocol:
      TRAIN : seen during training (candidates with learned calibration)
      NEW   : never seen in any form; enter the candidate table at test time
              via k-shot prototypes (the zero-retrain axis)
      NEG   : never seen as candidates; their images are OOD negatives used
              to train the abstain candidate (and as OOD at risk-coverage)
    Stratified over head/medium/tail so every partition sees the spectrum.
    """
    rng = np.random.RandomState(seed)
    counts = np.array(meta["cls_num_list"])
    buckets = np.array(meta["buckets"])
    order = np.argsort(-counts)
    pools = {b: order[buckets[order] == b] for b in ("head", "medium", "tail")}
    n_all = len(counts)

    def take(n):
        n = min(n, n_all - len(TRAIN) - len(NEW) - len(NEG))
        got = []
        for b in ("head", "medium", "tail"):
            pool = pools[b]
            t = int(round(n * len(pool) / n_all))
            got.extend(pool[:t])
        rng.shuffle(got)
        return sorted(got[:n])

    TRAIN, NEW, NEG = [], [], []
    TRAIN = take(n_train)
    rest = np.setdiff1d(np.arange(n_all), TRAIN)
    rest_by_bucket = {b: [c for c in rest if buckets[c] == b] for b in
                      ("head", "medium", "tail")}
    # NEG stratified over the remainder
    for b in ("head", "medium", "tail"):
        pool = np.array(rest_by_bucket[b])
        t = int(round(n_neg * len(pool) / len(rest)))
        NEG.extend(pool[:t])
    NEG = sorted(NEG[:n_neg])
    NEW = sorted(np.setdiff1d(rest, NEG).tolist())
    return np.array(TRAIN), np.array(NEW), np.array(NEG)


if __name__ == "__main__":
    tr, sup, _, te, meta = make_lt_datasets(rho=100)
    counts = np.array(meta["cls_num_list"])
    print("train size:", len(tr), "support:", len(sup), "test:", len(te))
    print("head/medium/tail buckets:",
          {b: meta["buckets"].count(b) for b in ("head", "medium", "tail")})
    print("n_max:", counts.max(), "n_min:", counts.min())
    assert len(sup) == 100 * 5
    print("OK")

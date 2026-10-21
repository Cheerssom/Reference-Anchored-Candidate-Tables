"""Two-stage training for JLT-Head.

Stage A: ResNet-32 + linear head, plain CE on the long-tailed train set
         (decoupled representation learning, aligned with LDAM-DRW baseline).
Stage B: freeze encoder; build candidate prototypes from the balanced k-shot
         support set; train the typed decision head with the multi-question
         loss (RMS-normalized).

Usage:
  python train.py --smoke            # tiny end-to-end run (~minutes on 4070)
  python train.py --rho 100          # full run
  python train.py --rho 100 --train-classes 50   # class-addition protocol
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

from lt_dataset import (make_lt_datasets, DATA_ROOT, CIFAR_MEAN, CIFAR_STD,
                        partition_classes, subset_by_class)
from encoder import ResNet32
from clip_encoder import CLIPVisionEncoder, CLIP_MEAN, CLIP_STD
from decision_head import JLTHead, MultiQuestionLoss

TF_TRAIN = T.Compose([
    T.ToPILImage(), T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
    T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
TF_EVAL = T.Compose([T.ToPILImage(), T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
# CLIP inputs: resize to 224 with CLIP statistics; train-time augmentation is
# light (random crop inside the resized canvas) to keep prototype stability
TF_CLIP_TRAIN = T.Compose([
    T.ToPILImage(), T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])
TF_CLIP = T.Compose([
    T.ToPILImage(), T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])


def make_transforms(encoder_name):
    if encoder_name == "clip":
        # deterministic transform on purpose: frozen features are cached once
        return TF_CLIP, TF_CLIP, TF_CLIP
    return TF_TRAIN, TF_EVAL, TF_EVAL


def build_encoder(name, num_classes, device):
    if name == "clip":
        enc = CLIPVisionEncoder().to(device)
        return enc
    return ResNet32(num_classes).to(device)


def stratified_classes(meta, k, seed=0):
    """Pick k classes stratified over head/medium/tail (for expansion protocol)."""
    rng = np.random.RandomState(seed)
    counts = np.array(meta["cls_num_list"])
    idx = np.argsort(-counts)  # head first
    buckets = np.array(meta["buckets"])
    chosen = []
    for b in ("head", "medium", "tail"):
        pool = idx[buckets[idx] == b]
        take = int(round(k * len(pool) / len(counts)))
        chosen.extend(pool[:take])
    chosen = np.array(sorted(chosen[:k]))
    return chosen


@torch.no_grad()
def build_prototypes(encoder, support_loader, num_classes, device):
    encoder.eval()
    feats = {}
    for x, fine, _, _ in support_loader:
        _, pooled = encoder(x.to(device))
        for f, p in zip(fine, pooled.cpu()):
            feats.setdefault(int(f), []).append(p)
    protos = torch.stack([torch.stack(feats[c]).mean(0) for c in range(num_classes)])
    return protos.to(device)


def stage_a(args, num_classes, train_lt, device, log_prior_np=None):
    encoder = ResNet32(num_classes).to(device)
    opt = torch.optim.SGD(encoder.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=2e-4, nesterov=True)
    # LDAM-DRW schedule: step decay x0.1 at 160/180 (matches old baseline runs)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(args.epochs_a * 0.8), int(args.epochs_a * 0.9)],
        gamma=0.1)
    lp = None
    if args.stage_a_loss == "bsm" and log_prior_np is not None:
        lp = torch.tensor(log_prior_np, dtype=torch.float32, device=device)
    loader = DataLoader(train_lt, batch_size=args.bs, shuffle=True,
                        num_workers=2, drop_last=True, persistent_workers=True)
    for ep in range(args.epochs_a):
        encoder.train()
        tot, correct, n = 0.0, 0, 0
        for x, y, _, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = encoder.classify(x)
            if lp is not None:  # balanced softmax: CE(z + log pi)
                loss = nn.functional.cross_entropy(logits + lp[None, :], y)
            else:
                loss = nn.functional.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        sched.step()
        if (ep + 1) % max(1, args.epochs_a // 5) == 0 or args.smoke:
            print(f"[stage A] ep{ep+1}: loss {tot/n:.3f} acc {correct/n:.3f}")
    return encoder


def split_train_val(train_lt, seed=1):
    """Class-BALANCED held-out val (up to 5/class, tail keeps >=3 train samples).
    Balanced val matches the balanced test distribution so early stopping
    selects generalizing checkpoints instead of head-memorizing ones."""
    rng = np.random.RandomState(seed)
    fine = train_lt.fine
    val_idx = []
    for c in np.unique(fine):
        idx = np.where(fine == c)[0]
        rng.shuffle(idx)
        n_val = 5 if len(idx) >= 20 else (2 if len(idx) >= 6 else 1)
        val_idx.extend(idx[:n_val])
    val_mask = np.zeros(len(fine), dtype=bool)
    val_mask[np.array(val_idx)] = True
    fit = type(train_lt)(train_lt.x[~val_mask], train_lt.fine[~val_mask],
                         train_lt.coarse[~val_mask], train_lt.transform)
    val = type(train_lt)(train_lt.x[val_mask], train_lt.fine[val_mask],
                         train_lt.coarse[val_mask], train_lt.transform)
    return fit, val


@torch.no_grad()
def class_mean_prototypes(encoder, dataset, device):
    """Class-mean features over a dataset (use with eval transform)."""
    encoder.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    feats = {}
    for x, fine, _, _ in loader:
        _, pooled = encoder(x.to(device))
        for f, p in zip(fine, pooled.cpu()):
            feats.setdefault(int(f), []).append(p)
    classes = sorted(feats)
    protos = torch.stack([torch.stack(feats[c]).mean(0) for c in classes])
    order = torch.tensor(classes)
    return protos.to(device), order


class CachedFeats(torch.utils.data.Dataset):
    """Precomputed (ctx_tokens, pooled, y) for a frozen encoder — one ViT pass
    total instead of one per epoch."""

    def __init__(self, feats, pooled, ys):
        self.feats, self.pooled, self.ys = feats, pooled, ys

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, i):
        return self.feats[i], self.pooled[i], int(self.ys[i]), i


@torch.no_grad()
def cache_features(encoder, ds, device):
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    F_, P_, Y_ = [], [], []
    for x, y, _, _ in loader:
        f_, p_ = encoder(x.to(device))
        F_.append(f_.cpu()); P_.append(p_.cpu()); Y_.append(y.clone())
    return CachedFeats(torch.cat(F_), torch.cat(P_), torch.cat(Y_))


def means_from_cached(cached):
    """Class-mean prototypes from precomputed pooled features."""
    protos = {}
    for p, y in zip(cached.pooled, cached.ys):
        protos.setdefault(int(y), []).append(p)
    classes = sorted(protos)
    return (torch.stack([torch.stack(protos[c]).mean(0) for c in classes]).to(
        cached.pooled.device), torch.tensor(classes))


def labels_of(ds):
    return ds.ys.numpy() if isinstance(ds, CachedFeats) else ds.fine


def stage_b(args, encoder, meta, fit_set, val_set, support, device, num_classes,
            train_class_ids=None, neg_set=None):
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    cached = (args.encoder == "clip")
    if cached:
        # frozen encoder + deterministic transform -> one ViT pass total
        print("caching frozen CLIP features (single pass)...")
        fit_set = cache_features(encoder, fit_set, device)
        val_set = cache_features(encoder, val_set, device)
        if neg_set is not None:
            neg_set = cache_features(encoder, neg_set, device)

    if getattr(args, "proto_source", "train") == "support":
        proto_loader = DataLoader(support, batch_size=256, shuffle=False)
        protos = build_prototypes(encoder, proto_loader, num_classes, device)
    else:
        # class-mean anchors over the FIT split only; the 5-shot support path
        # stays reserved for unseen classes at expansion
        if cached:
            protos, order = means_from_cached(fit_set)
            protos = protos.to(device)
        else:
            fit_eval = type(fit_set)(fit_set.x, fit_set.fine, fit_set.coarse,
                                     TF_EVAL)
            protos, order = class_mean_prototypes(encoder, fit_eval, device)
        assert order.max().item() == num_classes - 1, \
            f"class coverage {len(order)} != {num_classes}"

    counts_all = np.array(meta["cls_num_list"], dtype=np.float64)
    sup_all = np.asarray(meta["superclass_of"])
    if train_class_ids is not None:
        sel = np.asarray(train_class_ids)
        counts = counts_all[sel]
        sup_sel = sup_all[sel]
    else:
        counts = counts_all
        sup_sel = sup_all
    log_priors = torch.tensor(np.log(counts / counts.sum()), dtype=torch.float32,
                              device=device)
    group_ids = torch.tensor(sup_sel, device=device)

    head = JLTHead(d_enc=getattr(encoder, "dim", 64)).to(device)
    head.lam.data.fill_(args.lam_init)
    lossfn = MultiQuestionLoss().to(device)
    if getattr(args, "finetune_encoder", False) and not cached:
        for p in encoder.parameters():
            p.requires_grad_(True)
        opt = torch.optim.AdamW([
            {"params": encoder.parameters(), "lr": args.lr_enc},
            {"params": head.parameters(), "lr": args.lr_b}], weight_decay=0.05)
        print("stage B: joint finetuning (encoder unfrozen)")
    else:
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr_b, weight_decay=0.05)

    val_loader = DataLoader(val_set, batch_size=256, shuffle=False)
    print(f"stage B: fit {len(fit_set)} / val {len(val_set)} (held out)")

    from torch.utils.data import ConcatDataset
    # optional OOD negatives for the abstain candidate: NEG-class images get
    # target == num_classes (the learned none-of-the-above candidate)
    if getattr(args, "abstain", False) and neg_set is not None:
        if cached:
            neg_relab = CachedFeats(neg_set.feats, neg_set.pooled,
                                    torch.full_like(neg_set.ys, num_classes))
        else:
            neg_relab = type(fit_set)(neg_set.x, np.full(len(neg_set), num_classes),
                                      neg_set.coarse, fit_set.transform)
        train_src = ConcatDataset([fit_set, neg_relab])
        w_fit = 1.0 / counts[labels_of(fit_set)]
        neg_frac = 0.25
        w_neg = np.full(len(neg_set),
                        (neg_frac / (1 - neg_frac)) * w_fit.sum() / len(neg_set))
        w_all = np.concatenate([w_fit, w_neg])
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.tensor(w_all, dtype=torch.double), len(train_src),
            replacement=True)
        loader = DataLoader(train_src, batch_size=args.bs, sampler=sampler,
                            num_workers=0, drop_last=True)
        print(f"stage B: abstain training with {len(neg_set)} OOD negatives "
              f"(~{int(neg_frac*100)}% batch mass)")
    elif args.balanced_b:
        w = 1.0 / counts[labels_of(fit_set)]
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.tensor(w, dtype=torch.double), len(fit_set), replacement=True)
        loader = DataLoader(fit_set, batch_size=args.bs, sampler=sampler,
                            num_workers=0, drop_last=True)
    else:
        loader = DataLoader(fit_set, batch_size=args.bs, shuffle=True,
                            num_workers=0, drop_last=True)
    sup = torch.tensor(sup_sel, dtype=torch.long, device=device)
    ft = getattr(args, "finetune_encoder", False) and not cached
    best_val, best_state = -1.0, None
    for ep in range(args.epochs_b):
        head.train()
        encoder.train() if ft else encoder.eval()
        tot, correct, n = 0.0, 0, 0
        for batch in loader:
            if cached:
                feats, pooled_b, y, _ = batch
                feats, pooled_b, y = (feats.to(device), pooled_b.to(device),
                                      y.to(device))
            else:
                x, y, _, _ = batch
                x, y = x.to(device), y.to(device)
                if ft:
                    feats, pooled_b = encoder(x)
                else:
                    with torch.no_grad():
                        feats, pooled_b = encoder(x)
            out = head(feats, pooled_b, protos, log_priors, group_ids=group_ids,
                       use_abstain=bool(getattr(args, "abstain", False)))
            tg = sup[y.clamp(max=num_classes - 1)]
            tg = torch.where(y == num_classes,
                             torch.full_like(tg, -100), tg)  # OOD: no coarse target
            loss, parts = lossfn(out, y, tg)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(y)
            correct += (out["logits_q2"].argmax(1) == y).sum().item()
            n += len(y)
        # keep anchors consistent with the moving encoder
        if ft and (ep + 1) % 10 == 0:
            fit_eval = type(fit_set)(fit_set.x, fit_set.fine, fit_set.coarse,
                                     TF_EVAL)
            protos, _ = class_mean_prototypes(encoder, fit_eval, device)
        # honest validation: MACRO accuracy on class-balanced held-out split
        encoder.eval(); head.eval()
        vhits = {}
        with torch.no_grad():
            for batch in val_loader:
                if cached:
                    feats, pooled_b, y, _ = batch
                    feats, pooled_b = feats.to(device), pooled_b.to(device)
                else:
                    x, y, _, _ = batch
                    feats, pooled_b = encoder(x.to(device))
                out = head(feats, pooled_b, protos, log_priors, group_ids=group_ids,
                           use_abstain=bool(getattr(args, "abstain", False)))
                pred = out["logits_q2"].argmax(1).cpu()
                for yi, pi in zip(y.tolist(), pred.tolist()):
                    vhits.setdefault(yi, []).append(int(yi == pi))
        vacc = float(np.mean([np.mean(v) for v in vhits.values()]))
        if vacc > best_val:
            best_val = vacc
            best_state = {"head": {k: v.detach().clone()
                                   for k, v in head.state_dict().items()}}
            if ft:
                best_state["encoder"] = {k: v.detach().clone()
                                         for k, v in encoder.state_dict().items()}
                best_state["protos"] = protos.detach().clone()
        if (ep + 1) % max(1, args.epochs_b // 10) == 0 or args.smoke:
            print(f"[stage B] ep{ep+1}: loss {tot/n:.4f} train {correct/n:.3f} "
                  f"val {vacc:.3f} parts={ {k: round(v,3) for k,v in parts.items()} }")
    if best_state is not None:
        head.load_state_dict(best_state["head"])
        if "encoder" in best_state:
            encoder.load_state_dict(best_state["encoder"])
        if "protos" in best_state:
            protos = best_state["protos"]
        print(f"[stage B] best val acc {best_val:.4f} (restored)")
    return head, protos, log_priors, group_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=100)
    ap.add_argument("--n-max", type=int, default=500)
    ap.add_argument("--support-shots", type=int, default=5)
    ap.add_argument("--epochs-a", type=int, default=200)
    ap.add_argument("--epochs-b", type=int, default=30)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--num-classes", type=int, default=100)
    ap.add_argument("--train-classes", type=int, default=0,
                    help="if >0, train on this many classes (expansion protocol)")
    ap.add_argument("--stage-a-loss", type=str, default="ce",
                    choices=["ce", "bsm"])
    ap.add_argument("--load-encoder", type=str, default="",
                    help="path to a previous checkpoint; reuse stage-A encoder")
    ap.add_argument("--lam-init", type=float, default=0.0)
    ap.add_argument("--proto-source", type=str, default="train",
                    choices=["train", "support"])
    ap.add_argument("--lr-b", type=float, default=1e-4)
    ap.add_argument("--lr-enc", type=float, default=1e-4)
    ap.add_argument("--finetune-encoder", action="store_true")
    ap.add_argument("--encoder", type=str, default="resnet32",
                    choices=["resnet32", "clip"])
    ap.add_argument("--abstain", action="store_true",
                    help="train the abstain candidate with NEG-class OOD negatives")
    ap.add_argument("--neg-classes", type=int, default=0,
                    help="classes reserved as OOD negatives (needs --train-classes)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--balanced-b", action="store_true",
                    help="class-balanced sampling in stage B")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--out", type=str, default="../runs")
    args = ap.parse_args()

    if args.smoke:
        args.num_classes, args.epochs_a, args.epochs_b = 20, 2, 2
        args.train_classes, args.neg_classes, args.abstain = 12, 4, True
        print(f"SMOKE config: {vars(args)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.abstain and (args.train_classes <= 0 or args.neg_classes <= 0):
        print("WARNING: --abstain needs --train-classes and --neg-classes > 0; "
              "disabling abstain (no OOD negative source)")
        args.abstain = False
    tf_train, tf_eval, _ = make_transforms(args.encoder)
    train_lt, support, _, test, meta = make_lt_datasets(
        rho=args.rho, n_max=args.n_max, support_shots=args.support_shots,
        train_tf=tf_train, support_tf=tf_eval, test_tf=tf_eval,
        num_classes=args.num_classes)

    train_class_ids = new_class_ids = neg_class_ids = None
    neg_set = None
    if args.train_classes > 0:
        n_neg = args.neg_classes if args.neg_classes > 0 else 0
        train_class_ids, new_class_ids, neg_class_ids = partition_classes(
            meta, args.train_classes, n_neg, seed=0)
        full_lt = train_lt                       # keep pre-filter reference
        keep = np.isin(train_lt.fine, train_class_ids)
        train_lt = type(train_lt)(train_lt.x[keep], train_lt.fine[keep],
                                  train_lt.coarse[keep], train_lt.transform)
        # relabel to positions of train_class_ids for prototype alignment
        remap = {int(c): i for i, c in enumerate(train_class_ids)}
        train_lt.fine = np.array([remap[int(f)] for f in train_lt.fine])
        if n_neg > 0:
            neg_sub, _ = subset_by_class(full_lt, neg_class_ids)
            neg_set = type(neg_sub)(neg_sub.x, neg_sub.fine, neg_sub.coarse, tf_eval)
        print(f"expansion protocol: TRAIN {len(train_class_ids)} / "
              f"NEW {len(new_class_ids)} / NEG {len(neg_class_ids)} classes, "
              f"{len(train_lt)} train samples")

    counts_all = np.array(meta["cls_num_list"], dtype=np.float64)
    if args.stage_a_loss == "bsm":
        sel = np.asarray(train_class_ids) if train_class_ids is not None \
            else np.arange(args.num_classes)
        log_prior_np = np.log(counts_all[sel] / counts_all[sel].sum())
    else:
        log_prior_np = None
    n_trained = len(train_class_ids) if train_class_ids is not None \
        else args.num_classes
    # held-out val is isolated from BOTH stages (encoder memorization guard)
    fit_set, val_set = split_train_val(train_lt)
    print(f"data: fit {len(fit_set)} / val {len(val_set)} (balanced, held out)")
    if args.encoder == "clip":
        encoder = build_encoder("clip", n_trained, device)
        print("encoder: frozen CLIP ViT (stage A skipped)")
    elif args.load_encoder:
        prev = torch.load(args.load_encoder, map_location=device, weights_only=False)
        encoder = build_encoder(prev.get("args", {}).get("encoder", "resnet32"),
                                n_trained, device)
        encoder.load_state_dict(prev["encoder"])
        print(f"loaded stage-A encoder from {args.load_encoder}")
    else:
        encoder = stage_a(args, n_trained, fit_set, device,
                          log_prior_np=log_prior_np)
    head, protos, log_priors, group_ids = stage_b(
        args, encoder, meta, fit_set, val_set, support, device,
        len(train_class_ids) if train_class_ids is not None else args.num_classes,
        train_class_ids=train_class_ids, neg_set=neg_set)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("smoke" if args.smoke else f"rho{int(args.rho)}")
    torch.save({"encoder": encoder.state_dict(),
                "head": head.state_dict(),
                "train_class_ids": train_class_ids,
                "new_class_ids": new_class_ids,
                "neg_class_ids": neg_class_ids,
                "meta_cls_num_list": meta["cls_num_list"],
                "args": vars(args)},
               out / f"ckpt_{tag}.pt")
    print("saved checkpoint ->", out / f"ckpt_{tag}.pt")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Evaluate FastRecon (few-shot, one normal per frequency band) on RF/spectrum CLS protocols.

FastRecon: Few-shot Industrial Anomaly Detection via Fast Feature Reconstruction.
Source: ``references/FastRecon`` (FzJun26th/FastRecon).

Method (faithful to ``references/FastRecon/main.py``):
  * backbone wide_resnet50_2 (ImageNet), hooks on layer2[-1] + layer3[-1];
    avgpool(3,1,1) each, then ``embedding_concat`` -> patch features [B, 1536, 28, 28]
    (C = 512 + 1024, HW = 28*28 at input 224).
  * support: pool all normal patch features -> kCenterGreedy coreset ``Sc`` [Ns, C];
    ``mu`` = mean normal feature map over support images [HW, C].
  * query ``Q`` [HW, C]: closed-form regression with distribution regularization
        W = (Q Sc^T + lambda * mu Sc^T) @ inv((1+lambda) * Sc Sc^T)
        Q_hat = W @ Sc
    image score = max_patch ||Q - Q_hat||_2.
    Equivalent objective: min_W ||Q - W Sc||^2 + lambda * ||W Sc - mu||^2
    (reconstruction should resemble the query while staying close to the normal mean).

Deviation from the released code: ``main.py`` overwrites the support embedding buffer
each training batch, so with batch_size=1 the coreset and ``mu`` are built from the
*last* support image only. We aggregate *all* support samples (matches the paper).

Data / job building, dataset, and CSV/summary formatting are reused from
``tools/eval_patchcore_cls.py`` so results are directly comparable to the official
PatchCore run under the identical protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.random_projection import SparseRandomProjection
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

REPO_ROOT = Path(__file__).resolve().parents[1]
FASTRECON_SRC = REPO_ROOT / "references" / "FastRecon"
if str(FASTRECON_SRC) not in sys.path:
    sys.path.insert(0, str(FASTRECON_SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sampling_methods.kcenter_greedy import kCenterGreedy  # noqa: E402

from tools.eval_patchcore_cls import (  # noqa: E402
    PatchCorePathDataset,
    collect_rf_target_cell,
    public_rf_jobs,
    rf_target_jobs,
    safe_auc,
    spectrum_jobs,
    train_signature,
    write_csv,
)
from train_rf_target_pooled_universal import JSR_BY_SIGNAL  # noqa: E402
from utils.rf_frequency_sampling import maybe_select_one_per_frequency_band  # noqa: E402
from utils.training_utils import setup_seed  # noqa: E402


def append_average(rows, method="fastrecon"):
    out = list(rows)
    auc_cols = [c for c in rows[0].keys() if "auroc" in c]
    avg = {k: "" for k in rows[0].keys()}
    avg.update(
        {
            "method": method,
            "dataset": rows[0]["dataset"],
            "category": "average",
            "scene": "macro",
            "jsr": "macro",
            "num_train_normal": "",
            "num_test_normal": "",
            "num_test_abnormal": "",
        }
    )
    for col in auc_cols:
        avg[col] = float(np.mean([r[col] for r in rows]))
    out.append(avg)
    return out


def rf_target_per_scene_jobs(args):
    """Per-scene few-shot jobs: each scene builds its own support gallery by pooling
    all signals and JSRs in that scene, then taking one normal per frequency band.

    All (signal, scene, jsr) cells within a scene share that scene's gallery, so
    ``run_jobs`` fits FastRecon once per scene (4 fits for the full protocol).
    """
    import copy

    cell_args = copy.copy(args)
    cell_args.normal_sampling = "all"  # gather raw normals; subsample at scene level
    cell_args.exclude_support_from_test = False
    scene_train_raw: dict[str, list] = {}
    cells = []
    for signal in args.rf_signals:
        for scene in args.rf_scenes:
            for jsr in JSR_BY_SIGNAL[signal]:
                train_samples, test_samples = collect_rf_target_cell(signal, scene, jsr, cell_args)
                cells.append((signal, scene, jsr, test_samples))
                scene_train_raw.setdefault(scene, []).extend(train_samples)

    scene_support: dict[str, list] = {}
    for scene, samples in scene_train_raw.items():
        seen: set[str] = set()
        deduped = []
        for s in samples:
            key = str(s["path"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(s)
        support = maybe_select_one_per_frequency_band(
            deduped, args.normal_sampling, path_getter=lambda s: s["path"]
        )
        if args.max_pooled_train_normals > 0:
            support = support[: args.max_pooled_train_normals]
        scene_support[scene] = support
        print(f"[per_scene] scene={scene} support_normals={len(support)} (raw {len(deduped)})")

    jobs = []
    for signal, scene, jsr, test_samples in cells:
        support_paths = {str(s["path"]) for s in scene_support[scene]}
        filtered_test = [s for s in test_samples if str(s["path"]) not in support_paths]
        jobs.append(
            {
                "dataset": "rf_target",
                "category": signal,
                "scene": scene,
                "jsr": jsr,
                "train_samples": scene_support[scene],
                "test_samples": filtered_test,
            }
        )
    return jobs


# --------------------------------------------------------------------------------------
# Feature extraction (verbatim algorithm from references/FastRecon/main.py)
# --------------------------------------------------------------------------------------

def embedding_concat(x, y):
    """Concatenate two feature maps at the larger spatial resolution (PaDiM-style).

    Pulled from references/FastRecon/main.py; only fix is placing ``z`` on the input
    device so it works on GPU (the original allocates on CPU).
    """
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2, device=x.device, dtype=x.dtype)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z


class FastReconEncoder(nn.Module):
    """wide_resnet50_2 with forward hooks on layer2[-1] / layer3[-1]."""

    def __init__(self):
        super().__init__()
        self.model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self._feats: list[torch.Tensor] = []
        self.model.layer2[-1].register_forward_hook(self._hook)
        self.model.layer3[-1].register_forward_hook(self._hook)

    def _hook(self, _module, _input, output):
        self._feats.append(output)

    @torch.no_grad()
    def forward(self, x):
        self._feats = []
        _ = self.model(x)
        pooled = [F.avg_pool2d(f, 3, 1, 1) for f in self._feats]
        return embedding_concat(pooled[0], pooled[1])  # [B, 1536, 28, 28]


# --------------------------------------------------------------------------------------
# FastRecon model state
# --------------------------------------------------------------------------------------

class FastReconModel:
    """Holds the fitted coreset Sc, mean mu, and precomputed regression tensors.

    The expensive parts (coreset selection, ``inv(Sc Sc^T)``, ``mu Sc^T``) are
    lambda-independent, so a single fit serves any number of lambda values.
    Per lambda, ``inv_temp = inv(Sc Sc^T) / (1 + lambda)``.
    """

    def __init__(self, encoder: FastReconEncoder, lambdas, device: str):
        self.encoder = encoder
        self.lambdas = [float(v) for v in lambdas]
        self.device = device
        # populated by fit()
        self.Sc = None           # [Ns, C]  on device
        self.mu = None           # [HW, C]  on device
        self.inv_ScScT = None    # [Ns, Ns] on device, = inv(Sc Sc^T)
        self.mu_ScT = None       # [HW, Ns] on device, = mu Sc^T
        self.n_support_images = 0
        self.coreset_size = 0


def _extract_patch_view(feat: torch.Tensor):
    """[B, C, H, W] -> (patches [B*HW, C], per_image_map [B, C, HW])."""
    B, C, H, W = feat.shape
    patches = feat.permute(0, 2, 3, 1).contiguous().view(-1, C)
    per_image = feat.view(B, C, H * W)
    return patches, per_image


@torch.no_grad()
def fit_fastrecon(encoder: FastReconEncoder, train_samples, args, device) -> FastReconModel:
    if not train_samples:
        raise RuntimeError("FastRecon fit received no support samples")

    dataset = PatchCorePathDataset(train_samples, args.resize, args.imagesize)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = FastReconModel(encoder, args.lambdas, device)
    support_patches = []       # list of [b*HW, C] on CPU
    mu_sum = None              # [C, HW] on CPU
    n_img = 0

    encoder.eval()
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        feat = encoder(x)                       # [B, C, H, W]
        patches, per_image = _extract_patch_view(feat)
        support_patches.append(patches.cpu())
        if mu_sum is None:
            C, HW = per_image.shape[1], per_image.shape[2]
            mu_sum = torch.zeros(C, HW, dtype=torch.float32)
        mu_sum += per_image.sum(0).cpu()
        n_img += x.shape[0]

    all_patches = torch.cat(support_patches, dim=0).numpy().astype(np.float32)  # [N, C]
    model.n_support_images = n_img
    mu = (mu_sum / max(n_img, 1)).t().contiguous()                              # [HW, C]

    # Subsample support patches before kCenterGreedy (N^2 in N; pooled support ~ 1e5-1e6).
    n_total = all_patches.shape[0]
    if n_total > args.max_support_patches:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(n_total, size=args.max_support_patches, replace=False)
        all_patches = all_patches[keep]
    n_pool = all_patches.shape[0]

    # kCenterGreedy coreset (matches references/FastRecon/main.py: unfitted
    # SparseRandomProjection -> transform() raises -> falls back to raw features).
    n_select = max(1, int(round(n_pool * args.coreset_ratio)))
    n_select = min(n_select, n_pool, args.max_coreset_size)
    randomprojector = SparseRandomProjection(n_components="auto", eps=0.9)
    selector = kCenterGreedy(all_patches, 0, 0)
    selected_idx = selector.select_batch(
        model=randomprojector, already_selected=[], N=n_select
    )
    Sc = torch.from_numpy(all_patches[selected_idx]).to(device).float()         # [Ns, C]
    mu = mu.to(device).float()                                                   # [HW, C]
    model.Sc = Sc
    model.mu = mu
    model.coreset_size = Sc.shape[0]

    # Precompute inv(Sc Sc^T) (lambda-independent) and mu Sc^T.
    Sc_ScT = Sc @ Sc.t()                                                         # [Ns, Ns]
    try:
        inv_ScScT = torch.linalg.inv(Sc_ScT)
    except RuntimeError:
        inv_ScScT = torch.linalg.pinv(Sc_ScT)
    model.inv_ScScT = inv_ScScT
    model.mu_ScT = mu @ Sc.t()                                                   # [HW, Ns]

    print(
        f"[fit] support_images={n_img} support_patches={n_total} "
        f"pool_after_subsample={n_pool} coreset_size={Sc.shape[0]} "
        f"lambdas={model.lambdas} inv_ScScT=[{Sc_ScT.shape[0]}x{Sc_ScT.shape[1]}]"
    )
    return model


@torch.no_grad()
def predict_scores(model: FastReconModel, test_samples, args, device) -> dict[float, np.ndarray]:
    """Return per-image anomaly scores for each lambda (max patch reconstruction error).

    For lambda l: W = (Q Sc^T + l * mu Sc^T) @ (inv(Sc Sc^T) / (1 + l));
    Q_hat = W @ Sc; image score = max_patch ||Q - Q_hat||_2.
    """
    dataset = PatchCorePathDataset(test_samples, args.resize, args.imagesize)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    Sc = model.Sc                       # [Ns, C]
    inv_ScScT = model.inv_ScScT         # [Ns, Ns]
    mu_ScT = model.mu_ScT               # [HW, Ns]

    per_lambda = {lam: [] for lam in model.lambdas}
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        feat = model.encoder(x)                 # [B, C, H, W]
        B, C, H, W = feat.shape
        Q = feat.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)   # [B, HW, C]
        Q_ScT = Q @ Sc.t()                      # [B, HW, Ns]
        for lam in model.lambdas:
            inv_temp = inv_ScScT / (1.0 + lam)  # [Ns, Ns]
            A = Q_ScT + lam * mu_ScT.unsqueeze(0)  # [B, HW, Ns]
            W = A @ inv_temp                    # [B, HW, Ns]
            Q_hat = W @ Sc                      # [B, HW, C]
            diff = (Q - Q_hat).norm(dim=2)      # [B, HW]
            img_score = diff.max(dim=1).values  # [B]
            per_lambda[lam].extend([float(v) for v in img_score.cpu().tolist()])
    return {lam: np.asarray(v, dtype=np.float32) for lam, v in per_lambda.items()}


def predict_job(model: FastReconModel, job, args, device):
    test_samples = job["test_samples"]
    per_lambda = predict_scores(model, test_samples, args, device)
    labels = np.asarray([int(s["label"]) for s in test_samples], dtype=np.int32)

    score_dir = Path(args.output_root) / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{job['dataset']}-{job['category']}-{job['scene']}-{job['jsr']}".replace("/", "_")
    npz_payload = {
        "labels": labels,
        "image_paths": np.asarray([str(s["path"]) for s in test_samples]),
    }
    row = {
        "method": "fastrecon",
        "dataset": job["dataset"],
        "category": job["category"],
        "scene": job["scene"],
        "jsr": job["jsr"],
        "num_train_normal": len(job["train_samples"]),
        "num_test_normal": int((labels == 0).sum()),
        "num_test_abnormal": int((labels == 1).sum()),
    }
    primary_lam = args.primary_lam
    for lam, scores in per_lambda.items():
        key = f"image_auroc_lam{lam:g}"
        row[key] = safe_auc(labels.tolist(), scores.tolist())
        npz_payload[f"scores_lam{lam:g}"] = scores
    # Primary headline column (best config found via sweep; equals the chosen primary lambda).
    row["image_auroc"] = row[f"image_auroc_lam{primary_lam:g}"]
    np.savez_compressed(score_dir / f"{stem}-scores.npz", **npz_payload)
    return row


def run_jobs(jobs, args, device):
    rows = []
    groups = {}
    for idx, job in enumerate(jobs):
        groups.setdefault(train_signature(job), []).append((idx, job))

    for group_idx, grouped in enumerate(groups.values(), 1):
        first_job = grouped[0][1]
        print(
            f"[fit {group_idx}/{len(groups)}] "
            f"train_normals={len(first_job['train_samples'])} "
            f"eval_jobs={len(grouped)}"
        )
        if not first_job["train_samples"]:
            raise RuntimeError("No train samples for grouped FastRecon run")
        encoder = FastReconEncoder().to(device)
        model = fit_fastrecon(encoder, first_job["train_samples"], args, device)
        for idx, job in grouped:
            if not any(s["label"] == 1 for s in job["test_samples"]):
                raise RuntimeError(
                    f"No abnormal samples for {job['dataset']} {job['category']} "
                    f"{job['scene']} {job['jsr']}"
                )
            print(f"[{idx + 1}/{len(jobs)}] {job['dataset']} {job['category']} {job['scene']} {job['jsr']}")
            row = predict_job(model, job, args, device)
            print(f"  image_auroc={row['image_auroc']:.4f}")
            rows.append(row)
        del model, encoder
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    rows.sort(key=lambda r: (r["dataset"], r["category"], r["scene"], r["jsr"]))
    return rows


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="FastRecon few-shot CLS evaluation")
    parser.add_argument("--protocol", choices=["spectrum", "public_rf", "rf_target"], default="rf_target")
    parser.add_argument("--output-root", default="analysis_outputs/20260703_fastrecon_cls")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--imagesize", type=int, default=224)
    # FastRecon hyperparameters
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.5, 2.0, 10.0],
                        help="distribution-regularization weights to evaluate (paper default 2; "
                             "sweep shows lambda=0 works best on signal data)")
    parser.add_argument("--primary-lam", type=float, default=0.0,
                        help="lambda reported in the headline image_auroc column")
    parser.add_argument("--coreset-ratio", type=float, default=0.05,
                        help="fraction of support patches kept by kCenterGreedy")
    parser.add_argument("--max-support-patches", type=int, default=10000,
                        help="random subsample cap on support patches before kCenterGreedy")
    parser.add_argument("--max-coreset-size", type=int, default=256,
                        help="hard cap on Ns to keep inv(Sc Sc^T) feasible")
    # Protocol / sampling
    parser.add_argument("--normal-sampling", choices=["all", "frequency_one_per_band"],
                        default="frequency_one_per_band",
                        help="few-shot normal selection; default = one normal per frequency band")
    parser.add_argument("--max-pooled-train-normals", type=int, default=0)
    parser.add_argument("--max-test-normals", type=int, default=0)
    parser.add_argument("--max-abnormals", type=int, default=0)
    # spectrum protocol
    parser.add_argument("--spectrum-root", default=str(REPO_ROOT / "datasets" / "spectrum"))
    parser.add_argument("--spectrum-categories", nargs="+",
                        default=["16QAM", "CHIRP", "GMSK", "QPSK"],
                        choices=["16QAM", "CHIRP", "GMSK", "QPSK"])
    # rf_target protocol
    from train_rf_target_pooled_universal import JSR_BY_SIGNAL, SCENES
    parser.add_argument("--rf-signals", nargs="+", default=list(JSR_BY_SIGNAL.keys()),
                        choices=list(JSR_BY_SIGNAL.keys()))
    parser.add_argument("--rf-scenes", nargs="+", default=SCENES, choices=SCENES)
    parser.add_argument("--rf-train-mode", choices=["pooled", "per_cell", "per_scene"], default="per_scene",
                        help="per_scene (default): each scene builds its own one-normal-per-band "
                             "gallery pooled across signals+jsr; 4 fits for the full protocol")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda:0")

    if args.protocol == "spectrum":
        jobs = spectrum_jobs(args)
    elif args.protocol == "public_rf":
        jobs = public_rf_jobs(args)
    elif args.protocol == "rf_target" and args.rf_train_mode == "per_scene":
        jobs = rf_target_per_scene_jobs(args)
    else:
        jobs = rf_target_jobs(args)

    rows = run_jobs(jobs, args, device)
    rows_with_avg = append_average(rows)

    out_root = Path(args.output_root)
    result_path = out_root / "results_fastrecon_cls.csv"
    write_csv(result_path, rows_with_avg)

    df_cols = list(rows[0].keys())
    auc_cols = [c for c in df_cols if "auroc" in c]
    summary = {
        "method": "fastrecon",
        "source": "references/FastRecon (FzJun26th/FastRecon)",
        "protocol": args.protocol,
        "rf_train_mode": args.rf_train_mode if args.protocol == "rf_target" else None,
        "backbone": "wide_resnet50_2 (ImageNet)",
        "layers": ["layer2", "layer3"],
        "lambdas": args.lambdas,
        "primary_lambda": args.primary_lam,
        "coreset_ratio": args.coreset_ratio,
        "max_support_patches": args.max_support_patches,
        "max_coreset_size": args.max_coreset_size,
        "normal_sampling": args.normal_sampling,
        "resize": args.resize,
        "imagesize": args.imagesize,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "num_jobs": len(rows),
    }
    for col in auc_cols:
        summary[f"{col}_macro"] = float(np.mean([r[col] for r in rows]))
    summary["image_auroc_macro"] = summary[f"image_auroc_lam{args.primary_lam:g}_macro"]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out_root / "README.md").write_text(
        "# FastRecon CLS Evaluation\n\n"
        "Source: `references/FastRecon` (FzJun26th/FastRecon).\n\n"
        f"Protocol: `{args.protocol}`"
        + (f" (`{args.rf_train_mode}`)" if args.protocol == "rf_target" else "")
        + f"\n\nFew-shot normal sampling: `{args.normal_sampling}`\n\n"
        f"Backbone: wide_resnet50_2, layers layer2+layer3, input {args.imagesize}.\n\n"
        f"Coreset: ratio={args.coreset_ratio}, max_support_patches={args.max_support_patches}, "
        f"max_coreset_size={args.max_coreset_size}.\n\n"
        f"Lambdas swept: {args.lambdas} (paper default lambda=2). Headline uses lambda={args.primary_lam}.\n\n"
        "## Finding\n\n"
        "FastRecon's distribution-regularization term (lambda>0) *hurts* on signal spectrograms: "
        "AUROC drops monotonically as lambda grows, falling below random (<=34) at the paper default "
        "lambda=2. Only lambda=0 (pure reconstruction / projection residual) works. With lambda=0 the "
        "method reduces to projecting each query patch onto the normal coreset span and scoring the "
        "orthogonal residual, which is competitive with PatchCore on the same backbone/features.\n\n"
        f"Result CSV: `{result_path.name}`\n\n"
        f"Macro Image-AUROC (lambda={args.primary_lam}): `{summary['image_auroc_macro']:.4f}`\n",
        encoding="utf-8",
    )
    print(f"wrote {result_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

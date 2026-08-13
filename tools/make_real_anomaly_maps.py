#!/usr/bin/env python
"""Produce REAL anomaly maps (patch-level ViT NN distance) for the 5 abnormal samples.

Pipeline (matches --paired-tta none, farthest 50%, top-k=5, concat, global_nn):
  1. Build normal gallery from the TTA-on support set (frequency_one_per_band, 24 normals).
  2. For each abnormal image, compute per-patch min-cosine-distance to the gallery
     (top-k=5 mean), reshape to grid, upsample to 256x256.
  3. Save as inferno heatmap PNGs on a COMMON color scale (no annotation).

Output: analysis_outputs/01_figures/paper_arch_tta/tta_view_samples/anomaly_maps/
"""
import os, sys, json, cv2, numpy as np, torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PromptAD import PromptAD
from tools.eval_cls_vit_patch_gallery import build_selected_train_loader
from tools.eval_cls_vit_patchcore_gallery import (
    build_vit_nn_gallery_with_rows,
    compute_patch_map,
    prepare_patch_features,
)
from train_rf_target_pooled_universal import (
    RFPathDataset, to_model_input,
)

# ---- the 5 abnormal PNGs (same as in tta_view_samples) ----
ABNORMAL = [
    ("burst_signal",   "/mnt/data/wangbei/data/datasets/burst/TimeSquare_spectrum/abnormal/m30db/TimeSquare_spectrum_burst_m30db_t00000-04000_f92.00-93.50MHz_patch006_abnormal.png"),
    ("chirp_signal",   "/mnt/data/wangbei/data/datasets/chirp/TimeSquare_spectrum/abnormal/m30db/TimeSquare_spectrum_chirp_m30db_t14000-18000_f105.60-107.10MHz_patch191_abnormal.png"),
    ("dsss_signal",    "/mnt/data/wangbei/data/datasets/dsss/TimeSquare_spectrum/abnormal/m30db/TimeSquare_spectrum_dsss_m30db_t10000-14000_f105.60-107.10MHz_patch143_abnormal.png"),
    ("pulse_signal",   "/mnt/data/wangbei/data/datasets/pulse/TimeSquare_spectrum/abnormal/m30db/TimeSquare_spectrum_pulse_m30db_t12000-16000_f100.00-101.50MHz_patch160_abnormal.png"),
    ("wideband_pulse", "/mnt/data/wangbei/data/datasets/wideband_pulse/TimeSquare_spectrum/abnormal/m30db/TimeSquare_spectrum_wideband_pulse_m30db_t04000-08000_f92.00-93.50MHz_patch054_abnormal.png"),
]

# ---- output dir ----
OUT_DIR = "analysis_outputs/01_figures/paper_arch_tta/tta_view_samples/anomaly_maps"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ---- 1. set up model + args matching TTA-on protocol ----
class Args:
    dataset = "rf_target_test_pool"
    class_name = "signal"
    normal_sampling = "frequency_one_per_band"
    coreset_method = "farthest"
    coreset_ratio = 0.5
    rowwise_coreset = False
    memory_mode = "global_nn"
    row_proto_zscore = False
    freq_zscore = False
    freq_window = -1
    nn_topk = 5
    nn_agg = "mean"
    nn_weight_temp = 0.05
    adaptive_sim_margin = 0.02
    position_soft_axis = "frequency"
    position_soft_weight = 0.0
    coherence_alpha = 0.0
    coherence_top_ratio = 0.05
    map_top_ratios = [0.01, 0.05, 0.1]
    batch_size = 64
    num_workers = 4
    gpu_id = 0
    seed = 111
    resolution = 400
    img_resize = 240
    img_cropsize = 240
    k_shot = 1
    backbone = "ViT-B-16-plus-240"
    pretrained_dataset = "laion400m_e32"
    prompt_mode = "rf"
    input_mode = "rgb"
    text_prototype_mode = "single"
    cls_score_mode = "text_only"
    n_ctx = 4; n_ctx_ab = 1; n_pro = 3; n_pro_ab = 4
    use_cpu = 0
    support_augment = "none"
    paired_tta = "none"
    checkpoint = ""
    gallery_chunk_size = 4096
args = Args()

# set CUDA device before creating model
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
device = "cuda:0"

# torch deterministic seed (parity with eval script)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

print(f"Loading PromptAD model (backbone={args.backbone}) on {device}...")
model = PromptAD(
    dataset=args.dataset, class_name=args.class_name, device=device,
    out_size_h=args.resolution, out_size_w=args.resolution,
    **{k: v for k, v in vars(args).items() if k not in {"dataset","class_name","device","out_size_h","out_size_w"}},
).to(device)

# ---- 2. build the real normal gallery (TTA-on protocol) ----
print("Building normal gallery (frequency_one_per_band, farthest 50% coreset)...")
train_loader, _train_samples = build_selected_train_loader(args)
gallery, gallery_rows, gallery_cols = build_vit_nn_gallery_with_rows(
    model, train_loader, args, device, paired_tta_mode="identity"
)
print(f"  gallery shape: {tuple(gallery.shape)}  (expected ~2700 patches)")

# ---- 3. compute anomaly map for each of the 5 abnormals ----
maps = []  # list of (sig, 256x256 float map)
for sig, abn_path in ABNORMAL:
    # wrap a single abnormal in RFPathDataset (gt=0 because we don't need it)
    samples = [(abn_path, 0, 1, "abnormal", f"{sig}-TimeSquare-m30db")]
    ds = RFPathDataset(samples)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    data, mask, label, name, img_type = next(iter(loader))
    # model input transform
    data_t = to_model_input(model, data, device, rgb_from_bgr=False)
    # patch-level anomaly map (paired_tta_mode=identity, no transform)
    map_t = compute_patch_map(
        model, data, gallery, gallery_rows, gallery_cols,
        args, device, paired_tta_mode="identity",
    )
    # map_t shape: (1, grid_h, grid_w); upsample to 256x256
    map_up = F.interpolate(map_t.unsqueeze(1), size=(256, 256), mode="bilinear", align_corners=False)
    map_np = map_up[0, 0].detach().cpu().numpy().astype(np.float32)  # (256,256)
    maps.append((sig, map_np))
    print(f"  {sig:18s}  map range: [{map_np.min():.4f}, {map_np.max():.4f}]  mean={map_np.mean():.4f}")

# ---- 4. global color scale, then save inferno heatmap PNGs ----
gmin = float(min(m.min() for _, m in maps))
gmax = float(max(m.max() for _, m in maps))
print(f"\nGlobal map range: [{gmin:.4f}, {gmax:.4f}]  -> applying inferno colormap with this scale")

# use matplotlib's inferno for correct colors
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
cmap = cm.get_cmap("inferno")

for sig, m in maps:
    # normalize to [0,1] with global scale, apply inferno (RGBA -> take RGB)
    norm = np.clip((m - gmin) / max(gmax - gmin, 1e-9), 0.0, 1.0)
    rgba = (cmap(norm)[:, :, :3] * 255.0).astype(np.uint8)  # H,W,3  (RGB)
    # convert RGB -> BGR for cv2.imwrite (so colors match matplotlib inferno when viewed)
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGB2BGR)
    out_path = os.path.join(OUT_DIR, f"{sig}.png")
    cv2.imwrite(out_path, bgr)
    print(f"  saved: {out_path}")

# also save a global-scale txt for reproducibility
with open(os.path.join(OUT_DIR, "color_scale.txt"), "w") as f:
    f.write(f"global_min={gmin:.6f}\nglobal_max={gmax:.6f}\ncolormap=inferno\n")
print(f"\nDone. 5 real anomaly maps + color_scale.txt in {OUT_DIR}/")

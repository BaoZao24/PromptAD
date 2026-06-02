"""
VAE Reconstruction Error + PromptAD Score Fusion Experiment
===========================================================
Minimal experiment: train VAE on normal spectrograms, compute reconstruction
error, and fuse with PromptAD scores.

Protocol (matches main approach):
  - Dataset: burst_signal, chirp_signal, dsss_signal
  - Split: normal_75_25 (first 75% normal as train, rest 25% normal + all abnormal as test)
  - Seed: 111
  - Baseline: PromptAD main (rf_morph_fusion_gray_residual_a01, text_only)

Output:
  - VAE-only AUC
  - PromptAD-only AUC
  - Fusion AUC (beta in [0.05, 0.1, 0.2])
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

# --- Config ---
DATA_BASE = '/mnt/data/wangbei/data/datasets'
PROMPTAD_RESULT_DIRS = [
    # Main approach first, then fallbacks (same features, slight variants)
    '/mnt/data/wangbei/PromptAD/result_split_ablation/rf_morph_fusion_gray_residual_a01_threeclass',
    '/mnt/data/wangbei/PromptAD/result_split_ablation/rf_morph_fusion_gray_residual_no_contrast_a01_threeclass',
]
OUTPUT_DIR = '/mnt/data/wangbei/PromptAD/experiments/vae_fusion'
SEED = 111
BATCH_SIZE = 8
VAE_EPOCHS = 100
VAE_LR = 1e-4
IMAGE_SIZE = 256
LATENT_DIM = 256
KL_WEIGHT = 0.1
FUSION_BETAS = [0.05, 0.1, 0.2]

SCENES = ['Gymnasium_spectrum', 'Playground_spectrum', 'TimeSquare_spectrum', 'WeaponMuseum_spectrum']
ANOMALY_TYPES = ['burst_signal', 'chirp_signal', 'dsss_signal']
NOISE_LEVELS = ['m10db', 'm20db', 'm30db']

DATASET_DIR_MAP = {
    'burst_signal': 'burst',
    'chirp_signal': 'chirp',
    'dsss_signal': 'dsss',
}


# --- VAE Model (same architecture as vae_ism_ano) ---
def basic_conv(inc, out_c):
    return nn.Sequential(nn.Conv2d(inc, out_c, 4, 2, 1), nn.LeakyReLU(0.2))


def basic_deconv(inc, out_c):
    return nn.Sequential(nn.ConvTranspose2d(inc, out_c, 4, 2, 1), nn.LeakyReLU(0.2))


class VAE(nn.Module):
    def __init__(self, input_size=256, in_channels=1, base_channels=32,
                 latent_dim=256, fc_dim=4096, min_spatial=4, decoder_fc_dim=2048):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        depth = int(np.log2(input_size // min_spatial))
        flat = base_channels * (2 ** depth)
        last_ch = base_channels * (2 ** (depth - 1))
        encoder_channels = [base_channels * (2 ** i) for i in range(depth)]

        self.encoder_blocks = nn.ModuleList()
        ch = in_channels
        for out_ch in encoder_channels:
            self.encoder_blocks.append(basic_conv(ch, out_ch))
            ch = out_ch
        self.encoder_out = nn.Conv2d(last_ch, flat, min_spatial, min_spatial, 0)

        self.fc1 = nn.Sequential(nn.Linear(flat, fc_dim), nn.LeakyReLU(0.2))
        self.to_mean = nn.Sequential(nn.Linear(fc_dim, latent_dim))
        self.to_log = nn.Sequential(nn.Linear(fc_dim, latent_dim))

        if decoder_fc_dim is None:
            self.from_bottle = nn.Sequential(nn.Linear(latent_dim, flat), nn.LeakyReLU(0.2))
        else:
            self.from_bottle = nn.Sequential(
                nn.Linear(latent_dim, decoder_fc_dim), nn.LeakyReLU(0.2),
                nn.Linear(decoder_fc_dim, flat), nn.LeakyReLU(0.2),
            )

        self.decoder_input = nn.ConvTranspose2d(flat, last_ch, min_spatial, min_spatial, 0)
        self.decoder_blocks = nn.ModuleList()
        decoder_in_channels = list(reversed(encoder_channels))
        for idx, in_ch in enumerate(decoder_in_channels):
            out_ch = decoder_in_channels[idx + 1] if idx + 1 < len(decoder_in_channels) else in_channels
            self.decoder_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                nn.LeakyReLU(0.2) if out_ch != in_channels else nn.Identity(),
            ))
        self.output_activation = nn.Sigmoid()
        self.flat = flat

    def resample(self, mean, logvar):
        std = logvar.mul(0.5).exp()
        eps = torch.randn_like(std)
        return eps.mul(std).add(mean)

    def forward(self, x):
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_out(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        mean = self.to_mean(x)
        logvar = self.to_log(x)
        z = self.resample(mean, logvar)
        y = self.from_bottle(z)
        y = y.reshape(-1, self.flat, 1, 1)
        y = self.decoder_input(y)
        for block in self.decoder_blocks:
            y = block(y)
        y = self.output_activation(y)
        return y, mean, logvar


def vae_loss(y, mean, logvar, x):
    mse = ((y - x) ** 2).sum()
    logvar_clamped = logvar.clamp(-10, 10)
    kld = 0.5 * (1 + logvar_clamped - mean ** 2 - logvar_clamped.exp()).sum()
    return mse - KL_WEIGHT * kld


# --- Data Loading ---
class PathDataset(Dataset):
    def __init__(self, img_paths):
        self.img_paths = list(img_paths)
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('L')
        return self.transform(img)


class PathLabelDataset(Dataset):
    """Dataset with paths and labels for evaluation."""
    def __init__(self, img_paths, labels):
        self.img_paths = list(img_paths)
        self.labels = list(labels)
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('L')
        return self.transform(img), self.labels[idx], self.img_paths[idx]


def split_normal_paths(normal_paths, train_ratio=0.75):
    """Same split logic as rf_split_utils._split_normal_paths."""
    n_total = len(normal_paths)
    if n_total <= 1:
        return normal_paths[:1], []
    n_train = int(n_total * train_ratio)
    n_train = max(1, min(n_train, n_total - 1))
    return normal_paths[:n_train], normal_paths[n_train:]


def get_data_paths(dataset_name, scene, noise_level):
    """Get train/test image paths using normal_75_25 split (same as PromptAD)."""
    dir_name = DATASET_DIR_MAP[dataset_name]
    dataset_root = os.path.join(DATA_BASE, dir_name)

    normal_root = os.path.join(dataset_root, scene, 'normal', noise_level)
    abnormal_root = os.path.join(dataset_root, scene, 'abnormal', noise_level)

    normal_paths = []
    if os.path.isdir(normal_root):
        normal_paths = sorted(str(p) for p in Path(normal_root).glob('*.png'))

    abnormal_paths = []
    if os.path.isdir(abnormal_root):
        abnormal_paths = sorted(str(p) for p in Path(abnormal_root).glob('*.png'))

    train_normal_paths, test_normal_paths = split_normal_paths(normal_paths, 0.75)

    test_img_paths = list(test_normal_paths) + list(abnormal_paths)
    test_labels = [0] * len(test_normal_paths) + [1] * len(abnormal_paths)

    return train_normal_paths, test_img_paths, test_labels


# --- Scoring ---
def compute_vae_scores(model, img_paths, device, batch_size=16):
    """Compute per-image MSE reconstruction error."""
    dataset = PathDataset(img_paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model.eval()
    all_scores = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device).float()
            output, _, _ = model(data)
            mse = ((output - data) ** 2).reshape(data.size(0), -1).sum(dim=1)
            all_scores.extend(mse.cpu().numpy().tolist())
    return np.array(all_scores)


def load_promptad_scores(dataset_name, scene, noise_level):
    """Load PromptAD scores from saved .npz file, searching multiple result dirs."""
    for result_base in PROMPTAD_RESULT_DIRS:
        npz_path = os.path.join(
            result_base, dataset_name, scene, noise_level,
            'normal_75_25', 'k_1', 'scores', 'Seed_111-image_scores.npz'
        )
        if os.path.exists(npz_path):
            data = np.load(npz_path, allow_pickle=True)
            return data['names'], data['scores'], data['labels']
    return None, None, None


def normalize_scores(scores):
    """Min-max normalize to [0, 1]."""
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-8:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


def compute_auc(labels, scores):
    """Compute ROC-AUC. Labels: 0=normal, 1=abnormal. Higher score = more abnormal."""
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


# --- Training ---
def train_vae(model, train_paths, device, epochs=100, batch_size=BATCH_SIZE):
    """Train VAE on normal images only."""
    dataset = PathDataset(train_paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    optimizer = Adam(model.parameters(), lr=VAE_LR)
    best_loss = float('inf')

    model.train()
    for epoch in range(epochs):
        epoch_losses = []
        for data in loader:
            data = data.to(device).float()
            optimizer.zero_grad()
            output, mean, logvar = model(data)
            loss = vae_loss(output, mean, logvar, data)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = float(sum(epoch_losses) / len(epoch_losses))
        if avg_loss < best_loss:
            best_loss = avg_loss
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")

    return best_loss


# --- Main Experiment ---
def run_experiment(dataset_name, scene, noise_level, device):
    """Run one VAE training + evaluation + fusion experiment."""
    print(f"\n{'='*60}")
    print(f"Experiment: {dataset_name}/{scene}/{noise_level}")
    print(f"{'='*60}")

    # Get data paths
    train_paths, test_paths, test_labels = get_data_paths(dataset_name, scene, noise_level)
    if len(train_paths) < 2:
        print(f"  SKIP: only {len(train_paths)} training images")
        return None
    print(f"  Train normal: {len(train_paths)}, Test: {len(test_paths)} "
          f"(normal={test_labels.count(0)}, abnormal={test_labels.count(1)})")

    # Load PromptAD scores
    pa_names, pa_scores, pa_labels = load_promptad_scores(dataset_name, scene, noise_level)
    if pa_names is None:
        print(f"  SKIP: no PromptAD scores found")
        return None

    # Build path -> PromptAD score mapping
    pa_score_map = {}
    pa_label_map = {}
    for i, name in enumerate(pa_names):
        pa_score_map[str(name)] = float(pa_scores[i])
        pa_label_map[str(name)] = int(pa_labels[i])

    # Verify PromptAD test images match our test paths
    # (The .npz names include scene prefix; extract actual path)
    pa_basenames = set()
    for name in pa_names:
        # name format: "scene-scene_normal-filename_normal" or "scene-scene_abnormal-filename_abnormal"
        basename = str(name).split('-')[-1]  # Get filename part
        pa_basenames.add(basename)

    test_basenames = set(os.path.basename(p) for p in test_paths)
    overlap = pa_basenames & test_basenames
    if len(overlap) < 1:
        # Try matching by full path components
        print(f"  WARNING: basename overlap={len(overlap)}, trying path matching...")
        # Try to match by reconstructing paths from .npz names
        pass

    # Rebuild aligned test paths from .npz names
    # .npz name format: "{scene}-{scene}_{label}-{actual_filename_without_ext}"
    # e.g. "Playground_spectrum-Playground_spectrum_normal-Playground_spectrum_burst_m10db_t14000-18000_f89.60-91.10MHz_patch171_normal"
    #   -> file: "Playground_spectrum_burst_m10db_t14000-18000_f89.60-91.10MHz_patch171_normal.png"
    dir_name = DATASET_DIR_MAP[dataset_name]
    normal_root_dir = os.path.join(DATA_BASE, dir_name, scene, 'normal', noise_level)
    abnormal_root_dir = os.path.join(DATA_BASE, dir_name, scene, 'abnormal', noise_level)

    aligned_paths = []
    aligned_pa_scores = []
    aligned_labels = []
    missing = 0

    for i, name in enumerate(pa_names):
        name_str = str(name)

        label_type = 'abnormal' if int(pa_labels[i]) == 1 else 'normal'
        prefix = f"{scene}-{scene}_{label_type}-"
        if name_str.startswith(prefix):
            filename = name_str[len(prefix):] + '.png'
        else:
            missing += 1
            continue

        if label_type == 'normal':
            img_path = os.path.join(normal_root_dir, filename)
        else:
            img_path = os.path.join(abnormal_root_dir, filename)

        if os.path.exists(img_path):
            aligned_paths.append(img_path)
            aligned_pa_scores.append(float(pa_scores[i]))
            aligned_labels.append(int(pa_labels[i]))
        else:
            missing += 1

    if missing > 0:
        print(f"  WARNING: {missing}/{len(pa_names)} images not found on disk")

    if len(aligned_paths) < 10:
        print(f"  SKIP: too few aligned images ({len(aligned_paths)})")
        return None

    print(f"  Aligned images: {len(aligned_paths)} "
          f"(normal={aligned_labels.count(0)}, abnormal={aligned_labels.count(1)})")

    aligned_pa_scores = np.array(aligned_pa_scores)
    aligned_labels = np.array(aligned_labels)

    # Train VAE
    print(f"  Training VAE ({VAE_EPOCHS} epochs)...")
    vae = VAE(input_size=IMAGE_SIZE, latent_dim=LATENT_DIM).to(device)
    t0 = time.time()
    best_loss = train_vae(vae, train_paths, device, epochs=VAE_EPOCHS)
    train_time = time.time() - t0
    print(f"  Training done in {train_time:.0f}s, best_loss={best_loss:.4f}")

    # Compute VAE scores on test images
    print(f"  Computing VAE scores on {len(aligned_paths)} test images...")
    vae_scores = compute_vae_scores(vae, aligned_paths, device)
    vae_scores_norm = normalize_scores(vae_scores)

    # Compute PromptAD AUC (baseline)
    pa_auc = compute_auc(aligned_labels, aligned_pa_scores)

    # Compute VAE-only AUC
    vae_auc = compute_auc(aligned_labels, vae_scores_norm)

    # Fusion
    fusion_results = {}
    for beta in FUSION_BETAS:
        fused_scores = aligned_pa_scores + beta * vae_scores_norm
        fused_auc = compute_auc(aligned_labels, fused_scores)
        fusion_results[f'beta_{beta}'] = fused_auc

    # Report
    result = {
        'dataset': dataset_name,
        'scene': scene,
        'noise_level': noise_level,
        'n_train': len(train_paths),
        'n_test_normal': int((aligned_labels == 0).sum()),
        'n_test_abnormal': int((aligned_labels == 1).sum()),
        'vae_train_loss': best_loss,
        'vae_train_time_s': train_time,
        'vae_auc': vae_auc,
        'promptad_auc': pa_auc,
        'fusion': fusion_results,
        'best_fusion_beta': max(fusion_results, key=fusion_results.get),
        'best_fusion_auc': max(fusion_results.values()),
    }

    print(f"  Results:")
    print(f"    VAE-only AUC:       {vae_auc:.4f}")
    print(f"    PromptAD-only AUC:  {pa_auc:.4f}")
    for beta, auc in fusion_results.items():
        delta = auc - pa_auc
        sign = '+' if delta >= 0 else ''
        print(f"    Fusion {beta}:      {auc:.4f} ({sign}{delta:.4f} vs PromptAD)")
    best_beta = result['best_fusion_beta']
    best_auc = result['best_fusion_auc']
    delta_best = best_auc - pa_auc
    if delta_best > 0:
        print(f"    >>> BEST: {best_beta} improves PromptAD by +{delta_best:.4f}")
    else:
        print(f"    >>> No improvement over PromptAD-only")

    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Set seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    all_results = []

    for dataset_name in ANOMALY_TYPES:
        for scene in SCENES:
            for noise_level in NOISE_LEVELS:
                result = run_experiment(dataset_name, scene, noise_level, device)
                if result is not None:
                    all_results.append(result)

    # --- Aggregate Summary ---
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")

    # Per anomaly type summary
    for dataset_name in ANOMALY_TYPES:
        subset = [r for r in all_results if r['dataset'] == dataset_name]
        if not subset:
            continue
        pa_aucs = [r['promptad_auc'] for r in subset]
        vae_aucs = [r['vae_auc'] for r in subset]
        best_fusion_aucs = [r['best_fusion_auc'] for r in subset]

        avg_pa = np.mean(pa_aucs)
        avg_vae = np.mean(vae_aucs)
        avg_fusion = np.mean(best_fusion_aucs)

        print(f"\n{dataset_name} ({len(subset)} experiments):")
        print(f"  PromptAD avg AUC:     {avg_pa:.4f}")
        print(f"  VAE avg AUC:          {avg_vae:.4f}")
        print(f"  Best Fusion avg AUC:  {avg_fusion:.4f}")

        # Per noise level
        for nl in NOISE_LEVELS:
            nl_subset = [r for r in subset if r['noise_level'] == nl]
            if not nl_subset:
                continue
            pa_nl = np.mean([r['promptad_auc'] for r in nl_subset])
            vae_nl = np.mean([r['vae_auc'] for r in nl_subset])
            fusion_nl = np.mean([r['best_fusion_auc'] for r in nl_subset])
            improved = sum(1 for r in nl_subset if r['best_fusion_auc'] > r['promptad_auc'])
            print(f"    {nl}: PromptAD={pa_nl:.4f}, VAE={vae_nl:.4f}, "
                  f"Fusion={fusion_nl:.4f} (improved {improved}/{len(nl_subset)})")

    # Overall
    pa_all = np.mean([r['promptad_auc'] for r in all_results])
    vae_all = np.mean([r['vae_auc'] for r in all_results])
    fusion_all = np.mean([r['best_fusion_auc'] for r in all_results])
    improved_count = sum(1 for r in all_results if r['best_fusion_auc'] > r['promptad_auc'])
    print(f"\nOverall ({len(all_results)} experiments):")
    print(f"  PromptAD avg AUC:     {pa_all:.4f}")
    print(f"  VAE avg AUC:          {vae_all:.4f}")
    print(f"  Best Fusion avg AUC:  {fusion_all:.4f}")
    print(f"  Improved:             {improved_count}/{len(all_results)}")

    # Output detailed results JSON
    output_path = os.path.join(OUTPUT_DIR, 'fusion_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {output_path}")

    # Summary CSV
    csv_path = os.path.join(OUTPUT_DIR, 'fusion_summary.csv')
    with open(csv_path, 'w') as f:
        f.write('dataset,scene,noise_level,n_train,n_test_normal,n_test_abnormal,'
                'vae_auc,promptad_auc,fusion_best_auc,best_beta,delta\n')
        for r in all_results:
            delta = r['best_fusion_auc'] - r['promptad_auc']
            f.write(f"{r['dataset']},{r['scene']},{r['noise_level']},"
                    f"{r['n_train']},{r['n_test_normal']},{r['n_test_abnormal']},"
                    f"{r['vae_auc']:.4f},{r['promptad_auc']:.4f},"
                    f"{r['best_fusion_auc']:.4f},{r['best_fusion_beta']},{delta:+.4f}\n")
    print(f"CSV summary saved to: {csv_path}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Extract normal visual maps for band-aware calibration."""
import os, sys
sys.path.insert(0, '.')
import torch, numpy as np
from PIL import Image
from tqdm import tqdm
os.environ.setdefault('OMP_NUM_THREADS', '1')
torch.set_num_threads(1)

from PromptAD import PromptAD
from datasets.dataset import CLIPDataset
from datasets import load_function_dict
from torch.utils.data import DataLoader

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

DS = args.dataset
DEVICE = f'cuda:{args.gpu}'
CHECKPOINT = f'./result/{DS}/Playground_spectrum/m30db/normal_75_25/k_1/checkpoint/CLS-Seed_111-Playground_spectrum-check_point.pt'

model = PromptAD(
    out_size_h=400, out_size_w=400, device=DEVICE,
    backbone='ViT-B-16-plus-240', pretrained_dataset='laion400m_e32',
    n_ctx=4, n_pro=3, n_ctx_ab=1, n_pro_ab=4,
    class_name='Playground_spectrum',
    k_shot=1, dataset=DS, prompt_mode='rf',
    input_mode='morph_fusion_gray_residual_a01',
    img_resize=240, img_cropsize=240, seed=111,
    cls_score_mode='text_only', text_prototype_mode='single',
    visual_adapter=False, visual_lora=False,
)
model = model.to(DEVICE)
model.eval_mode()
model.eval()
sd = torch.load(CHECKPOINT, map_location='cpu')
matched = {k: v for k, v in sd.items() if 'prompt_learner' in k}
model.load_state_dict(matched, strict=False)
for k in ['feature_gallery1', 'feature_gallery2']:
    if k in sd:
        setattr(model, k, sd[k].to(DEVICE))

dataset_inst = CLIPDataset(
    load_function=load_function_dict[DS],
    category='Playground_spectrum', phase='train', k_shot=1,
    noise_level='m30db', split_mode='normal_75_25', normal_train_ratio=0.75,
)
loader = DataLoader(dataset_inst, batch_size=400, shuffle=False, num_workers=0)
all_maps = []
with torch.no_grad():
    for data, mask, label, name, img_type in tqdm(loader, desc=f'{DS} normal'):
        data_t = [model.transform(Image.fromarray(f.numpy())) for f in data]
        data_t = torch.stack(data_t, dim=0).to(DEVICE)
        vf = model.encode_image(data_t)
        vm = model.calculate_visual_anomaly_score(vf)
        all_maps.append(vm.cpu().numpy())
maps = np.concatenate(all_maps, axis=0)
base = os.path.basename(DS).replace('_signal', '')
np.save(f'experiments/band_aware_scoring/normal_maps_{DS}_Playground_m30db.npy', maps)
print(f'{DS}: {maps.shape[0]} normal maps saved.')

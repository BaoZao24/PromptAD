#!/usr/bin/env bash
set -e

if ! command -v conda >/dev/null 2>&1; then
	echo "conda not found. Please initialize conda first, for example: source ~/miniconda3/etc/profile.d/conda.sh"
	exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx 'prompt_ad'; then
	conda create -n prompt_ad python=3.10 -y
fi

conda activate prompt_ad

# PyTorch
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python -m pip install setuptools==59.5.0
python -m pip install --upgrade diffusers[torch]
python -m pip install opencv-python pycocotools matplotlib ipykernel
python -m pip install transformers
python -m pip install addict
python -m pip install yapf
python -m pip install timm
python -m pip install loguru
python -m pip install tqdm
python -m pip install scikit-image
python -m pip install scikit-learn
python -m pip install pandas
python -m pip install tensorboard
python -m pip install seaborn
python -m pip install open_clip_torch
python -m pip install SciencePlots

# visa datasets
# cd ..
# python datasets/prepare_visa_public.py


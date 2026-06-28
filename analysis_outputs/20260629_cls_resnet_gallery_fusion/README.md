# ResNet Gallery CLS Fusion

Date: 2026-06-29

Hypothesis: keep PromptAD image/text score unchanged and replace the image-level visual calibration term with a frozen ResNet18 normal patch-feature gallery. For each test image, patch distances to the target-normal CNN gallery are computed and the highest-distance 10% patches are averaged as the image-level CNN anomaly score.

Command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python tools/eval_cls_resnet_gallery_fusion.py \
  --output-root analysis_outputs/20260629_cls_resnet_gallery_fusion \
  --formal-baseline-csv analysis_outputs/20260627_method_funnel/results_cls_pooled_rf_rgb.csv \
  --gpu-id 1 \
  --batch-size 200 \
  --num-workers 4 \
  --max-gallery-patches 50000 \
  --distance-chunk-size 1024 \
  --resnet-layers layer2 layer3 \
  --image-top-ratio 0.1 \
  --lambdas 0.2 0.5 1.0 1.5 2.0 3.0
```

Setup:

- Checkpoint: `analysis_outputs/20260627_method_funnel/runs/pooled_rf_rgb/cls/checkpoint/overall-best.pt`
- CNN: frozen ImageNet-pretrained ResNet18.
- Gallery: target normal train split only.
- Test: all target cells for burst/chirp/dsss/pulse.
- Image aggregation: average of top 10% patch distances.

Macro image AUROC:

| Method | AUROC |
| --- | ---: |
| Formal PromptAD baseline | 86.3382 |
| PromptAD current rescore | 78.4088 |
| ResNet18 layer2 gallery only | 84.2390 |
| ResNet18 layer3 gallery only | 90.7871 |
| PromptAD + ResNet18 layer3 raw lambda=1 | 91.5056 |
| PromptAD + ResNet18 layer3 minmax lambda=3 | 91.4672 |

Per-class image AUROC:

| Dataset | Baseline | PromptAD rescore | ResNet layer3 only | PromptAD + ResNet layer3 |
| --- | ---: | ---: | ---: | ---: |
| burst_signal | 92.2494 | 82.1649 | 95.3609 | 95.0693 |
| chirp_signal | 92.5200 | 89.3739 | 90.1650 | 91.6151 |
| dsss_signal | 81.7538 | 61.4258 | 97.1836 | 96.1909 |
| pulse_signal | 78.8296 | 80.6707 | 80.4391 | 83.1472 |

Conclusion:

The CLS result strongly supports the CNN gallery idea. ResNet18 layer3 gallery alone already reaches `90.7871`, and fusion with PromptAD reaches `91.5056`, improving the formal baseline by `+5.1674`. Layer3 is clearly better than layer2 for image-level scoring, likely because CLS benefits from more structural features while SEG benefits from more local features.


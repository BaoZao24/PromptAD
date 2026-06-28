# ResNet Gallery SEG Fusion

Date: 2026-06-29

Hypothesis: keep PromptAD textual anomaly maps unchanged, replace the visual normal-distance branch with a frozen ResNet18 local-feature gallery. If CNN local features capture RF spectral morphology better than CLIP patch features, `text_map * (1 + beta * cnn_distance_map)` should improve pixel AUROC.

Command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python tools/eval_seg_resnet_gallery_fusion.py \
  --output-root analysis_outputs/20260629_seg_resnet_gallery_fusion \
  --formal-baseline-csv analysis_outputs/20260627_method_funnel/results_seg_pooled_rf_rgb.csv \
  --gpu-id 1 \
  --batch-size 100 \
  --num-workers 4 \
  --max-gallery-patches 50000 \
  --distance-chunk-size 1024 \
  --resnet-layers layer2 layer3 \
  --betas 0.25 0.5 1.0 2.0
```

Setup:

- Checkpoint: `analysis_outputs/20260627_method_funnel/runs/pooled_rf_rgb/seg/checkpoint/overall-best.pt`
- CNN: frozen ImageNet-pretrained ResNet18.
- Gallery: target normal train split only.
- Test: all target cells for burst/chirp/dsss/pulse.
- Fusion: `final_map = textual_map * (1 + beta * minmax(cnn_gallery_distance_map))`.

Macro pROC:

| Method | pROC |
| --- | ---: |
| Formal PromptAD baseline | 89.3342 |
| Textual only | 89.3921 |
| CLIP gallery only | 89.1617 |
| Text + CLIP gallery, beta=0.25 | 91.5322 |
| ResNet18 layer2 gallery only | 91.0585 |
| Text + ResNet18 layer2 gallery, beta=0.25 | 91.5577 |
| Text + ResNet18 layer3 gallery, beta=0.25 | 91.5564 |

Per-class pROC:

| Dataset | Baseline | Text+CLIP | ResNet layer2 only | Text+ResNet layer2 |
| --- | ---: | ---: | ---: | ---: |
| burst_signal | 98.0407 | 98.1705 | 96.8841 | 98.0316 |
| chirp_signal | 95.0768 | 96.1052 | 95.5554 | 96.6795 |
| dsss_signal | 72.6408 | 76.0434 | 79.3896 | 77.9449 |
| pulse_signal | 91.5786 | 95.8096 | 92.4050 | 93.5749 |

Conclusion:

The ResNet18 gallery branch supports the hypothesis, but only slightly at the macro level. It improves the best macro pROC from `91.5322` to `91.5577`. The useful signal is strongest on dsss and chirp; pulse is worse than the CLIP-gallery gate. This suggests CNN local morphology is helpful, but a universal replacement for CLIP gallery is not yet clearly better.


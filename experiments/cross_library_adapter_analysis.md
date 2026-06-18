# Cross-Library Adapter Experiment Analysis

Date: 2026-06-18

## Background

This experiment evaluates whether lightweight adapters can improve RF anomaly detection under cross-library transfer. The key question is whether source-library supervised signals can help PromptAD generalize better to a different target library, without using target abnormal samples during training.

The tested adapters are:

- `VCPA` / visual class prompt adapter: maps the normal visual prototype into soft class-name prompt tokens.
- `visual_adapter`: residual adapter applied to CLIP visual features.
- `VCPA + visual_adapter`: both adapters enabled.

The current main RF setting is:

```text
prompt_mode = rf
input_mode = morph_fusion_gray_residual_a01
cls_score_mode = text_only
split_mode = normal_75_25
seed = 111
k-shot = 1
noise_level = m10db
```

## Cross-Library Protocol

The protocol separates source supervision from target evaluation:

1. Source train split is used to build the source normal gallery.
2. Source test split, containing normal and abnormal samples, is used for supervised training.
3. Target train split is used only to build the target normal gallery.
4. Target test split is used only for evaluation.
5. Target abnormal samples are never used during training.

This setting tests cross-library transfer rather than ordinary in-domain few-shot adaptation.

## 50-Epoch Ablation Results

The longer 50-epoch run compared four settings on three difficult cross-library pairs.

| Source -> Target | Scene | Base | VCPA only | Visual adapter only | VCPA + visual adapter |
|---|---|---:|---:|---:|---:|
| `burst_signal -> chirp_signal` | `Gymnasium_spectrum` | 83.69 | 83.76 | 83.69 | 83.76 |
| `dsss_signal -> burst_signal` | `WeaponMuseum_spectrum` | 94.11 | 94.33 | 94.11 | 94.28 |
| `dsss_signal -> burst_signal` | `Gymnasium_spectrum` | 93.48 | 93.98 | 93.48 | 93.94 |
| Mean | - | 90.43 | 90.69 | 90.43 | 90.66 |

Relative to the baseline:

| Method | Mean AUROC | Mean delta |
|---|---:|---:|
| Base | 90.43 | +0.00 |
| VCPA only | 90.69 | +0.26 |
| Visual adapter only | 90.43 | +0.00 |
| VCPA + visual adapter | 90.66 | +0.23 |

## Main Interpretation

The hypothesis is partially supported.

VCPA shows a small but consistent gain across all three difficult pairs. This suggests that injecting a normal visual prototype into the prompt side does carry useful cross-library information. However, the gain is modest, indicating that the current VCPA design only weakly changes the final anomaly score.

The visual adapter alone does not improve performance in this setting. Under `cls_score_mode=text_only`, the final image score mainly depends on image-text similarity against normal and abnormal text prototypes. A residual visual feature adapter can affect the encoded image feature, but it is not directly supervised through a final visual-distance score. In practice, it appears to stay close to the baseline behavior.

The combined setting remains better than baseline but slightly worse than VCPA-only on two of the three pairs. This means the current visual adapter does not add complementary information to VCPA under the present scoring protocol.

## Why the Gain Is Small

Several factors limit the current adapter effect:

1. The final scoring path is narrow.
   - With `cls_score_mode=text_only`, VCPA only influences prompt embeddings.
   - It does not directly reshape the visual anomaly map or visual gallery distance.

2. VCPA uses only a mean normal prototype.
   - A single averaged normal vector loses intra-class normal variation.
   - Cross-library differences may be distributional, not representable by one mean vector.

3. Target normal gallery already absorbs part of the domain shift.
   - During evaluation, the target normal gallery is rebuilt from target normal samples.
   - This already adapts the visual reference distribution, leaving less room for the adapter.

4. Source supervision may not transfer perfectly.
   - The adapter is trained with source abnormal labels.
   - Source abnormal boundaries may not match the target library's abnormal patterns.

5. The visual adapter is not matched to the current score mode.
   - It is more likely to help when visual-distance or final-score supervision participates in training.
   - In text-only scoring, its effect is indirect and weak.

## Can More Epochs Help?

More epochs can help, but the expected gain is limited.

The 50-epoch run already shows that VCPA needs sufficient training to reveal its effect. Continuing to 100 or 150 epochs may produce small additional gains if the target AUROC curve is still rising. However, because training uses source labels, longer training may also overfit the source domain and reduce cross-library transfer.

A better next experiment is to record per-epoch target AUROC curves for `base` and `VCPA only`. If VCPA is still improving at epoch 50, longer training is justified. If the curve has plateaued, changing the training objective or adapter design is more important than increasing epochs.

## Can Larger Networks Help?

Increasing VCPA capacity may help, but it should be tested conservatively.

Recommended order:

1. Increase `visual_class_token_num`.
   - Current value: `2`.
   - Suggested values: `4`, `8`.
   - This directly increases the number of prompt tokens available to express visual-domain information.

2. Increase `visual_class_prompt_bottleneck_ratio`.
   - Current value: `0.25`.
   - Suggested values: `0.5`, `1.0`.
   - This increases adapter capacity without changing the prompt length.

3. Increase `visual_class_prompt_alpha`.
   - Current value: `0.2`.
   - Suggested values: `0.5`, `1.0`.
   - This strengthens the residual prompt injection, but too large a value may disturb CLIP's text embedding space.

4. Avoid immediately deepening the MLP.
   - A deeper adapter may overfit source labels.
   - Token count and residual strength are simpler and more interpretable first steps.

## More Promising Improvements

The most promising next changes are not only larger networks or more epochs, but stronger alignment between the adapter and the final anomaly score.

Recommended directions:

1. Add final-score supervision.
   - Train with BCE or ranking loss on the final `score_img`, not only image-text classification logits.
   - This directly optimizes the quantity used by evaluation.

2. Use visual-aware scoring when testing the visual adapter.
   - Evaluate settings where visual anomaly score participates in the final score.
   - This is necessary to fairly test whether the visual adapter is useful.

3. Replace mean prototype with richer normal statistics.
   - Use multiple normal prototypes, top-k normal prototypes, or covariance-aware normal distribution summaries.
   - This may represent cross-library normal variation better than one averaged vector.

4. Add source-target normal alignment.
   - Use source normal and target normal samples together to constrain the adapter.
   - This can reduce domain shift without using target abnormal labels.

5. Run a full matrix validation.
   - First compare `base` vs `VCPA only` on the full 12-case matrix.
   - Then add multiple seeds to verify statistical stability.

## Current Conclusion

The current evidence supports VCPA as a valid but modest cross-library improvement. It is not yet a strong standalone contribution by magnitude, but it reveals a useful direction: normal visual prototypes can improve text-side adaptation.

The current visual adapter is not validated under `text_only` scoring. To test it fairly, the training and evaluation protocol should include visual-score or final-score supervision.

The next practical step is to run a focused `base` vs `VCPA only` full-matrix experiment with larger VCPA token counts, while separately designing a visual-score-supervised experiment for the visual adapter.

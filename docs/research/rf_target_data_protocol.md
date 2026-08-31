# self-RF 训练/测试数据协议

In-house RF 的正常背景来自本项目自测/自采集的 RF 记录；异常样本是在这些记录上注入
合成干扰后生成，并由项目同时生成对应 ground truth。因此该数据集属于本项目自测的
派生数据集，不是外部公开数据集。

正式主比较的配置由 `datasets/rf_target.py` 固定：场景为
`WeaponMuseum_spectrum`、`Playground_spectrum`、`TimeSquare_spectrum` 和
`Gymnasium_spectrum`；注入类型为 `burst_signal`、`chirp_signal`、`dsss_signal`、
`pulse_signal` 和 `deceptive_signal`。其中前三类使用 `m10db/m20db/m30db`，pulse 正式
使用 `m20db/m30db/m40db`，deceptive 使用 `strong/medium/weak`，并按场景映射到对应的
存储强度（Gymnasium 为 `0db/10db/20db`）。因此正式协议是 4 个场景 × 5 类注入 × 3 个
强度单元，共 60 个评估单元；`wideband_pulse` 不属于当前正式五类主比较。

四个场景的原始正常数组均覆盖 88–108 MHz、20001 个频率 bin；TF 切片使用 4000 帧窗口、
2000 帧步长、1.5 MHz 频率窗口和 0.8 MHz 频率步长，输出 256×256 PNG。论文按原始观测
计数：每个场景取1个宽带正常观测（1-shot），再因网络输入尺寸限制沿频率轴裁切为24个
输入块，并复用于该场景的所有signal/strength单元。这24个输入块不属于24个独立shot。
代码中的`per_frequency`仅是裁切块与清单的内部组织名称。

## 当前唯一正式口径

self-RF 没有单独的 `/datasets/normal/{scene}` 训练目录。真实数据都位于：

```text
/mnt/data/wangbei/data/datasets/
└── {signal}/{scene}/
    ├── normal/{jsr}/*.png
    ├── abnormal/{jsr}/*.png
    └── groundtruth/{jsr}/*.png
```

其中 `signal` 和 `jsr` 表示注入的异常条件，正常背景属于 `scene`。因此：

1. 每个场景单独建立一套正常 support；
2. 同一场景下的所有 signal/JSR 单元共用这套 support；
3. 不同场景不共用 support；
4. 不做 full-shot。

## 时间切分

频谱图的时间窗长度为 4000，步长为 2000，相邻窗口会重叠。正式协议规定：

- support 候选：仅正常图 `t=0–4000`；
- test normal：仅 `t_start >= 4000`；
- test abnormal：仅 `t_start >= 4000`；
- `t=2000–6000` 与 support 重叠，因此不进入测试。

区间按半开区间理解，所以 `[0, 4000)` 与 `[4000, 8000)` 不重叠。

当前限制：每个场景现有数据仍来自同一段原始 recording。本协议消除了 support/test
的时间重叠，但不是 recording-level holdout。若论文要求更严格的跨 recording
泛化，需要为每个场景补采至少一段独立正常 recording。

## 少样本定义

- 论文正式口径：每个场景1个宽带正常观测，即1-shot；
- `per_frequency`：上述宽带观测裁切得到的24个频率窗口，是内部输入组织方式，不单独计为shot；
- `1shot`、`2shot`、`4shot`：每个场景分别选择 1、2、4 张；
- 所有选择先按解码后像素哈希去重；
- support 与正常/异常 test 都由同一个 manifest 固定；
- ViT、CNN 和置信度融合必须记录并核对同一个 manifest SHA-256。

## 代码入口

- 数据路径与兼容加载器：`datasets/rf_target.py`
- support/test 清单：`utils/rf_scene_support.py`
- 清单生成：`tools/build_rf_target_scene_support_manifest.py`
- ViT 证据：`tools/eval_cls_vit_patchcore_gallery.py`
- Ours 辅助 CNN 证据：`tools/eval_cls_aux_cnn_gallery.py`
- 置信度融合：`tools/eval_cls_dual_visual_evidence_fusion.py`
- 独立 PatchCore baseline：`tools/eval_patchcore_cls.py`

旧的逐信号加载器仍保留原函数名，供旧命令兼容，但内部统一调用
`datasets/rf_target.py`。正式论文实验使用 `target_scene` / `per_scene`。

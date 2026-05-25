# signal_adaptive 小白说明

## 1. 一句话解释

`signal_adaptive` 的意思是：

> 根据不同的信号异常类型，自动选择更合适的输入图片形式。

它不是一个新的模型，也不是一个新的 prompt。它只是告诉程序：

- 遇到 `burst_signal` 或 `chirp_signal`，用 `spectral_gradient` 输入；
- 遇到 `dsss_signal`，用 `rgb` 输入；
- 其他频谱类数据，默认多用 `spectral_gradient`；
- 普通视觉数据，仍然用 `rgb`。

在代码里，它是 `--input-mode` 的一个选项：

```bash
--input-mode signal_adaptive
```

## 2. 为什么要有 input-mode

PromptAD 原本是做工业图片异常检测的，比如瓶子、螺丝、木板缺陷。

这些图片一般是普通 RGB 图，也就是常见的三通道彩色图片：

```text
R 通道：红色信息
G 通道：绿色信息
B 通道：蓝色信息
```

但是我们现在做的是频谱异常检测。频谱图不是普通照片，它更像一张“信号地图”：

```text
横轴：时间
纵轴：频率
颜色/亮度：信号强度
```

所以如果直接把频谱图当成普通 RGB 图片，模型可能看不到最重要的信息。

因此项目里加了几种输入模式：

| 输入模式 | 意思 |
|---|---|
| `rgb` | 把频谱图当普通彩色图输入 |
| `spectral_gradient` | 把频谱图改成“强度 + 时间变化 + 频率变化”三通道 |
| `auto` | 程序按数据集自动选择 |
| `signal_adaptive` | 按信号异常类型自动选择 |

## 3. rgb 是什么

`rgb` 就是最普通的图像输入方式。

可以理解成：

> 模型看到的是原始频谱图长什么样。

优点：

- 保留原图的整体纹理；
- 对宽带、大片、连续的频谱结构比较友好；
- 和 CLIP 原始训练时的图片格式更接近。

缺点：

- 对很弱、很细、很短的异常信号不一定敏感；
- 容易把频谱图当成普通图片纹理，而不是信号结构。

## 4. spectral_gradient 是什么

`spectral_gradient` 可以理解成：

> 不只看原图，还看频谱图在时间方向和频率方向上哪里变化明显。

它把一张频谱图变成三个通道：

```text
第 1 个通道：原始灰度强度
第 2 个通道：时间方向变化
第 3 个通道：频率方向变化
```

更直白一点：

- 原始灰度强度：哪里信号强；
- 时间方向变化：信号有没有突然出现、突然消失；
- 频率方向变化：信号在频率上有没有明显边界。

这对频谱异常很有用，因为很多异常不是“颜色不一样”，而是“形状和变化不一样”。

## 5. 为什么 burst 和 chirp 用 spectral_gradient

### burst 的特点

`burst` 通常是短时间突然出现的窄带信号。

它像这样：

```text
时间上：很短，突然冒出来
频率上：比较窄，集中在某个频段
```

所以它的关键特征是：

- 突然出现；
- 持续时间短；
- 边界明显；
- 可能很弱，不容易直接看出来。

`spectral_gradient` 会突出“突然变化”和“边界”，所以更适合 burst。

### chirp 的特点

`chirp` 通常是扫频信号，在时频图上像一条斜线。

它像这样：

```text
随着时间变化，频率也在变化
在频谱图上形成一条斜着的轨迹
```

所以它的关键特征是：

- 斜线；
- 扫频轨迹；
- 时间和频率同时变化；
- 形状结构很重要。

`spectral_gradient` 能更明显地突出这种斜线和边缘，所以更适合 chirp。

## 6. 为什么 dsss 现在用 rgb

`dsss` 是扩频类信号，通常不是一条很细的线，也不是一个很短的脉冲。

它更像一个宽带区域：

```text
频率范围：比较宽
能量分布：可能比较散
纹理：像一片宽带噪声结构
```

对 dsss 来说，重要的可能不是某一条清晰边缘，而是整体宽带纹理和能量分布。

如果强行用 `spectral_gradient`，模型会更关注边缘和变化，反而可能破坏 dsss 的整体宽带纹理。

所以当前代码里选择：

```text
dsss_signal -> rgb
```

这不是说 `rgb` 永远比 `spectral_gradient` 适合 dsss，而是当前实验和实现下，dsss 保留原图 RGB 更稳。

## 7. signal_adaptive 到底做了什么

当前代码逻辑可以简化成：

```python
if input_mode == "signal_adaptive":
    if dataset_name in {"burst_signal", "chirp_signal"}:
        input_mode = "spectral_gradient"
    elif dataset_name == "dsss_signal":
        input_mode = "rgb"
    else:
        input_mode = "rgb" 或 "spectral_gradient"
```

更口语化地说：

```text
如果是 burst：用更会看突变和边缘的输入
如果是 chirp：用更会看斜线和变化的输入
如果是 dsss：用更能保留原始宽带纹理的输入
```

## 8. 和 auto 有什么区别

`auto` 是按“是不是频谱数据集”来选。

当前逻辑大致是：

```text
只要是频谱类数据集 -> spectral_gradient
普通工业图像数据集 -> rgb
```

也就是说，`auto` 不太关心具体是哪种异常。

`signal_adaptive` 更细一点，它会看具体信号类型：

```text
burst_signal -> spectral_gradient
chirp_signal -> spectral_gradient
dsss_signal  -> rgb
```

所以：

| 模式 | 选择依据 |
|---|---|
| `auto` | 看是不是频谱数据 |
| `signal_adaptive` | 看是哪一种频谱异常类型 |

## 9. signal_adaptive 和 prompt-mode 是两件事

`signal_adaptive` 控制的是输入图片怎么变成模型要看的三通道。

也就是：

```bash
--input-mode signal_adaptive
```

它回答的是：

> 模型看原始 RGB 图，还是看强度 + 时间梯度 + 频率梯度？

`prompt-mode` 控制的是文本提示词怎么写。

也就是：

```bash
--prompt-mode rf
```

它回答的是：

> 文本分支用什么语言去描述正常频谱和异常频谱？

所以这两个参数不要混在一起：

| 参数 | 控制对象 | 影响内容 |
|---|---|---|
| `--input-mode` | 图像输入 | RGB、梯度通道、按信号类型自适应 |
| `--prompt-mode` | 文本提示词 | legacy、rf、rf_object_agnostic 等 prompt 写法 |

常见组合是：

```bash
--prompt-mode rf --input-mode signal_adaptive
```

意思是：

> 文本上用 RF 频谱任务 prompt，图像上按 burst / chirp / dsss 自动选输入表示。

## 10. legacy 和 rf 有什么区别

`legacy` 是旧版 PromptAD 的 prompt 写法。

它更像是把当前类别当成普通物体或普通类别来描述。例如类别名是 `Playground_spectrum`，旧逻辑会围绕这个类别名去拼 prompt。

简单理解：

```text
legacy 更关心“这个类别名是什么”
```

它的优点是：

- 保留原始 PromptAD 的写法；
- 适合作为 baseline；
- 方便证明新 prompt 到底有没有收益。

它的问题是：

- `Playground_spectrum`、`Gymnasium_spectrum` 这类名字本质上是场景名，不是异常结构；
- 对 burst / chirp / dsss 这种信号异常，旧 prompt 不一定能表达“时频图异常”；
- 容易把频谱图当成普通图片类别，而不是 RF 信号检测任务。

`rf` 是频谱任务专用 prompt。

它会把 burst / chirp / dsss 这类数据统一描述成：

```text
radio frequency spectrogram
```

然后再加入 RF 任务相关的异常描述，例如：

```text
abnormal radio frequency spectrogram
radio frequency spectrogram with anomalous signal energy
radio frequency spectrogram with injected radio-frequency interference
radio frequency spectrogram with abnormal time-frequency structure
```

如果能识别出具体信号类型，还会加入对应结构词：

```text
burst: short-duration burst, transient pulse, narrowband burst
chirp: slanted trace, swept-frequency signal, chirp-like slope
dsss: wideband spread-spectrum-like band, wideband energy spread
```

简单理解：

```text
rf 更关心“这是一个射频频谱异常检测任务”
```

所以：

| prompt-mode | 核心区别 |
|---|---|
| `legacy` | 沿用原始 PromptAD 类别 prompt，把类别名当主要描述对象 |
| `rf` | 把任务改写成 RF spectrogram 异常检测，并加入信号结构描述 |

举个例子，如果做 `chirp_signal`，两者大致区别是：

```text
legacy:
abnormal Playground_spectrum
Playground_spectrum with defect

rf:
abnormal radio frequency spectrogram
radio frequency spectrogram with abnormal time-frequency structure
radio frequency spectrogram with a slanted narrowband interference line
```

这就是为什么 `rf` 更适合写进论文方法部分：它不是简单换词，而是把 prompt 从“普通图像类别”改成了“频谱异常结构”。

## 11. rf_object_agnostic 是什么

`rf_object_agnostic` 可以翻译成：

> 不绑定具体对象或具体信号类型的 RF prompt。

这里的 `object` 不是说频谱里真的有一个物体，而是借用视觉异常检测里的说法：

```text
object-specific prompt: 针对某个具体类别写 prompt
object-agnostic prompt: 不依赖具体类别，写更通用的异常描述
```

在本项目里，`rf_object_agnostic` 的意思是：

```text
只保留 RF 频谱正常/异常的公共描述，
不再额外加入 burst / chirp / dsss 的专属结构词。
```

也就是说，它会使用类似这些通用 RF 异常 prompt：

```text
abnormal radio frequency spectrogram
radio frequency spectrogram with anomalous signal energy
radio frequency spectrogram with unexpected interference
radio frequency spectrogram with abnormal time-frequency structure
```

但它不会再额外强调：

```text
burst 的短时突发
chirp 的斜线扫频
dsss 的宽带扩频
```

为什么要这样做？

因为在跨场景实验里，太具体的异常描述有时会带来偏置。

例如：

- 测试场景里的 chirp 很弱，未必像 prompt 里描述的清晰斜线；
- dsss 是弥散宽带能量，不一定能被某几个结构词稳定概括；
- 不同站点背景差异大，过细的 prompt 可能学到某种固定外观。

`rf_object_agnostic` 的目标是让文本分支更关注：

```text
这是不是正常 RF 频谱背景？
有没有异常信号能量或异常时频结构？
```

而不是过度关注：

```text
它是不是长得像某个手写 prompt 里的 burst / chirp / dsss？
```

所以三者关系可以总结成：

| prompt-mode | 关注点 | 适合作用 |
|---|---|---|
| `legacy` | 原始类别名 | baseline 对照 |
| `rf` | RF 任务 + 具体信号结构 | 主推的结构化频谱 prompt |
| `rf_object_agnostic` | RF 任务公共异常语义 | 跨场景泛化候选方案 |

注意：`rf_object_agnostic` 不是比 `rf` 永远更好。

它更像是一个泛化性更强、但结构指向更弱的版本。正式实验里应该和 `rf` 一起做对比，而不是直接替代。

## 12. 一个生活类比

可以把输入模式想成“给模型戴不同的眼镜”。

`rgb` 像普通眼镜：

> 看到原图本来的样子。

`spectral_gradient` 像边缘增强眼镜：

> 更容易看到哪里突然变化、哪里有边界、哪里有线条。

`signal_adaptive` 像自动换眼镜：

> 看 burst/chirp 的时候戴边缘增强眼镜；看 dsss 的时候戴普通眼镜。

如果把 prompt-mode 也放进这个类比里：

```text
input-mode 像模型戴什么眼镜；
prompt-mode 像你用什么语言告诉模型要找什么异常。
```

`rf` 是告诉模型：

> 你现在看的是射频频谱图，要找异常信号结构。

`rf_object_agnostic` 是告诉模型：

> 你现在看的是射频频谱图，只判断有没有异常信号能量或异常时频结构，先不要强行套某一种具体信号形状。

## 13. 什么时候应该用 signal_adaptive

建议在下面情况使用：

- 同时跑 burst、chirp、dsss 多种信号；
- 想让程序自动给不同信号选输入；
- 已经确认 `spectral_gradient` 对 burst/chirp 有帮助；
- dsss 用 `rgb` 更稳定。

运行示例：

```bash
python run_rf_split_all.py \
    --prompt-mode rf \
    --input-mode signal_adaptive
```

或者单独训练：

```bash
python train_cls.py \
    --dataset chirp_signal \
    --class_name Playground_spectrum \
    --train-site WeaponMuseum_spectrum \
    --noise-level m30db \
    --prompt-mode rf \
    --input-mode signal_adaptive
```

如果这里的 dataset 是 `chirp_signal`，实际输入会自动变成 `spectral_gradient`。

如果 dataset 是 `dsss_signal`，实际输入会自动变成 `rgb`。

## 14. 容易误解的点

### 误解 1：signal_adaptive 是一个新模型

不是。

它只是输入预处理策略。

模型还是 PromptAD。

### 误解 2：signal_adaptive 会修改 prompt

不会。

prompt 由 `--prompt-mode` 控制，例如：

```bash
--prompt-mode rf
```

输入由 `--input-mode` 控制，例如：

```bash
--input-mode signal_adaptive
```

这两个东西是分开的。

### 误解 3：rf_object_agnostic 是 signal_adaptive 的一部分

不是。

`rf_object_agnostic` 是 prompt 写法，属于：

```bash
--prompt-mode rf_object_agnostic
```

`signal_adaptive` 是输入写法，属于：

```bash
--input-mode signal_adaptive
```

它们可以一起用：

```bash
--prompt-mode rf_object_agnostic --input-mode signal_adaptive
```

意思是：

```text
文本上用更通用的 RF 异常描述；
图像上按信号类型自动选择输入通道。
```

### 误解 4：rf 一定比 legacy 好

不一定。

`rf` 更符合频谱任务，但最终要看实验结果。

所以报告里最好同时比较：

```text
legacy
rf
rf_object_agnostic
```

这样才能说明收益到底来自哪里。

### 误解 5：signal_adaptive 永远最好

不一定。

它是根据当前实验观察设计的实用策略。最终是否更好，要看完整实验结果。

所以论文或报告里要做对比：

```text
rgb
spectral_gradient
signal_adaptive
```

不能只报 `signal_adaptive`。

## 15. 最简总结

`signal_adaptive` 就是：

> 根据信号类型自动选择输入方式。

当前规则是：

```text
burst  -> spectral_gradient
chirp  -> spectral_gradient
dsss   -> rgb
```

原因是：

```text
burst/chirp 更像线条、边缘、突变，所以用 spectral_gradient；
dsss 更像宽带纹理和整体能量分布，所以用 rgb。
```

它的目标是让模型看到更适合当前信号类型的信息，从而提高频谱异常检测效果。

`legacy`、`rf`、`rf_object_agnostic` 是 prompt-mode：

```text
legacy            -> 原始 PromptAD 类别 prompt
rf                -> RF 频谱任务 prompt + 具体信号结构词
rf_object_agnostic -> RF 频谱任务通用 prompt，不绑定具体信号结构
```

最推荐先做的完整对比是：

```text
legacy + rgb
rf + rgb
rf + spectral_gradient
rf + signal_adaptive
rf_object_agnostic + signal_adaptive
```

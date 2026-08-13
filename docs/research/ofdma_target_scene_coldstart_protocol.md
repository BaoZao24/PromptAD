# OFDMA target-scene cold-start v1

## 目标

验证目标场景冷启动：方法进入一个新的通信配置后，只使用该配置的少量正常观测
建立 normal memory，再检测同一配置下的后续正常与干扰观测。

旧 OFDMA 协议使用少量正常配置检测大量独立随机配置，属于跨配置压力测试，不作为
本协议的主结果。

## 实验单位

- 上游 Sionna `scene0` 是所有数据共享的物理工厂。
- 本协议的 `target_scene_id` 表示一个固定的目标通信配置。
- 一次 `observation` 包含同一观测窗口下 21 个 SU 的同步频谱图。
- 1/2/4-shot 分别使用前 1/2/4 次正常 support observation，且严格嵌套。
- 每个 SU 先将四个同源视图融合后的 ViT anomaly map 取第三高 patch，抑制孤立
  极值响应；再对 21 个 SU 分数取最大值，保留少数 SU 可见的干扰证据。该固定
  规则在 validation 场景上完成设计确认，test 场景只用于最终报告。

## 场景内固定与变化

一个 target scene 内固定：

- 合法发射机数量和位置；
- 每个合法发射机的频率资源范围；
- 合法发射机到各 SU 的信道频率响应；
- 物理环境和 SU 位置。

每次 observation 独立变化：

- 合法数据符号；
- 热噪声；
- 固定频率范围内的时隙活动。

异常 observation 使用与某个 normal-test observation 相同的合法资源状态，并额外
加入一个随机位置、方向和功率的 jammer。合法数据符号和噪声使用记录在 manifest
中的确定性随机种子生成。

## 正式规模

| Split | Target scenes | Support normal | Test normal | Test anomaly |
|---|---:|---:|---:|---:|
| validation | 5 | 4/scene | 100/scene | 5 types × 20/scene |
| test | 30 | 4/scene | 100/scene | 5 types × 20/scene |

总计 7,140 次 observation 和 149,940 张 SU 频谱图。validation 用于方案设计确认；
test 标签只在方法冻结后计算最终指标。

## 数据清单

每个 target scene 独立保存配置、manifest 和完成标记。全局 manifest 至少包含：

- `target_scene_id`、`split`；
- `observation_id`、`role`、`support_rank`；
- `paired_normal_observation_id`；
- `label`、`jammer_type`、`jammer_power`、`jammer_location`；
- `su_id`、`image_path`；
- `context_seed`、`legitimate_seed`、`jammer_seed`；
- `context_fingerprint`、`resource_fingerprint`。

所有图使用官方 OFDMA 数据集固定的功率范围
`[-183.38843, 27.147293] dB` 转换为 uint8，不根据新数据集的 validation/test
样本重新估计归一化范围。

## 完整性要求

- validation/test target scene 不重合；
- 每个 observation 恰有 21 个唯一 SU；
- support 全部为 normal；
- 每个场景严格包含 4 support、100 normal test、100 anomaly test；
- 每类 jammer 每场景恰有 20 次；
- anomaly 与记录的 normal observation 使用相同资源状态；
- 1/2/4-shot support 嵌套；
- 图片数组形状为 `1320 × 70`，值域为 uint8；
- test 标签不参与建库、调参或分数校准。

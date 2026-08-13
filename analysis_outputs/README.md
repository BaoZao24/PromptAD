# 实验结果索引

原始运行目录仍按日期保留在本目录根下，以保证旧脚本、论文链接和复现实验命令继续有效。日常查看请优先进入下面四类索引目录；其中的条目是指向原始运行目录的轻量链接，不复制数据。

## 正式论文结果

- [`formal/inhouse_rf/`](formal/inhouse_rf/)：In-house RF 正式比较、传统方法和已发表基线。
- [`formal/public_rf/`](formal/public_rf/)：Public RF 的 k-per-frequency 主实验和对比方法。
- [`formal/ofdma/`](formal/ofdma/)：OFDMA target-scene cold-start 主结果。
- [`formal/fedjam/`](formal/fedjam/)：FedJam 少样本补充实验。
- [`formal/paper_figures/`](formal/paper_figures/)：论文统一图表来源。

## 当前探索与验证

- [`exploratory/tta/`](exploratory/tta/)：尚未进入论文主线的频谱 TTA 候选与预览。
- [`smoke_validation/`](smoke_validation/)：最近的 smoke、协议验证和性能排查结果；这些目录不能直接作为主表依据。

## 历史材料

- [`archive/`](archive/)：早期汇报、旧视觉基线、旧协议和候选分支的入口。它们保留用于追溯，不作为当前论文结果。

## 约定

- 新的正式实验：在原始日期目录完成后，新增一个链接到 `formal/<dataset>/`，并在对应 README 写清协议。
- 新的探索：放在日期目录，完成筛选前只在 `exploratory/` 或 `smoke_validation/` 建链接。
- 不删除原始目录，除非结果已确认可从脚本和数据重新生成且不再被论文、报告或代码引用。

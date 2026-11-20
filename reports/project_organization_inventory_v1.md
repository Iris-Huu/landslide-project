# 阶段性文件整理与本地实验索引

## 结构和归档

- `src/data/`、`src/data_audit/`、`src/grid_reconstruction/`、`src/registration/`：数据读取、审计、相对网格和局部配准；`data/interim/`保留统一清单、划分、网格及配准证据。原始数据、压缩包、清单与审批字段均未改动。
- `src/models/`、`src/metrics/`、`src/training/`与`tests/`：轻量U-Net、DeepLabV3+、SegFormer、损失、评价、检查点及合成测试。`scripts/training/uav_exploratory.py`为轻量U-Net入口；`scripts/training/uav_deeplabv3plus.py`是DeepLab与SegFormer共享入口。前者保留早期结果的执行路径，不与后者简单合并。
- `reports/`与`reports/tables/`：方案、结构化审计与实验说明；`reports/tables/exploration_run_index_v1.json`是新增小型摘要，可纳入Git，包含关键配置、样本及源码哈希、指标、日志与检查点路径及SHA-256。原始运行记录的完整配置与逐轮指标仍在本地。
- CPU真实冒烟：`experiments/uav_cpu_smoke_doy2b1bq/`；轻量U-Net原始GPU运行：`experiments/uav_train_0ev4mlp7/`；BCE与加权BCE对照：`experiments/uav_train_xj_kf48q/`、`experiments/uav_train_s47c7j29/`；诊断：`experiments/uav_diagnostic_zftoy29_/`、`experiments/uav_val_error_y69erm3a/`。
- DeepLabV3+/ResNet-50：`experiments/uav_deeplab_train_83uohcoi/`，best第4轮、last第5轮；SegFormer-B2：`experiments/uav_segformer_train_ydt54mjz/`，best和last均第5轮。两个目录各自的`run.json`、`best.pt`、`last.pt`保留；四个检查点合计约1.15 GiB。每个模型仅seed 17五轮，多随机种子尚未执行。
- 四张UAV人工核查图 `reports/figures/uav_train_val_review_v1/`、诊断图、实验原始日志和检查点、`.venv/`及原始数据保留本地、被Git忽略；本次未将其暂存。报告、轻量索引和代码可以审查后纳入Git，但当前未自动提交。

## 保留与未合并

历史审计、候选划分、早期轻量U-Net与DeepLab/SegFormer入口服务于不同阶段及复现记录，未发现足以安全合并并保持全部历史调用的依据，故保留。重复误报分析目录包含不同中间记录、样本数修正和日志格式，虽有相同图片但保留；旧notebook检查点、缺少对应源码的字节码及`src/data_audit/__pycache__/build_source_grid.py`来源不明，均保留。未删除、移动或合并独特代码、报告、结果、实验日志、图片或检查点。

## 删除的可重建缓存

以下字节码均有对应`.py`源码、未被Git追踪，运行时可重新生成；仅删除这些文件：

- `src/data_audit/__pycache__/compare_sample_spectra.cpython-311.pyc`
- `src/grid_reconstruction/__pycache__/find_horizontal_vertical_matches.cpython-311.pyc`
- `src/grid_reconstruction/__pycache__/build_source_grid.cpython-311.pyc`
- `src/grid_reconstruction/__pycache__/validate_grid_overlaps.cpython-311.pyc`
- `src/grid_reconstruction/__pycache__/find_diagonal_matches_all.cpython-311.pyc`
- `src/grid_reconstruction/__pycache__/check_diagonal_consistency_all.cpython-311.pyc`
- `src/registration/__pycache__/match_sift_pair.cpython-311.pyc`
- `src/registration/__pycache__/screen_selected_pairs.cpython-311.pyc`
- `src/registration/__pycache__/screen_satellite_anchors.cpython-311.pyc`

候选验证已参与模型选择，既有测试候选曾被探索；结果并非正式E1或独立盲测。`all_pixels_provisional`不是已验证有效区域；NoData、标签对齐、跨区域/平台空间隔离、独立评价和多种子等仍待协议及证据。

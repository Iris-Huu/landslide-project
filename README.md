# 卫星到无人机的跨平台滑坡分割

本项目研究卫星影像与无人机影像之间的**域偏移**对滑坡语义分割的影响，并计划在同址影像基础上探索跨平台模型适配。

**当前进度：已冻结UAV划分v1，完成DeepLab与SegFormer的三种子A单元测试评价，以及DINOv3两个预训练版本的三种子冻结特征训练。** DINOv3测试、四单元等样本量析因及适配实验尚未完成。

## 项目数据

- **CAS Landslide Dataset**：已下载16个影像子集及1个研究区矢量压缩包，17个ZIP均通过内部CRC检查；尚未逐包与发布者校验值比对。
- **Moxitaidi（CAS重点研究区域）**：卫星（SAT）652张、无人机（UAV-0.6m）984张，均为512×512图块。
- **GDCLD**：已整理Lushan_1、Lushan_2、Lushan_3三幅测试影像及标签，完成基础元数据与对齐检查。

原始遥感数据体积较大，保存在AutoDL服务器，**不包含在本Git仓库中**。

## 已完成的工作

| 工作 | 目前结果 |
|---|---|
| 数据下载与检查 | CAS的16个影像子集已获取，完成压缩包完整性、文件结构、Moxitaidi影像/掩码配对及基础统计 |
| 无数据区域审计 | 已统计黑白边界等疑似无效区域，发现部分候选区域与真实滑坡标签重叠，暂不能直接作为最终NoData掩码 |
| 相对网格重建 | 通过图块重叠恢复Moxitaidi SAT的164个、UAV的8个相对连通组，并完成组内重叠一致性检查 |
| UAV空间划分v1 | 按学长意见冻结最大连通组划分：训练484张、验证71张、测试149张、缓冲57张；组内空间间隔复核未发现违规，其他223张暂不使用 |
| 同址配准尝试 | 保留SAT100局部配准证据；新增连通组拼图配准，第154、158组叠加支持同址定位，第148组尚未可靠定位 |
| 分辨率分析 | UAV-1m到UAV-0.6m像素尺度中位数为1.6389；SAT第154、158组当前尺度约为1，此前SAT100局部约为2；绝对地面分辨率尚未独立确认 |
| GDCLD来源检查 | Lushan_1/2的GeoTIFF像素间距为0.5米，Lushan_3为3米；结合论文提出可能的数据来源，但具体传感器尚未确认 |
| UAV读取与指标 | 已核验训练候选484对、验证候选71对影像与掩码的读取、尺寸和0/1标签，建立数据读取及NumPy二分类指标模块 |
| CPU流程测试 | 轻量U-Net和带有效像素掩码的训练步骤通过合成测试；另用2张训练候选各完成1次CPU冒烟更新，仅验证流程，未运行验证集精度或正式E1实验 |
| GPU探索实验 | DeepLabV3+/ResNet-50与SegFormer-B2随机初始化、seed 17、全图各训练5轮；候选验证最佳IoU分别为0.3487和0.3411，仅供开发诊断 |
| 多随机种子探索 | DeepLabV3+/ResNet-50与SegFormer-B2各完成三个随机种子的5轮探索；验证集已参与选模，结果仅供开发诊断 |
| NoData规则v1 | 只忽略边缘连通纯黑背景，保留纯白与全部正类标签；已完成训练、验证统计并接入A单元训练 |
| A单元训练v1 | DeepLabV3+/ResNet-50与SegFormer-B2随机初始化，各完成三种子训练与测试；测试IoU均值分别为17.11%和18.09% |
| B单元空间筛选 | 第154组234张成为候选；512像素缓冲及额外32/64/128像素敏感性检查的候选数量一致，暂未批准训练 |
| A单元测试评价 | 已实现拼接概率平均、重叠像素去重、Boundary F1（2像素）与10,000次空间bootstrap；DeepLab和SegFormer已评价，DINOv3待评价 |
| DINOv3冻结特征基线 | 两版本各完成seed 17/42/123的100轮线性头训练；LVD与SAT最佳验证IoU均值为45.14%、46.83%，尚未测试 |
| 模型准备 | DINOv3两版本权重已下载、校验并通过GPU合成检查；已缓存训练/验证特征并完成六次线性头训练，SAM 3访问权限已获得 |

早期探索保留原候选划分与全像素暂用口径；新增A单元使用冻结划分v1及NoData v1。DeepLab、SegFormer结果已在测试拼图去重后计算；DINOv3目前只有验证集选模结果。冻结预训练编码器与全参数随机初始化模型的训练策略不同，不能直接归因为骨干或预训练优劣。

## 当前困难与未完成事项

1. **B单元样本量待确定**：当前234张为空间隔离候选，尚未批准正式训练；低于300张参考线，需与学长确认四单元等样本量方案。
2. **E1尚未全部完成**：DeepLab与SegFormer已完成测试；DINOv3两版本只完成冻结编码器线性头训练，测试待执行。
3. **尺度不能统一推广**：不同SAT连通组出现不同相对尺度；UAV-1m校验提供相对尺度证据，不能单独证明绝对分辨率。
4. **测试结果较低且比较条件有限**：当前随机初始化结果不能作为CAS论文的公平复现对照；后续不得用已知测试结果选择训练超参数。
5. **E2与适配尚未完成**：四单元等样本量训练、效应表、分辨率降质曲线和适配实验仍待推进。
6. **可复现性与标签局限仍需记录**：环境复用部分基础依赖；NoData规则、标签对齐和地域代表性保留既有说明。GDCLD进一步处理按学长意见延期。

> 已明确区分候选、验证结果和测试结果；当前进度不代表全部E1、析因或跨平台适配实验已经完成。

## 接下来的计划

1. 固定DINOv3两版本六个检查点的测试协议，检查恢复一致性后完成统一测试评价。
2. 与学长确认B单元234张候选的使用范围、分组尺度和四单元等样本量方案。
3. 准备E2训练池及Moxi town、Jiuzhai 0.2m子集的网格与划分，再开展析因与降质曲线实验。

## 项目文件说明

| 目录 | 主要内容 |
|---|---|
| `src/data/`、`src/metrics/` | UAV候选与正式训练/验证读取、NoData规则、二分类指标及拼接去重 |
| `src/models/`、`src/training/` | U-Net、DeepLabV3+、SegFormer及带有效像素掩码的训练、验证步骤 |
| `src/data_audit/` | CAS、GDCLD影像、标签和NoData检查代码 |
| `src/grid_reconstruction/` | 图块重叠匹配、相对网格重建、空间检查 |
| `src/registration/` | SAT/UAV特征匹配、配准和验证 |
| `scripts/` | 配准、空间划分、分辨率汇总及探索训练/诊断入口 |
| `tests/` | 合成数据读取、指标、损失、模型与检查点测试 |
| `data/interim/reconstructed_grids/` | 重建的相对网格及诊断CSV/JSON |
| `data/interim/splits/` | 空间划分候选、统一清单和泄漏风险记录 |
| `data/interim/registration/` | 局部配准、筛查与可靠性判定结果 |
| `reports/tables/` | 数据审计、分辨率与验证结果表 |
| `reports/figures/` | 可视化检查结果 |

建议优先查看以下文件：

- `reports/e1_stage_status_v1.md`：当前A单元结果、评价边界与下次工作。
- `reports/tables/uav_A_v1_test_summary.json`：DeepLab三种子去重测试结果及空间区间。
- `reports/tables/uav_segformer_A_v1_test_summary.json`：SegFormer三种子去重测试结果及空间区间。
- `reports/tables/dinov3_linear_A_v1_training_summary.json`：DINOv3两版本六次冻结特征训练的验证结果与检查点索引。
- `reports/uav_1m_relative_scale_v1.md`：UAV不同分辨率子集的相对尺度校验及限制。

- `reports/tables/uav_A_v1_training_summary.json`：A单元三个随机种子的训练汇总与检查点追溯。
- `reports/tables/uav_A_v1_test_protocol.json`：初版测试协议；实际DeepLab评价使用补充Boundary F1和10,000次bootstrap的v2。
- `data/interim/splits/moxitaidi_sat_B_candidates_v1.csv`：B单元234张卫星候选，未批准为正式训练源。
- `reports/satellite_B_selection_status_v1.md`：B候选筛选、分组尺度与配准诊断的证据和限制。
- `scripts/training/uav_a_v1.py`：A单元训练入口，使用冻结划分和NoData v1。

- `data/interim/splits/moxitaidi_unified_spatial_manifest_v1.csv`：统一影像清单及候选划分；**所有正式训练/评估批准字段目前均为False**。
- `data/interim/splits/moxitaidi_uav_spatial_split_formal_v1.csv`：按学长意见冻结的UAV单平台划分v1。
- `reports/tables/uav_split_validation_recheck_v1.json`：UAV划分的组内缓冲验证。
- `reports/tables/uav_nodata_v1_train_val_stats.json`：NoData v1规则在训练和验证集上的统计。
- `reports/uav_multiseed_exploratory_v1.md`：DeepLab和SegFormer多随机种子探索结果及限制。
- `reports/uav_exploratory_baseline_plan_v1.md`：UAV探索性实验方案及使用限制。
- `reports/e1_user_taskbook_resnet50_dice_exploration_v1.md`：用户提供的任务书转述与DeepLab探索记录。
- `reports/e1_segformer_b2_exploratory_v1.md`：SegFormer与DeepLab探索配置及结果对照。
- `reports/uav_val_false_positive_full_image_v1.md`：U-Net误报及全图/中心裁剪敏感性分析。
- `reports/tables/exploration_run_index_v1.json`：关键探索运行、日志、样本清单和本地检查点哈希索引。
- `scripts/training/uav_exploratory.py`：轻量U-Net候选探索入口；`scripts/training/uav_deeplabv3plus.py`：DeepLab与SegFormer共享入口。
- `reports/tables/uav_cpu_real_smoke_run_v1.json`：2张训练候选的CPU冒烟运行记录，不是正式实验结果。
- `reports/uav_data_reader_and_visual_review_v1.md`：训练/验证候选的读取与人工核查图说明。
- `data/interim/registration/sat100_registration_reliability_decision_v1.csv`：局部配准证据等级。
- `reports/tables/effective_resolution_comparison_v1.csv`：分辨率对照及未知状态。
- `reports/tables/lushan_sensor_provenance_v1.csv`：芦山影像来源推断与待确认状态。

审计环境为AutoDL Linux、Conda `landslide-audit`、Python 3.11.18；独立 `.venv/torch-cpu` 使用PyTorch 2.6.0+cpu，`.venv/torch-gpu` 使用PyTorch 2.8.0+cu128与SMP 0.5.0。虚拟环境复用部分基础依赖，尚非完整锁定的正式训练环境。原始数据、虚拟环境、完整实验日志和检查点留在服务器，不随Git仓库分发。

## 数据来源

- [CAS Landslide Dataset](https://doi.org/10.5281/zenodo.10294997)
- [CAS数据集论文](https://doi.org/10.1038/s41597-023-02847-z)
- [GDCLD](https://doi.org/10.5281/zenodo.13612636)
- [GDCLD论文](https://doi.org/10.5194/essd-16-4817-2024)

**说明**：当前阶段成果包括数据审计、UAV划分v1、A单元三种子训练和B单元空间筛选候选；正式测试评价与跨平台实验尚未完成。

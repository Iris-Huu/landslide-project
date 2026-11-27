# 卫星到无人机的跨平台滑坡分割

本项目研究卫星影像与无人机影像之间的**域偏移**对滑坡语义分割的影响，并计划在同址影像基础上探索跨平台模型适配。

**当前进度：第一周数据审计已阶段性完成；第二周已冻结UAV单平台划分v1，并完成A单元DeepLabV3+/ResNet-50三个随机种子的训练。** B单元已整理234张卫星候选；A单元测试评价尚未执行。

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
| 分辨率分析 | 第154、158组当前估计的原始SAT到UAV像素尺度约为1，此前SAT100局部结果约为2；尺度需分组核实，绝对地面分辨率尚未确认 |
| GDCLD来源检查 | Lushan_1/2的GeoTIFF像素间距为0.5米，Lushan_3为3米；结合论文提出可能的数据来源，但具体传感器尚未确认 |
| UAV读取与指标 | 已核验训练候选484对、验证候选71对影像与掩码的读取、尺寸和0/1标签，建立数据读取及NumPy二分类指标模块 |
| CPU流程测试 | 轻量U-Net和带有效像素掩码的训练步骤通过合成测试；另用2张训练候选各完成1次CPU冒烟更新，仅验证流程，未运行验证集精度或正式E1实验 |
| GPU探索实验 | DeepLabV3+/ResNet-50与SegFormer-B2随机初始化、seed 17、全图各训练5轮；候选验证最佳IoU分别为0.3487和0.3411，仅供开发诊断 |
| 多随机种子探索 | DeepLabV3+/ResNet-50与SegFormer-B2各完成三个随机种子的5轮探索；验证集已参与选模，结果仅供开发诊断 |
| NoData规则v1 | 只忽略边缘连通纯黑背景，保留纯白与全部正类标签；已完成训练、验证统计并接入A单元训练 |
| A单元训练v1 | DeepLabV3+/ResNet-50随机初始化，seed 17/42/123均运行至第30轮；最佳验证IoU均值及样本标准差为0.3938±0.0109，测试评价待执行 |
| B单元空间筛选 | 第154组234张成为候选；512像素缓冲及额外32/64/128像素敏感性检查的候选数量一致，暂未批准训练 |
| 测试评价准备 | 已固定三个best检查点及测试协议；拼接去重5项合成测试通过，空间bootstrap实现与真实测试推理尚未完成 |
| 模型准备 | 据任务书第8页，模型包含DeepLabV3+/ResNet-50、SegFormer-B2及DINOv3 ViT-L/16；DINOv3两个版本与SAM 3已获得访问权限，DINOv3接入工作尚需核实完成状态 |

早期U-Net、DeepLab和SegFormer探索使用候选划分及`all_pixels_provisional`口径；新增A单元训练使用冻结UAV划分v1和NoData v1。两类记录应分别解读。A单元三个种子的最佳验证IoU为0.4063、0.3862、0.3891，均值及样本标准差为0.3938±0.0109；这些是验证集选模结果，尚无测试IoU。任务书模型信息来自用户提供的第8页转述；DINOv3与SAM 3权限已获批，后续接入尚未完成。

## 当前困难与未完成事项

1. **B单元清单尚待定稿**：第154组234张通过当前变换下的空间隔离筛选，已保存候选清单；第158组81张均未通过筛选，第148组69张尚未可靠定位。候选未批准为正式训练源；数量低于300张参考线，需与学长确认四单元等样本量方案。
2. **配准证据范围有限**：第154组四处局部梯度相关位移约为4–16.5个UAV原始像素，第2处相关峰区分度较弱。这是局部诊断，不是独立人工精度测量或全组误差上界；未定位图块暂不使用。
3. **尺度与绝对分辨率需区分**：不同SAT连通组出现不同相对尺度，不能统一采用2倍上采样。UAV-1m与UAV-0.6m的尺度校验尚未完成；以实测证据为主，来源未核实处保留未知状态。
4. **有效像素与标签仍有局限**：A训练已使用保守NoData v1，但尺寸一致和叠加图不能证明标签整体准确。GDCLD坐标系与标签对齐按学长意见延期处理。
5. **A单元测试评价待执行**：三个best检查点与测试协议已固定；拼接去重模块5项合成测试通过。测试推理、重叠去重后的测试指标及空间bootstrap尚未执行，bootstrap实现尚未创建。不得用测试结果调参。
6. **训练配置与比较边界**：当前A单元使用484张训练图块、随机初始化和单一冻结划分；不是预训练模型对照，也尚未形成四单元等样本量析因结果。环境复用部分基础依赖，尚非完整锁定环境。

> 目前仍在进一步学习、核查相关方法与代码。已运行的探索性实验不代表所用方法和结果已达到正式论文实验的验证标准。

## 接下来的计划

1. 完成A单元独立测试评价入口与合成检查，按固定协议拼接测试预测、去除重叠计数，并实现空间块bootstrap。
2. 向学长同步B单元234张候选及分组尺度差异，确认候选使用范围和四单元等样本量方案。
3. 核查UAV-1m相对尺度；在析因实验前完成Moxi town与Jiuzhai 0.2m子集的网格与划分，再推进DINOv3接入。

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

- `reports/tables/uav_A_v1_training_summary.json`：A单元三个随机种子的训练汇总与检查点追溯。
- `reports/tables/uav_A_v1_test_protocol.json`：测试推理前固定的检查点、拼接规则与拟采用的bootstrap设置；评价尚未执行。
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

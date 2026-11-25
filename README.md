# 卫星到无人机的跨平台滑坡分割

本项目研究卫星影像与无人机影像之间的**域偏移**对滑坡语义分割的影响，并计划在同址影像基础上探索跨平台模型适配。

**当前进度：第一周数据审计已阶段性完成；第二周已完成UAV数据读取、模型流程和多随机种子探索实验，并依据学长意见冻结UAV单平台划分v1。** 正式E1尚未开始，SAT/UAV跨平台空间隔离仍待验证。

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
| 同址配准尝试 | 使用SIFT与RANSAC找到SAT100与部分UAV图块的局部对应关系；两组人工核验误差分别约为3.59和6.17 UAV像素 |
| 分辨率分析 | 建立标称分辨率、局部像素尺度和探索性模糊参数的对照记录；绝对有效分辨率尚未测得 |
| GDCLD来源检查 | Lushan_1/2的GeoTIFF像素间距为0.5米，Lushan_3为3米；结合论文提出可能的数据来源，但具体传感器尚未确认 |
| UAV读取与指标 | 已核验训练候选484对、验证候选71对影像与掩码的读取、尺寸和0/1标签，建立数据读取及NumPy二分类指标模块 |
| CPU流程测试 | 轻量U-Net和带有效像素掩码的训练步骤通过合成测试；另用2张训练候选各完成1次CPU冒烟更新，仅验证流程，未运行验证集精度或正式E1实验 |
| GPU探索实验 | DeepLabV3+/ResNet-50与SegFormer-B2随机初始化、seed 17、全图各训练5轮；候选验证最佳IoU分别为0.3487和0.3411，仅供开发诊断 |
| 多随机种子探索 | DeepLabV3+/ResNet-50与SegFormer-B2各完成三个随机种子的5轮探索；验证集已参与选模，结果仅供开发诊断 |
| NoData规则v1 | 实现边缘连通纯黑像素忽略规则，保留所有正类标签；已对训练与验证候选完成统计，尚未接入正式训练 |
| 模型准备 | 据任务书第8页，模型包含DeepLabV3+/ResNet-50、SegFormer-B2及DINOv3 ViT-L/16；DINOv3两个版本与SAM 3已获得访问权限，DINOv3接入工作尚需核实完成状态 |

上述结果均来自候选划分及`all_pixels_provisional`全像素暂用口径；验证集已参与模型选择与诊断，不是独立测试，也不是正式E1上界。两种模型已完成三个随机种子的5轮探索，**尚不能作为正式E1结果**。任务书模型信息来自用户提供的第8页转述，服务器尚无原文；SAM 3文本提示与SegEarth-OV3为零样本对照。

## 当前困难与未完成事项

1. **空间划分仍不完整**：CAS图块缺少可用的完整地理定位信息。虽然恢复了部分相对位置，但不同连通组及SAT/UAV之间的空间关系还不清楚。UAV最大连通组划分已按学长意见冻结为单平台v1，**跨平台空间泄漏尚未排除**；其他223张UAV暂不使用，SAT图块仍需完成定位与筛选。
2. **配准只在局部有效**：部分卫星与无人机影像差异较大，SIFT/RANSAC难以获得足够可靠的匹配。现有人工核验点少、独立性有限，尚不能推广到整个研究区，也不能据此确定正式训练配对。
3. **实际分辨率与影像来源不明确**：CAS论文将Moxitaidi SAT标注为0.6米、来源为Sentinel-2/L2A，与Sentinel-2原生分辨率存在待解释差异。GDCLD的Lushan影像也缺少文件级传感器身份信息，已联系作者确认。
4. **无数据区和标签对齐仍需处理**：目前的NoData候选规则可能误删滑坡像素；GDCLD部分影像与标签还存在网格/尺寸对齐问题。
5. **实验协议与模型评估**：已根据学长反馈冻结UAV单平台划分v1；旧统一清单的批准字段保持不变。测试区此前做过标签统计，后续不得用测试结果调参；SAT图块隔离及正式E1结果仍待完成。
6. **训练环境和有效像素仍需确认**：已验证RTX 5090及PyTorch 2.8.0+cu128可用于探索；正式模型权重及训练环境仍需确认。`all_pixels_provisional`暂时计入全部像素，不代表有效区域已验证；UAV影像和标签缺少可用地理参考，尺寸一致及叠加图不能证明真实空间对齐。

> 目前仍在进一步学习、核查相关方法与代码。已运行的探索性实验不代表所用方法和结果已达到正式论文实验的验证标准。

## 接下来的计划

1. 按学长建议，继续进行SAT/UAV连通组拼图配准与空间隔离检查，统计B单元可用卫星图块数量。
2. 将NoData v1规则接入训练与评价；GPU可用后运行DeepLabV3+/ResNet-50的A单元E1实验，并按协议补充随机种子。
3. 继续核查UAV不同分辨率子集的相对尺度，并准备DINOv3与后续适配实验。

## 项目文件说明

| 目录 | 主要内容 |
|---|---|
| `src/data/`、`src/metrics/` | UAV候选数据读取与NumPy二分类指标 |
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

**说明**：当前阶段成果包括数据审计、UAV单平台划分v1、局部配准和多随机种子探索实验，不代表正式跨平台模型或E1实验已经完成。

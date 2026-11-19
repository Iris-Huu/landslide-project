# 卫星到无人机的跨平台滑坡分割

本项目研究卫星影像与无人机影像之间的**域偏移**对滑坡语义分割的影响，并计划在同址影像基础上探索跨平台模型适配。

**当前进度：第一周数据审计与实验准备已阶段性完成；第二周已完成UAV数据读取、模型准备及CPU冒烟测试，尚未开始正式模型训练。** 现有空间划分和配准结果属于候选或局部验证，不能视为已完成全数据集无泄漏验证。

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
| 空间划分候选 | 对UAV最大连通组建立接近7:1:2的划分：训练484张、验证71张、测试149张，另有缓冲区57张；组内空间间隔复核未发现违规 |
| 同址配准尝试 | 使用SIFT与RANSAC找到SAT100与部分UAV图块的局部对应关系；两组人工核验误差分别约为3.59和6.17 UAV像素 |
| 分辨率分析 | 建立标称分辨率、局部像素尺度和探索性模糊参数的对照记录；绝对有效分辨率尚未测得 |
| GDCLD来源检查 | Lushan_1/2的GeoTIFF像素间距为0.5米，Lushan_3为3米；结合论文提出可能的数据来源，但具体传感器尚未确认 |
| UAV读取与指标 | 已核验训练候选484对、验证候选71对影像与掩码的读取、尺寸和0/1标签，建立数据读取及NumPy二分类指标模块 |
| CPU流程测试 | 轻量U-Net和带有效像素掩码的训练步骤通过合成测试；另用2张训练候选各完成1次CPU冒烟更新，仅验证流程，未运行验证集精度或正式E1实验 |
| 模型准备 | 已提交DINOv3 SAT-493M、DINOv3 LVD-1689M和SAM 3.1的访问申请，仍在等待结果 |

上述工作及检查结果已保存为脚本、CSV和JSON，详细证据以仓库中的结构化记录为准。

## 当前困难与未完成事项

1. **空间划分仍不完整**：CAS图块缺少可用的完整地理定位信息。虽然恢复了部分相对位置，但不同连通组及SAT/UAV之间的空间关系还不清楚。当前划分只是UAV局部候选方案，**跨平台空间泄漏尚未排除**；其他223张UAV仍未分配，652张SAT暂不用于正式跨平台训练。
2. **配准只在局部有效**：部分卫星与无人机影像差异较大，SIFT/RANSAC难以获得足够可靠的匹配。现有人工核验点少、独立性有限，尚不能推广到整个研究区，也不能据此确定正式训练配对。
3. **实际分辨率与影像来源不明确**：CAS论文将Moxitaidi SAT标注为0.6米、来源为Sentinel-2/L2A，与Sentinel-2原生分辨率存在待解释差异。GDCLD的Lushan影像也缺少文件级传感器身份信息，已联系作者确认。
4. **无数据区和标签对齐仍需处理**：目前的NoData候选规则可能误删滑坡像素；GDCLD部分影像与标签还存在网格/尺寸对齐问题。
5. **实验协议与模型评估待确认**：统一清单的正式训练和评估批准字段均为False，老师与学长意见仍待回复。测试候选区域曾被探索，不能称为未接触盲测集；尚无正式U-Net基线、真实验证集IoU/F1或跨域适配结果。
6. **训练环境和有效像素仍需确认**：CPU环境已通过流程测试，GPU及其兼容环境尚未检查。`all_pixels_provisional`暂时计入全部像素，不代表有效区域已验证；UAV影像和标签缺少可用地理参考，尺寸一致及叠加图不能证明真实空间对齐。

> 目前仍在进一步学习、核查相关方法与代码。已运行的探索性实验不代表所用方法和结果已达到正式论文实验的验证标准。

## 接下来的计划

1. 明天启动GPU后，先检查设备、驱动及PyTorch/CUDA兼容性；当前CPU专用环境不是GPU训练环境。
2. 根据老师与学长反馈明确UAV探索性训练和正式E1的数据使用、有效像素及受控评估规则，再推进可复现的训练准备。
3. 继续完善跨平台空间对应关系、局部配准和NoData处理，再考虑卫星到无人机的正式域适配实验。

## 项目文件说明

| 目录 | 主要内容 |
|---|---|
| `src/data/`、`src/metrics/` | UAV候选数据读取与NumPy二分类指标 |
| `src/models/`、`src/training/` | 轻量U-Net及带有效像素掩码的训练、验证步骤 |
| `src/data_audit/` | CAS、GDCLD影像、标签和NoData检查代码 |
| `src/grid_reconstruction/` | 图块重叠匹配、相对网格重建、空间检查 |
| `src/registration/` | SAT/UAV特征匹配、配准和验证 |
| `scripts/` | 配准、空间划分、分辨率汇总及CPU冒烟测试入口 |
| `tests/` | 合成数据读取、指标和CPU模型流程测试 |
| `data/interim/reconstructed_grids/` | 重建的相对网格及诊断CSV/JSON |
| `data/interim/splits/` | 空间划分候选、统一清单和泄漏风险记录 |
| `data/interim/registration/` | 局部配准、筛查与可靠性判定结果 |
| `reports/tables/` | 数据审计、分辨率与验证结果表 |
| `reports/figures/` | 可视化检查结果 |

建议优先查看以下文件：

- `data/interim/splits/moxitaidi_unified_spatial_manifest_v1.csv`：统一影像清单及候选划分；**所有正式训练/评估批准字段目前均为False**。
- `reports/tables/uav_split_validation_recheck_v1.json`：UAV候选划分的组内缓冲验证。
- `reports/uav_exploratory_baseline_plan_v1.md`：UAV探索性实验方案及使用限制。
- `reports/tables/uav_cpu_real_smoke_run_v1.json`：2张训练候选的CPU冒烟运行记录，不是正式实验结果。
- `reports/uav_data_reader_and_visual_review_v1.md`：训练/验证候选的读取与人工核查图说明。
- `data/interim/registration/sat100_registration_reliability_decision_v1.csv`：局部配准证据等级。
- `reports/tables/effective_resolution_comparison_v1.csv`：分辨率对照及未知状态。
- `reports/tables/lushan_sensor_provenance_v1.csv`：芦山影像来源推断与待确认状态。

审计环境为AutoDL Linux、Conda `landslide-audit`、Python 3.11.18。独立 `.venv/torch-cpu` 使用PyTorch 2.6.0+cpu，并通过 `--system-site-packages` 复用部分审计环境依赖；它不是完整锁定或GPU环境。正式训练环境尚未验收。

## 数据来源

- [CAS Landslide Dataset](https://doi.org/10.5281/zenodo.10294997)
- [CAS数据集论文](https://doi.org/10.1038/s41597-023-02847-z)
- [GDCLD](https://doi.org/10.5281/zenodo.13612636)
- [GDCLD论文](https://doi.org/10.5194/essd-16-4817-2024)

**说明**：当前阶段成果包括数据审计、空间划分候选、局部配准和CPU探索性流程测试，不代表正式跨平台模型或E1实验已经完成。

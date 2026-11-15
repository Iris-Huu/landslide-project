# 卫星到无人机的跨平台滑坡分割：同址影像下的域偏移分解与适配

> **项目状态（截至 2026-11-15）**：第一周数据审计、局部配准可行性研究与实验协议整理已形成**阶段性交付**，相关结果已归档至 Git。尚未启动正式深度学习训练；**跨平台无泄漏数据划分、全域同址配准和绝对有效分辨率均未得到验证**。本 README 是仓库唯一的说明文档，实验数据表、脚本、JSON 和图片继续保留。

## 1. 研究目标与阶段边界

研究卫星影像（源域）与无人机影像（目标域）上的滑坡语义分割差异，分析成像平台、空间尺度、纹理/光谱与地域差异所引起的域偏移，并在有可靠同址证据的前提下探索跨平台适配。

目前完成的是**数据与实验设计的可行性审计**，而非已实现卫星到无人机的泛化模型。局部匹配成功不能代表所有影像可配对；模型间一致不等于独立地理真值；数据划分候选通过组内检查不等于全域空间无泄漏。

### 第一周七项任务验收

| 任务 | 当前完成程度 | 还缺什么 |
|---|---|---|
| 1. 三项受限权重申请 | **申请动作完成**：DINOv3 SAT-493M、DINOv3 LVD-1689M、SAM 3.1 已提交 | 尚未收到审批/获得可用权重；不应记录为访问获批 |
| 2. CAS 16 个影像子集下载 | **归档完整性检查完成**：16 个影像 ZIP + 1 个研究区矢量 ZIP，17/17 通过 ZIP CRC；保存子集结构检查表 | 未逐包与发布者公布的 MD5/SHA 校验值比对；部分子集图像/标签数量存在差异 |
| 3. 切片相对网格重建 | **阶段完成**：Moxitaidi SAT 652 张→164 个相对连通分量；UAV-0.6m 984 张→8 个分量 | 各分量之间、SAT 与 UAV 之间的绝对空间位置尚未统一 |
| 4. 空间分块 70/10/20 与缓冲 | **候选交付完成**：UAV 第 2 分量内部划分及缓冲距离检查通过；全 1636 张影像已进入统一清单 | 仍有 223 张 UAV 未分配、652 张 SAT 未批准；正式跨平台划分未冻结，全域泄漏未排除 |
| 5. Moxitaidi 同址配准 | **局部可行性有支持证据**：SAT100 对 UAV0519/0533 的人工核验与模型一致性检查 | 自动匹配内点较少、人工点独立性未排除、全域可靠配准未成立；不能批准正式跨平台训练配对 |
| 6. 有效分辨率对照 | **证据表及缺失解释完成**：区分论文标称、相对尺度、模糊参数及未知的绝对有效分辨率 | CAS SAT 的 0.6 m 报告值与标注的 Sentinel-2 传感器原生分辨率存在待解释差异；真实有效GSD未测定 |
| 7. GDCLD Lushan_1/2 传感器 | **元数据和论文来源核查完成，文件级未确认**：0.5 m 的 Lushan_1/2 与论文 Map World 数据一致，尚属推断 | 底层具体卫星/传感器未知；已联系作者，尚无回复；Lushan_3 的 PlanetScope 对应也未文件级证实 |

## 2. 数据、完整性与标签审计

### 2.1 CAS

- 原始数据放在服务器 `data/raw/CAS/`，**不随 Git 仓库分发**。
- 已检查 **17 个 ZIP** 的内部完整性（CRC 通过 17、失败 0），其中 16 个影像子集和 1 个 `study_areas_shp.zip`；这不能替代官方文件级哈希校验。
- 16 个影像子集的文件数量已经清点；例如 Longxi River UAV 的 `label` 数量比 `img/mask` 多 49，Tiburon Sentinel 的 `label` 比 `img/mask` 少 15。数据包通过 CRC 不代表标注配对完美。
- CAS 的 Moxitaidi SAT 有 **652** 组三通道 512×512 影像及二值掩码；Moxitaidi UAV-0.6m 有 **984** 组；已进行影像/掩码配对、掩码取值与基础像素统计审计。
- 滑坡像素均值：SAT **7.18%**，UAV **10.49%**（按掩码全图计算，**尚未剔除无数据候选区**）。
- 黑白边界连通规则仅是 **NoData 候选检测**，不应自动作为训练中的最终有效像素掩码。候选区与正类标注有交集：SAT 11,028 像素、UAV 274,141 像素，存在误删真实滑坡的风险。
- 抽查的 CAS Moxitaidi 切片无 CRS/GCP/RPC；Rasterio 的默认单位变换 `(1,1)` **不是 1 米/像素**，TIFF 的 96 DPI 亦非地面分辨率。

**主要证据：**

- `reports/tables/cas_archive_integrity_final_v1.json`
- `reports/tables/cas_16_subsets_structure_v1.csv`
- `reports/tables/moxitaidi_basics_per_image.csv`
- `reports/tables/moxitaidi_nodata_audit_v1.csv`
- `reports/tables/moxitaidi_candidate_mask_overlap.csv`
- `src/data_audit/`（审计脚本）

### 2.2 GDCLD / Lushan

GDCLD 测试包已上传、完成 ZIP 校验和本地/服务器 SHA-256 一致性检查；包含 `Lushan_1/2/3.tif` 及对应标签。图像与标签之间存在**尺寸差 1 像素及局部网格偏移/对齐问题**，不得未经对齐检验直接视作像素严格对应。

| 影像 | 图像尺寸（像素） | GeoTIFF CRS | 投影网格像素间距 | 来源判读 |
|---|---|---|---|---|
| Lushan_1 | 14745×13365 | EPSG:4544 | 0.5 m | 与论文 Map World 0.5 m 一致；**未确证** |
| Lushan_2 | 12375×13621 | EPSG:32648 | 0.5 m | 与论文 Map World 0.5 m 一致；**未确证** |
| Lushan_3 | 10274×9420 | EPSG:32648 | 3.0 m | 与论文 PlanetScope 3 m 一致；**未确证** |

GDCLD 论文还提到芦山区域 0.2 m UAV 影像（Sony ILCE-5100）；**不能据此假定现有三张测试 TIFF 中一定包含这份 UAV 影像**。Map World 是影像集成来源，不是唯一的传感器型号。作者已经收到联系请求，尚未取得文件级确认。

证据：`reports/tables/gdcld_lushan_metadata_v1.csv`、`reports/tables/lushan_sensor_provenance_v1.csv`、`reports/tables/gdcld_lushan_crs_label_overlap_v1.json` 及 `src/data_audit/` 相关脚本。

## 3. 相对网格重建（已达阶段性目标）

CAS 官方发布的裁剪示例代码使用 **512×512** 窗口、**256** 像素步长，符合本项目相邻切片可能重叠 256 像素的处理假设。但公开示例**不能证明**全部数据子集都使用完全相同的生产流水线。

| 平台 | 影像 | 相对连通分量 | 组内预测重叠且像素一致的图块对 |
|---|---:|---:|---:|
| Moxitaidi SAT | 652 | 164 | 1,409 |
| Moxitaidi UAV-0.6m | 984 | 8 | 3,162 |

- SAT 的 **149 个分量是单张切片**，另 15 个为多图分量；最大 SAT 分量 154 有 234 张，分量 158 有 81 张，分量 148 有 69 张。
- UAV 最大分量 **2** 有 761 张，其余 7 个分量共 223 张。
- 已对恢复出的组内坐标进行重复、冲突与重叠一致性检查。
- **不可恢复的内容**：缺少可靠连接证据的不同分量间偏移、跨平台统一地理位置；不应把相对网格 `grid_row/grid_col` 误认为地图坐标。

主要文件：`data/interim/reconstructed_grids/satellite_grid_coordinates_v2.csv`、`uav_grid_coordinates_v2.csv`、两份 `*_grid_overlap_validation_v2.csv`；脚本在 `src/grid_reconstruction/`。

## 4. 空间划分候选（未冻结）

当前只对 **UAV 第 2 相对连通分量**（761 张）构建训练/验证/测试候选与缓冲；其余 UAV 分量未进入候选实验。全 1,636 张 SAT+UAV 图块已进入统一清单，每张恰好一条记录。

| UAV 状态 | 数量 | 在 704 张有效候选中的比例 |
|---|---:|---:|
| `train_candidate` | 484 | 68.75% |
| `val_candidate` | 71 | 10.09% |
| `test_candidate` | 149 | 21.16% |
| `buffer` | 57 | 不计入 |
| `unassigned` | 223 | 不计入 |

基于重建网格（切片512像素、步长256像素）检查了训练–验证、训练–测试、验证–测试图块的边界最小间隔，分别为 **512、3840、512 UAV像素**；对应 34,364、72,116、10,579 组比较，总计 **117,059** 组，违规 **0**。该通过结论仅适用于**UAV 第2连通分量内部**。

### 使用限制（必须遵守）

1. `buffer` 与 `unassigned` **不得自动纳入**当前实验。
2. SAT 全部 **652** 张的跨平台训练批准为 **False**；统一清单中所有影像的 `approved_for_training` 与 `approved_for_evaluation` 均为 **False**，这是跨平台严格协议的保守状态，**不是所有UAV内部基线都不可能开展**。
3. 不同 UAV 分量的真实相对位置、SAT-UAV 全局覆盖、无数据区有效性未验证，**不得宣称全局无泄漏**。
4. `UAV0846` 等测试候选图块曾用于探索性配准/审计，**不是严格未接触的盲测集**。正式报告前需重新定义受控评估协议，避免以测试数据选模型或调参。
5. 现有文件是 **candidate/探索性清单**，不是已冻结的最终跨平台 train/val/test。

证据：`data/interim/splits/moxitaidi_uav_split_candidate_v1.csv`、`moxitaidi_unified_spatial_manifest_v1.csv`、`satellite_leakage_risk_v1.csv`、`reports/tables/uav_split_validation_recheck_v1.json`；脚本 `scripts/splits/build_unified_spatial_split_manifest.py`、`src/grid_reconstruction/check_split.py`。

## 5. Moxitaidi SAT→UAV 局部配准（局部可行、全域未证）

以 SIFT 特征、目标特征去重、RANSAC 局部仿射估计等方法筛查跨平台同名区域；部分早期批量结果存在退化变换，**不能将匹配失败解释成地理无重叠**。SAT168 到测试候选 UAV0846 的九张卫星邻域筛查均少于4个RANSAC内点，没有建立可靠配准，也无法据此排除空间泄漏。

SAT100 与 UAV0519、UAV0533 的可辨识地物在视觉和几何上具有局部对应证据：

| 配对 | RANSAC内点 | 人工核验点数 | 人工点 RMSE（UAV px） | 证据标记 |
|---|---:|---:|---:|---|
| SAT100 → UAV0519 | 7 | 4（第二轮） | **3.595** | `A_candidate` |
| SAT100 → UAV0533 | 5 | 4 | **6.172** | `B_candidate` |
| SAT100 → UAV0518 | 5 | 无 | 未测 | `review_required` |

UAV0519 与 UAV0533 两套模型在同一 UAV 相对网格上的 SAT 中心预测差 **2.53 像素**、五个检查位置最大差 **9.86 像素**；UAV0518 与 UAV0519 中心差 **18.89 像素**、最大差 **70.19 像素**，故保持待复核。

**重要解释**：人工点只是探索性核验，可能存在判读/点击偏差，尚未验证其与原SIFT内点完全独立；UAV0519 第一轮4点也已保留（不应只公布误差更低的第二轮）。等级只是项目内部证据标签，**并非正式精度认证**，三个组合 `training_pair_approved=False`。

主要证据：`data/interim/registration/sat100_registration_reliability_decision_v1.csv`、`sat100_uav0519_anchor_check_result.json`、`sat100_uav0533_neighbor_result.json`、`reports/tables/sat100_uav0519_manual_validation_v1.csv`（8条/两轮）、`sat100_uav0533_manual_validation_v1.csv`（4条）、`data/interim/registration/uav_group2_train_anchor_screening_v1.csv`；脚本位于 `src/registration/`、`scripts/registration/`。

## 6. 分辨率与影像来源证据（已交付对照表，物理真值未得）

| 对象 | 现有证据 | 尚不能得出的结论 |
|---|---|---|
| CAS Moxitaidi SAT | 论文表格报告 **0.6 m/px**，传感器标为 Sentinel-2/L2A；局部配准尺度约 **2.000178–2.000300 UAV px/SAT px** | 不能确定真实/原生 GSD 或绝对有效分辨率 |
| CAS Moxitaidi UAV-0.6m | 数据包命名（以及论文报告）标称 **0.6 m/px** | 不能确认切片的实测有效空间分辨率 |
| SAT001相关探索窗口 | 4条高斯模糊匹配记录，最优 σ **0.9–1.1 UAV像素** | σ 不是米/像素，也不能单独当作真实有效GSD |
| GDCLD Lushan_1/2/3 | GeoTIFF投影网格间距分别为 **0.5/0.5/3 m/px** | 网格间距本身不能证明底层具体传感器或原生光学分辨率 |

**关键待核实矛盾**：公开 CAS 论文中 Moxitaidi SAT 的0.6米报告值与 Sentinel-2 MSI 原生10/20/60米波段规格存在待解释差异；不能直接假定两平台实际分辨率一致。发布者公开的 `Cropping.py` 示例以 PIL `crop()` 按像素窗口裁剪、保存TIFF，**没有显式重采样或地理参考保留**；这只能解释公开裁剪示例的行为，**不能证明原始影像没有预处理**。需要向发布方核实卫星来源、报告值的含义、原始GSD和处理链。

证据：`reports/tables/effective_resolution_comparison_v1.csv`、`effective_resolution_audit_v1.csv`、`scripts/resolution/build_effective_resolution_comparison.py`。

## 7. 复现环境与仓库结构

审计环境（已记录的主要版本，**不是完整锁定环境**）：

| 项目 | 版本 |
|---|---|
| 服务器 | AutoDL Linux |
| Conda环境 | `landslide-audit` |
| Python | 3.11.18 |
| NumPy | 2.4.6 |
| Rasterio | 1.4.4 |
| SciPy | 1.17.1 |
| Matplotlib | 3.11.2 |
| Pillow | 13.0.0 |

```bash
cd /root/autodl-tmp/landslide_project
conda activate landslide-audit

# 已完成结果的轻量级报告再生成（需本地原始输入CSV/JSON仍在指定位置）
python scripts/registration/generate_manual_validation_report.py
python scripts/registration/generate_registration_reliability_decision.py
python scripts/splits/build_unified_spatial_split_manifest.py
python scripts/resolution/build_effective_resolution_comparison.py
```

目录用途：

| 路径 | 内容 |
|---|---|
| `src/data_audit/` | CAS/GDCLD基础审计、标签与无数据区核查 |
| `src/grid_reconstruction/` | 切片关系、相对网格、重叠验证与划分检查 |
| `src/registration/` | SIFT匹配、筛查、局部变换和可视化 |
| `scripts/registration/`, `scripts/splits/`, `scripts/resolution/` | 阶段性表格/决策汇总的可复现脚本 |
| `data/interim/reconstructed_grids/` | 已重建的相对网格和诊断记录 |
| `data/interim/registration/` | 配准候选、变换JSON、可靠性判定 |
| `data/interim/splits/` | 候选划分、统一清单、SAT风险记录 |
| `reports/tables/` | 结构化审计/验证 CSV 和 JSON（**保留**） |
| `reports/figures/` | 已纳入版本管理的检查图片 |
| `data/raw/` | 仅服务器端的原始数据，不包含在Git仓库中 |

**训练环境尚未验收**：曾有 `torch` 导入失败记录；是否有CUDA/GPU、可用PyTorch及受限模型权重，需要在启动训练前重新检查。尚无可信的正式训练权重、IoU/F1基线或跨域适配实验结果。

## 8. 仍未完成的工作（下一阶段优先级）

1. **最高优先级：正式实验协议**。为单平台UAV基线明确可用候选集合与受控评估规则，处理已被探索的测试候选图块；跨平台正式协议必须先有可信的SAT-UAV空间隔离证据并冻结版本。
2. **跨平台几何覆盖扩展**。在更多区域增加独立验证控制点、稳健/失败案例分析，检查匹配模型与网格的偏差；完善跨分量空间关联，不能以低SIFT内点或匹配失败证明无重叠。
3. **NoData与标签对齐**。复核候选无效区和正类交叉；处理 GDCLD 芦山影像与标签尺寸/网格偏移，定义可靠的评估掩码。
4. **真实地面分辨率与来源**。解释 CAS SAT 的0.6 m/Sentinel-2矛盾；Lushan_1/2实际提供方与传感器需文件级证据，Lushan_3的PlanetScope身份亦需确认；绝对有效分辨率未测定。
5. **数据真实性核查**。可进一步比对17个CAS ZIP与发布者原始校验值；核查有标签数差异的子集是否能严格一一配对。
6. **模型训练准备**。确认GPU、CUDA、PyTorch、数据加载器、指标和权重授权；从严格描述限制的UAV单平台基线起步，再讨论SAT→UAV域适配。三项受限模型权重申请仍待回复。

**禁止将当前项目描述为**“全数据集无泄漏划分完成”“卫星与无人机全域同址配准完成”“绝对有效分辨率已测得”“受限权重已获批”或“已经完成模型训练”。

## 9. 数据、论文与可追溯性

- CAS Landslide Dataset（数据）：https://doi.org/10.5281/zenodo.10294997
- CAS论文（Xu等，*Scientific Data*）：https://doi.org/10.1038/s41597-023-02847-z
- CAS官方示例制作代码：https://github.com/Aizu0/CAS-Landslide-Dataset-production-code
- GDCLD（数据）：https://doi.org/10.5281/zenodo.13612636
- GDCLD论文（Fang等，*Earth System Science Data*）：https://doi.org/10.5194/essd-16-4817-2024
- Sentinel-2 MSI官方规格：https://s2.pages.eopf.copernicus.eu/msi/s2msi/main/PDFS_ADFS/index.html

当前阶段性成果归档基线提交：`cd5e959`（2026-11-15）。本次整理仅改变仓库说明文档的组织方式，保留代码、结构化证据、图片和历史提交；Git 历史中仍能找到已移除的旧报告文本。

> **说明**：本 README 中的统计和证据等级来自截至上述提交的项目审计记录。它们应视作当前的探索性/阶段性交付，而非独立复现实验或已发表结论。使用第三方数据前应核查其许可与发布条款。

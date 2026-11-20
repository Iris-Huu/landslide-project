# UAV 数据读取与叠加图核查 v1

## 范围与复用

只使用统一清单 `UAV` 第 2 连通组 `train_candidate` 和 `val_candidate`，不打开其他划分的影像或 mask。新增 `src/data/uav_data.py`，后续新代码通过 `select_samples` 统一筛选，通过 `read_sample` 读取；历史审计脚本保持原样，其筛选逻辑仍是独立的历史实现。筛选函数检查必要字段、UAV、分量、允许的两个划分、候选文件名唯一性与安全性，并记录清单哈希。读取时逐对检查文件存在、RGB 波段标记、三通道/单通道、uint8、影像和 mask 尺寸一致、mask 取值 0/1；异常包括文件名和原因。RGB 波段解释必须明确为红、绿、蓝；遇到无标记或其他顺序会拒绝而非猜测。函数允许指定分量，但本轮只使用 2。只做数组转换，不改清单或审批状态。

输出 `image` 为 `(3,H,W)` RGB、`float32`、每个 uint8 数值除以 255 得到 [0,1]；`mask` 为 `(H,W)`、`uint8`、0/1；`valid_mask` 为 `(H,W)`、`bool` 全 True，策略名为 `all_pixels_provisional`。这只是暂时计入所有像素的约定，不是有效像素已验证。返回元数据记录样本名、划分、分量、清单路径及 SHA-256、预处理和有效像素策略。无疑似区域自动排除。

```python
from pathlib import Path
from src.data.uav_data import select_samples, read_sample
manifest = Path('data/interim/splits/moxitaidi_unified_spatial_manifest_v1.csv')
base = Path('data/raw/CAS/Moxitaidi_UAV_0.6m')
records = select_samples(manifest, 'train_candidate', component='2')
sample = read_sample(records[0], base)
image, mask, valid, provenance = (sample[k] for k in ('image', 'mask', 'valid_mask', 'metadata'))
```

合成测试在临时目录创建小型 TIFF 与清单；10 项测试通过，覆盖合法 RGB 缩放、尺寸、影像及 mask 通道和 dtype、非法标签、缺失文件、清单重复及缺字段、禁用划分、错误有效区域策略和错误 RGB 波段标记。另从每组按文件名字典序取前三张，共 6 张真实影像及对应 mask，通过形状、类型、数值范围及全像素策略检查；见 `reports/tables/uav_train_val_review_selection_v1.json`。原有指标 9 项测试亦可独立运行。

## 人工核查图

每组选择两张：在既有疑似区域审计中，疑似像素为 0 的文件名字典序第一张；以及疑似区域与正类重叠像素数最大的样本（数量相同时按文件名字典序）。这种选法是可复现的示例，不是随机样本或质量代表。审计输入哈希、文件、数值和绘图参数见 `reports/tables/uav_train_val_review_selection_v1.json`。四张图保存在 `reports/figures/uav_train_val_review_v1/`；该路径受 `.gitignore` 忽略；四张核查图保存在服务器，当前未纳入Git。

图像从左至右为影像、二值标签、红色标签叠加、疑似区域叠加。黄色是边缘连通的全通道纯黑或纯白的疑似区域，品红是疑似区域与正类标签交集；颜色的含义也在图例中写明。这里直接复用 `src/data_audit/audit_nodata_candidate.py` 的 `find_candidate`，并逐图对比本次计算的候选与交集像素数与既有审计记录一致；规则未经验证。

已打开四张图目视检查，标题、图例与颜色可辨：

- `luding_UAV0164`（训练，普通）：红色标签覆盖两块可见的浅色裸露区域。此视觉对应不足以证实边界或空间对齐精度。
- `luding_UAV0305`（训练，疑似交集最大）：右上有大片白色区域，黄色疑似区覆盖该区域，品红标出了正类延伸到白色区域的一小段。建议核查该白色区域成因及标注边界，不自动判断其无效。
- `luding_UAV0651`（验证，普通）：图底部红色标签覆盖一片植被较多的区域，单靠该图难以判断标注边界是否准确，建议结合原始标注与上下文人工核查。
- `luding_UAV0779`（验证，疑似交集最大）：底部亮白区域的正类标签内出现明显品红交集；建议重点检查高亮区域与标签真实性。疑似区域和正类相交不单独证明规则误判或标签错误。

四图都只是局部可视化，不证明配准正确、无空间泄漏或标签准确。正式训练与评估审批仍为 False，协议和有效区域处理仍待确认。本轮未训练或推理。

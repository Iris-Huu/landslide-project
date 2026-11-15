# 第一周空间划分阶段性交付说明（v1）

## 交付性质

本文件是受限范围的候选划分交付，不是正式冻结或全局无泄漏证明。

## 数据范围

- UAV总计984张，8个相对连通组。
- 第2连通组761张参与候选划分及缓冲区设计。
- 其余223张UAV影像保持unassigned，不进入当前实验。
- SAT总计652张，本次检查批准训练数量为0。

## 候选划分

| 类别 | 切片数量 | 占704张候选实验影像比例 |
|---|---:|---:|
| 训练候选 | 484 | 68.75% |
| 验证候选 | 71 | 10.09% |
| 测试候选 | 149 | 21.16% |
| 缓冲区 | 57 | 不计入 |
| 未分配 | 223 | 不计入 |

## 第2连通组的空间间隔复核

| 集合对 | 最小切片边界间距（UAV像素） | 违反数量 |
|---|---:|---:|
| train_candidate / val_candidate | 512 | 0 |
| train_candidate / test_candidate | 3840 | 0 |
| val_candidate / test_candidate | 512 | 0 |

## 尚未解决的风险

- 其他UAV连通组相对于第2组的真实地理位置尚未验证。
- SAT与UAV的全局空间覆盖关系尚未可靠建立。
- 局部SIFT配准失败不能用于证明两张影像地理上不重叠。
- 有效无数据区掩码与真实地理位置仍存在不确定性。
- 测试候选影像曾参与探索性空间审核，不属于完全盲测数据。

## 使用规则

- 不将buffer或unassigned影像纳入当前训练及评估。
- 不将未经验证的SAT影像纳入正式训练。
- 在完成跨平台泄漏审核与划分冻结前，不报告最终无泄漏泛化性能。
- 若研究必须使用SAT训练，则本划分不能直接作为正式实验协议。

## 可复核依据

- 候选划分：`data/interim/splits/moxitaidi_uav_split_candidate_v1.csv`
- UAV相对网格：`data/interim/reconstructed_grids/uav_grid_coordinates_v2.csv`
- 距离检查：`reports/tables/uav_split_validation_recheck_v1.json`
- SAT风险清单：`data/interim/splits/satellite_leakage_risk_v1.csv`

状态：阶段性候选交付完成；正式划分冻结未完成。

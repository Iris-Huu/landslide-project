# Moxitaidi统一空间划分候选清单校验

## 1. 完整性

- SAT图块：652
- UAV图块：984
- 总记录数：1636
- 每张图块在统一清单中恰好出现一次。

## 2. UAV划分

| 状态 | 数量 |
|---|---:|
| train_candidate | 484 |
| val_candidate | 71 |
| test_candidate | 149 |
| buffer | 57 |
| unassigned | 223 |

## 3. 有效候选划分比例

- Train：68.75%
- Val：10.09%
- Test：21.16%

## 4. 风险与使用限制

- 仅UAV最大连通分量的704张图块进入训练/验证/测试候选集。
- UAV缓冲区57张、未分配223张均不得自动进入候选训练或测试。
- SAT全部652张暂不分配训练/验证/测试角色。
- 当前未证实所有SAT与UAV图块之间的空间无泄漏。
- 此文件是候选划分清单，不是已经冻结批准的全平台实验划分。
- 本清单全部approved_for_training和approved_for_evaluation均为False。

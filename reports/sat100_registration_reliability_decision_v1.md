# SAT100局部跨平台配准可靠性判定

## 1. 证据分级

| 配准组合 | 等级 | RANSAC内点 | 人工RMSE(px) | 中心一致性差(px) |
|---|---|---:|---:|---:|
| SAT100 → luding_UAV0519.tif | A_candidate | 7 | 3.595 | 2.5316 |
| SAT100 → luding_UAV0533.tif | B_candidate | 5 | 6.1718 | 2.5316 |
| SAT100 → luding_UAV0518.tif | review_required | 5 | 未验证 | 18.8939 |

## 2. 判定依据

- A/B仅表示局部探索证据的相对强弱，并非正式精度认证。
- UAV0519和UAV0533模型相互一致，但仍存在人工核验误差。
- UAV0518与其他模型尺度及几何位置不一致，暂不批准。
- 人工点尚未证明独立于原SIFT/RANSAC拟合内点。
- 当前尚未证明SAT与UAV全区域无空间泄漏。
- 所有组合的training_pair_approved均为False。

## 3. 阶段结论

SAT100与UAV0519、UAV0533具有局部同址配准可行性证据。
目前仅作为方法可行性研究，不作为正式无泄漏训练配对。
UAV0518保持待复核状态。

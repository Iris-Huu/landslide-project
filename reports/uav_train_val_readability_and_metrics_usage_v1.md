# UAV 读取核验与二分类指标：使用说明

本轮仅检查统一清单中 UAV 第 2 连通分量的 `train_candidate` 和 `val_candidate`，没有读取其他划分影像或标签。逐图结果及异常详情见 `reports/tables/uav_train_val_readability_details_v1.csv`；汇总、运行环境、清单和旧无数据审计表的 SHA-256、读取错误、限制见 `reports/tables/uav_train_val_readability_v1.json`。滑坡像素占比是全图口径；旧无数据疑似区域的统计没有重新计算，也没有用于排除像素。可读和尺寸一致不能证明真实地理对齐。

重跑时指定两个**尚不存在**的新输出路径；脚本不会覆盖现有报告。它完整读取影像和 mask 的每个通道，并阻止 GDAL 在原始数据旁创建 PAM 辅助文件。单个文件读取错误会记录后继续；若存在异常，汇总写出后进程以非零码结束。

```bash
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/landslide-audit/bin/python -B src/data_audit/check_uav_train_val_readability.py --summary /tmp/uav_read_summary_new.json --details /tmp/uav_read_details_new.csv
```

指标模块 `src/metrics/binary_segmentation.py` 仅依赖 NumPy；每次 `update` 接收形状相同的预测二值数组、真实二值数组及可选布尔型有效掩码。未提供掩码时统计全部像素，并不表示有效区域已验证。可对每个样本调用 `update`，`result()` 给出整个样本集累计的 TP、FP、FN、TN、分母和指标。它返回的是像素汇总而非逐图平均。无有效像素的样本数记录在 `empty_samples`；如果整个集合没有有效像素，调用方应停止评价并排查。

```python
import json
import numpy as np
from src.metrics.binary_segmentation import ConfusionMatrix

meter = ConfusionMatrix()
meter.update(np.array([1, 1, 0]), np.array([1, 0, 1]),
             valid_mask=np.array([True, True, False]))
result = meter.result()  # TP=1, FP=1, FN=0, TN=0
print(json.dumps(result, allow_nan=False))
```

分母为零的指标是 Python `None`，JSON 中合法表示为 `null`；使用 `allow_nan=False` 防止输出非标准 NaN。模块拒绝非法标签及非布尔有效掩码，包括被排除区域中的非法标签。自动测试：

```bash
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/landslide-audit/bin/python -B -m unittest discover -s tests -p test_binary_segmentation.py -v
```

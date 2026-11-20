# UAV候选划分GPU探索性流程与结果 v1

本次仅用于探索性流程检查，不是正式E1、独立测试或跨平台迁移结果。数据来自统一清单UAV第2相对连通组的训练和验证候选；未读取test_candidate、buffer、unassigned或SAT影像与标签。此前候选测试区域曾被探索，不能称作未接触的盲测集。所有正式批准字段仍为False。

## 环境与运行

- RTX 5090，32,607 MiB；驱动595.71.05，`nvidia-smi`显示驱动支持CUDA 13.2；现有PyTorch 2.8.0+cu128使用CUDA 12.8构建，实际CUDA张量运算成功。GPU初始报告占用2 MiB、利用率100%，未列出进程；该瞬时利用率不能据此解释。
- 基于 `/root/miniconda3/bin/python` 新建 `.venv/torch-gpu`，使用 `--system-site-packages` 复用base中的GPU PyTorch，独立安装Rasterio 1.4.4；不修改base、landslide-audit或CPU虚拟环境。它不是完全隔离或锁定的环境。

```bash
/root/miniconda3/bin/python -m venv --system-site-packages .venv/torch-gpu
.venv/torch-gpu/bin/python -m pip install --no-cache-dir --disable-pip-version-check --index-url https://pypi.org/simple 'rasterio==1.4.4'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/torch-gpu/bin/python -B -m unittest discover -s tests -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/torch-gpu/bin/python -B scripts/training/uav_exploratory.py --mode train --crop 256 --base-channels 8 --batch-size 4 --epochs 3
```

32项测试通过，包含合成检查、验证不改变模型参数和状态、禁止划分、CPU/GPU检查点与随机状态恢复；另有GPU合成前向、反向及参数更新。初次GPU冒烟在读取数据前因CUDA上下文未初始化而重置显存统计失败，修正初始化顺序后通过。训练结束后检查点GPU恢复曾因随机状态被映射到GPU失败，已修复为将随机状态移回CPU，添加GPU回归测试并实际恢复best检查点；训练结果与原检查点未覆盖。训练时的旧辅助代码快照保存在该运行目录，哈希与当次记录一致。

GPU真实样本冒烟：两张训练中心裁剪、两步更新，损失0.87218→0.86684。第一次小样本检查所选两个裁剪没有正类，损失下降仅说明背景拟合；后改选按文件名排序最早的两个含正类训练中心裁剪UAV0096、UAV0098（正类像素704、278），30步损失0.86204→0.06066，仅为实现检查。

## 受控候选验证

基准使用16个训练候选与4个验证候选，256×256中心裁剪、batch 4、基础通道8；训练0.739秒、验证0.029秒，峰值分配约124.5 MiB。因基准用时和显存占用较低，选择全部484个训练候选和71个验证候选的中心裁剪、batch 4、3轮。配置：种子17，Adam学习率0.001，每轮基于种子+轮次的随机采样顺序，无增强、无额外归一化常数，二值BCEWithLogits，仅在暂用有效像素上计数，预测阈值0.5。影像、标签、有效掩码共享中心裁剪；验证无梯度及参数更新。每张仅使用256×256中心区域，**不代表整张512×512图块的评价**。

| 轮次 | 训练损失 | 验证损失 | 训练+验证秒数 |
|---|---:|---:|---:|
| 1 | 0.392493 | 0.344027 | 4.214 |
| 2 | 0.268947 | 0.329263 | 2.815 |
| 3 | 0.261789 | 0.332163 | 2.817 |

三个轮次阈值0.5的验证汇总相同：TP=0、FP=0、FN=534758、TN=4118298；滑坡IoU=0、F1=0、Recall=0、Precision分母为0，表示为JSON `null`。模型预测全背景；损失下降不代表正类分割有效。IoU选择最佳检查点的规则严格大于，三轮并列时保留第1轮；最低验证损失在第2轮，不应混淆两种选择标准。

训练主过程开始到结束约10.76秒，GPU峰值分配130,500,096字节约124.5 MiB（PyTorch分配量，不等同全部GPU占用）。运行日志、样本名、哈希、环境、每轮结果及可信本地`best.pt`、`last.pt`位于 `experiments/uav_train_0ev4mlp7/`。`torch.load(..., weights_only=False)`仅能加载可信的本地产物。恢复训练可用相同配置加 `--resume experiments/uav_train_0ev4mlp7/last.pt --epochs 目标总轮数`；恢复创建新运行目录。实验目录和模型检查点由Git忽略。

`all_pixels_provisional`把所有裁剪像素暂作有效，未经验证；疑似无数据规则可能误删正类，未用于本次屏蔽。相对网格只做过组内部分间隔检查，跨连通组/跨平台泄漏、标签地理对齐、真实有效区域与正式测试协议仍未验证。结果仅反映探索性候选划分及暂用全像素口径，等待老师和学长确认正式协议。

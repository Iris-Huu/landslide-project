# 轻量 U-Net 的 CPU 合成流程验证

## 环境

隔离环境：项目根目录 `.venv/torch-cpu`（被 Git 忽略）。上次创建时使用：

```bash
/root/miniconda3/envs/landslide-audit/bin/python -m venv --system-site-packages .venv/torch-cpu
.venv/torch-cpu/bin/python -m pip install --no-cache-dir --disable-pip-version-check --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0+cpu'
```

PyTorch CPU wheel 下载自官方 `download.pytorch.org` / `download-r2.pytorch.org`；安装到虚拟环境中，没有修改 landslide-audit。此环境启用了 `--system-site-packages`，会**只读复用**基础审计环境里可见的包（包括 NumPy），不是完全自包含的锁定环境；基础环境改变可能影响复现。实际验证：Python 3.11.18、torch 2.6.0+cpu（`torch.version.cuda` 为 `None`）、NumPy 2.4.6。显式使用 CPU 和 1 个线程，计算 `[1,2]*2+1` 得到 `[3,5]`，设备为 `cpu`。

进入环境和检查（本次未申请或使用 GPU）：

```bash
source .venv/torch-cpu/bin/activate
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -c 'import torch; torch.set_num_threads(1); x=torch.tensor([1.,2.],device="cpu"); print(torch.__version__, torch.version.cuda, (x*2+1).tolist(), x.device)'
```

上次 base 环境的 CUDA 构建 `2.8.0+cu128` 导入进程曾以 137 退出，原因未确认；不能仅根据 137 判定原因。本次 CPU 环境单独运行成功。此前可读取的 cgroup 内存上限是 2 GiB；它不等于该次退出的确切原因。

## 接口与限制

`src/models/unet.py` 的 `SmallUNet(base_channels=8)` 是两次下采样的轻量 U-Net。输入为 `N×3×H×W` 浮点 RGB，H、W 均至少 4 且为 4 的倍数；输出为 `N×1×H×W` 的 logits，末尾不加 sigmoid，权重是本地随机初始化。真实预处理的单张 CHW 数据还需由未来调用者组 batch，并把 mask 和有效像素掩码调整为 `N×1×H×W`；本轮没有实现真实数据训练入口。

`src/training/segmentation_steps.py` 的 `masked_bce` 只对布尔有效像素计算 `binary_cross_entropy_with_logits` 均值。整个批次没有有效像素时抛出明确错误；不会静默计算 NaN。它检查形状、有限 logits 及 0/1 标签。`train_step` 负责前向、反向和优化器更新；`validation_step` 在无梯度模式下运行，sigmoid 后以给定阈值（默认 0.5）取得二值预测，复用既有 NumPy 混淆矩阵及指标。`ExperimentConfig` 保存基础通道、学习率、随机种子、阈值、输入尺寸和暂用有效像素策略，调用者须将配置显式传递给模型及步骤；阈值不是针对真实数据调优的结果。

`all_pixels_provisional` 是探索性全像素约定，并未验证真实有效区域；实现接收调用者提供的有效掩码，不自动排除疑似无数据区域。当前没有检查点保存/恢复实现，也没有正式训练循环、真实数据适配或模型权重文件。

## 实际验证和复现

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/torch-cpu/bin/python -B -m unittest discover -s tests -p 'test_unet_cpu.py' -v
```

6 项 CPU 合成测试通过：输入输出形状和有限损失；前向、反向和参数更新；有效像素损失与梯度（无效位置梯度为零）以及全无效时报错；手工 logits 下 TP/FP/FN/TN 均为 1、IoU 为 1/3、F1/Precision/Recall 为 0.5；非法输入；同环境、固定种子、单线程的两次运行损失和更新后的参数完全相等。独立一次种子 17、16×16、base_channels=2、SGD 学习率 0.01 的合成步骤损失为 0.6929681897163391，26/26 参数张量改变，最大绝对变化 0.0003808364272117615。数字仅用于记录一次功能测试，不是模型精度。固定种子不保证跨硬件、版本或线程配置均完全一致。没有检查点实现，所以未执行保存恢复测试。

上轮审计环境的数据模块 10 项、指标模块 9 项测试通过；本轮 U-Net 测试只使用合成张量，不访问真实影像。下一步接真实数据前，仍需获得探索性使用许可、确认数据与受控评价协议、检查标签对齐与有效区域策略、明确真实训练配置和计算资源。现有正式批准字段为 False；不得据合成测试宣称真实滑坡精度、全域无泄漏或正式上界。GPU 对这些小型合成检查不是必需；真实规模训练时可能显著缩短耗时，是否必要应根据模型大小、批量和时间要求评估。

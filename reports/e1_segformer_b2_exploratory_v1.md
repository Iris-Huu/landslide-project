# SegFormer-B2候选全图探索与DeepLab配置对照 v1

本报告为候选划分上的开发性探索，不是正式E1、独立测试或论文性能比较。用户提供的任务书第8页转述列出DeepLabV3+/ResNet-50、SegFormer-B2、DINOv3 ViT-L/16（LVD-1689M、SAT-493M）；未在服务器核验原文。DINOv3两种权重及SAM访问据用户说明已重新申请、等待审核，本轮未申请、下载或绕过权限。

## 结构与工程

使用segmentation-models-pytorch 0.5.0的`Segformer`与`mit_b2`编码器、随机初始化、无权重来源或哈希，24,722,369参数。编码器为`MixVisionTransformerEncoder`、解码器为`SegformerDecoder`；多尺度特征先双线性插值到输入四分之一大小（`align_corners=False`），经融合及1×1单通道logits头后，SMP自带双线性4倍上采样（`align_corners=True`），无末端激活。输入512×512时输出512×512；代码不额外插值标签。结构实现与预训练权重是不同问题，本轮没有自动下载权重。`src/models/segformer.py`支持明确指定可信且已获许可的本地encoder state_dict；权重来源/哈希由运行记录保存。

SegFormer通过 `scripts/training/uav_deeplabv3plus.py --architecture segformer --encoder mit_b2` 使用与DeepLab相同的`CandidateDataset`、`train_step`、`evaluate`、检查点和日志流程；旧入口默认DeepLab，保留旧检查点兼容。损失为训练/验证均使用批内有效像素池化的soft Dice + BCE，Dice平滑1、两个系数各1；验证损失是逐批损失按有效像素数加权平均，混淆矩阵和指标先汇总验证像素再计算。`all_pixels_provisional`不是经过验证的NoData掩码。

48项测试通过，包括随机合成张量输出尺寸、有限损失/梯度、GPU更新、无有效像素报错、eval不修改模型状态、禁止划分、模型及优化器状态检查点恢复；4张真实训练候选UAV0094–0097的128×128、batch 2冒烟完成两步、耗时2.401秒、首个参数改变，峰值PyTorch分配517623296字节。训练候选前16张的512全图、batch 4基准4步2.681秒、峰值3955378688字节约3.68 GiB，显存足够采用与DeepLab相同的batch 4。冒烟与基准均只用于检查，不是性能评估。

## 配置与可比性

两组使用统一清单SHA-256 `54ab9a9fd80ab321bc4f0a18ad259b5a467b3a5502c1c70a0c184e4a7c3b281a`；训练样本名单SHA-256 `4df89c112f475e2e3fb868b5b4f3e3b5ba274c90a84bcce4a5eb68511a8b50`（484张）；验证名单SHA-256 `5a644079b5f40d6def42e60bd32f298b62da73ca72a1cd4a0e21eeee92cb7af5`（71张）。两组均为512×512全图、随机初始化、RGB除255、无增强、种子17、每轮seed+epoch打乱、batch 4、Adam学习率0.0001、5轮、Dice+BCE、阈值0.5、全像素暂用、验证像素汇总。环境均为RTX 5090、PyTorch 2.8.0+cu128、SMP 0.5.0。

不同项：DeepLab使用ResNet-50及DeepLabV3+ ASPP/低层解码结构、26,677,585参数；SegFormer使用MiT-B2和多尺度MLP解码、24,722,369参数，各自默认内部结构、插值及随机初始化不同。只运行一个种子和5轮、预训练均未使用、收敛未确认、验证集已用于模型/轮次选择，因此只能作此配置下的开发对照，不能归因为模型能力差异或视为正式论文比较。

## 逐轮结果

| 轮次 | SegFormer训练损失 | 验证损失 | 验证IoU@0.5 | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.153842 | 1.321997 | 0.002877 | 0.005737 | 0.335456 | 0.002893 |
| 2 | 0.928903 | 1.332295 | 0.000248 | 0.000496 | 0.491620 | 0.000248 |
| 3 | 0.823912 | 1.077109 | 0.171450 | 0.292715 | 0.766165 | 0.180917 |
| 4 | 0.738404 | 0.965162 | 0.322621 | 0.487851 | 0.426626 | 0.569592 |
| 5 | 0.654841 | 0.883737 | 0.341075 | 0.508659 | 0.516523 | 0.501031 |

SegFormer第5轮最佳验证TP=1067165、FP=998890、FN=1062772、TN=15483397；比较DeepLab第4轮IoU=0.348708、F1=0.517100、Precision=0.549729、Recall=0.488126。best使用每轮阈值0.5验证IoU严格大于才替换，平分保留最早；last为第5轮。SegFormer运行总耗时69.58秒、PyTorch峰值分配3955378688字节；DeepLab总耗时45.08秒、峰值3012159488字节。这是本机单次运行计时而非标准基准。

`experiments/uav_segformer_train_ydt54mjz/`保存逐轮完整混淆矩阵及指标、配置、样本/清单哈希、源码哈希、Git状态、环境、`best.pt`及`last.pt`；已在GPU加载best并核对模型参数及345项优化器状态。仅可信本地检查点可反序列化。冒烟 `experiments/uav_segformer_smoke_8xnmal_9/`、基准 `experiments/uav_segformer_benchmark_thavosci/`；这些目录及检查点被Git忽略。

正式实验前仍需解决任务书权重及许可证、真实有效像素/NoData、标签对齐、组间和跨平台空间隔离、测试候选既往探索的处置、独立评估和多随机种子协议；具体见 `reports/uav_independent_evaluation_e1_preparation_v1.md`。本轮没有读取test_candidate、buffer或unassigned影像和标签。

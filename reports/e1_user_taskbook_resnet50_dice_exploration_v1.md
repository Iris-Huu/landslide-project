# E1任务书转述与DeepLabV3+/ResNet-50探索准备 v1

## 信息来源和配置边界

以下为**用户提供的课题任务书第8页转述**，服务器本轮没有发现原文，未自行核验原文：三类模型、四种具体配置分别为DeepLabV3+ / ResNet-50（与CAS论文可比基准）、SegFormer-B2、DINOv3 ViT-L/16 + 简单解码头（分别采用LVD-1689M与SAT-493M预训练版本）。SAM 3文本提示和SegEarth-OV3为零样本对照，不属于上述三类骨干。任务书规定Dice + BCE，损失和评价忽略无数据像素。用户表示DINOv3及SAM访问权限已重新申请、等待审核；未核验审批状态，未下载受限权重。

仍待核实：Dice/BCE权重、Dice平滑项与批次/逐图聚合、训练轮数/优化器/学习率/增强/随机种子与重复次数、输入分辨率与分块、预训练来源及授权、参数冻结与解码头细节、NoData真值、验证/独立测试使用方式和阈值。以下数值是探索性工程选择，不假称任务书指定。

## ResNet-50与损失

使用 `segmentation-models-pytorch==0.5.0` 的DeepLabV3Plus，encoder ResNet-50、output stride 16、ASPP率12/24/36、RGB输入及单通道logits，26,677,585参数。随机初始化，无预训练权重（来源/哈希为空）；不与CAS论文声称公平可比。

[TorchVision ResNet-50官方文档](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50)提供IMAGENET1K_V1/V2及RGB除255后均值`[0.485,0.456,0.406]`、标准差`[0.229,0.224,0.225]`的推理归一化；其分类输入缩放裁剪不能直接套用于分割标签。[TorchVision预训练权重许可说明](https://github.com/pytorch/vision#pre-trained-model-license)指出权重可能受训练数据的许可/条款约束，当前未确认本项目适用条件，故未下载，使用随机初始化与RGB除255。后续确认具体权重、数据许可和分割输入预处理后再记录来源、权重标识及哈希。

本轮损失 `1.0*BCE + 1.0*(1-Dice)`；BCE在批次内有效像素上取均值，soft Dice对批次所有有效像素汇总：`(2*sum(p*y)+1)/(sum(p)+sum(y)+1)`，`p=sigmoid(logits)`、平滑项1。全无效像素报错，两项都只计入有效像素；本轮未使用正类权重。验证损失是各批次Dice+BCE按有效像素数加权平均，Dice并非全验证集合并后一次计算；验证指标由全验证集累计混淆矩阵计算。`all_pixels_provisional`仅暂时计入所有像素，不代表任务书所需NoData忽略已实现或验证。

## 测试、基准和探索运行

46项测试通过，含Dice+BCE完美预测/全背景、部分/全部无效、非法形状和有限梯度；ResNet-50 GPU合成前后向、输出尺寸、eval参数及BatchNorm状态不变、模型与优化器检查点恢复及恢复后更新。初次检查点测试将CPU/GPU张量直接比较而报错，修正测试为跨设备比较数值后通过，未发现训练实现错误。

GPU真实冒烟仅4张训练候选、128×128中心裁剪、batch 2、两步；训练计时0.769秒，PyTorch峰值分配547283968字节（约522 MiB），参数更新成功。随后仅16张训练候选做512×512、batch 4基准，4步0.938秒，峰值3012159488字节（约2.81 GiB），因此探索运行选batch 4、5轮；此基准不是正式训练时间承诺。

RTX 5090，驱动595.71.05，PyTorch 2.8.0+cu128，SMP 0.5.0；种子17、Adam学习率0.0001、无增强、512×512全图、阈值0.5，训练484张/验证71张，最多5轮无早停。候选验证集已用于过去及本轮模型选择，不能称独立测试、正式E1上界或CAS论文公平对照。执行总耗时45.08秒，最高PyTorch分配约2.81 GiB；验证像素级统计包含像素重叠与全像素暂用的限制。

| 轮 | 训练Dice+BCE | 验证Dice+BCE | 验证IoU@0.5 |
|---|---:|---:|---:|
| 1 | 1.251832 | 1.295036 | 0.247616 |
| 2 | 0.890226 | 1.096664 | 0.130416 |
| 3 | 0.712652 | 1.379442 | 0.015063 |
| 4 | 0.687893 | 0.874103 | 0.348708 |
| 5 | 0.613448 | 0.974061 | 0.287899 |

第4轮按验证IoU严格大于更新、并列保留最早，作为best；TP=1039678、FP=851576、FN=1090259、TN=15630711，IoU=0.3487、F1=0.5171、Precision=0.5497、Recall=0.4881。`experiments/uav_deeplab_train_83uohcoi/` 保存配置、代码状态/哈希、清单与样本列表哈希、环境、逐轮结果、`best.pt`及`last.pt`；GPU已恢复best并校验模型参数和优化器状态。可信本地检查点才允许反序列化，实验目录和权重均被Git忽略。`experiments/uav_deeplab_smoke_gya8q8cm/`、`experiments/uav_deeplab_benchmark_kjgfer9j/`保留冒烟及基准记录。

## SegFormer和DINOv3后续

SMP 0.5.0提供`Segformer`与`mit_b2`，当前环境已用随机初始化合成2×3×128×128验证GPU前向、Dice+BCE反向和参数更新，输出2×1×128×128，24,722,369参数，峰值分配517688832字节约494 MiB，训练步骤1.216秒；见`scripts/training/segformer_b2_synthetic_smoke.py`与`experiments/segformer_b2_synthetic_jrge38vu/run.json`。未读取真实样本或下载权重。[NVIDIA mit-b2模型卡](https://huggingface.co/nvidia/mit-b2)列有ImageNet预训练编码器及独立许可链接，具体使用授权及与SMP实现兼容性待核实。

DINOv3两版本可分别使用同一编码器接口、各自许可的本地预训练权重和版本/哈希；冻结策略、token到空间网格转换与简单解码头结构待任务书及老师确认后实现并测试。当前只记录接口方案；无可用受限权重，也未尝试绕过权限。SAM 3和SegEarth-OV3仅作待确定的零样本对照。

在正式E1前仍需确认候选划分与跨连通组/跨平台空间隔离、测试候选曾被探索的处置、有效区域及标签对齐，冻结独立评价和多种子协议。全图探索只是开发用途，老师和学长意见仍待回复；本轮未访问test_candidate、buffer或unassigned影像及标签。

# E1 DeepLabV3+ 工程准备与待确认事项 v1

## 任务书核实状态

项目目录、服务器`/root/autodl-tmp`的常见任务书文件（PDF/DOCX及带任务字样文件）与仓库历史未发现任务书原文。`README.md`记录了DINOv3 SAT-493M、DINOv3 LVD-1689M和SAM 3.1三项受限权重申请仍待回复，但这不是三种E1骨干的任务书证据，不能据此认定任务书模型、DeepLabV3+参数或权限。`reports/uav_independent_evaluation_e1_preparation_v1.md`也明确列出E1配置尚待老师及学长确认。

| 任务书三种骨干 | 实现 | 预训练权重及访问权限 |
|---|---|---|
| 第1种，名称待核实 | 待任务书确认后选用 | 待确认来源、授权与状态 |
| 第2种，名称待核实 | 待任务书确认后选用 | 待确认来源、授权与状态 |
| 第3种，名称待核实 | 待任务书确认后选用 | 待确认来源、授权与状态 |

请提供任务书原文，或向老师与学长核实名称、版本、是否冻结、预训练权重来源及访问要求；本次没有把已有三项申请等同于任务书骨干。

## 实现与环境

使用 `segmentation-models-pytorch==0.5.0` 的 `DeepLabV3Plus`（[项目及许可](https://github.com/qubvel-org/segmentation_models.pytorch)、[MIT许可证](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/LICENSE)）。本地检查为`DeepLabV3PlusDecoder`，包含ASPP与低层特征解码融合；与[DeepLabV3+论文](https://arxiv.org/abs/1802.02611)的编码器—解码器思想一致。项目主体MIT；不同编码器及其依赖/权重可能有各自许可，正式使用须逐项核查。

工程测试配置：SMP `DeepLabV3Plus`、`resnet18`编码器、输出步长16、ASPP空洞率12/24/36、输入3通道、输出单通道logits、无末端激活，参数12,329,297。编码器可配置，但与任务书三种骨干是否一致未知。`encoder_weights=None`随机初始化，无预训练权重、来源或哈希；没有下载权重。封装支持明确指定可信本地encoder state_dict并记录SHA-256，权重许可由调用者预先核实，不隐式下载。

已有 `.venv/torch-gpu` 继承base站点包；在该环境中新增：`segmentation-models-pytorch==0.5.0`、`timm==1.0.31`、`huggingface-hub==2.4.0`、`safetensors==0.8.0`、`hf-xet==1.8.0`、`httpx2==2.16.2`、`httpcore2==2.16.2`、`idna==3.20`、`truststore==0.10.5`。其中SMP的依赖timm与huggingface-hub的包元数据为Apache-2.0，其他依赖许可按各包检查；安装覆盖于虚拟环境内，未修改base或原CPU环境。现有PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128、Rasterio 1.4.4，RTX 5090、驱动595.71.05。该环境复用base依赖，不是完整锁定环境。

```bash
.venv/torch-gpu/bin/python -m pip install --no-cache-dir --disable-pip-version-check --index-url https://pypi.org/simple 'segmentation-models-pytorch==0.5.0'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/torch-gpu/bin/python -B -m unittest discover -s tests -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/torch-gpu/bin/python -B scripts/training/uav_deeplabv3plus.py --mode smoke --crop 128 --batch-size 2
```

`scripts/training/uav_deeplabv3plus.py`默认只做最多4张训练候选的冒烟，显式`--mode train`提供后续候选训练和验证入口，但本次未运行该模式；调用`CandidateDataset`只允许train_candidate/val_candidate。同步中心裁剪、RGB除255、`all_pixels_provisional`、掩码BCE、Adam、阈值0.5、逐像素混淆矩阵、随机种子、日志及可信本地检查点复用已有模块。训练分支每轮验证并保存best/last；最佳按验证IoU严格大于更新、平分保留最早。恢复只加载可信本地检查点（含模型、优化器、轮次、随机状态），并校验配置。训练BatchNorm要求batch至少2且最后训练批次不可为1；不静默丢弃样本。

## 本轮检查与结果

42项测试通过。DeepLabV3+合成检查包含输出形状、有限损失、掩码、反向/参数更新、eval时参数与BatchNorm缓冲不变、非法输入、禁止划分、模型与优化器检查点状态恢复，以及GPU合成前后向更新；没有以真实验证数据运行模型。

仅用训练候选按文件名字典序前4对（UAV0094、0095、0096、0097）做GPU冒烟：128×128中心裁剪、batch 2、2步、Adam 0.0001、seed 17、未加权掩码BCE、全像素暂用；0.690秒（读取及两步训练计时）、峰值PyTorch分配253305344字节约241.6 MiB，首个参数发生变化、损失有限。运行记录 `experiments/uav_deeplab_smoke_ffhe984j/run.json` 保存样本名、清单与样本列表哈希、参数、代码状态、环境和无预训练权重状态；实验目录被Git忽略。此耗时仅是4张128裁剪的流程基准，不能可靠预测512全图或正式训练时间；待协议确定后应做短时受控基准。

## 后续E1准备

已固定的是此工程冒烟的模型结构和运行配置，不是任务书配置。候选验证集已用于选择/分析，不能作为独立最终评价；测试候选过去已探索，本轮未读取，后续处置须与老师学长确认。正式方案仍需明确三骨干、输入全图或分块/重叠、编码器权重及授权、是否冻结、损失和采样、评价指标和阈值、随机种子及重复次数。

在正式使用前还需独立核验标签对齐、有效像素及NoData与正类冲突，确认连通组之间和跨平台的空间隔离、避免影像重叠泄漏并冻结数据清单。可先做合成或候选范围内工程测试、权重许可核查与完整输入资源基准；正式E1结论、独立测试处置和批准状态须待协议与证据确认。本轮仅探索性工程验证，不是E1上界。

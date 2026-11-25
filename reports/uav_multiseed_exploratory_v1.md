# UAV 双模型多随机种子探索结果

固定方案见 [uav_multiseed_protocol_v1.md](uav_multiseed_protocol_v1.md)；机器可读索引及检查点 SHA-256 见 [uav_multiseed_run_index_v1.json](tables/uav_multiseed_run_index_v1.json)。旧 seed 17 的日志、两个检查点、清单与原索引哈希一致；新旧六次运行的训练、验证样本和清单哈希均一致。

GPU：RTX 5090；PyTorch 2.8.0+cu128，SMP 0.5.0。训练前 48 项测试通过，包括两个模型 GPU 合成前后向、检查点和划分限制。新运行全部正常完成，每轮首个参数均发生变化。

以下均为候选验证集全像素暂用口径、阈值 0.5，混淆矩阵依次为 TP / FP / FN / TN。best 依据该验证集最高 IoU 选择，严格改进才更新；last 为第 5 轮。

| 模型 | seed | 检查点(轮次) | IoU | F1 | Precision | Recall | TP / FP / FN / TN |
|---|---:|---|---:|---:|---:|---:|---|
| deeplabv3plus | 17 | best (4) | 0.3487 | 0.5171 | 0.5497 | 0.4881 | 1039678 / 851576 / 1090259 / 15630711 |
| deeplabv3plus | 17 | last (5) | 0.2879 | 0.4471 | 0.5820 | 0.3629 | 773055 / 555224 / 1356882 / 15927063 |
| deeplabv3plus | 42 | best (2) | 0.3686 | 0.5387 | 0.7678 | 0.4149 | 883678 / 267277 / 1246259 / 16215010 |
| deeplabv3plus | 42 | last (5) | 0.3148 | 0.4788 | 0.6079 | 0.3950 | 841297 / 542680 / 1288640 / 15939607 |
| deeplabv3plus | 123 | best (5) | 0.3629 | 0.5325 | 0.6087 | 0.4733 | 1008066 / 648069 / 1121871 / 15834218 |
| deeplabv3plus | 123 | last (5) | 0.3629 | 0.5325 | 0.6087 | 0.4733 | 1008066 / 648069 / 1121871 / 15834218 |
| segformer | 17 | best (5) | 0.3411 | 0.5087 | 0.5165 | 0.5010 | 1067165 / 998890 / 1062772 / 15483397 |
| segformer | 17 | last (5) | 0.3411 | 0.5087 | 0.5165 | 0.5010 | 1067165 / 998890 / 1062772 / 15483397 |
| segformer | 42 | best (3) | 0.3090 | 0.4721 | 0.3316 | 0.8190 | 1744417 / 3515698 / 385520 / 12966589 |
| segformer | 42 | last (5) | 0.2963 | 0.4572 | 0.3978 | 0.5373 | 1144474 / 1732316 / 985463 / 14749971 |
| segformer | 123 | best (4) | 0.3565 | 0.5256 | 0.5710 | 0.4869 | 1036978 / 779184 / 1092959 / 15703103 |
| segformer | 123 | last (5) | 0.3357 | 0.5027 | 0.5244 | 0.4827 | 1028154 / 932363 / 1101783 / 15549924 |

| 模型 | best IoU 均值 | 样本标准差 (n−1) | 范围 |
|---|---:|---:|---|
| deeplabv3plus | 0.3601 | 0.0103 | 0.3487–0.3686 |
| segformer | 0.3355 | 0.0242 | 0.3090–0.3565 |

| 模型 | seed | 运行目录 | 总耗时秒 | 显存峰值 GiB |
|---|---:|---|---:|---:|
| deeplabv3plus | 17 | `experiments/uav_deeplab_train_83uohcoi` | 45.1 | 2.81 |
| deeplabv3plus | 42 | `experiments/uav_deeplabv3plus_train_be14p2ab` | 45.0 | 2.81 |
| deeplabv3plus | 123 | `experiments/uav_deeplabv3plus_train_o6kfrz6d` | 47.1 | 2.81 |
| segformer | 17 | `experiments/uav_segformer_train_ydt54mjz` | 69.6 | 3.68 |
| segformer | 42 | `experiments/uav_segformer_train_ffsz6jh6` | 68.7 | 3.68 |
| segformer | 123 | `experiments/uav_segformer_train_k8fntit3` | 63.9 | 3.68 |

每个目录内保留配置、完整逐轮日志、样本名、环境、代码状态以及 best.pt 和 last.pt；这些目录被 Git 忽略，文件仍在服务器。原 seed 17 的 DeepLab 入口哈希与当前入口不同，当前入口扩展了 SegFormer 支持；其他原索引源文件一致，因此不声称旧、新 DeepLab 入口完全相同。

三种种子并非三个独立地理测试集；五轮不保证收敛，验证集已反复用于开发和检查点选择，不能视为独立测试或正式 E1 上界。all_pixels_provisional 并未验证 NoData；空间隔离、跨平台泄漏、标签对齐和曾被探索的测试候选处置仍待确认。DINOv3 与 SAM 权限审核状态无法从本地运行记录确认，未下载受限权重。

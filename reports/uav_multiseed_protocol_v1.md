# UAV 双模型多随机种子探索方案

固定于新增运行前：复用经运行索引核验日志、best/last 和清单 SHA-256 的 seed 17；两模型各增加 seed 42、123，不依据验证结果调整超参数。DeepLab 旧入口的源文件哈希与当前入口不同（后者扩展 SegFormer 支持），其他已索引源文件哈希一致；两次 DeepLab 的源代码版本不能声称完全相同。

- 数据：UAV component 2，train_candidate 484，val_candidate 71；统一清单 SHA-256 `54ab9a9fd80ab321bc4f0a18ad259b5a467b3a5502c1c70a0c184e4a7c3b281a`。不读取其他划分影像及标签。
- 模型：SMP 0.5.0 DeepLabV3+ / ResNet-50 和 Segformer / mit_b2；均随机初始化，未用预训练权重。两者架构不同，其余下列训练配置与各自 seed 17 记录一致。
- 512×512 全图；RGB / 255；无增强；all_pixels_provisional，不代表有效区域已验证。有效像素上 BCE 均值 + soft Dice 损失，按批次汇总有效像素，平滑项 1，两项权重均为 1，未使用正类权重；验证损失为批损失按有效像素数加权。
- Adam、学习率 0.0001、batch size 4、5 epochs、workers 0、阈值 0.5；验证像素汇总混淆矩阵。best 为验证 IoU 严格改善时更新，并列保留较早轮次；另保存 last。
- Python、NumPy、PyTorch 和 CUDA 设置种子，每轮加载顺序使用 seed + epoch；cuDNN benchmark 关闭，deterministic 开启，但未强制所有 PyTorch 算子确定性，跨设备和软件版本不保证完全复现。
- 原两次运行均为 RTX 5090、PyTorch 2.8.0+cu128；克隆环境同型号和 PyTorch 构建、SMP 0.5.0。GPU 合成前后向与检查点测试通过后依次运行各新种子，不覆盖旧结果。
- 候选验证集已经用于开发及选择；多种子只反映固定候选划分下部分训练随机性，并非独立地理测试或正式 E1。五轮未必收敛。数据隔离、标签对齐、NoData 与正式测试协议仍需确认；测试候选曾被探索。

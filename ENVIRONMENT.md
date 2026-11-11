# 环境记录

本文件记录第一周数据审计时检查到的环境版本。

## 基础环境

- 操作系统：AutoDL Linux 服务器
- Conda 环境名称：landslide-audit
- Python：3.11.18

## 数据审计依赖

| 软件包 | 版本 |
|---|---|
| NumPy | 2.4.6 |
| Rasterio | 1.4.4 |
| SciPy | 1.17.1 |
| Matplotlib | 3.11.2 |
| Pillow | 13.0.0 |

## 使用说明

在服务器终端中激活环境：

    conda activate landslide-audit

进入项目目录：

    cd /root/autodl-tmp/landslide_project

## 注意事项

- 本记录仅列出已检查的主要审计依赖，并非完整环境锁定文件。
- 环境重建时还需要确认依赖之间的兼容性。
- 当前记录不代表 PyTorch 已在此环境中安装或完成 GPU 测试。
- 模型训练、配准及其他后续任务可能需要额外依赖。
- 原始数据和模型权重未包含在 GitHub 仓库中。
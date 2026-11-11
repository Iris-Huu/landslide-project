"""
Moxitaidi 基础数据审计。

检查：
1. img、label、mask 的文件名配对；
2. 影像和 mask 的尺寸及通道数；
3. mask 的像素取值和滑坡像素比例。

注意：
- 本脚本完整读取 mask，但只读取影像的元数据；
- 滑坡像素比例未排除候选无数据区；
- 不修改原始数据。
"""

from pathlib import Path
import csv
import warnings

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning


PROJECT_DIR = Path(__file__).resolve().parents[2]
CAS_DIR = PROJECT_DIR / "data" / "raw" / "CAS"
OUTPUT_PATH = (
    PROJECT_DIR
    / "reports"
    / "tables"
    / "moxitaidi_basics_per_image.csv"
)

SUBSETS = [
    "Moxitaidi_SAT",
    "Moxitaidi_UAV_0.6m",
]

warnings.filterwarnings(
    "ignore",
    category=NotGeoreferencedWarning,
)


def list_tif_files(folder):
    """返回 TIFF 文件路径，键为不带扩展名的文件名。"""
    return {
        path.stem: path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    }


def audit_subset(subset):
    """审计一个子集，返回逐张结果与文件配对异常。"""
    root = CAS_DIR / subset

    folders = {
        name: root / name
        for name in ["img", "label", "mask"]
    }

    for name, folder in folders.items():
        if not folder.is_dir():
            raise FileNotFoundError(
                f"{subset} 缺少 {name} 目录：{folder}"
            )

    files = {
        name: list_tif_files(folder)
        for name, folder in folders.items()
    }

    image_names = set(files["img"])
    pairing_issues = []

    print(f"\n数据集：{subset}")
    for name in ["img", "label", "mask"]:
        print(f"  {name}: {len(files[name])} 个 TIFF 文件")

    for name in ["label", "mask"]:
        missing = sorted(image_names - set(files[name]))
        extra = sorted(set(files[name]) - image_names)

        print(
            f"  {name}：缺少 {len(missing)} 个，"
            f"多出 {len(extra)} 个"
        )

        for item in missing:
            pairing_issues.append(f"{name} 缺少：{item}")
        for item in extra:
            pairing_issues.append(f"{name} 多出：{item}")

    results = []

    for index, stem in enumerate(sorted(image_names), start=1):
        image_path = files["img"][stem]
        mask_path = files["mask"].get(stem)

        row = {
            "subset": subset,
            "filename": image_path.name,
            "image_height": "",
            "image_width": "",
            "image_channels": "",
            "mask_height": "",
            "mask_width": "",
            "mask_channels": "",
            "mask_values": "",
            "landslide_percent": "",
            "issues": "",
        }

        issues = []

        if mask_path is None:
            row["issues"] = "缺少同名 mask"
            results.append(row)
            continue

        try:
            with rasterio.open(image_path) as image:
                row["image_height"] = image.height
                row["image_width"] = image.width
                row["image_channels"] = image.count

            with rasterio.open(mask_path) as mask:
                row["mask_height"] = mask.height
                row["mask_width"] = mask.width
                row["mask_channels"] = mask.count
                mask_array = mask.read(1)

            image_shape = (
                row["image_height"],
                row["image_width"],
            )
            mask_shape = (
                row["mask_height"],
                row["mask_width"],
            )

            values = np.unique(mask_array)
            row["mask_values"] = " ".join(
                str(value) for value in values
            )

            if image_shape != (512, 512):
                issues.append("影像尺寸不是 512×512")

            if image_shape != mask_shape:
                issues.append("影像与 mask 尺寸不一致")

            if row["image_channels"] != 3:
                issues.append("影像不是 3 通道")

            if row["mask_channels"] != 1:
                issues.append("mask 不是 1 通道")

            if not np.isin(values, [0, 1]).all():
                issues.append("mask 存在 0 和 1 以外的值")
            else:
                row["landslide_percent"] = (
                    float(np.mean(mask_array == 1)) * 100
                )

        except Exception as error:
            issues.append(f"读取失败：{error}")

        row["issues"] = "；".join(issues)
        results.append(row)

        if index % 200 == 0 or index == len(image_names):
            print(f"  已检查 {index}/{len(image_names)}", flush=True)

    return results, pairing_issues


def main():
    if OUTPUT_PATH.exists():
        raise SystemExit(
            f"输出文件已存在，为避免覆盖而停止：{OUTPUT_PATH}"
        )

    all_results = []

    for subset in SUBSETS:
        results, pairing_issues = audit_subset(subset)
        all_results.extend(results)

        ratios = [
            row["landslide_percent"]
            for row in results
            if row["landslide_percent"] != ""
        ]
        issue_count = sum(bool(row["issues"]) for row in results)

        print(f"\n{subset} 汇总：")
        print(f"  文件配对异常：{len(pairing_issues)} 条")
        print(f"  逐张检查有问题：{issue_count} 张")

        if pairing_issues:
            print("  前 5 条配对异常：")
            for issue in pairing_issues[:5]:
                print(f"    {issue}")

        if ratios:
            print(f"  平均滑坡像素占比：{np.mean(ratios):.2f}%")
            print(f"  最小滑坡像素占比：{np.min(ratios):.2f}%")
            print(f"  最大滑坡像素占比：{np.max(ratios):.2f}%")
            print(f"  无滑坡像素的掩码：{sum(r == 0 for r in ratios)} 张")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "subset",
        "filename",
        "image_height",
        "image_width",
        "image_channels",
        "mask_height",
        "mask_width",
        "mask_channels",
        "mask_values",
        "landslide_percent",
        "issues",
    ]

    with OUTPUT_PATH.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n逐张结果已保存：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
"""
检查 GDCLD Lushan 影像与标签的基础元数据。

- 不读取整景影像像素；
- 标签取值使用缩小后的抽样结果，不代表全部像素；
- 分辨率单位由坐标系决定，不一定是米；
- 元数据不能直接证明传感器来源。
"""

from pathlib import Path
import csv

import numpy as np
import rasterio
from rasterio.enums import Resampling


PROJECT = Path(__file__).resolve().parents[2]

ROOT = PROJECT / "data/raw/GDCLD/Lushan_extracted"

OUTPUT = (
    PROJECT
    / "reports/tables"
    / "gdcld_lushan_metadata_v1.csv"
)

SCENES = ["Lushan_1.tif", "Lushan_2.tif", "Lushan_3.tif"]


def main():
    if OUTPUT.exists():
        raise SystemExit(
            f"输出文件已存在，为避免覆盖而停止：{OUTPUT}"
        )

    fields = [
        "scene",
        "status",
        "detail",
        "image_width",
        "image_height",
        "image_bands",
        "image_crs",
        "image_resolution",
        "image_bounds",
        "label_width",
        "label_height",
        "label_bands",
        "label_crs",
        "label_bounds",
        "same_dimensions",
        "same_crs",
        "same_transform",
        "label_sample_shape",
        "label_sample_values",
    ]

    results = []

    for name in SCENES:
        image_path = ROOT / "test_data/Lushan" / name
        label_path = ROOT / "test_label/Lushan" / name

        row = {field: "" for field in fields}
        row["scene"] = name

        try:
            with rasterio.open(image_path) as image:
                with rasterio.open(label_path) as label:
                    same_dimensions = (
                        image.width == label.width
                        and image.height == label.height
                    )

                    same_crs = (
                        image.crs is not None
                        and label.crs is not None
                        and image.crs == label.crs
                    )

                    same_transform = (
                        image.transform.almost_equals(label.transform)
                    )

                    # 只抽样读取标签第一波段
                    sample_height = min(label.height, 512)
                    sample_width = min(label.width, 512)

                    sample = label.read(
                        1,
                        out_shape=(sample_height, sample_width),
                        resampling=Resampling.nearest,
                    )

                    values = np.unique(sample)

                    if len(values) <= 20:
                        values_text = " ".join(str(v) for v in values)
                    else:
                        values_text = (
                            " ".join(str(v) for v in values[:20])
                            + f" ... 共 {len(values)} 种抽样值"
                        )

                    row.update({
                        "image_width": image.width,
                        "image_height": image.height,
                        "image_bands": image.count,
                        "image_crs": str(image.crs),
                        "image_resolution": str(image.res),
                        "image_bounds": str(tuple(image.bounds)),
                        "label_width": label.width,
                        "label_height": label.height,
                        "label_bands": label.count,
                        "label_crs": str(label.crs),
                        "label_bounds": str(tuple(label.bounds)),
                        "same_dimensions": same_dimensions,
                        "same_crs": same_crs,
                        "same_transform": same_transform,
                        "label_sample_shape": str(sample.shape),
                        "label_sample_values": values_text,
                    })

                    if same_dimensions and same_crs and same_transform:
                        row["status"] = "grid_metadata_matches"
                    else:
                        row["status"] = "needs_review"
                        row["detail"] = (
                            "尺寸、坐标系或网格变换不完全一致，"
                            "或缺少坐标系；需要进一步检查"
                        )

        except Exception as error:
            row["status"] = "read_error"
            row["detail"] = f"{type(error).__name__}: {error}"

        results.append(row)

        print(f"\n{name}: {row['status']}")

        if row["status"] == "read_error":
            print(f"  错误：{row['detail']}")
            continue

        print(
            f"  影像尺寸：{row['image_width']} × "
            f"{row['image_height']}，波段数：{row['image_bands']}"
        )
        print(
            f"  标签尺寸：{row['label_width']} × "
            f"{row['label_height']}，波段数：{row['label_bands']}"
        )
        print(f"  影像坐标系：{row['image_crs']}")
        print(f"  标签坐标系：{row['label_crs']}")
        print(f"  影像分辨率：{row['image_resolution']}")
        print(
            f"  尺寸一致：{row['same_dimensions']}；"
            f"坐标系一致：{row['same_crs']}；"
            f"网格变换一致：{row['same_transform']}"
        )
        print(
            f"  标签第一波段抽样取值："
            f"{row['label_sample_values']}"
        )

        if row["detail"]:
            print(f"  说明：{row['detail']}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n结果已保存：{OUTPUT}")
    print("注意：标签取值仅为抽样；分辨率单位以坐标系为准。")

    if any(row["status"] == "read_error" for row in results):
        raise SystemExit("有文件读取失败，请检查上方错误。")


if __name__ == "__main__":
    main()

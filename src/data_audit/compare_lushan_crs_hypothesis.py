"""
探索性检查 Lushan_1 的 EPSG:4543 假设。

仅在内存中的虚拟网格上比较 Lushan_1 与 Lushan_3；
不修改原始影像、标签或 CRS 元数据。
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window


PROJECT = Path(__file__).resolve().parents[2]
FOLDER = PROJECT / "data/raw/GDCLD/Lushan_extracted/test_data/Lushan"

FIGURE = (
    PROJECT / "reports/figures/"
    "gdcld_lushan_crs_hypothesis.png"
)
RESULT = (
    PROJECT / "reports/tables/"
    "gdcld_lushan_crs_hypothesis.json"
)

WINDOW_SIZE = 1024
DISPLAY_SIZE = 512
GRID = 7
MAX_WINDOWS = 3


def main():
    for path in [FIGURE, RESULT]:
        if path.exists():
            raise SystemExit(f"输出已存在，未覆盖：{path}")

    items = []

    with rasterio.open(FOLDER / "Lushan_1.tif") as image1, \
         rasterio.open(FOLDER / "Lushan_3.tif") as image3:

        if image1.crs is None or image3.crs is None:
            raise SystemExit("至少一景缺少 CRS，已停止。")

        if image1.count < 3 or image3.count < 3:
            raise SystemExit("影像波段数不足 3。")

        if image1.width < WINDOW_SIZE or image1.height < WINDOW_SIZE:
            raise SystemExit("Lushan_1 尺寸不足。")

        # 不修改 image1：只在目标虚拟网格中假设其数值属于 4543。
        with WarpedVRT(
            image3,
            crs="EPSG:4543",
            transform=image1.transform,
            width=image1.width,
            height=image1.height,
            resampling=Resampling.bilinear,
        ) as warped3:

            rows = np.unique(
                np.linspace(
                    0, image1.height - WINDOW_SIZE, GRID, dtype=int
                )
            )
            cols = np.unique(
                np.linspace(
                    0, image1.width - WINDOW_SIZE, GRID, dtype=int
                )
            )

            for row in rows:
                for col in cols:
                    window = Window(
                        int(col), int(row),
                        WINDOW_SIZE, WINDOW_SIZE,
                    )

                    first = np.moveaxis(
                        image1.read(
                            [1, 2, 3],
                            window=window,
                            out_shape=(3, DISPLAY_SIZE, DISPLAY_SIZE),
                            resampling=Resampling.bilinear,
                        ),
                        0, -1,
                    )

                    third = np.moveaxis(
                        warped3.read(
                            [1, 2, 3],
                            window=window,
                            out_shape=(3, DISPLAY_SIZE, DISPLAY_SIZE),
                            resampling=Resampling.bilinear,
                        ),
                        0, -1,
                    )

                    # 只用于挑选可视化窗口，不作为有效区判断。
                    first_nonblack = float(
                        np.any(first != 0, axis=2).mean()
                    )
                    third_nonblack = float(
                        np.any(third != 0, axis=2).mean()
                    )

                    if first_nonblack < 0.70 or third_nonblack < 0.70:
                        continue

                    items.append({
                        "row": int(row),
                        "col": int(col),
                        "first_nonblack_fraction": first_nonblack,
                        "third_nonblack_fraction": third_nonblack,
                        "image1": first,
                        "image3": third,
                    })

                    print(
                        f"找到窗口：行 {row}，列 {col}；"
                        f"Lushan_1 非纯黑 {first_nonblack:.1%}，"
                        f"Lushan_3 非纯黑 {third_nonblack:.1%}",
                        flush=True,
                    )

                    if len(items) >= MAX_WINDOWS:
                        break

                if len(items) >= MAX_WINDOWS:
                    break

    if not items:
        raise SystemExit(
            "未找到满足展示条件的窗口；"
            "这不能单独证明 CRS 假设错误。"
        )

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    RESULT.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        len(items), 2,
        figsize=(11, 5 * len(items)),
        squeeze=False,
    )

    for index, item in enumerate(items):
        axes[index, 0].imshow(item["image1"])
        axes[index, 0].set_title(
            f"Lushan_1: row {item['row']}, col {item['col']}"
        )

        axes[index, 1].imshow(item["image3"])
        axes[index, 1].set_title(
            "Lushan_3 on hypothetical Lushan_1 grid"
        )

        axes[index, 0].axis("off")
        axes[index, 1].axis("off")

    fig.suptitle("Exploratory CRS hypothesis: EPSG:4543")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=150, bbox_inches="tight")
    plt.close(fig)

    report = {
        "hypothesis": (
            "Interpret Lushan_1 numeric grid coordinates as EPSG:4543 "
            "for visualization only"
        ),
        "reference_scene": "Lushan_3.tif",
        "windows": [
            {
                key: value
                for key, value in item.items()
                if key not in ("image1", "image3")
            }
            for item in items
        ],
        "note": (
            "Nonblack fraction is only a display-selection heuristic. "
            "Visual similarity does not independently establish the "
            "correct CRS or sensor source."
        ),
    }

    with RESULT.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n对照图：{FIGURE}")
    print(f"检查记录：{RESULT}")
    print("注意：CRS 假设未确认，原始文件没有修改。")


if __name__ == "__main__":
    main()

"""
可视化 Moxitaidi UAV 切片的相对网格。

每个连通组单独绘制：
- 蓝色方格表示该位置有一张切片；
- 空白表示该组在该位置没有记录到切片；
- 不同组分别有自己的原点，位置不能直接比较。
"""

from collections import defaultdict
from pathlib import Path
import csv
import math

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[2]

COORDINATE_PATH = (
    PROJECT_DIR
    / "data"
    / "interim"
    / "reconstructed_grids"
    / "uav_grid_coordinates_v1.csv"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "reports"
    / "figures"
    / "uav_grid_components.png"
)


def main():
    if OUTPUT_PATH.exists():
        raise SystemExit(
            f"输出图片已存在，为避免覆盖而停止：{OUTPUT_PATH}"
        )

    groups = defaultdict(list)

    with COORDINATE_PATH.open(
        encoding="utf-8",
        newline="",
    ) as file:
        for row in csv.DictReader(file):
            component = int(row["component"])
            grid_row = int(row["grid_row"])
            grid_col = int(row["grid_col"])

            groups[component].append(
                (grid_row, grid_col, row["filename"])
            )

    if not groups:
        raise SystemExit("坐标表为空，无法生成网格图。")

    components = sorted(groups)

    # 每行最多画两个连通组
    plot_cols = 2
    plot_rows = math.ceil(len(components) / plot_cols)

    fig, axes = plt.subplots(
        plot_rows,
        plot_cols,
        figsize=(14, 4 * plot_rows),
        squeeze=False,
    )

    for index, component in enumerate(components):
        ax = axes[index // plot_cols][index % plot_cols]
        items = groups[component]

        rows = [item[0] for item in items]
        cols = [item[1] for item in items]

        ax.scatter(
            cols,
            rows,
            marker="s",
            s=35,
            color="steelblue",
        )

        ax.set_title(
            f"Component {component}: {len(items)} tiles"
        )
        ax.set_xlabel("Grid column")
        ax.set_ylabel("Grid row")

        # 影像中行号通常自上而下增加
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)

        print(
            f"第 {component} 组：{len(items)} 张，"
            f"行 {min(rows)}–{max(rows)}，"
            f"列 {min(cols)}–{max(cols)}"
        )

    # 隐藏未使用的子图
    for index in range(len(components), plot_rows * plot_cols):
        axes[index // plot_cols][index % plot_cols].axis("off")

    fig.suptitle(
        "Moxitaidi UAV: relative grid components",
        fontsize=15,
    )
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n网格图已保存：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
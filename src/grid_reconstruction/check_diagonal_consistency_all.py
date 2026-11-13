"""
检查无人机对角候选与已有水平、垂直网格的关系。

只做诊断：
- 不修改已有坐标；
- 不自动合并连通组；
- 对方向歧义候选单独统计。
"""

from collections import Counter, defaultdict
from pathlib import Path
import csv


PROJECT = Path(__file__).resolve().parents[2]
GRID_DIR = (
    PROJECT / "data" / "interim" / "reconstructed_grids"
)

COORD_PATH = GRID_DIR / "uav_grid_coordinates_v1.csv"
DIAG_PATH = GRID_DIR / "uav_diagonal_candidates.csv"
OUTPUT_PATH = GRID_DIR / "uav_diagonal_diagnostics.csv"


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def main():
    if OUTPUT_PATH.exists():
        raise SystemExit(f"输出文件已存在，已停止：{OUTPUT_PATH}")

    coord_rows = read_csv(COORD_PATH)
    diagonal_rows = read_csv(DIAG_PATH)

    coords = {}
    for row in coord_rows:
        name = row["filename"]

        if name in coords:
            raise ValueError(f"坐标表存在重复文件名：{name}")

        coords[name] = (
            int(row["component"]),
            int(row["grid_row"]),
            int(row["grid_col"]),
        )

    offsets = {
        "down_right": (1, 1),
        "down_left": (1, -1),
    }

    results = []
    summary = Counter()

    # 记录方向唯一的跨组候选所支持的组间平移
    proposals = defaultdict(list)

    for row in diagonal_rows:
        source = row["source_image"]
        target = row["target_image"]
        direction = row["direction"]
        candidate_status = row["status"]

        if source not in coords or target not in coords:
            raise ValueError(f"坐标表缺少文件：{source} / {target}")

        if direction not in offsets:
            raise ValueError(f"未知方向：{direction}")

        source_group, source_row, source_col = coords[source]
        target_group, target_row, target_col = coords[target]
        dr, dc = offsets[direction]

        if source_group == target_group:
            actual = (
                target_row - source_row,
                target_col - source_col,
            )
            relation = (
                "same_group_consistent"
                if actual == (dr, dc)
                else "same_group_conflict"
            )

            shift_row = ""
            shift_col = ""

        else:
            relation = "cross_group"

            # 将目标组坐标平移到源组坐标系所需的偏移
            shift_row = source_row + dr - target_row
            shift_col = source_col + dc - target_col

            if candidate_status == "unique_candidate":
                proposals[
                    (
                        source_group,
                        target_group,
                        shift_row,
                        shift_col,
                    )
                ].append((source, target, direction))

        summary[(candidate_status, relation)] += 1

        results.append({
            "source_image": source,
            "target_image": target,
            "direction": direction,
            "candidate_status": candidate_status,
            "source_component": source_group,
            "target_component": target_group,
            "relation": relation,
            "target_shift_row": shift_row,
            "target_shift_col": shift_col,
            "pure_black_white_percent": row[
                "pure_black_white_percent"
            ],
        })

    fields = [
        "source_image",
        "target_image",
        "direction",
        "candidate_status",
        "source_component",
        "target_component",
        "relation",
        "target_shift_row",
        "target_shift_col",
        "pure_black_white_percent",
    ]

    with OUTPUT_PATH.open(
        "x", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"对角候选总数：{len(results)}")

    print("\n按候选状态和网格关系统计：")
    for (status, relation), count in sorted(summary.items()):
        print(f"  {status} / {relation}: {count}")

    print("\n方向唯一的跨组平移线索（按支持条数排序，前 10 项）：")
    if not proposals:
        print("  无")
    else:
        ranked = sorted(
            proposals.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )

        for key, evidence in ranked[:10]:
            source_group, target_group, dr, dc = key
            print(
                f"  组 {source_group} -> 组 {target_group}："
                f"目标组平移 ({dr}, {dc})，"
                f"{len(evidence)} 条候选"
            )

    print(f"\n诊断表已保存：{OUTPUT_PATH}")
    print("注意：跨组线索尚未验证，不代表已经完成合并。")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="检查对角候选与已有网格的关系。"
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["uav", "satellite"],
        help="选择无人机或卫星结果",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="新的诊断 CSV 路径",
    )

    args = parser.parse_args()

    prefix = "uav" if args.dataset == "uav" else "satellite"

    COORD_PATH = GRID_DIR / f"{prefix}_grid_coordinates_v1.csv"
    DIAG_PATH = GRID_DIR / f"{prefix}_diagonal_candidates.csv"
    OUTPUT_PATH = args.output

    main()
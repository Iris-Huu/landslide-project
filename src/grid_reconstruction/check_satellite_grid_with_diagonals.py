"""
试合并 Moxitaidi 卫星切片网格。

使用：
- 方向唯一的水平、垂直候选；
- 5 条已目视检查的跨组对角候选。

不修改原始影像，也不覆盖 v1 坐标表。
"""

from collections import deque
from pathlib import Path
import csv


PROJECT = Path(__file__).resolve().parents[2]

IMAGE_DIR = (
    PROJECT / "data" / "raw" / "CAS"
    / "Moxitaidi_SAT" / "img"
)

GRID_DIR = (
    PROJECT / "data" / "interim" / "reconstructed_grids"
)

HV_PATH = GRID_DIR / "satellite_horizontal_vertical_candidates.csv"
DIAGONAL_PATH = GRID_DIR / "satellite_diagonal_candidates.csv"

COORD_PATH = GRID_DIR / "satellite_grid_coordinates_v2.csv"
CONFLICT_PATH = GRID_DIR / "satellite_grid_conflicts_v2.csv"

SELECTED_DIAGONALS = {
    ("Moxitaidi_Satellite245.tif", "Moxitaidi_Satellite251.tif", "down_right"),
    ("Moxitaidi_Satellite263.tif", "Moxitaidi_Satellite264.tif", "down_right"),
    ("Moxitaidi_Satellite548.tif", "Moxitaidi_Satellite551.tif", "down_right"),
    ("Moxitaidi_Satellite557.tif", "Moxitaidi_Satellite563.tif", "down_right"),
    ("Moxitaidi_Satellite582.tif", "Moxitaidi_Satellite593.tif", "down_right"),
}

OFFSETS = {
    "right": (0, 1),
    "down": (1, 0),
    "down_right": (1, 1),
    "down_left": (1, -1),
}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def add_edge(graph, source, target, direction):
    if source not in graph or target not in graph:
        raise ValueError(f"候选表中有未知影像：{source} / {target}")

    if direction not in OFFSETS:
        raise ValueError(f"未知方向：{direction}")

    dr, dc = OFFSETS[direction]
    graph[source].append((target, dr, dc))
    graph[target].append((source, -dr, -dc))


def main():
    for path in [COORD_PATH, CONFLICT_PATH]:
        if path.exists():
            raise SystemExit(f"输出文件已存在，已停止：{path}")

    filenames = sorted(path.name for path in IMAGE_DIR.glob("*.tif"))

    if not filenames:
        raise SystemExit(f"未找到影像：{IMAGE_DIR}")

    graph = {name: [] for name in filenames}

    hv_rows = read_csv(HV_PATH)
    diagonal_rows = read_csv(DIAGONAL_PATH)

    used_hv = 0
    skipped_hv = 0

    for row in hv_rows:
        if row["status"] != "unique_candidate":
            skipped_hv += 1
            continue

        add_edge(
            graph,
            row["source_image"],
            row["target_image"],
            row["direction"],
        )
        used_hv += 1

    found_diagonals = set()

    for row in diagonal_rows:
        key = (
            row["source_image"],
            row["target_image"],
            row["direction"],
        )

        if key not in SELECTED_DIAGONALS:
            continue

        if row["status"] != "unique_candidate":
            raise ValueError(f"所选对角关系不是唯一候选：{key}")

        if key in found_diagonals:
            raise ValueError(f"对角候选表中有重复关系：{key}")

        add_edge(graph, *key)
        found_diagonals.add(key)

    missing = SELECTED_DIAGONALS - found_diagonals
    if missing:
        raise ValueError(f"没有找到以下所选对角关系：{missing}")

    # 文件名 -> (连通组, 行, 列)
    positions = {}
    conflicts = []
    component_sizes = []
    component = 0

    for start in filenames:
        if start in positions:
            continue

        component += 1
        positions[start] = (component, 0, 0)
        queue = deque([start])
        size = 0

        while queue:
            current = queue.popleft()
            size += 1

            _, current_row, current_col = positions[current]

            for neighbor, dr, dc in graph[current]:
                expected = (
                    component,
                    current_row + dr,
                    current_col + dc,
                )

                if neighbor not in positions:
                    positions[neighbor] = expected
                    queue.append(neighbor)

                elif positions[neighbor] != expected:
                    conflicts.append({
                        "type": "inconsistent_position",
                        "component": component,
                        "image": neighbor,
                        "other_image": current,
                        "detail": (
                            f"已有 {positions[neighbor]}；"
                            f"候选关系要求 {expected}"
                        ),
                    })

        component_sizes.append(size)

    # 检查不同影像是否占据同一组内的同一坐标
    occupied = {}

    for filename, position in positions.items():
        if position in occupied:
            conflicts.append({
                "type": "duplicate_position",
                "component": position[0],
                "image": filename,
                "other_image": occupied[position],
                "detail": f"两张影像占据位置 {position}",
            })
        else:
            occupied[position] = filename

    # 每组单独平移坐标，使最小行、列分别为 0
    minimums = {}

    for group, row, col in positions.values():
        if group not in minimums:
            minimums[group] = [row, col]
        else:
            minimums[group][0] = min(minimums[group][0], row)
            minimums[group][1] = min(minimums[group][1], col)

    GRID_DIR.mkdir(parents=True, exist_ok=True)

    with COORD_PATH.open(
        "x", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "component",
                "grid_row",
                "grid_col",
                "neighbor_count",
            ],
        )
        writer.writeheader()

        for filename in filenames:
            group, row, col = positions[filename]
            min_row, min_col = minimums[group]

            writer.writerow({
                "filename": filename,
                "component": group,
                "grid_row": row - min_row,
                "grid_col": col - min_col,
                "neighbor_count": len(graph[filename]),
            })

    with CONFLICT_PATH.open(
        "x", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "type",
                "component",
                "image",
                "other_image",
                "detail",
            ],
        )
        writer.writeheader()
        writer.writerows(conflicts)

    isolated = sum(
        not neighbors for neighbors in graph.values()
    )

    print("卫星网格试合并完成。")
    print(f"影像总数：{len(filenames)}")
    print(f"使用的水平、垂直关系：{used_hv} 条")
    print(f"跳过的水平、垂直歧义关系：{skipped_hv} 条")
    print(f"加入的已检查对角关系：{len(found_diagonals)} 条")
    print(f"连通组数量：{component}")
    print(f"最大的 10 个组：{sorted(component_sizes, reverse=True)[:10]}")
    print(f"没有邻居的影像：{isolated} 张")
    print(f"冲突记录：{len(conflicts)} 条")
    print(f"\n坐标表：{COORD_PATH}")
    print(f"冲突表：{CONFLICT_PATH}")

    if conflicts:
        print("\n前 5 条冲突：")
        for item in conflicts[:5]:
            print(
                f"  {item['type']}：{item['image']}；"
                f"{item['detail']}"
            )

    print(
        "\n注意：坐标一致性不代替组内重叠像素验证；"
        "不同连通组之间也可能存在未识别的重叠。"
    )


if __name__ == "__main__":
    main()
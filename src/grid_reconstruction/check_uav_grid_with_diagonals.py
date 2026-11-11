"""
使用水平、垂直关系及两条已目视检查的对角关系，
试算 Moxitaidi UAV 的相对网格。

本脚本只生成试验结果：
- 不修改原始影像；
- 不覆盖原候选表和 v1 坐标；
- 坐标一致不等于已经完成所有重叠验证。
"""

from collections import defaultdict, deque
from pathlib import Path
import csv


PROJECT = Path(__file__).resolve().parents[2]

IMAGE_DIR = (
    PROJECT / "data" / "raw" / "CAS"
    / "Moxitaidi_UAV_0.6m" / "img"
)

GRID_DIR = (
    PROJECT / "data" / "interim" / "reconstructed_grids"
)

HORIZONTAL_PATH = GRID_DIR / "uav_horizontal_candidates.csv"
VERTICAL_PATH = GRID_DIR / "uav_vertical_candidates.csv"
DIAGONAL_PATH = GRID_DIR / "uav_diagonal_candidates.csv"

COORDINATE_PATH = GRID_DIR / "uav_grid_coordinates_v2.csv"
CONFLICT_PATH = GRID_DIR / "uav_grid_conflicts_v2.csv"

# 已逐像素核对，并对拼接预览进行目视检查的两条关系
SELECTED_DIAGONALS = {
    ("luding_UAV0150.tif", "luding_UAV0154.tif", "down_right"),
    ("luding_UAV0802.tif", "luding_UAV0820.tif", "down_right"),
}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def add_edge(graph, source, target, dr, dc):
    """添加正向和反向的相对位移关系。"""
    graph[source].append((target, dr, dc))
    graph[target].append((source, -dr, -dc))


def main():
    for output in [COORDINATE_PATH, CONFLICT_PATH]:
        if output.exists():
            raise SystemExit(
                f"输出文件已存在，为避免覆盖而停止：{output}"
            )

    image_names = sorted(
        path.name for path in IMAGE_DIR.glob("*.tif")
    )

    if not image_names:
        raise SystemExit(f"未找到影像：{IMAGE_DIR}")

    image_set = set(image_names)
    graph = {name: [] for name in image_names}

    horizontal = read_csv(HORIZONTAL_PATH)
    vertical = read_csv(VERTICAL_PATH)
    diagonal = read_csv(DIAGONAL_PATH)

    skipped_ambiguous = 0

    # 水平关系：向右，列 +1
    for row in horizontal:
        if row["status"] != "unique_candidate":
            skipped_ambiguous += 1
            continue

        source = row["left_image"]
        target = row["right_image"]

        if source not in image_set or target not in image_set:
            raise ValueError(
                f"水平表包含未知文件：{source}, {target}"
            )

        add_edge(graph, source, target, 0, 1)

    # 垂直关系：向下，行 +1
    for row in vertical:
        if row["status"] != "unique_candidate":
            skipped_ambiguous += 1
            continue

        source = row["top_image"]
        target = row["bottom_image"]

        if source not in image_set or target not in image_set:
            raise ValueError(
                f"垂直表包含未知文件：{source}, {target}"
            )

        add_edge(graph, source, target, 1, 0)

    # 只加入指定的两条跨组对角关系
    found_diagonals = set()

    for row in diagonal:
        key = (
            row["source_image"],
            row["target_image"],
            row["direction"],
        )

        if key not in SELECTED_DIAGONALS:
            continue

        if row["status"] != "unique_candidate":
            raise ValueError(
                f"选中的对角关系不是唯一候选：{key}"
            )

        source, target, direction = key

        if source not in image_set or target not in image_set:
            raise ValueError(
                f"对角表包含未知文件：{source}, {target}"
            )

        if direction == "down_right":
            add_edge(graph, source, target, 1, 1)
        elif direction == "down_left":
            add_edge(graph, source, target, 1, -1)
        else:
            raise ValueError(f"未知方向：{direction}")

        found_diagonals.add(key)

    if found_diagonals != SELECTED_DIAGONALS:
        missing = SELECTED_DIAGONALS - found_diagonals
        raise ValueError(
            f"对角候选表缺少已选关系：{missing}"
        )

    positions = {}
    conflicts = []
    component_sizes = []
    component_id = 0

    for start in image_names:
        if start in positions:
            continue

        component_id += 1
        positions[start] = (component_id, 0, 0)

        queue = deque([start])
        members = []

        while queue:
            current = queue.popleft()
            members.append(current)

            _, current_row, current_col = positions[current]

            for neighbor, dr, dc in graph[current]:
                expected = (
                    component_id,
                    current_row + dr,
                    current_col + dc,
                )

                if neighbor not in positions:
                    positions[neighbor] = expected
                    queue.append(neighbor)

                elif positions[neighbor] != expected:
                    conflicts.append({
                        "type": "inconsistent_position",
                        "component": component_id,
                        "image": neighbor,
                        "other_image": current,
                        "detail": (
                            f"已有位置 {positions[neighbor]}；"
                            f"根据关系应为 {expected}"
                        ),
                    })

        component_sizes.append(len(members))

    # 检查同一连通组中，不同影像是否占据同一网格位置
    occupied = {}

    for filename, position in positions.items():
        if position in occupied:
            conflicts.append({
                "type": "duplicate_position",
                "component": position[0],
                "image": filename,
                "other_image": occupied[position],
                "detail": f"两张影像占据同一位置 {position}",
            })
        else:
            occupied[position] = filename

    # 每组独立平移，使最小行、列均从 0 开始
    minimums = {}

    for component, row, col in positions.values():
        if component not in minimums:
            minimums[component] = [row, col]
        else:
            minimums[component][0] = min(
                minimums[component][0], row
            )
            minimums[component][1] = min(
                minimums[component][1], col
            )

    GRID_DIR.mkdir(parents=True, exist_ok=True)

    with COORDINATE_PATH.open(
        "x", newline="", encoding="utf-8"
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

        for filename in image_names:
            component, row, col = positions[filename]
            min_row, min_col = minimums[component]

            writer.writerow({
                "filename": filename,
                "component": component,
                "grid_row": row - min_row,
                "grid_col": col - min_col,
                "neighbor_count": len(graph[filename]),
            })

    with CONFLICT_PATH.open(
        "x", newline="", encoding="utf-8"
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

    print("试合并网格检查完成。")
    print(f"影像总数：{len(image_names)}")
    print(f"加入的已检查对角关系：{len(found_diagonals)} 条")
    print(f"跳过的水平、垂直歧义关系：{skipped_ambiguous} 条")
    print(f"连通组数量：{component_id}")
    print(f"各组大小：{sorted(component_sizes, reverse=True)}")
    print(f"冲突记录：{len(conflicts)} 条")
    print(f"\n试验坐标表：{COORDINATE_PATH}")
    print(f"冲突表：{CONFLICT_PATH}")

    if conflicts:
        print("\n前 5 条冲突：")
        for conflict in conflicts[:5]:
            print(
                f"  {conflict['type']}："
                f"{conflict['image']}；{conflict['detail']}"
            )

    print(
        "\n注意：仍需检查合并后可能存在的其他影像重叠，"
        "以及不同连通组之间的潜在空间关系。"
    )


if __name__ == "__main__":
    main()
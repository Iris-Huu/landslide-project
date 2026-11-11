"""
检查 Moxitaidi 卫星切片的相对网格。

坐标规则：
- 向右：列号 +1；
- 向下：行号 +1；
- 每个连通组独立设置原点。

注意：相对网格坐标不是地理坐标，
不同连通组的坐标不能直接比较。
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

CANDIDATE_PATH = (
    GRID_DIR / "satellite_horizontal_vertical_candidates.csv"
)

COORDINATE_PATH = GRID_DIR / "satellite_grid_coordinates_v1.csv"
CONFLICT_PATH = GRID_DIR / "satellite_grid_conflicts_v1.csv"


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def add_edge(graph, source, target, dr, dc):
    """记录正向和反向的相对位移。"""
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
        raise SystemExit(f"未找到 TIFF 影像：{IMAGE_DIR}")

    image_set = set(image_names)
    candidates = read_csv(CANDIDATE_PATH)

    # 包含全部影像，即使某张没有候选邻居
    graph = {name: [] for name in image_names}

    offsets = {
        "right": (0, 1),
        "down": (1, 0),
    }

    skipped_ambiguous = 0
    used_edges = 0

    for row in candidates:
        if row["status"] != "unique_candidate":
            skipped_ambiguous += 1
            continue

        source = row["source_image"]
        target = row["target_image"]
        direction = row["direction"]

        if source not in image_set or target not in image_set:
            raise ValueError(
                f"候选表包含未知影像：{source} / {target}"
            )

        if direction not in offsets:
            raise ValueError(f"未知方向：{direction}")

        dr, dc = offsets[direction]
        add_edge(graph, source, target, dr, dc)
        used_edges += 1

    # 影像名称 -> (连通组, 相对行, 相对列)
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
                            f"候选关系给出 {expected}"
                        ),
                    })

        component_sizes.append(len(members))

    # 检查不同切片是否被分配到同一个位置
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

    # 将每组的最小行、列分别平移到 0
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
        len(neighbors) == 0
        for neighbors in graph.values()
    )

    print("卫星网格一致性检查完成。")
    print(f"影像总数：{len(image_names)}")
    print(f"输入候选关系：{len(candidates)} 条")
    print(f"使用的唯一候选关系：{used_edges} 条")
    print(f"跳过的歧义候选关系：{skipped_ambiguous} 条")
    print(f"连通组数量：{component_id}")
    print(f"各组大小：{sorted(component_sizes, reverse=True)}")
    print(f"没有邻居的影像：{isolated} 张")
    print(f"冲突记录：{len(conflicts)} 条")
    print(f"\n坐标表：{COORDINATE_PATH}")
    print(f"冲突表：{CONFLICT_PATH}")

    if conflicts:
        print("\n前 5 条冲突：")
        for conflict in conflicts[:5]:
            print(
                f"  {conflict['type']}："
                f"{conflict['image']}；"
                f"{conflict['detail']}"
            )

    print(
        "\n注意：结果仍需进行目视检查、对角关系检查"
        "和实际重叠验证。"
    )


if __name__ == "__main__":
    main()
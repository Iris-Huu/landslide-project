"""
检查 Moxitaidi UAV 水平、垂直候选关系的网格一致性。

坐标说明：
- 向右：列号 +1
- 向下：行号 +1
- 每个连通组独立设置原点，不能直接比较不同组的坐标
- 网格坐标不是地理坐标
"""

from collections import defaultdict, deque
from pathlib import Path
import csv


PROJECT_DIR = Path(__file__).resolve().parents[2]

IMAGE_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "CAS"
    / "Moxitaidi_UAV_0.6m"
    / "img"
)

GRID_DIR = (
    PROJECT_DIR
    / "data"
    / "interim"
    / "reconstructed_grids"
)

HORIZONTAL_PATH = GRID_DIR / "uav_horizontal_candidates.csv"
VERTICAL_PATH = GRID_DIR / "uav_vertical_candidates.csv"

COORDINATE_PATH = GRID_DIR / "uav_grid_coordinates_v1.csv"
CONFLICT_PATH = GRID_DIR / "uav_grid_conflicts_v1.csv"


def read_table(path):
    """读取 CSV 表格。"""
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def add_edge(graph, source, target, delta_row, delta_col):
    """
    添加双向连接关系。

    source 到 target 的位移为 (delta_row, delta_col)；
    反向位移取相反数。
    """
    graph[source].append((target, delta_row, delta_col))
    graph[target].append((source, -delta_row, -delta_col))


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
    horizontal = read_table(HORIZONTAL_PATH)
    vertical = read_table(VERTICAL_PATH)

    # 初始化所有切片，包括没有任何邻居的切片
    graph = {name: [] for name in image_names}

    ambiguous_edges = 0

    for row in horizontal:
        if row["status"] != "unique_candidate":
            ambiguous_edges += 1
            continue

        left = row["left_image"]
        right = row["right_image"]

        if left not in image_set or right not in image_set:
            raise ValueError(f"水平候选表包含未知文件：{left}, {right}")

        add_edge(graph, left, right, 0, 1)

    for row in vertical:
        if row["status"] != "unique_candidate":
            ambiguous_edges += 1
            continue

        top = row["top_image"]
        bottom = row["bottom_image"]

        if top not in image_set or bottom not in image_set:
            raise ValueError(f"垂直候选表包含未知文件：{top}, {bottom}")

        add_edge(graph, top, bottom, 1, 0)

    # filename -> (component_id, row, col)
    positions = {}

    # 各种异常或冲突记录
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
                            f"根据邻接关系应为 {expected}"
                        ),
                    })

        component_sizes.append(len(members))

    # 检查同一个连通组中，是否有多张图占据同一位置
    occupied = {}

    for filename, position in positions.items():
        if position in occupied:
            conflicts.append({
                "type": "duplicate_position",
                "component": position[0],
                "image": filename,
                "other_image": occupied[position],
                "detail": f"两张影像被分配到同一位置 {position}",
            })
        else:
            occupied[position] = filename

    # 为每个连通组平移坐标，使最小行、列均为 0
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

    isolated = sum(
        len(neighbors) == 0
        for neighbors in graph.values()
    )

    print("网格一致性检查完成。")
    print(f"影像总数：{len(image_names)}")
    print(f"水平候选关系：{len(horizontal)}")
    print(f"垂直候选关系：{len(vertical)}")
    print(f"跳过的歧义候选关系：{ambiguous_edges}")
    print(f"连通组数量：{component_id}")
    print(f"最大连通组包含影像：{max(component_sizes)} 张")
    print(f"没有邻居的影像：{isolated} 张")
    print(f"冲突记录：{len(conflicts)} 条")
    print(f"\n坐标表：{COORDINATE_PATH}")
    print(f"冲突表：{CONFLICT_PATH}")

    if conflicts:
        print("\n前 5 条冲突：")
        for conflict in conflicts[:5]:
            print(
                f"  {conflict['type']}："
                f"{conflict['image']}，"
                f"{conflict['detail']}"
            )

    print(
        "\n提示：即使没有冲突，网格仍需结合影像内容"
        "和空间划分要求进一步检查。"
    )


if __name__ == "__main__":
    main()
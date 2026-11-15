"""
解压三个异址子集的 img 和 mask 到独立目录。

不解压 label，不覆盖已有文件。
若中途中断，不自动删除已解压内容。
"""

import shutil
from collections import Counter
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


PROJECT = Path(__file__).resolve().parents[2]
CAS = PROJECT / "data/raw/CAS"

SUBSETS = [
    ("Longxi River(SAT).zip", "Longxi_SAT"),
    ("Longxi River（UAV）.zip", "Longxi_UAV"),
    ("Jiuzhai valley (UAV-0.5m).zip", "Jiuzhai_UAV_0.5m"),
]

RESERVE_BYTES = 2 * 1024 ** 3


def selected_entries(archive):
    """只接受顶层 img/mask 目录下的 TIFF 文件。"""
    selected = []

    for item in archive.infolist():
        if item.is_dir():
            continue

        path = PurePosixPath(item.filename.replace("\\", "/"))

        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"压缩包包含异常路径：{item.filename}")

        if len(path.parts) != 2:
            continue

        folder, filename = path.parts

        if folder not in {"img", "mask"}:
            continue

        if path.suffix.lower() not in {".tif", ".tiff"}:
            continue

        selected.append((item, folder, filename))

    return selected


def preflight():
    plans = []
    total_bytes = 0

    for archive_name, directory_name in SUBSETS:
        archive_path = CAS / archive_name
        destination = CAS / directory_name

        if not archive_path.is_file():
            raise FileNotFoundError(f"没有找到压缩包：{archive_path}")

        if destination.exists():
            raise FileExistsError(
                f"目标目录已存在，未覆盖：{destination}"
            )

        with ZipFile(archive_path) as archive:
            entries = selected_entries(archive)

        if not entries:
            raise ValueError(f"{archive_name} 没有找到 img/mask TIFF")

        names = [(folder, filename) for _, folder, filename in entries]

        if len(names) != len(set(names)):
            raise ValueError(f"{archive_name} 存在重复的目标文件名")

        groups = {
            folder: {
                Path(filename).stem
                for _, kind, filename in entries
                if kind == folder
            }
            for folder in ("img", "mask")
        }

        image_count = sum(kind == "img" for _, kind, _ in entries)
        mask_count = sum(kind == "mask" for _, kind, _ in entries)

        if image_count != len(groups["img"]) or mask_count != len(groups["mask"]):
            raise ValueError(
                f"{archive_name} 存在同名不同扩展名文件，需先检查"
            )

        if groups["img"] != groups["mask"]:
            raise ValueError(
                f"{archive_name} img/mask 文件名不完全匹配："
                f"缺少 mask {len(groups['img'] - groups['mask'])}，"
                f"多余 mask {len(groups['mask'] - groups['img'])}"
            )

        size = sum(item.file_size for item, _, _ in entries)
        total_bytes += size

        plans.append({
            "archive": archive_path,
            "destination": destination,
            "image_count": image_count,
            "mask_count": mask_count,
            "bytes": size,
        })

    free = shutil.disk_usage(CAS).free

    print(f"预计解压 img/mask：{total_bytes / 1024**3:.2f} GiB")
    print(f"当前可用空间：{free / 1024**3:.2f} GiB")
    print(f"要求保留空间：{RESERVE_BYTES / 1024**3:.2f} GiB")

    if free < total_bytes + RESERVE_BYTES:
        raise SystemExit("磁盘空间不足，未开始解压。")

    return plans


def extract(plan):
    archive_path = plan["archive"]
    destination = plan["destination"]

    destination.mkdir(parents=False, exist_ok=False)
    (destination / "img").mkdir()
    (destination / "mask").mkdir()

    with ZipFile(archive_path) as archive:
        entries = selected_entries(archive)

        for index, (item, folder, filename) in enumerate(entries, start=1):
            target = destination / folder / filename

            with archive.open(item) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)

            if index % 500 == 0 or index == len(entries):
                print(
                    f"  已解压 {index}/{len(entries)} 个文件",
                    flush=True,
                )

    counts = Counter(
        path.parent.name
        for path in destination.rglob("*")
        if path.is_file()
    )

    if (
        counts["img"] != plan["image_count"]
        or counts["mask"] != plan["mask_count"]
    ):
        raise RuntimeError(f"{destination} 解压后文件数不一致")

    print(
        f"  完成：img={counts['img']}，mask={counts['mask']}\n"
        f"  目录：{destination}",
        flush=True,
    )


def main():
    plans = preflight()

    for plan in plans:
        print(f"\n开始解压：{plan['archive'].name}", flush=True)
        extract(plan)

    print("\n三个子集的 img 和 mask 已解压并核对文件数。")
    print("注意：此步骤不等于逐像素完整性或空间划分验证。")


if __name__ == "__main__":
    main()

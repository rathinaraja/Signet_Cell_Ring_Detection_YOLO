"""
python tools/digestpath_box_stats.py
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import statistics as stats
import csv
from PIL import Image

DATASET_ROOT = Path("c")
POS_DIR = DATASET_ROOT / "sig-train-pos"
OUT_DIR = Path("outputs/stats")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_boxes(xml_path: Path):
    boxes = []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall(".//object"):
        bnd = obj.find(".//bndbox")
        if bnd is not None:
            xmin = int(float(bnd.findtext("xmin", "0")))
            ymin = int(float(bnd.findtext("ymin", "0")))
            xmax = int(float(bnd.findtext("xmax", "0")))
            ymax = int(float(bnd.findtext("ymax", "0")))
            if xmax > xmin and ymax > ymin:
                boxes.append((xmin, ymin, xmax, ymax))

    if not boxes:
        for bnd in root.findall(".//bndbox"):
            xmin = int(float(bnd.findtext("xmin", "0")))
            ymin = int(float(bnd.findtext("ymin", "0")))
            xmax = int(float(bnd.findtext("xmax", "0")))
            ymax = int(float(bnd.findtext("ymax", "0")))
            if xmax > xmin and ymax > ymin:
                boxes.append((xmin, ymin, xmax, ymax))

    return boxes

def safe_stat(arr, fn, default=0):
    return fn(arr) if arr else default

def main():
    xml_paths = sorted(POS_DIR.glob("*.xml"))

    widths, heights, areas, boxes_per_image = [], [], [], []
    rel_widths, rel_heights, rel_areas = [], [], []

    per_image_csv = OUT_DIR / "per_image_box_counts.csv"
    with open(per_image_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "image_width", "image_height", "box_count"])

        for xml_path in xml_paths:
            img_path = xml_path.with_suffix(".jpeg")
            if not img_path.exists():
                continue

            img = Image.open(img_path)
            W, H = img.size
            boxes = parse_boxes(xml_path)
            boxes_per_image.append(len(boxes))
            writer.writerow([img_path.name, W, H, len(boxes)])

            for xmin, ymin, xmax, ymax in boxes:
                w = xmax - xmin
                h = ymax - ymin
                a = w * h
                widths.append(w)
                heights.append(h)
                areas.append(a)
                rel_widths.append(w / W)
                rel_heights.append(h / H)
                rel_areas.append(a / (W * H))

    summary_txt = OUT_DIR / "box_stats_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("=== DigestPath Box Statistics ===\n")
        f.write(f"Positive XML files             : {len(xml_paths)}\n")
        f.write(f"Total boxes                    : {len(widths)}\n")
        f.write(f"Mean boxes per positive image  : {safe_stat(boxes_per_image, stats.mean):.2f}\n")
        f.write(f"Median boxes per image         : {safe_stat(boxes_per_image, stats.median):.2f}\n")
        f.write(f"Min boxes per image            : {safe_stat(boxes_per_image, min)}\n")
        f.write(f"Max boxes per image            : {safe_stat(boxes_per_image, max)}\n\n")

        f.write(f"Mean box width (px)            : {safe_stat(widths, stats.mean):.2f}\n")
        f.write(f"Median box width (px)          : {safe_stat(widths, stats.median):.2f}\n")
        f.write(f"Min/Max box width (px)         : {safe_stat(widths, min)} / {safe_stat(widths, max)}\n\n")

        f.write(f"Mean box height (px)           : {safe_stat(heights, stats.mean):.2f}\n")
        f.write(f"Median box height (px)         : {safe_stat(heights, stats.median):.2f}\n")
        f.write(f"Min/Max box height (px)        : {safe_stat(heights, min)} / {safe_stat(heights, max)}\n\n")

        f.write(f"Mean relative width            : {safe_stat(rel_widths, stats.mean):.6f}\n")
        f.write(f"Mean relative height           : {safe_stat(rel_heights, stats.mean):.6f}\n")
        f.write(f"Mean relative area             : {safe_stat(rel_areas, stats.mean):.8f}\n")

    print(f"Saved box summary to: {summary_txt}")
    print(f"Saved per-image counts to: {per_image_csv}")

if __name__ == "__main__":
    main()
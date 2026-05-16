"""
python tools/visualize_digestpath_xml.py
"""
from pathlib import Path
import random
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

DATASET_ROOT = Path("/data_64T_1/Barathi/Projects/SRC_detection/digestpath_dataset")
POS_DIR = DATASET_ROOT / "sig-train-pos"
OUT_DIR = Path("outputs/viz")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_SAMPLES = 20
RANDOM_SEED = 42

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

def main():
    random.seed(RANDOM_SEED)
    img_paths = sorted(POS_DIR.glob("*.jpeg"))
    chosen = random.sample(img_paths, min(NUM_SAMPLES, len(img_paths)))

    for img_path in chosen:
        xml_path = img_path.with_suffix(".xml")
        if not xml_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        boxes = parse_boxes(xml_path)

        for box in boxes:
            draw.rectangle(box, outline="red", width=3)

        out_path = OUT_DIR / f"{img_path.stem}_boxed.jpeg"
        img.save(out_path, quality=95)

    print(f"Saved {len(chosen)} visualization samples to: {OUT_DIR}")

if __name__ == "__main__":
    main()
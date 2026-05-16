"""
python tools/visualize_yolo_tiles.py
"""
from pathlib import Path
import random
from PIL import Image, ImageDraw

DATASET_ROOT = Path("/path/digestpath_yolo")
IMG_DIR = DATASET_ROOT / "images" / "train"
LBL_DIR = DATASET_ROOT / "labels" / "train"
OUT_DIR = Path("outputs/yolo_tile_viz")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_SAMPLES = 20
RANDOM_SEED = 42

def read_yolo_labels(lbl_path, img_w, img_h):
    boxes = []
    if not lbl_path.exists():
        return boxes

    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id, xc, yc, w, h = parts
            xc = float(xc) * img_w
            yc = float(yc) * img_h
            w = float(w) * img_w
            h = float(h) * img_h

            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2
            boxes.append((x1, y1, x2, y2))
    return boxes

def main():
    random.seed(RANDOM_SEED)
    img_paths = sorted(IMG_DIR.glob("*.jpg"))
    chosen = random.sample(img_paths, min(NUM_SAMPLES, len(img_paths)))

    for img_path in chosen:
        lbl_path = LBL_DIR / f"{img_path.stem}.txt"
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        boxes = read_yolo_labels(lbl_path, W, H)

        draw = ImageDraw.Draw(img)
        for box in boxes:
            draw.rectangle(box, outline="lime", width=3)

        out_path = OUT_DIR / f"{img_path.stem}_viz.jpg"
        img.save(out_path, quality=95)

    print(f"Saved {len(chosen)} visualizations to: {OUT_DIR}")

if __name__ == "__main__":
    main()
"""
python tools/audit_digestpath.py
"""
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET
import csv

DATASET_ROOT = Path("/data_64T_1/Barathi/Projects/SRC_detection/digestpath_dataset")
POS_DIR = DATASET_ROOT / "sig-train-pos"
NEG_DIR = DATASET_ROOT / "sig-train-neg"
OUT_DIR = Path("outputs/audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def is_image_readable(img_path: Path):
    try:
        with Image.open(img_path) as img:
            img.verify()
        return True, ""
    except Exception as e:
        return False, str(e)

def is_xml_readable(xml_path: Path):
    try:
        ET.parse(xml_path)
        return True, ""
    except Exception as e:
        return False, str(e)

def main():
    pos_imgs = sorted(POS_DIR.glob("*.jpeg"))
    pos_xmls = sorted(POS_DIR.glob("*.xml"))
    neg_imgs = sorted(NEG_DIR.glob("*.jpeg"))

    pos_img_stems = {p.stem for p in pos_imgs}
    pos_xml_stems = {p.stem for p in pos_xmls}

    missing_xml = sorted(pos_img_stems - pos_xml_stems)
    orphan_xml = sorted(pos_xml_stems - pos_img_stems)

    broken_images = []
    broken_xmls = []

    for img_path in pos_imgs + neg_imgs:
        ok, err = is_image_readable(img_path)
        if not ok:
            broken_images.append((str(img_path), err))

    for xml_path in pos_xmls:
        ok, err = is_xml_readable(xml_path)
        if not ok:
            broken_xmls.append((str(xml_path), err))

    report_path = OUT_DIR / "audit_report.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "item", "details"])

        writer.writerow(["summary", "positive_images", len(pos_imgs)])
        writer.writerow(["summary", "positive_xmls", len(pos_xmls)])
        writer.writerow(["summary", "negative_images", len(neg_imgs)])
        writer.writerow(["summary", "missing_xml_count", len(missing_xml)])
        writer.writerow(["summary", "orphan_xml_count", len(orphan_xml)])
        writer.writerow(["summary", "broken_image_count", len(broken_images)])
        writer.writerow(["summary", "broken_xml_count", len(broken_xmls)])

        for stem in missing_xml:
            writer.writerow(["missing_xml", stem, "positive image has no matching xml"])

        for stem in orphan_xml:
            writer.writerow(["orphan_xml", stem, "xml has no matching positive image"])

        for path, err in broken_images:
            writer.writerow(["broken_image", path, err])

        for path, err in broken_xmls:
            writer.writerow(["broken_xml", path, err])

    print("=== DigestPath Audit Summary ===")
    print(f"Positive images : {len(pos_imgs)}")
    print(f"Positive XMLs   : {len(pos_xmls)}")
    print(f"Negative images : {len(neg_imgs)}")
    print(f"Missing XML     : {len(missing_xml)}")
    print(f"Orphan XML      : {len(orphan_xml)}")
    print(f"Broken images   : {len(broken_images)}")
    print(f"Broken XMLs     : {len(broken_xmls)}")
    print(f"Saved report to: {report_path}")

if __name__ == "__main__":
    main()
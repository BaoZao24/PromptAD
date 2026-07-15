#!/usr/bin/env python
"""Create a pure-visual Visio architecture from the original source diagram."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET


NS = "{http://schemas.microsoft.com/office/visio/2012/main}"


def set_text(shape, value):
    text = shape.find(f"{NS}Text")
    for child in list(text):
        text.remove(child)
    ET.SubElement(text, f"{NS}cp", {"IX": "0"})
    text.text = value


def main():
    source = Path("纯视觉架构图.vsdx")
    output = Path("Confidence_Gated_Dual_Visual_Architecture.vsdx")
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        with zipfile.ZipFile(source) as archive:
            archive.extractall(root)
        page = root / "visio/pages/page1.xml"
        tree = ET.parse(page)
        shapes = {shape.get("ID"): shape for shape in tree.findall(f".//{NS}Shape")}
        for shape_id, label in {
            "11": "Confidence-Gated CNN Fusion",
            "15": "Farthest Coreset",
            "16": "CNN Normal Gallery",
            "20": "Farthest Coreset",
            "22": "ViT Normal Gallery",
            "39": "Final Anomaly Score",
        }.items():
            set_text(shapes[shape_id], label)
        tree.write(page, encoding="utf-8", xml_declaration=True)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root).as_posix())


if __name__ == "__main__":
    main()

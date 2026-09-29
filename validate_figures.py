"""Read-only PDF/raster checks and PDF rendering for human visual review.

Requires optional pypdf and Pillow plus a configured Poppler pdftoppm.
All outputs are saved outside the prospective public repository.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
from pypdf import PdfReader
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdftoppm", required=True, type=Path)
    args = parser.parse_args()
    work = Path(os.environ["CPO_WORK_DIR"]).resolve()
    supplement = work / "CPO_SUBMISSION_SUPPLEMENT_20260917"
    figures = supplement / "EHJACC_FIGURES"
    output = work / "reproduction_qa/figure_previews"
    output.mkdir(parents=True, exist_ok=True)
    paths = [figures / "Graphical_Abstract.pdf", *[figures / f"Figure_{i}.pdf" for i in range(1, 5)],
             *sorted((supplement / "figures_r1").glob("*.pdf"))]
    checks = []
    for path in paths:
        pdf = PdfReader(path)
        assert len(pdf.pages) == 1
        page = pdf.pages[0]
        image_count = len(page.images)
        assert image_count == 0, path.name + " includes raster objects"
        fonts = page["/Resources"].get_object().get("/Font", {}).get_object()
        for font_reference in fonts.values():
            font = font_reference.get_object()
            if "/DescendantFonts" in font:
                font = font["/DescendantFonts"][0].get_object()
            descriptor = font.get("/FontDescriptor")
            assert descriptor is not None
            assert any(key in descriptor.get_object() for key in ("/FontFile", "/FontFile2", "/FontFile3"))
        if path.name == "Figure_1.pdf":
            text = page.extract_text()
            assert "7,753 landmarks" in text and "7,754" not in text
        for extension in (".png", ".tiff") if path.parent == figures else (".png",):
            raster = path.with_suffix(extension)
            with Image.open(raster) as image:
                recorded_dpi = image.info.get("dpi")
                dpi = [float(value) for value in recorded_dpi] if recorded_dpi else None
                if path.parent == figures:
                    assert dpi and min(dpi) > 599
                checks.append({"file": str(raster.relative_to(work)), "pixels": list(image.size), "dpi": dpi})
        subprocess.run([str(args.pdftoppm), "-singlefile", "-scale-to", "1800", "-png", str(path), str(output / path.stem)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        checks.append({"file": str(path.relative_to(work)), "pages": 1, "raster_objects": image_count,
                       "embedded_fonts": True, "technical_status": "PASS"})
    (output.parent / "figure_technical_checks.json").write_text(json.dumps(checks, indent=2))
    print(f"{len(paths)} one-page vector PDFs passed technical checks and rendered for visual review.")
    print("Human visual inspection is required before delivery; this command does not assert it.")


if __name__ == "__main__":
    main()

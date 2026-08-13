#!/usr/bin/env python3
"""Render the Markdown experiment section as an IEEE-style PDF.

The renderer is intentionally local and deterministic: Markdown is converted
to HTML, display equations are embedded as local SVGs, and Chrome's print
engine emits the final PDF.  The document uses a US-Letter, two-column, 10 pt layout with
Times New Roman for Latin text and a CJK fallback for Chinese glyphs.
"""

from __future__ import annotations

import argparse
import base64
import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import markdown
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.mathtext import math_to_image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "docs/论文实验部分.md"
DEFAULT_OUTPUT = ROOT / "docs/论文实验部分.pdf"
TIMES_FONT_FILES = (
    Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold_Italic.ttf"),
)
for _font_path in TIMES_FONT_FILES:
    if _font_path.is_file():
        fontManager.addfont(str(_font_path))
FORMULA_FONT = FontProperties(
    fname=str(TIMES_FONT_FILES[0]) if TIMES_FONT_FILES[0].is_file() else None,
    family="Times New Roman",
    size=10,
)

CSS = r"""
@page {
  size: Letter;
  margin: 0.75in 0.625in 1.0in;
}

* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  color: #000;
  background: #fff;
  font-family: "Times New Roman", "Noto Serif CJK SC", "Noto Serif CJK TC", serif;
  font-size: 10pt;
  line-height: 1.13;
  text-align: justify;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}

.paper {
  column-count: 2;
  column-gap: 0.25in;
  column-fill: balance;
}

h1, h2, h3, h4 {
  font-family: "Times New Roman", "Noto Serif CJK SC", serif;
  text-align: left;
  break-after: avoid;
  -webkit-column-break-after: avoid;
}
h1 {
  column-span: all;
  font-size: 14pt;
  line-height: 1.1;
  margin: 0 0 8pt;
  padding-bottom: 3pt;
  border-bottom: 0.6pt solid #000;
}
h2 {
  font-size: 11pt;
  line-height: 1.1;
  margin: 9pt 0 4pt;
}
h3 {
  font-size: 10pt;
  line-height: 1.1;
  margin: 7pt 0 3pt;
}
h4 { font-size: 10pt; margin: 5pt 0 2pt; }
p { margin: 0 0 4pt; orphans: 2; widows: 2; }
ul, ol { margin: 2pt 0 5pt 15pt; padding-left: 9pt; }
li { margin: 1pt 0; }
blockquote {
  margin: 5pt 8pt;
  padding-left: 7pt;
  border-left: 1.5pt solid #777;
}
code, pre {
  font-family: "Courier New", "Noto Sans Mono CJK SC", monospace;
  font-size: 8pt;
}
code { white-space: nowrap; }
pre code {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  word-break: break-word;
}
pre {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  word-break: break-word;
  margin: 4pt 0 6pt;
  padding: 4pt 5pt;
  border: 0.4pt solid #aaa;
  background: #f7f7f7;
  break-inside: avoid;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 5pt 0 7pt;
  font-size: 8.2pt;
  line-height: 1.08;
  text-align: center;
  break-inside: avoid;
  -webkit-column-break-inside: avoid;
}
.table-block {
  break-inside: avoid;
  -webkit-column-break-inside: avoid;
  margin: 5pt 0 7pt;
}
.table-block.wide-table { column-span: all; }
.table-block p { margin: 0 0 2pt; text-align: left; }
.table-block table { margin: 0; }
th, td {
  padding: 2.5pt 3pt;
  border-top: 0.45pt solid #000;
  border-bottom: 0.45pt solid #000;
  vertical-align: middle;
}
th { font-weight: bold; }
tr + tr td { border-top: 0.25pt solid #888; }

img {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 4pt auto;
  break-inside: avoid;
  -webkit-column-break-inside: avoid;
}
p:not(.equation):has(> img:not(.inline-equation)),
p:not(.equation):has(> img:not(.inline-equation)) + p {
  column-span: all;
  text-align: center;
  break-inside: avoid;
}
p:not(.equation):has(> img:not(.inline-equation)) { margin: 7pt 0 2pt; }
p:not(.equation):has(> img:not(.inline-equation)) + p { margin: 0 0 7pt; font-size: 8.5pt; }

hr { border: 0; border-top: 0.5pt solid #000; margin: 6pt 0; }
strong { font-weight: bold; }
em { font-style: italic; }

/* Keep equations in the current text column, as in a standard two-column paper. */
.equation { column-span: none; text-align: center; margin: 5pt 0; }
.equation img { max-width: 92%; }
.inline-equation {
  display: inline;
  width: auto;
  height: 1.05em;
  margin: 0 1pt;
  vertical-align: -0.18em;
}
.figure-section {
  column-span: all;
  break-inside: avoid;
  page-break-inside: avoid;
}
"""


def find_chrome() -> str:
    for candidate in ("google-chrome", "chromium", "chromium-browser"):
        found = shutil.which(candidate)
        if found:
            return found
    raise RuntimeError("Chrome/Chromium is required for PDF rendering")


def write_math_svg(formula: str, equation_path: Path) -> None:
    """Write all equations with the same Times New Roman, 10 pt settings."""

    # Markdown display equations may wrap across source lines.  Matplotlib's
    # mathtext parser treats a literal newline as an end-of-expression, so
    # collapse source whitespace while preserving the LaTeX commands.
    formula = " ".join(formula.split())

    with mpl.rc_context(
        {
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "mathtext.tt": "Times New Roman",
            "svg.fonttype": "none",
        }
    ):
        math_to_image(
            f"${formula}$",
            equation_path,
            prop=FORMULA_FONT,
            dpi=220,
            format="svg",
        )


def replace_display_math(source: str, temp_dir: Path) -> str:
    """Replace Markdown ``$$...$$`` blocks with crisp local SVG equations."""

    def replacement(match: re.Match[str]) -> str:
        formula = match.group(1).strip()
        equation_path = temp_dir / f"equation_{replacement.counter:02d}.svg"
        replacement.counter += 1
        write_math_svg(formula, equation_path)
        encoded = base64.b64encode(equation_path.read_bytes()).decode("ascii")
        return (
            '<p class="equation">'
            f'<img src="data:image/svg+xml;base64,{encoded}" alt="equation">'
            "</p>"
        )

    replacement.counter = 1
    return re.sub(r"\$\$(.*?)\$\$", replacement, source, flags=re.DOTALL)


def replace_inline_math(source: str, temp_dir: Path) -> str:
    """Replace simple inline ``$...$`` expressions with inline SVGs."""

    def replacement(match: re.Match[str]) -> str:
        formula = match.group(1).strip()
        equation_path = temp_dir / f"inline_equation_{replacement.counter:02d}.svg"
        replacement.counter += 1
        write_math_svg(formula, equation_path)
        encoded = base64.b64encode(equation_path.read_bytes()).decode("ascii")
        return (
            '<img class="inline-equation" '
            f'src="data:image/svg+xml;base64,{encoded}" alt="equation">'
        )

    replacement.counter = 1
    return re.sub(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", replacement, source)


def group_table_captions(body: str) -> str:
    """Keep each Markdown table caption attached to its table."""

    pattern = re.compile(r"(<p><strong>表[^<]*?</strong></p>\s*<table>.*?</table>)", re.DOTALL)

    def replacement(match: re.Match[str]) -> str:
        block = match.group(1)
        table = re.search(r"<table>.*?</table>", block, flags=re.DOTALL)
        header_count = len(re.findall(r"<th(?:\s|>)", table.group(0))) if table else 0
        class_name = "table-block wide-table" if header_count >= 6 else "table-block"
        return f'<div class="{class_name}">{block}</div>'

    return pattern.sub(replacement, body)


def group_figure_section(body: str) -> str:
    """Keep the anomaly-type heading with its first figure."""

    pattern = re.compile(
        r"(<h2>4\.4 不同异常类型分析</h2>\s*"
        r"<p><img.*?</p>\s*<p><em>.*?</em></p>)",
        re.DOTALL,
    )
    return pattern.sub(r'<div class="figure-section">\1</div>', body, count=1)


def render(input_path: Path, output_path: Path) -> None:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    base_uri = input_path.parent.as_uri() + "/"
    raw_source = input_path.read_text(encoding="utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="experiment_pdf_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        source = replace_display_math(raw_source, temp_dir)
        source = replace_inline_math(source, temp_dir)
        body = markdown.markdown(
            source,
            extensions=["tables", "fenced_code", "sane_lists"],
            output_format="html5",
        )
        body = group_table_captions(body)
        body = group_figure_section(body)
        document = f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>论文实验部分</title>
<base href="{html.escape(base_uri, quote=True)}">
<style>{CSS}</style>
</head>
<body><main class="paper">{body}</main></body>
</html>
'''
        html_path = temp_dir / "experiment.html"
        html_path.write_text(document, encoding="utf-8")
        chrome = find_chrome()
        command = [
            chrome,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--allow-file-access-from-files",
            "--no-pdf-header-footer",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=1000",
            f"--print-to-pdf={output_path}",
            html_path.as_uri(),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0 or not output_path.is_file():
            raise RuntimeError(
                "Chrome PDF rendering failed:\n"
                + completed.stdout
                + completed.stderr
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.input, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

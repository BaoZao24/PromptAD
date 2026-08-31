#!/usr/bin/env python3
"""Render the Markdown experiment section as an IEEE-style PDF.

The renderer is intentionally local and deterministic: Markdown is converted
to HTML, display equations are embedded as local SVGs, and Chrome's print
engine emits the final PDF.  The document uses a US-Letter, two-column, 10 pt layout with
Times New Roman for Latin text and a CJK fallback for Chinese glyphs.  Display
and inline equations are rendered with the same Times New Roman math settings.
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
DEFAULT_INPUT = ROOT / "docs/paper/论文实验部分.md"
DEFAULT_OUTPUT = ROOT / "docs/paper/论文实验部分.pdf"
TIMES_FONT_FILES = (
    Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold_Italic.ttf"),
)
for _font_path in TIMES_FONT_FILES:
    if _font_path.is_file():
        fontManager.addfont(str(_font_path))
FORMULA_SIZE_PT = 9.5
FORMULA_FONT = FontProperties(
    fname=str(TIMES_FONT_FILES[0]) if TIMES_FONT_FILES[0].is_file() else None,
    family="Times New Roman",
    math_fontfamily="custom",
    size=FORMULA_SIZE_PT,
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
  line-height: 1.17;
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
  text-align: center;
}
h3 {
  font-size: 10pt;
  line-height: 1.1;
  margin: 7pt 0 3pt;
}
h4 { font-size: 10pt; margin: 5pt 0 2pt; }
p { margin: 0 0 0.5pt; orphans: 2; widows: 2; text-indent: 2em; }
ul, ol { margin: 2pt 0 5pt; padding-left: 2em; }
li { margin: 1pt 0; padding-left: 0; text-indent: 0; }
li p { margin: 0; text-indent: 0; }
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
.table-block p { margin: 0 0 2pt; text-align: left; text-indent: 0; }
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
.figure-block {
  text-align: center;
  break-inside: avoid;
  -webkit-column-break-inside: avoid;
  margin: 6pt 0;
}
.figure-block.wide-figure { column-span: all; }
.figure-block.compact-figure {
  margin: 3pt 0 5pt;
}
.figure-block.compact-figure img {
  width: auto;
  max-width: 100%;
  max-height: 2.25in;
  margin: 2pt auto;
}
.figure-block p { margin: 0; text-align: center; text-indent: 0; }
.figure-block p + p { margin-top: 2pt; font-size: 8.5pt; }

hr { border: 0; border-top: 0.5pt solid #000; margin: 6pt 0; }
strong { font-weight: bold; }
em { font-style: italic; }

/* Keep equations in the current text column, as in a standard two-column paper. */
.equation {
  column-span: none;
  font-family: "Times New Roman", serif;
  font-size: 9.5pt;
  line-height: 1;
  text-align: center;
  text-indent: 0;
  margin: 7pt 0;
}
.equation img { width: auto; max-width: 90%; margin: 0 auto; }
.equation.numbered {
  position: relative;
  padding: 0 18pt;
  box-sizing: border-box;
}
.equation.numbered img { max-width: calc(100% - 24pt); }
.equation-number {
  position: absolute;
  right: 0;
  top: 50%;
  transform: translateY(-50%);
  font-family: "Times New Roman", serif;
  font-size: 9.5pt;
  font-style: normal;
}
.inline-equation {
  display: inline-block;
  width: auto;
  height: auto;
  margin: 0 1pt;
  line-height: 1;
  vertical-align: -0.18em;
}
.figure-section {
  column-span: all;
  break-inside: avoid;
  page-break-inside: avoid;
}
"""


# The scheme overview contains wide comparison tables and architecture figures.
# A single-column layout keeps these elements readable and avoids Chromium's
# column balancing placing a wide table across two narrow columns.
SINGLE_COLUMN_CSS = CSS + r"""
@page {
  size: A4;
  margin: 0.62in 0.55in 0.72in;
}

body {
  font-size: 10.5pt;
  line-height: 1.38;
  text-align: left;
}

.paper {
  column-count: 1;
  column-gap: 0;
}

p { margin: 0 0 7pt; }
h2 { margin-top: 12pt; }
h3 { margin-top: 9pt; }

/* Give equations enough vertical and horizontal room in the single-column
   manuscript. This prevents tall symbols and fractions from crowding the
   surrounding Chinese text. */
.equation {
  line-height: 1.25;
  margin: 10pt 0 11pt;
  overflow: visible;
}

.equation img {
  max-width: 96%;
  margin: 1pt auto;
}

table {
  font-size: 8.4pt;
  line-height: 1.15;
}

.table-block.wide-table {
  width: 100%;
}

.table-block.wide-table table {
  width: 100%;
  table-layout: auto;
}

.table-block.wide-table th,
.table-block.wide-table td {
  padding: 3pt 3.5pt;
  overflow-wrap: anywhere;
  word-break: break-word;
}

img {
  max-width: 100%;
}

"""


def find_chrome() -> str:
    for candidate in ("google-chrome", "chromium", "chromium-browser"):
        found = shutil.which(candidate)
        if found:
            return found
    raise RuntimeError("Chrome/Chromium is required for PDF rendering")


def write_math_svg(formula: str, equation_path: Path) -> None:
    """Write all equations with the same Times New Roman, 9.5 pt settings."""

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
            "mathtext.sf": "Times New Roman",
            "mathtext.tt": "Times New Roman",
            "mathtext.cal": "Times New Roman:italic",
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

    # Matplotlib crops math SVGs tightly to their glyph bounds. Descenders,
    # fraction denominators, and stacked subscripts can then lose their bottom
    # edge when Chrome embeds the SVG in a line box. Expand the SVG canvas
    # uniformly while preserving the rendered formula size.
    svg = equation_path.read_text(encoding="utf-8")
    root_pattern = re.compile(
        r'(<svg\b[^>]*\bwidth=")([0-9.]+)pt(" height=")([0-9.]+)pt'
        r'(" viewBox=")0 0 ([0-9.]+) ([0-9.]+)(")'
    )

    def padded_root(match: re.Match[str]) -> str:
        width = float(match.group(2))
        height = float(match.group(4))
        view_width = float(match.group(6))
        view_height = float(match.group(7))
        pad_x = 0.75
        pad_y = 1.5
        return (
            f'{match.group(1)}{width + 2 * pad_x:g}pt'
            f'{match.group(3)}{height + 2 * pad_y:g}pt'
            f'{match.group(5)}{-pad_x:g} {-pad_y:g} '
            f'{view_width + 2 * pad_x:g} {view_height + 2 * pad_y:g}'
            f'{match.group(8)}'
        )

    svg, count = root_pattern.subn(padded_root, svg, count=1)
    if count != 1:
        raise RuntimeError(f"Could not pad math SVG canvas: {equation_path}")
    equation_path.write_text(svg, encoding="utf-8")


def replace_display_math(source: str, temp_dir: Path) -> str:
    """Replace Markdown ``$$...$$`` blocks with crisp local SVG equations."""

    def replacement(match: re.Match[str]) -> str:
        formula = match.group(1).strip()
        tag_match = re.search(r"\\tag\{([^{}]+)\}\s*$", formula)
        tag = None
        if tag_match:
            tag = tag_match.group(1).strip()
            formula = formula[: tag_match.start()].rstrip()
        equation_path = temp_dir / f"equation_{replacement.counter:02d}.svg"
        replacement.counter += 1
        write_math_svg(formula, equation_path)
        encoded = base64.b64encode(equation_path.read_bytes()).decode("ascii")
        if tag is not None:
            return (
                '<p class="equation numbered">'
                f'<img src="data:image/svg+xml;base64,{encoded}" alt="equation">'
                f'<span class="equation-number">({html.escape(tag)})</span>'
                "</p>"
            )
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


def collapse_chinese_soft_wraps(source: str) -> str:
    """Remove editor-only line wraps inside Chinese paragraphs.

    Markdown converts a single newline to a whitespace character.  This is
    useful for wrapped English prose, but the manuscript is written as Chinese
    paragraphs whose source lines are manually wrapped; retaining that
    whitespace produces visible gaps between Chinese words in Chromium's PDF.
    Paragraph breaks and Markdown block boundaries remain untouched.
    """

    def is_cjk(text: str) -> bool:
        return any("\u3400" <= char <= "\u9fff" for char in text)

    def is_structural_line(text: str) -> bool:
        stripped = text.lstrip()
        return (
            not stripped
            or stripped.startswith(("#", "|", "```", "$$", ">", "![", "---", "***"))
        )

    def starts_new_block(text: str) -> bool:
        stripped = text.lstrip()
        return is_structural_line(text) or bool(re.match(r"^[-+*]\s+", stripped)) or bool(
            re.match(r"^\d+[.)]\s+", stripped)
        )

    lines = source.splitlines()
    collapsed: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        while index + 1 < len(lines):
            next_line = lines[index + 1]
            current = line.strip()
            if (
                not current
                or current.endswith("  ")
                or is_structural_line(current)
                or starts_new_block(next_line)
                or not is_cjk(current)
            ):
                break
            line += next_line.strip()
            index += 1
        collapsed.append(line)
        index += 1
    return "\n".join(collapsed) + ("\n" if source.endswith("\n") else "")


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


def group_figure_blocks(body: str) -> str:
    """Keep figures with captions and span wide multi-panel figures across columns."""

    pattern = re.compile(
        r"(<p><img.*?</p>\s*<p><em>(.*?)</em></p>)",
        re.DOTALL,
    )

    def replacement(match: re.Match[str]) -> str:
        caption = re.sub(r"<.*?>", "", match.group(2)).strip()
        figure_html = match.group(1)
        is_wide = caption.startswith("图1")
        class_name = "figure-block wide-figure" if is_wide else "figure-block compact-figure"
        return f'<div class="{class_name}">{match.group(1)}</div>'

    return pattern.sub(replacement, body)


def render(input_path: Path, output_path: Path, single_column: bool = False) -> None:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    base_uri = input_path.parent.as_uri() + "/"
    raw_source = input_path.read_text(encoding="utf-8")
    raw_source = collapse_chinese_soft_wraps(raw_source)

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
        body = group_figure_blocks(body)
        document = f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(input_path.stem)}</title>
<base href="{html.escape(base_uri, quote=True)}">
<style>{SINGLE_COLUMN_CSS if single_column else CSS}</style>
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
    parser.add_argument(
        "--single-column",
        action="store_true",
        help="Use a readable single-column A4 layout for scheme overviews",
    )
    args = parser.parse_args()
    render(args.input, args.output, single_column=args.single_column)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

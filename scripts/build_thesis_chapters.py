from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

from evidence_fashion.manifest import git_commit, sha256_file, utc_timestamp, write_json

ROOT = Path(__file__).parents[1]
THESIS = ROOT / "thesis"
RUNTIME = THESIS / ".cache"
MANIFEST = ROOT / "artifacts/manifests/thesis_chapters_manifest.json"
WORD_RE = re.compile(r"\b[\w’'-]+\b")

CHAPTERS: dict[int, dict[str, Any]] = {
    1: {
        "source": THESIS / "thesis_chapter_1.md",
        "output": THESIS / "CHPT1.docx",
        "title": "Introduction",
        "figures": {},
        "table_captions": [],
    },
    2: {
        "source": THESIS / "thesis_chapter_2.md",
        "output": THESIS / "CHPT2.docx",
        "title": "Literature Review",
        "figures": {},
        "table_captions": [
            "Representative research and the gaps leading to the present study",
        ],
    },
    3: {
        "source": THESIS / "thesis_chapter_3.md",
        "output": THESIS / "CHPT3.docx",
        "title": "Methodology",
        "figures": {
            "3.2": ("fig_01_system_architecture.svg", "Frozen multimodal recommendation and explanation boundary"),
            "3.3": ("fig_03_evidence_trace.svg", "Exact antecedent-applicable V3 rule trace"),
        },
        "table_captions": [],
    },
    4: {
        "source": THESIS / "thesis_chapter_4.md",
        "output": THESIS / "CHPT4.docx",
        "title": "Results",
        "figures": {
            "4.2": ("fig_05_recommendation_metrics.svg", "Held-out recommendation effectiveness"),
            "4.4": ("stage12_support_rates.svg", "Exact-trace and full-KB claim support with paired 95% confidence intervals"),
            "4.5": ("stage12_uifr.svg", "Unsupported Item-Fact Rate; lower is better"),
            "4.6": ("stage12_supported_per_100.svg", "Exact-trace supported claims per 100 words"),
        },
        "table_captions": [
            "Final paired Stage 12 explanation metrics with 95% bootstrap confidence intervals",
        ],
    },
    5: {
        "source": THESIS / "thesis_chapter_5.md",
        "output": THESIS / "CHPT5.docx",
        "title": "Discussion and Conclusion",
        "figures": {},
        "table_captions": [],
    },
}


def set_cell_shading(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = properties.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        properties.append(shading)
    shading.set(qn("w:fill"), fill)


def set_repeat_table_header(row) -> None:
    properties = row._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    properties.append(repeat)


def add_field(paragraph, instruction: str) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, end])


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_field(paragraph, "PAGE")


def add_inline(paragraph, text: str) -> None:
    pattern = re.compile(r"(\*\*.+?\*\*|(?<!\*)\*[^*]+?\*|`[^`]+`)")
    cursor = 0
    for match in pattern.finditer(text):
        if match.start() > cursor:
            paragraph.add_run(text[cursor:match.start()])
        token = match.group(0)
        run = paragraph.add_run(token.strip("*`"))
        if token.startswith("**"):
            run.bold = True
        elif token.startswith("*"):
            run.italic = True
        else:
            run.font.name = "Consolas"
            run.font.size = Pt(9.5)
        cursor = match.end()
    if cursor < len(text):
        paragraph.add_run(text[cursor:])


def add_native_equation(document: Document, lines: list[str]) -> None:
    raw = " ".join(line.strip() for line in lines if line.strip())
    raw = raw.replace("\\qquad", "    ").replace("\\,", " ")
    raw = raw.replace("\\lVert", "‖").replace("\\rVert", "‖")
    raw = raw.replace("\\sum", "∑").replace("\\frac", "frac")
    raw = raw.replace("\\cos", "cos").replace("\\min", "min").replace("\\max", "max")
    raw = raw.replace("\\models", "⊨").replace("\\land", "∧").replace("\\varnothing", "∅")
    raw = raw.replace("\\in", "∈").replace("\\text", "text")
    raw = raw.replace("\\overline", "overline").replace("\\mathbb", "")
    raw = raw.replace("\\left", "").replace("\\right", "")
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(4)
    paragraph.paragraph_format.space_after = Pt(7)
    run = paragraph.add_run(raw)
    run.font.name = "Cambria Math"
    run.font.size = Pt(11.5)
    run.italic = True


def convert_figure(path: Path) -> Path:
    if path.suffix.lower() != ".svg":
        return path
    RUNTIME.mkdir(parents=True, exist_ok=True)
    output = RUNTIME / f"{path.stem}.png"
    edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
    if not edge.exists():
        raise FileNotFoundError("Microsoft Edge is required to render canonical SVG figures.")
    subprocess.run(
        [
            str(edge),
            "--headless",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--screenshot={output}",
            "--window-size=1800,1000",
            path.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
    )
    if not output.exists():
        raise FileNotFoundError(f"Edge did not render {path}.")
    return output


def add_figure(document: Document, source: Path, caption: str, number: int, chapter: int) -> None:
    path = convert_figure(source)
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(path), width=Inches(6.25))
    cap = document.add_paragraph(style="Caption")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run(f"Figure {chapter}.{number}. {caption}")
    run.bold = True
    source_para = document.add_paragraph(style="Figure Source")
    source_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    source_para.add_run("Source: author-generated from frozen project artifacts.").italic = True


def configure_document(document: Document, chapter: int, title: str) -> None:
    section = document.sections[0]
    section.page_height = Cm(29.7)
    section.page_width = Cm(21)
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(3.2)
    section.right_margin = Cm(2.54)
    section.header_distance = Cm(1.25)
    section.footer_distance = Cm(1.25)

    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(12)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    normal.paragraph_format.space_after = Pt(7)
    normal.paragraph_format.widow_control = True

    heading_colors = {1: "17365D", 2: "1F4E79", 3: "365F91"}
    for level in range(1, 4):
        style = document.styles[f"Heading {level}"]
        style.font.name = "Arial"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
        style.font.color.rgb = RGBColor.from_string(heading_colors[level])
        style.font.bold = True
        style.font.size = Pt({1: 18, 2: 15, 3: 13}[level])
        style.paragraph_format.space_before = Pt({1: 18, 2: 14, 3: 10}[level])
        style.paragraph_format.space_after = Pt(6)
        style.paragraph_format.keep_with_next = True

    caption = document.styles["Caption"]
    caption.font.name = "Arial"
    caption.font.size = Pt(9.5)
    caption.font.color.rgb = RGBColor.from_string("333333")
    caption.paragraph_format.space_before = Pt(4)
    caption.paragraph_format.space_after = Pt(2)
    if "Figure Source" not in [style.name for style in document.styles]:
        source_style = document.styles.add_style("Figure Source", WD_STYLE_TYPE.PARAGRAPH)
        source_style.font.name = "Times New Roman"
        source_style.font.size = Pt(8.5)
        source_style.font.color.rgb = RGBColor.from_string("666666")
        source_style.paragraph_format.space_after = Pt(8)

    title_para = document.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_para.paragraph_format.space_before = Pt(110)
    run = title_para.add_run(f"CHAPTER {chapter}")
    run.font.name = "Arial"
    run.font.size = Pt(24)
    run.font.bold = True
    run.font.color.rgb = RGBColor.from_string("17365D")
    title_text = document.add_paragraph()
    title_text.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_text.paragraph_format.space_before = Pt(20)
    run = title_text.add_run(title.upper())
    run.font.name = "Arial"
    run.font.size = Pt(20)
    run.font.bold = True
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_before = Pt(40)
    subtitle.add_run("Evidence-Constrained Multimodal Fashion Recommendation\nMPhil Thesis").italic = True
    document.add_page_break()

    toc_heading = document.add_paragraph("Contents", style="Heading 1")
    toc_heading.alignment = WD_ALIGN_PARAGRAPH.LEFT
    toc_note = document.add_paragraph()
    toc_note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    toc_note.add_run("Update the table of contents in the assembled thesis after final pagination.").italic = True
    document.add_page_break()

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    header_run = header.add_run(f"Chapter {chapter}  |  {title}")
    header_run.font.name = "Arial"
    header_run.font.size = Pt(8.5)
    header_run.font.color.rgb = RGBColor.from_string("666666")
    add_page_number(section.footer.paragraphs[0])

    properties = document.core_properties
    properties.title = f"Chapter {chapter}: {title}"
    properties.subject = "Evidence-Constrained Multimodal Fashion Recommendation"
    properties.author = "Thesis Researcher"
    properties.keywords = "fashion recommendation, multimodal retrieval, rule RAG, faithfulness"


def add_table(document: Document, rows: list[list[str]], caption: str | None, chapter: int, number: int) -> None:
    if caption:
        cap = document.add_paragraph(style="Caption")
        cap.add_run(f"Table {chapter}.{number}. {caption}").bold = True
    table = document.add_table(rows=len(rows), cols=len(rows[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    table.autofit = True
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            cell = table.cell(row_index, col_index)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.space_before = Pt(2)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_index == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(value)
            run.font.name = "Arial"
            run.font.size = Pt(8.5)
            if row_index == 0:
                run.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
                set_cell_shading(cell, "1F4E79")
            elif row_index % 2 == 0:
                set_cell_shading(cell, "EAF2F8")
    set_repeat_table_header(table.rows[0])
    document.add_paragraph()


def build_chapter(chapter: int, spec: dict[str, Any]) -> dict[str, Any]:
    document = Document()
    configure_document(document, chapter, spec["title"])
    lines = spec["source"].read_text(encoding="utf-8").splitlines()
    figure_number = 0
    table_number = 0
    table_captions = iter(spec["table_captions"])
    skip_top_titles = 2
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        if not line:
            index += 1
            continue
        heading_match = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading_match:
            if skip_top_titles:
                skip_top_titles -= 1
                index += 1
                continue
            level = len(heading_match.group(1))
            heading = heading_match.group(2)
            if heading == "References":
                document.add_page_break()
            document.add_paragraph(heading, style=f"Heading {level}")
            key = heading.split()[0]
            if key in spec["figures"]:
                figure_number += 1
                filename, caption = spec["figures"][key]
                add_figure(document, ROOT / "artifacts/figures" / filename, caption, figure_number, chapter)
            index += 1
            continue
        if line == "\\[":
            equation_lines = []
            index += 1
            while index < len(lines) and lines[index].strip() != "\\]":
                equation_lines.append(lines[index])
                index += 1
            add_native_equation(document, equation_lines)
            index += 1
            continue
        if line.startswith("|") and index + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[index + 1]):
            table_rows = []
            while index < len(lines) and lines[index].startswith("|"):
                if not re.match(r"^\|[\s:|-]+\|$", lines[index]):
                    table_rows.append([cell.strip() for cell in lines[index].strip("|").split("|")])
                index += 1
            table_number += 1
            add_table(document, table_rows, next(table_captions, None), chapter, table_number)
            continue
        if re.match(r"^[-*]\s+", line):
            paragraph = document.add_paragraph(style="List Bullet")
            add_inline(paragraph, re.sub(r"^[-*]\s+", "", line))
            index += 1
            continue
        paragraph_lines = [line]
        index += 1
        while index < len(lines):
            candidate = lines[index].rstrip()
            if not candidate or candidate.startswith("#") or candidate == "\\[" or candidate.startswith("|") or re.match(r"^[-*]\s+", candidate):
                break
            paragraph_lines.append(candidate)
            index += 1
        text = " ".join(paragraph_lines)
        style = "Normal"
        if document.paragraphs and document.paragraphs[-1].text == "References":
            style = "Normal"
        paragraph = document.add_paragraph(style=style)
        add_inline(paragraph, text)
        if re.match(r"^\[\d+\]", text):
            paragraph.paragraph_format.left_indent = Cm(0.7)
            paragraph.paragraph_format.first_line_indent = Cm(-0.7)
            paragraph.paragraph_format.line_spacing = 1.0
            paragraph.paragraph_format.space_after = Pt(5)

    spec["output"].parent.mkdir(parents=True, exist_ok=True)
    document.save(spec["output"])
    source_text = spec["source"].read_text(encoding="utf-8")
    return {
        "chapter": chapter,
        "title": spec["title"],
        "source_words": len(WORD_RE.findall(source_text)),
        "source_sha256": sha256_file(spec["source"]),
        "docx_sha256": sha256_file(spec["output"]),
        "docx_bytes": spec["output"].stat().st_size,
        "figures_embedded": figure_number,
        "tables_embedded": table_number,
    }


def main() -> None:
    results = [build_chapter(chapter, spec) for chapter, spec in CHAPTERS.items()]
    manifest = {
        "stage": "thesis_chapter_build",
        "status": "complete",
        "generated_at_utc": utc_timestamp(),
        "git_commit_at_generation": git_commit(),
        "format": {
            "page": "A4",
            "body": "Times New Roman 12 pt, 1.5 spacing, justified",
            "headings": "Arial, three-level hierarchy",
            "features": ["title page", "contents placeholder", "page numbers", "Cambria Math equation blocks", "captioned figures", "styled tables", "chapter references"],
        },
        "source_context": {
            "note": "All five chapters are generated from versioned Markdown sources.",
        },
        "chapters": results,
        "output_artifact_hashes": {
            str(spec["output"].relative_to(ROOT)).replace("\\", "/"): sha256_file(spec["output"])
            for spec in CHAPTERS.values()
        },
    }
    write_json(MANIFEST, manifest)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

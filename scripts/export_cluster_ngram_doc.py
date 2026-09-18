"""Export the editable cluster/n-gram Markdown guide to a polished PDF."""

from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "Giai_Thich_Clusters_va_Ngrams.md"
OUTPUT = ROOT / "output" / "pdf" / "Giai_Thich_Clusters_va_Ngrams.pdf"


def register_fonts():
    candidates = [
        ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for regular, bold in candidates:
        if Path(regular).exists() and Path(bold).exists():
            pdfmetrics.registerFont(TTFont("DocSans", regular))
            pdfmetrics.registerFont(TTFont("DocSansBold", bold))
            return
    raise FileNotFoundError("Không tìm thấy font Unicode Arial hoặc DejaVu Sans")


def inline(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font name="DocSans">\1</font>', text)
    return text


def styles():
    base = getSampleStyleSheet()
    navy, blue, ink, muted = colors.HexColor("#12304A"), colors.HexColor("#147D92"), colors.HexColor("#243746"), colors.HexColor("#5C7180")
    return {
        "title": ParagraphStyle("Title", parent=base["Title"], fontName="DocSansBold", fontSize=23, leading=28, textColor=navy, alignment=TA_CENTER, spaceAfter=8),
        "subtitle": ParagraphStyle("Subtitle", parent=base["BodyText"], fontName="DocSans", fontSize=10.2, leading=14, textColor=muted, alignment=TA_CENTER, spaceAfter=18),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName="DocSansBold", fontSize=14, leading=18, textColor=navy, spaceBefore=12, spaceAfter=7, keepWithNext=True),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontName="DocSans", fontSize=9.3, leading=13.3, textColor=ink, spaceAfter=5),
        "bullet": ParagraphStyle("Bullet", parent=base["BodyText"], fontName="DocSans", fontSize=9.2, leading=13, leftIndent=13, firstLineIndent=-7, textColor=ink, spaceAfter=2.5),
        "quote": ParagraphStyle("Quote", parent=base["BodyText"], fontName="DocSans", fontSize=9.6, leading=14, leftIndent=12, rightIndent=8, borderColor=blue, borderWidth=0, borderPadding=8, backColor=colors.HexColor("#EAF6F8"), textColor=navy, spaceBefore=5, spaceAfter=9),
        "code": ParagraphStyle("Code", parent=base["Code"], fontName="DocSans", fontSize=8.4, leading=12, leftIndent=8, rightIndent=8, borderPadding=8, backColor=colors.HexColor("#F2F5F7"), textColor=ink, spaceBefore=3, spaceAfter=8),
        "small": ParagraphStyle("Small", parent=base["BodyText"], fontName="DocSans", fontSize=8.2, leading=11, textColor=muted),
    }


def parse_table(lines, sty):
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
            continue
        rows.append([Paragraph(inline(c), sty["small"]) for c in cells])
    widths = [35 * mm, 64 * mm, 77 * mm] if len(rows[0]) == 3 else None
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12304A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "DocSansBold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F7FAFB")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD8DE")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def markdown_to_story(text, sty):
    lines, story, i = text.splitlines(), [], 0
    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("# "):
            story.append(Spacer(1, 8 * mm))
            story.append(Paragraph(inline(line[2:]), sty["title"]))
        elif line.startswith("> "):
            block = []
            while i < len(lines) and lines[i].startswith("> "):
                block.append(lines[i][2:].rstrip("  "))
                i += 1
            story.append(Paragraph("<br/>".join(inline(part) for part in block), sty["quote"]))
            i -= 1
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), sty["h2"]))
        elif line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), sty["h2"]))
        elif line.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            story.append(Preformatted("\n".join(code), sty["code"]))
        elif line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1
            story.append(parse_table(table_lines, sty))
            story.append(Spacer(1, 5))
            i -= 1
        elif re.match(r"^[-*] ", line):
            story.append(Paragraph("• " + inline(line[2:]), sty["bullet"]))
        elif re.match(r"^\d+\. ", line):
            num, content = line.split(". ", 1)
            story.append(Paragraph(f"{num}. " + inline(content), sty["bullet"]))
        elif line == "---":
            # Content after the divider is editing/export guidance for the
            # Markdown source, not part of the stakeholder-facing PDF.
            break
        elif line.strip():
            story.append(Paragraph(inline(line.rstrip("  ")), sty["subtitle"] if line.startswith("Dành cho") else sty["body"]))
        i += 1
    return story


def decorate(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(colors.HexColor("#147D92"))
    canvas.rect(0, h - 7 * mm, w, 7 * mm, fill=1, stroke=0)
    canvas.setFont("DocSans", 7.5)
    canvas.setFillColor(colors.HexColor("#667B88"))
    canvas.drawString(18 * mm, 10 * mm, "HiFPT Journey Clustering - Tài liệu giải thích nhanh")
    canvas.drawRightString(w - 18 * mm, 10 * mm, f"Trang {doc.page}")
    canvas.restoreState()


def main():
    register_fonts()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=17 * mm, rightMargin=17 * mm, topMargin=17 * mm, bottomMargin=17 * mm, title="Hiểu nhanh Clusters và N-grams")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="guide", frames=[frame], onPage=decorate)])
    doc.build(markdown_to_story(SOURCE.read_text(encoding="utf-8"), styles()))
    print(OUTPUT)


if __name__ == "__main__":
    main()

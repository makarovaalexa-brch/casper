"""Deck helpers built on the author's own Google Slides template (Simple Light).

Everything is a NATIVE PowerPoint object so the deck survives a Google Slides
import and can be edited there.  Style rules live in STYLE.md next to this file.

Canvas: 10 x 5.625 in (16:9).  Fonts: Arial (theme).  Logo: layout-provided.
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.oxml.ns import qn
from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "template", "reading_template.pptx")
IMG_DIR = os.path.join(HERE, "img")

# ---------------------------------------------------------------- palette ----
BLACK   = RGBColor(0x00, 0x00, 0x00)
INK     = RGBColor(0x00, 0x00, 0x00)
GREY    = RGBColor(0x59, 0x59, 0x59)       # theme dk2
LIGHT   = RGBColor(0xEE, 0xEE, 0xEE)       # theme lt2  (table header fill)
ZEBRA   = RGBColor(0xF7, 0xF7, 0xF7)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
ORANGE  = RGBColor(0xF9, 0xCB, 0x9C)       # flow-chart box (author's)
ORANGE_D = RGBColor(0xE6, 0x91, 0x38)      # dark orange box (author's)
CREAM   = RGBColor(0xFC, 0xE5, 0xCD)       # container (author's)
GOLD    = RGBColor(0xFF, 0xAB, 0x40)       # theme accent1, used sparingly

FONT = "Arial"

SLIDE_W = Inches(10)
SLIDE_H = Inches(5.625)
MARGIN_L = Inches(0.34)
CONTENT_W = Inches(9.32)
BODY_TOP = Inches(1.26)
BODY_H = Inches(3.74)
BODY_BOTTOM = BODY_TOP + BODY_H          # 5.00 in

TITLE_SIZE = 28
BODY_SIZE = 18
SMALL_BODY = 16
CAPTION_SIZE = 12


# ------------------------------------------------------------------ deck ----
def new_deck():
    """Open the author's template and strip its slides, keeping master/layouts."""
    prs = Presentation(TEMPLATE)
    lst = prs.slides._sldIdLst
    for sldId in list(lst):
        prs.part.drop_rel(sldId.rId)
        lst.remove(sldId)
    return prs


def _layout(prs, name):
    for l in prs.slide_layouts:
        if l.name == name:
            return l
    raise KeyError(name)


def _set_runs(paragraph, text, size, bold=False, italic=False, color=INK):
    r = paragraph.add_run()
    r.text = text
    r.font.name = FONT
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    return r


def _drop_shape(sh):
    el = sh._element
    el.getparent().remove(el)


def title_slide(prs, title, name, date, subtitle=None):
    s = prs.slides.add_slide(_layout(prs, "TITLE"))
    t = s.placeholders[0]
    t.left, t.top, t.width, t.height = Inches(0.34), Inches(0.6), Inches(9.32), Inches(2.1)
    t.text_frame.text = ""
    t.text_frame.vertical_anchor = MSO_ANCHOR.BOTTOM
    _set_runs(t.text_frame.paragraphs[0], title, 32, bold=True)
    st = s.placeholders[1]
    st.left, st.top, st.width, st.height = Inches(0.34), Inches(2.85), Inches(9.32), Inches(1.6)
    st.text_frame.text = ""
    first = True
    if subtitle:
        _set_runs(st.text_frame.paragraphs[0], subtitle, 16, color=GREY)
        first = False
    p = st.text_frame.paragraphs[0] if first else st.text_frame.add_paragraph()
    p.space_before = Pt(10)
    _set_runs(p, name, 24)
    p = st.text_frame.add_paragraph()
    _set_runs(p, date, 16, color=GREY)
    # logo bottom right, as in the author's 2023 title slide
    s.shapes.add_picture(os.path.join(IMG_DIR, "icons", "reading_logo.png"),
                         Inches(7.55), Inches(4.55), width=Inches(2.0))
    return s


def section_slide(prs, title, line=None):
    s = prs.slides.add_slide(_layout(prs, "SECTION_HEADER"))
    t = s.placeholders[0]
    t.text_frame.text = ""
    _set_runs(t.text_frame.paragraphs[0], title, 34, bold=True)
    # sub-lines deliberately not rendered: the author's decks carry bare section titles
    return s


def slide(prs, title, body=None, size=BODY_SIZE, body_box=None):
    """Content slide on TITLE_AND_BODY.

    body: optional list of bullet items: str, or (str, level).  If None the
    body placeholder is removed and the caller places shapes freely.
    body_box: (left, top, width, height) in inches to resize the placeholder.
    Returns (slide, y_below_body_in_EMU).
    """
    s = prs.slides.add_slide(_layout(prs, "TITLE_AND_BODY"))
    t = s.placeholders[0]
    t.text_frame.text = ""
    _set_runs(t.text_frame.paragraphs[0], title, TITLE_SIZE, bold=True)
    ph = s.placeholders[1]
    if body is None:
        _drop_shape(ph)
        return s, BODY_TOP
    if body_box:
        l, tp, w, h = body_box
        ph.left, ph.top, ph.width, ph.height = Inches(l), Inches(tp), Inches(w), Inches(h)
    tf = ph.text_frame
    tf.word_wrap = True
    tf.text = ""
    y = ph.top
    first = True
    for it in body:
        text, level = (it, 0) if isinstance(it, str) else it
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = level
        p.space_before = Pt(6 if level == 0 else 2)
        sz = size if level == 0 else size - 2
        _set_runs(p, text, sz)
        chars = int(ph.width / Inches(1) * (13.5 * 18 / sz))
        n = 1 + len(text) // max(1, chars)
        y += Inches(sz / 72 * 1.2 * n + (6 if level == 0 else 2) / 72)
    return s, y


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text


# ------------------------------------------------------------------ text ----
def textbox(s, left, top, width, height, text, size=SMALL_BODY, bold=False,
            italic=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
            bullets=False):
    tb = s.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    lines = text if isinstance(text, (list, tuple)) else [text]
    for i, ln in enumerate(lines):
        level = 0
        if isinstance(ln, tuple):
            ln, level = ln
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.level = level
        p.space_after = Pt(3)
        if bullets:
            _bullet(p, level)
        _set_runs(p, ln, size - (2 if level else 0), bold=bold, italic=italic, color=color)
    return tb


def _bullet(p, level=0):
    pPr = p._p.get_or_add_pPr()
    ind = 0.22 + 0.22 * level
    pPr.set("marL", str(int(Inches(ind))))
    pPr.set("indent", str(-int(Inches(0.22))))
    for tag in ("a:buNone", "a:buChar", "a:buAutoNum"):
        for e in pPr.findall(qn(tag)):
            pPr.remove(e)
    bu = etree.SubElement(pPr, qn("a:buChar"))
    bu.set("char", "•" if level == 0 else "–")


def bullets(s, left, top, width, items, size=SMALL_BODY, height=None):
    """Free-placed bullet list (for two-column layouts). Returns bottom y."""
    h = height or Inches(0.4 * len(items))
    tb = textbox(s, left, top, width, h, items, size=size, bullets=True)
    y = top
    for it in items:
        text, level = (it, 0) if isinstance(it, str) else it
        sz = size - (2 if level else 0)
        chars = int(width / Inches(1) * (13.5 * 18 / sz))
        n = 1 + len(text) // max(1, chars)
        y += Inches(sz / 72 * 1.2 * n + 3 / 72)
    return y


def caption(s, left, top, width, text, size=CAPTION_SIZE, align=PP_ALIGN.LEFT):
    """'Figure N. ...' line, in the author's convention."""
    return textbox(s, left, top, width, Inches(0.3), text, size=size,
                   color=GREY, align=align)


# ----------------------------------------------------------------- table ----
def table(s, rows, left, top, width, col_w=None, font_size=11, row_h=0.26,
          align=None, first_col_bold=False, header_rows=1, zebra=True,
          bold_cells=()):
    """rows: list of lists of str.  bold_cells: set of (i, j) to bold.
    Returns the GraphicFrame; its .height is the declared minimum."""
    nr, nc = len(rows), len(rows[0])
    row_h = max(0.20, row_h)
    gf = s.shapes.add_table(nr, nc, left, top, width, Inches(row_h * nr))
    tbl = gf.table
    tbl.first_row = header_rows > 0
    tbl.horz_banding = False
    # neutral style: kill the theme's coloured table style
    tblPr = tbl._tbl.tblPr
    for e in tblPr.findall(qn("a:tableStyleId")):
        tblPr.remove(e)
    if col_w:
        total = sum(col_w)
        for j, w in enumerate(col_w):
            tbl.columns[j].width = Emu(int(width * w / total))
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        is_hdr = i < header_rows
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.05)
            c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            c.fill.fore_color.rgb = LIGHT if is_hdr else (
                ZEBRA if zebra and (i - header_rows) % 2 == 1 else WHITE)
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.space_after = Pt(0)
            if align:
                p.alignment = align[j]
            _set_runs(p, str(val), font_size,
                      bold=is_hdr or (first_col_bold and j == 0) or (i, j) in bold_cells)
    _borders(tbl, nr, nc)
    return gf


def _borders(tbl, nr, nc):
    """Thin grey horizontal rules only (academic table look)."""
    for i in range(nr):
        for j in range(nc):
            tc = tbl.cell(i, j)._tc
            tcPr = tc.get_or_add_tcPr()
            for side in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
                for e in tcPr.findall(qn(side)):
                    tcPr.remove(e)
            for side, on in (("a:lnL", False), ("a:lnR", False),
                             ("a:lnT", i == 0 or i == 1), ("a:lnB", i == nr - 1 or i == 0)):
                ln = etree.SubElement(tcPr, qn(side))
                if on:
                    ln.set("w", "9525")
                    sf = etree.SubElement(ln, qn("a:solidFill"))
                    clr = etree.SubElement(sf, qn("a:srgbClr"))
                    clr.set("val", "7F7F7F")
                else:
                    etree.SubElement(ln, qn("a:noFill"))


# ---------------------------------------------------------------- figure ----
def figure(s, name, left, top, width=None, height=None, cap=None, max_bottom=None):
    """PNG from img/ with an optional caption below. Returns y below caption."""
    from PIL import Image
    path = os.path.join(IMG_DIR, name)
    iw, ih = Image.open(path).size
    if width is not None:
        height = int(width * ih / iw)
    elif height is not None:
        width = int(height * iw / ih)
    if max_bottom is not None and top + height > max_bottom - (Inches(0.35) if cap else 0):
        height = int(max_bottom - (Inches(0.35) if cap else 0) - top)
        width = int(height * iw / ih)
    s.shapes.add_picture(path, left, top, width, height)
    y = top + height
    if cap:
        caption(s, left, y + Inches(0.05), max(width, Inches(4)), cap)
        y += Inches(0.35)
    return y


def icon(s, name, left, top, size=Inches(0.6)):
    return s.shapes.add_picture(os.path.join(IMG_DIR, "icons", name + ".png"),
                                left, top, size, size)


# ------------------------------------------------------- flow-chart parts ----
def _shape(s, kind, left, top, width, height, fill, line=None, line_w=0.75, radius=0.18):
    sh = s.shapes.add_shape(kind, left, top, width, height)
    sh.shadow.inherit = False
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(line_w)
    try:
        sh.adjustments[0] = radius
    except (IndexError, ValueError):
        pass
    return sh


def _label(sh, text, size=11, bold=False, color=INK, align=PP_ALIGN.CENTER):
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    lines = text if isinstance(text, (list, tuple)) else [text]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(0)
        _set_runs(p, ln, size, bold=bold, color=color)


def obox(s, left, top, width, height, text, size=11, dark=False, bold=False):
    """Author-style orange rounded box."""
    sh = _shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height,
                ORANGE_D if dark else ORANGE, radius=0.22)
    _label(sh, text, size=size, bold=bold, color=WHITE if dark else INK)
    return sh


def wbox(s, left, top, width, height, text, size=11):
    """White box with thin black outline (utterances, data)."""
    sh = _shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height,
                WHITE, line=BLACK, radius=0.18)
    _label(sh, text, size=size)
    return sh


def container(s, left, top, width, height, label=None, size=12):
    """Cream rounded container grouping a subsystem; label sits top-left."""
    sh = _shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height, CREAM, radius=0.08)
    if label:
        textbox(s, left + Inches(0.12), top + Inches(0.06), width - Inches(0.24),
                Inches(0.3), label, size=size, bold=True)
    return sh


def bubble(s, left, top, width, height, text, size=11, tail="left"):
    """Speech bubble (white, thin black outline)."""
    sh = _shape(s, MSO_SHAPE.RECTANGULAR_CALLOUT, left, top, width, height,
                WHITE, line=BLACK)
    try:
        # adjustments: x, y of the tail tip relative to the box (fractions)
        if tail == "left":
            sh.adjustments[0] = -0.62; sh.adjustments[1] = 0.25
        elif tail == "right":
            sh.adjustments[0] = 0.62; sh.adjustments[1] = 0.25
        elif tail == "down":
            sh.adjustments[0] = -0.2; sh.adjustments[1] = 0.85
        elif tail == "up":
            sh.adjustments[0] = -0.2; sh.adjustments[1] = -0.85
    except (IndexError, ValueError):
        pass
    _label(sh, text, size=size)
    return sh


def arrow(s, x1, y1, x2, y2, label=None, size=10, width=1.0, curved=False,
          label_dx=0, label_dy=Inches(-0.22), color=BLACK, head=True):
    kind = MSO_CONNECTOR.CURVE if curved else MSO_CONNECTOR.STRAIGHT
    c = s.shapes.add_connector(kind, x1, y1, x2, y2)
    c.line.color.rgb = color
    c.line.width = Pt(width)
    if head:
        el = c.line._get_or_add_ln()
        tail = etree.SubElement(el, qn("a:tailEnd"))
        tail.set("type", "triangle"); tail.set("w", "med"); tail.set("len", "med")
    if label:
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        textbox(s, mx - Inches(0.9) + label_dx, my + label_dy, Inches(1.8),
                Inches(0.24), label, size=size, align=PP_ALIGN.CENTER)
    return c


def person(s, left, top, size=Inches(0.5)):
    """Black silhouette person icon from two shapes."""
    head = _shape(s, MSO_SHAPE.OVAL, left + size * 0.3, top, size * 0.4, size * 0.4, BLACK)
    body = _shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, left, top + size * 0.45, size, size * 0.55,
                  BLACK, radius=0.5)
    # flatten the bottom: chord would be nicer, rounded rect is fine at this size
    return head, body


def database(s, left, top, width=Inches(0.45), height=Inches(0.5)):
    sh = _shape(s, MSO_SHAPE.CAN, left, top, width, height, WHITE, line=BLACK, line_w=1.25)
    return sh


# ---------------------------------------------------------------- checks ----
def bounds(sh):
    if sh.top is None or sh.left is None:
        return None
    return (sh.left, sh.top, sh.width or 0, sh.height or 0)

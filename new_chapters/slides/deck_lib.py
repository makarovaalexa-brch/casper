"""Shared pptx helpers for the CASPER supervisor deck.

Everything here emits NATIVE PowerPoint objects (text frames, tables, autoshapes,
connectors) so the deck stays fully editable after a Google Slides import.
No images, no rendered tables.
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR

# ---------------------------------------------------------------- palette ----
INK        = RGBColor(0x1A, 0x1A, 0x1A)   # body text
MUTED      = RGBColor(0x5F, 0x6B, 0x76)   # secondary text
RULE       = RGBColor(0xC8, 0xD0, 0xD8)   # hairlines
BG         = RGBColor(0xFF, 0xFF, 0xFF)
ACCENT     = RGBColor(0x1F, 0x4E, 0x79)   # paper A / primary
ACCENT_B   = RGBColor(0x2E, 0x75, 0x5B)   # paper B
ACCENT_C   = RGBColor(0x8C, 0x4A, 0x0F)   # paper C
BAND       = RGBColor(0xEE, 0xF3, 0xF7)   # table header fill
BAND_ALT   = RGBColor(0xF7, 0xF9, 0xFB)   # zebra fill

FONT = "Calibri"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

MARGIN_L = Inches(0.62)
CONTENT_W = SLIDE_W - 2 * MARGIN_L


def new_deck():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def _blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


# ------------------------------------------------------------------ text ----
def textbox(slide, left, top, width, height, text, size=16, bold=False,
            color=INK, align=PP_ALIGN.LEFT, italic=False, space_after=4,
            line_spacing=1.0, anchor=MSO_ANCHOR.TOP, font=FONT):
    """text may be a str or a list of (str, dict-of-overrides) / plain str lines."""
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0

    lines = text if isinstance(text, (list, tuple)) else [text]
    for i, line in enumerate(lines):
        over = {}
        if isinstance(line, tuple):
            line, over = line
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = over.get("align", align)
        p.space_after = Pt(over.get("space_after", space_after))
        p.line_spacing = over.get("line_spacing", line_spacing)
        if over.get("level"):
            p.level = over["level"]
        r = p.add_run()
        r.text = line
        f = r.font
        f.name = over.get("font", font)
        f.size = Pt(over.get("size", size))
        f.bold = over.get("bold", bold)
        f.italic = over.get("italic", italic)
        f.color.rgb = over.get("color", color)
    return tb


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# ----------------------------------------------------------------- chrome ----
def _rule(slide, top, color=RULE, width=None):
    ln = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN_L, top,
                                width or CONTENT_W, Pt(1.1))
    ln.fill.solid()
    ln.fill.fore_color.rgb = color
    ln.line.fill.background()
    ln.shadow.inherit = False
    return ln


def header(slide, title, claim=None, kicker=None, accent=ACCENT):
    """Standard slide chrome: kicker / title / one-line claim / hairline."""
    top = Inches(0.42)
    if kicker:
        textbox(slide, MARGIN_L, top, CONTENT_W, Inches(0.24), kicker.upper(),
                size=11, bold=True, color=accent)
        top = top + Inches(0.30)
    textbox(slide, MARGIN_L, top, CONTENT_W, Inches(0.5), title,
            size=27, bold=True, color=INK)
    top = top + Inches(0.52)
    if claim:
        textbox(slide, MARGIN_L, top, CONTENT_W, Inches(0.42), claim,
                size=14.5, color=MUTED, line_spacing=1.05)
        top = top + Inches(0.40)
    _rule(slide, top + Inches(0.06), accent)
    return top + Inches(0.26)   # y where content may begin


def slide(prs, title, claim=None, kicker=None, accent=ACCENT):
    s = _blank(prs)
    y = header(s, title, claim, kicker, accent)
    return s, y


def footer(slide, text, accent=MUTED):
    tb = textbox(slide, MARGIN_L, SLIDE_H - Inches(0.46), CONTENT_W, Inches(0.28),
                 text, size=10.5, color=accent, italic=True)
    tb.name = "FOOTER"          # pinned: relayout() never moves it
    return tb


# ------------------------------------------------------------------ table ----
def table(slide, rows, left, top, width, col_w=None, font_size=11.5,
          header_size=None, row_h=0.28, header_fill=BAND, zebra=True,
          align=None, accent=ACCENT, first_col_bold=False, header_rows=1):
    """rows: list of lists of str. Returns the GraphicFrame."""
    nr, nc = len(rows), len(rows[0])
    # Rows are laid out a little tighter than requested: PowerPoint grows a row
    # to fit its content anyway, so the declared height only needs to be a
    # sensible minimum, and a generous one wastes vertical space.
    row_h = max(0.215, row_h * 0.85)
    height = Inches(row_h * nr)
    gf = slide.shapes.add_table(nr, nc, left, top, width, height)
    tbl = gf.table
    tbl.first_row = header_rows > 0
    tbl.horz_banding = False

    if col_w:
        total = sum(col_w)
        for j, w in enumerate(col_w):
            tbl.columns[j].width = Emu(int(width * w / total))

    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        is_hdr = i < header_rows
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.05)
            cell.margin_top = cell.margin_bottom = Inches(0.015)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if is_hdr:
                cell.fill.fore_color.rgb = header_fill
            elif zebra and (i - header_rows) % 2 == 1:
                cell.fill.fore_color.rgb = BAND_ALT
            else:
                cell.fill.fore_color.rgb = BG

            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.space_after = Pt(0)
            if align:
                p.alignment = align[j] if not is_hdr else (
                    PP_ALIGN.LEFT if j == 0 else align[j])
            r = p.add_run()
            r.text = str(val)
            r.font.name = FONT
            r.font.size = Pt(header_size or font_size if is_hdr else font_size)
            r.font.bold = is_hdr or (first_col_bold and j == 0)
            r.font.color.rgb = accent if is_hdr else INK
    return gf


# ------------------------------------------------------------ diagram bits ---
def box(slide, left, top, width, height, text, fill=BG, line=ACCENT,
        size=12, bold=True, color=INK, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
        line_w=1.25):
    sh = slide.shapes.add_shape(shape, left, top, width, height)
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = line
    sh.line.width = Pt(line_w)
    sh.shadow.inherit = False
    try:
        sh.adjustments[0] = 0.10
    except (IndexError, ValueError):
        pass
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.margin_top = tf.margin_bottom = Inches(0.04)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    lines = text if isinstance(text, (list, tuple)) else [text]
    for i, ln in enumerate(lines):
        over = {}
        if isinstance(ln, tuple):
            ln, over = ln
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        p.space_after = Pt(1)
        r = p.add_run()
        r.text = ln
        r.font.name = FONT
        r.font.size = Pt(over.get("size", size))
        r.font.bold = over.get("bold", bold)
        r.font.color.rgb = over.get("color", color)
    return sh


def arrow(slide, x1, y1, x2, y2, color=MUTED, width=1.5, label=None,
          label_dy=Inches(-0.26), label_size=10):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    c.line.color.rgb = color
    c.line.width = Pt(width)
    el = c.line._get_or_add_ln()
    from pptx.oxml.ns import qn
    from lxml import etree
    tail = etree.SubElement(el, qn('a:tailEnd'))
    tail.set('type', 'triangle')
    tail.set('w', 'med')
    tail.set('len', 'med')
    if label:
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        textbox(slide, mx - Inches(0.85), my + label_dy, Inches(1.7),
                Inches(0.24), label, size=label_size, color=color,
                align=PP_ALIGN.CENTER)
    return c


def chip(slide, left, top, width, height, text, fill, color=RGBColor(0xFF, 0xFF, 0xFF),
         size=11, bold=True):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    sh.shadow.inherit = False
    tf = sh.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    r.font.name = FONT
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    return sh


IMG_DIR = None          # set by the caller; defaults to ./img


def figure(slide, name, left, top, width=None, height=None, caption=None,
           caption_size=10.5, align=PP_ALIGN.LEFT):
    """Place a PNG from the img/ directory, with a plain caption underneath.

    Exactly one of width/height is honoured; the other follows the image's own
    aspect ratio.  Returns the y coordinate just below the caption.
    """
    import os
    from PIL import Image
    d = IMG_DIR or os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
    path = os.path.join(d, name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    iw, ih = Image.open(path).size
    if width is not None:
        height = int(width * ih / iw)
    elif height is not None:
        width = int(height * iw / ih)
    else:
        raise ValueError("figure() needs width or height")
    pic = slide.shapes.add_picture(path, left, top, width, height)
    y = top + height
    if caption:
        tb = textbox(slide, left, y + Inches(0.07), width, Inches(0.3), caption,
                     size=caption_size, color=MUTED, italic=True,
                     line_spacing=1.1, align=align)
        n = 1 + len(caption) // max(1, int(width / Inches(1) * 17))
        y = y + Inches(0.07 + caption_size / 72 * 1.1 * n)
    pic.name = "FIGURE"
    return y


def _bounds(sh):
    if sh.top is None or sh.left is None:
        return None
    return (sh.left, sh.top, sh.width or 0, sh.height or 0)


def relayout(prs, gap=Inches(0.14), footer_top=SLIDE_H - Inches(0.52), min_row=Inches(0.20)):
    """Repair vertical collisions caused by tables being taller than assumed.

    Tables are laid out with a declared height of row_h * n_rows; content placed
    beneath one can therefore start inside it. This pass (a) shrinks any table
    that runs into the footer strip, then (b) pushes every shape that starts
    inside a table down past it, preserving relative offsets within the group.
    Returns a list of human-readable warnings for anything it could not fix.
    """
    warn = []
    for idx, s in enumerate(prs.slides, 1):
        tables = [sh for sh in s.shapes if sh.has_table]

        # (a) shrink tables that overrun the footer strip
        for gf in tables:
            b = _bounds(gf)
            if not b:
                continue
            over = (b[1] + b[3]) - footer_top
            if over > 0:
                tbl = gf.table
                n = len(tbl.rows)
                target = max(min_row, (footer_top - b[1]) // n)
                for r in tbl.rows:
                    r.height = int(target)
                gf.height = int(target * n)

        # (b) push colliding content below each table
        for _ in range(4):
            moved = False
            for gf in sorted(tables, key=lambda x: x.top):
                gl, gt, gw, gh = _bounds(gf)
                gb = gt + gh
                group = []
                for sh in s.shapes:
                    if sh is gf or sh.name == "FOOTER":
                        continue
                    b = _bounds(sh)
                    if not b:
                        continue
                    l, t, w, h = b
                    if w < Inches(0.1) or min(gl + gw, l + w) - max(gl, l) <= Inches(0.1):
                        continue
                    if t > gt + Inches(0.05):
                        group.append(sh)
                if not group:
                    continue
                # Move the WHOLE group by one offset so relative spacing inside
                # it is preserved. Moving only the shapes that start inside the
                # table collapses the gap between a section label and the
                # content beneath it, and repeats of this pass compound it.
                need = gb + gap - min(sh.top for sh in group)
                if need > Inches(0.02):
                    for sh in group:
                        sh.top = int(sh.top + need)
                    moved = True
            if not moved:
                break

        for sh in s.shapes:
            b = _bounds(sh)
            if b and not sh.has_text_frame and b[1] + b[3] > SLIDE_H:
                warn.append("slide %d: shape runs past the canvas" % idx)
            if b and sh.has_table and b[1] + b[3] > footer_top + Inches(0.05):
                warn.append("slide %d: table still overruns the footer" % idx)
    return warn


def bullets(slide, left, top, width, items, size=14.5, gap=0.055,
            marker_color=ACCENT, line_spacing=1.06):
    """items: list of str, or (head, tail) where head is bolded."""
    y = top
    for it in items:
        head, tail = (it, None) if isinstance(it, str) else it
        d = slide.shapes.add_shape(MSO_SHAPE.OVAL, left, y + Pt(size * 0.42),
                                   Pt(4.6), Pt(4.6))
        d.fill.solid()
        d.fill.fore_color.rgb = marker_color
        d.line.fill.background()
        d.shadow.inherit = False
        tb = slide.shapes.add_textbox(left + Inches(0.20), y,
                                      width - Inches(0.20), Inches(0.3))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.line_spacing = line_spacing
        r = p.add_run()
        r.text = head
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.bold = tail is not None
        r.font.color.rgb = INK
        if tail:
            r2 = p.add_run()
            r2.text = " " + tail
            r2.font.name = FONT
            r2.font.size = Pt(size)
            r2.font.color.rgb = MUTED
        n_lines = 1 + (len(head) + len(tail or "")) // int(width / Inches(1) * 15.5)
        y = y + Inches(size / 72 * line_spacing * n_lines + gap)
    return y

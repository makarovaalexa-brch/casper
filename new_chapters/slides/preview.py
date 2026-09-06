"""Crude rasteriser for a pptx so slides can be eyeballed without PowerPoint.

Draws shapes (fill + outline), pictures, tables (grid + text) and text frames
with PIL. Fonts and wrapping are approximate; geometry is exact. Usage:

    python preview.py deck.pptx out_dir [slide numbers...]
"""
import os, sys, io
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Emu

DPI = 110
FONT_DIR = "C:/Windows/Fonts"


def _font(pt, bold=False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(os.path.join(FONT_DIR, name), max(6, int(pt * DPI / 72)))
    except Exception:
        return ImageFont.load_default()


def _rgb(color, default=(0, 0, 0)):
    try:
        if color.type is not None and color.rgb is not None:
            return tuple(int(str(color.rgb)[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        pass
    return default


def _wrap(draw, text, font, width):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= width or not cur:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def _draw_text_frame(draw, tf, box, default_pt=18, bold_default=False, anchor="top", indent_bullets=False):
    x, y, w, h = box
    pad = 4
    y_cur = y + pad
    lines_out = []
    for p in tf.paragraphs:
        runs = p.runs
        text = "".join(r.text for r in runs) or p.text
        if not text.strip():
            continue
        pt = None; bold = bold_default; color = (0, 0, 0)
        for r in runs:
            if r.font.size: pt = r.font.size.pt
            if r.font.bold is not None: bold = r.font.bold
            color = _rgb(r.font.color, color)
        pt = pt or default_pt
        f = _font(pt, bold)
        lvl = p.level or 0
        ind = 0
        pPr = p._p.pPr
        has_bullet = False
        if pPr is not None:
            marL = pPr.get("marL")
            if marL: ind = Emu(int(marL)) / 914400 * DPI
            if pPr.find("{http://schemas.openxmlformats.org/drawingml/2006/main}buChar") is not None:
                has_bullet = True
        if indent_bullets and not has_bullet:
            has_bullet = True; ind = ind or (0.25 + 0.25 * lvl) * DPI
        for k, ln in enumerate(_wrap(draw, text, f, w - pad * 2 - ind)):
            if k == 0 and has_bullet:
                draw.text((x + pad + ind - 0.16 * DPI, y_cur), "\u2022", font=f, fill=color)
            lines_out.append((x + pad + ind, y_cur, ln, f, color))
            y_cur += int(pt * DPI / 72 * 1.2)
        y_cur += int(pt * DPI / 72 * 0.25)
    total = y_cur - y
    off = 0
    if anchor == "middle":
        off = max(0, (h - total) // 2)
    for (lx, ly, ln, f, color) in lines_out:
        draw.text((lx, ly + off), ln, font=f, fill=color)
    return y_cur + off


def render_slide(prs, s, idx, out):
    W = int(prs.slide_width / 914400 * DPI); H = int(prs.slide_height / 914400 * DPI)
    im = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(im)
    # layout pictures (logo)
    for sh in list(s.slide_layout.shapes):
        if sh.shape_type == 13:
            _pic(im, sh)
    for sh in s.shapes:
        _shape(im, draw, sh)
    im.save(os.path.join(out, "s%02d.png" % idx))


def _pic(im, sh):
    try:
        p = Image.open(io.BytesIO(sh.image.blob)).convert("RGBA")
    except Exception:
        return
    x, y = int(sh.left / 914400 * DPI), int(sh.top / 914400 * DPI)
    w, h = max(1, int(sh.width / 914400 * DPI)), max(1, int(sh.height / 914400 * DPI))
    # honour crop
    try:
        cl, ct, cr, cb = sh.crop_left, sh.crop_top, sh.crop_right, sh.crop_bottom
        if any((cl, ct, cr, cb)):
            pw, ph = p.size
            p = p.crop((int(pw * cl), int(ph * ct), int(pw * (1 - cr)), int(ph * (1 - cb))))
    except Exception:
        pass
    p = p.resize((w, h))
    im.paste(p, (x, y), p)


def _shape(im, draw, sh):
    if sh.left is None:
        return
    x, y = int(sh.left / 914400 * DPI), int(sh.top / 914400 * DPI)
    w, h = int((sh.width or 0) / 914400 * DPI), int((sh.height or 0) / 914400 * DPI)
    if sh.shape_type == 13:
        _pic(im, sh); return
    if sh.shape_type == 6:  # group
        for sub in sh.shapes:
            _shape(im, draw, sub)
        return
    if getattr(sh, "has_table", False) and sh.has_table:
        tbl = sh.table
        cx = x
        col_px = [int(c.width / 914400 * DPI) for c in tbl.columns]
        ry = y
        for i, row in enumerate(tbl.rows):
            rh = int(row.height / 914400 * DPI)
            # grow row to fit text
            need = rh
            cx = x
            cells = []
            for j, col in enumerate(tbl.columns):
                c = tbl.cell(i, j)
                fill = (255, 255, 255)
                try:
                    if c.fill.type == 1: fill = _rgb(c.fill.fore_color, fill)
                except Exception:
                    pass
                pt = 11; bold = False; text = c.text
                for p in c.text_frame.paragraphs:
                    for r in p.runs:
                        if r.font.size: pt = r.font.size.pt
                        if r.font.bold: bold = True
                f = _font(pt, bold)
                lines = _wrap(draw, text, f, col_px[j] - 8)
                hh = int(len(lines) * pt * DPI / 72 * 1.2 + 6)
                need = max(need, hh)
                cells.append((cx, col_px[j], fill, lines, f))
                cx += col_px[j]
            for (cx0, cw, fill, lines, f) in cells:
                draw.rectangle([cx0, ry, cx0 + cw, ry + need], fill=fill, outline=(200, 200, 200))
                ty = ry + 3
                for ln in lines:
                    draw.text((cx0 + 4, ty), ln, font=f, fill=(0, 0, 0)); ty += int(f.size * 1.2)
            ry += need
        if ry > y + h + 2:
            draw.rectangle([x, y + h, x + sum(col_px), ry], outline=(255, 0, 0), width=2)
        return
    if sh.shape_type in (9, 10):  # connector / line
        try:
            col = _rgb(sh.line.color, (0, 0, 0))
        except Exception:
            col = (0, 0, 0)
        x2, y2 = int(sh.left / 914400 * DPI + (sh.width or 0) / 914400 * DPI), int(sh.top / 914400 * DPI + (sh.height or 0) / 914400 * DPI)
        flipH = sh._element.xpath(".//p:spPr/a:xfrm/@flipH")
        flipV = sh._element.xpath(".//p:spPr/a:xfrm/@flipV")
        x1, y1 = x, y
        if flipH and flipH[0] == "1": x1, x2 = x2, x1
        if flipV and flipV[0] == "1": y1, y2 = y2, y1
        draw.line([x1, y1, x2, y2], fill=col, width=2)
        # arrow head
        import math
        ang = math.atan2(y2 - y1, x2 - x1)
        L = 9
        pts = [(x2, y2), (x2 - L * math.cos(ang - 0.4), y2 - L * math.sin(ang - 0.4)),
               (x2 - L * math.cos(ang + 0.4), y2 - L * math.sin(ang + 0.4))]
        draw.polygon(pts, fill=col)
        return
    # autoshape / textbox / placeholder
    fill = None; outline = None
    try:
        if sh.fill.type == 1:
            fill = _rgb(sh.fill.fore_color, (200, 200, 200))
    except Exception:
        pass
    try:
        if sh.line.fill.type == 1:
            outline = _rgb(sh.line.color, (0, 0, 0))
    except Exception:
        pass
    if fill is not None or outline is not None:
        try:
            ast = sh.auto_shape_type
        except Exception:
            ast = None
        if ast is not None and "ROUNDED" in str(ast):
            draw.rounded_rectangle([x, y, x + w, y + h], radius=max(4, int(min(w, h) * 0.18)), fill=fill, outline=outline)
        elif ast is not None and "OVAL" in str(ast):
            draw.ellipse([x, y, x + w, y + h], fill=fill, outline=outline)
        else:
            draw.rectangle([x, y, x + w, y + h], fill=fill, outline=outline)
    if sh.has_text_frame and sh.text_frame.text.strip():
        is_ph = sh.is_placeholder
        default_pt = 18
        bold_default = False
        anchor = "top"
        indent = False
        if is_ph:
            t = sh.placeholder_format.type
            if t in (1, 3):
                default_pt = 28; bold_default = True; anchor = "middle"
            elif t == 2:
                indent = True
        else:
            try:
                if sh.text_frame.vertical_anchor is not None and "MIDDLE" in str(sh.text_frame.vertical_anchor):
                    anchor = "middle"
            except Exception:
                pass
        yend = _draw_text_frame(draw, sh.text_frame, (x, y, w, h), default_pt, bold_default, anchor, indent)
        if yend > y + h + 6 and not is_ph:
            draw.rectangle([x, y, x + w, yend], outline=(255, 0, 0), width=1)


def main():
    path, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    want = set(int(a) for a in sys.argv[3:])
    prs = Presentation(path)
    for i, s in enumerate(prs.slides, 1):
        if want and i not in want:
            continue
        render_slide(prs, s, i, out)
    print("rendered to", out)


if __name__ == "__main__":
    main()

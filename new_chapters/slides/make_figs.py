"""Rasterise chapter figures into PNGs for the supervisor deck.

    python new_chapters/slides/make_figs.py

Writes new_chapters/slides/img/*.png.  Two modes:

  standalone  -- a one-figure PDF; render page 0 and trim whitespace.
  in-chapter  -- a page of a compiled chapter; locate the LaTeX caption,
                 take the band between the previous text block and the
                 caption, render it, and trim whitespace.

The LaTeX caption is deliberately excluded: the deck writes its own
caption line under each image.
"""
import os
import numpy as np
import pymupdf as fitz

HERE = os.path.dirname(os.path.abspath(__file__))
NC = os.path.dirname(HERE)
IMG = os.path.join(HERE, "img")
os.makedirs(IMG, exist_ok=True)

DPI = 200
PAD = 6          # px of white margin kept after trimming


def _trim(pix, thresh=248):
    """Crop a rendered pixmap to its non-white bounding box."""
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n >= 3:
        g = a[:, :, :3].min(axis=2)
    else:
        g = a[:, :, 0]
    mask = g < thresh
    if not mask.any():
        return None
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    y0, y1 = rows[0], rows[-1] + 1
    x0, x1 = cols[0], cols[-1] + 1
    y0 = max(0, y0 - PAD); x0 = max(0, x0 - PAD)
    y1 = min(pix.height, y1 + PAD); x1 = min(pix.width, x1 + PAD)
    return a[y0:y1, x0:x1, :3].copy()


def _save(arr, out):
    from PIL import Image
    Image.fromarray(arr).save(out)
    return out


def standalone(pdf, out):
    d = fitz.open(pdf)
    pix = d[0].get_pixmap(dpi=DPI)
    arr = _trim(pix)
    d.close()
    if arr is None:
        print("  !! blank: %s" % pdf)
        return None
    p = os.path.join(IMG, out)
    _save(arr, p)
    print("  %-34s %4d x %4d  <- %s" % (out, arr.shape[1], arr.shape[0],
                                        os.path.basename(pdf)))
    return p


def in_chapter(pdf, page_no, caption_anchor, out, glue=26.0, full_width=False):
    """Crop the artwork sitting immediately above a LaTeX caption.

    The artwork is a cluster of vector paths, embedded images and small text
    runs (axis tick labels, legend entries).  We seed the cluster with the
    lowest such element above the caption and grow upward while the vertical
    gap to the next element stays under `glue` points; the first real gap is
    the boundary between the figure and the body text above it.
    """
    d = fitz.open(pdf)
    pg = d[page_no - 1]
    hits = pg.search_for(caption_anchor)
    if not hits:
        d.close()
        print("  !! caption not found on p%d: %r" % (page_no, caption_anchor))
        return None
    cap_top = min(h.y0 for h in hits)

    items = []                                   # (y0, y1, x0, x1)
    for dr in pg.get_drawings():
        r = dr["rect"]
        if r.y1 <= cap_top - 1 and r.height < pg.rect.height * 0.9:
            items.append((r.y0, r.y1, r.x0, r.x1))
    for blk in pg.get_text("dict")["blocks"]:
        r = fitz.Rect(blk["bbox"])
        if r.y1 <= cap_top - 1:
            items.append((r.y0, r.y1, r.x0, r.x1))
    items = [it for it in items if it[1] > pg.rect.y0 + 40]
    if not items:
        d.close()
        print("  !! nothing above the caption on p%d" % page_no)
        return None

    items.sort(key=lambda it: it[1])
    band = [items[-1]]
    top = items[-1][0]
    changed = True
    while changed:
        changed = False
        for it in items:
            if it in band:
                continue
            if it[1] >= top - glue and it[0] < top:
                band.append(it)
                top = min(top, it[0])
                changed = True

    if full_width:
        x0, x1 = pg.rect.x0, pg.rect.x1
    else:
        x0 = min(it[2] for it in band)
        x1 = max(it[3] for it in band)
    clip = fitz.Rect(max(pg.rect.x0, x0 - 4), max(pg.rect.y0, top - 4),
                     min(pg.rect.x1, x1 + 4), cap_top - 2)
    if clip.height < 30 or clip.width < 60:
        d.close()
        print("  !! band too thin on p%d for %r" % (page_no, caption_anchor))
        return None
    pix = pg.get_pixmap(dpi=DPI, clip=clip)
    arr = _trim(pix)
    d.close()
    if arr is None or arr.shape[0] < 40:
        print("  !! empty band: %s p%d" % (os.path.basename(pdf), page_no))
        return None
    p = os.path.join(IMG, out)
    _save(arr, p)
    print("  %-34s %4d x %4d  <- %s p%d" % (out, arr.shape[1], arr.shape[0],
                                            os.path.basename(pdf), page_no))
    return p


if __name__ == "__main__":
    A = os.path.join(NC, "chapterA_v2")
    B = os.path.join(NC, "chapterB_v2")
    C = os.path.join(NC, "chapterC_v2")
    L = os.path.join(NC, "paperC_lbr")

    print("standalone:")
    standalone(os.path.join(A, "fig_graded_curve.pdf"), "a_graded_curve.png")
    standalone(os.path.join(B, "m7d_route_atlas.pdf"), "b_route_atlas.png")

    print("in-chapter:")
    ap = os.path.join(A, "chapterA_v2.pdf")
    bp = os.path.join(B, "chapterB_v2.pdf")
    in_chapter(ap, 14, "The gap map", "a_gap_map.png")
    in_chapter(ap, 19, "The channel-agnostic interface", "a_architecture.png")
    in_chapter(bp, 4, "The measurement contract", "b_contract.png")
    in_chapter(bp, 12, "Full NDCG@10 against question budget", "b_budget_curves.png")
    in_chapter(bp, 29, "The reach pipeline", "b_act5_pipeline.png", full_width=True)

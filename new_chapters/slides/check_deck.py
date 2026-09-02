"""Verify the built deck: shape overlaps, image references, speaker notes.

    python new_chapters/slides/check_deck.py
"""
import os
from pptx import Presentation
from pptx.util import Emu, Inches

HERE = os.path.dirname(os.path.abspath(__file__))
PPTX = os.path.join(HERE, "casper_supervisor_2026-09.pptx")
TOL = Emu(int(Inches(0.04)))          # shapes may touch by this much

def eff_h(sh):
    """Height a shape actually occupies.

    A plain textbox is declared with a generous placeholder height, so its box
    bounds overstate what the eye sees.  Estimate the wrapped height from the
    run sizes instead and take the smaller of the two.  Tables, pictures and
    autoshapes are trusted at their declared height.
    """
    if not sh.has_text_frame or sh.shape_type not in (17, None):
        return sh.height
    if sh.has_table if hasattr(sh, "has_table") else False:
        return sh.height
    cols = max(8.0, sh.width / 914400 * 15.0)      # chars per line, ~ Calibri
    total = 0.0
    for p in sh.text_frame.paragraphs:
        txt = "".join(r.text for r in p.runs)
        pts = max([r.font.size.pt for r in p.runs if r.font.size] or [14])
        ls = p.line_spacing if isinstance(p.line_spacing, float) else 1.0
        lines = max(1, -(-len(txt) // int(cols / (pts / 14.0))))
        total += lines * pts * ls * 1.22 + (p.space_after.pt if p.space_after else 0)
    return min(sh.height, int(total / 72.0 * 914400))


prs = Presentation(PPTX)
titles, bad_overlap, bad_notes, imgs = [], [], [], []

for i, s in enumerate(prs.slides, 1):
    shapes = []
    for sh in s.shapes:
        if sh.top is None or sh.left is None or not sh.width or not sh.height:
            continue
        if sh.shape_type == 13:                       # PICTURE
            imgs.append((i, sh.name))
        shapes.append(sh)

    # first sizeable bold text block on the slide is its title
    for sh in s.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip():
            t = sh.text_frame.text.strip().splitlines()[0]
            runs = [r for p in sh.text_frame.paragraphs for r in p.runs]
            if runs and runs[0].font.size and runs[0].font.size.pt >= 25:
                titles.append((i, t))
                break
    else:
        titles.append((i, "(no title)"))

    boxes = [(sh, sh.left, sh.top, sh.width, eff_h(sh)) for sh in shapes]
    boxes.sort(key=lambda r: (r[2], r[1]))
    for a in range(len(boxes)):
        A, al, at, aw, ah = boxes[a]
        for b in range(a + 1, len(boxes)):
            B, bl, bt, bw, bh = boxes[b]
            if bt >= at + ah - TOL:
                break
            ox = min(al + aw, bl + bw) - max(al, bl)
            oy = min(at + ah, bt + bh) - max(at, bt)
            if ox > TOL and oy > TOL:
                # bullet markers sit inside their own text line by design
                # bullet markers are 4.6 pt dots that sit inside their own
                # text line by design
                if min(aw, ah, bw, bh) < Emu(int(Inches(0.12))):
                    continue
                bad_overlap.append(
                    (i, str(A.shape_type), str(B.shape_type),
                     round(ox / 914400, 2), round(oy / 914400, 2),
                     (A.text_frame.text[:34] if A.has_text_frame else ""),
                     (B.text_frame.text[:34] if B.has_text_frame else "")))

    if s.has_notes_slide:
        txt = s.notes_slide.notes_text_frame.text.strip()
        if txt and not txt.startswith("NOTE TO AUTHOR:"):
            bad_notes.append((i, txt[:70]))

print("slides: %d" % len(prs.slides))
print("\nTITLES")
for i, t in titles:
    print("%3d  %s" % (i, t))

print("\nIMAGES PLACED: %d" % len(imgs))
missing = [f for f in os.listdir(os.path.join(HERE, "img"))] if os.path.isdir(
    os.path.join(HERE, "img")) else []
print("img/ contains: %s" % ", ".join(sorted(missing)))

print("\nNOTES")
n_notes = sum(1 for s in prs.slides
              if s.has_notes_slide and s.notes_slide.notes_text_frame.text.strip())
print("slides carrying notes: %d (all must start 'NOTE TO AUTHOR:')" % n_notes)
for i, t in bad_notes:
    print("  !! slide %d non-conforming note: %s" % (i, t))

print("\nOVERLAPS: %d" % len(bad_overlap))
for row in bad_overlap:
    line = ("  !! slide %d  %s x %s  (%.2f x %.2f in)  %r / %r" %
            (row[0], row[1], row[2], row[3], row[4], row[5], row[6]))
    print(line.encode("ascii", "replace").decode("ascii"))

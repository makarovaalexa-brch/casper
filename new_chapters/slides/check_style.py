"""Mechanical checks against STYLE.md.  python check_style.py deck.pptx"""
import re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pptx import Presentation
from pptx.util import Inches

BANNED = re.compile(r"\b(Paper [ABC]|Act \d|arc|walkthrough|story|hook|one ruler|"
                    r"CASPER-[RU]|and yet|this is what|the moment)\b", re.I)
EM = "—"


def text_shapes(s):
    for sh in s.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip():
            yield sh


def main(path):
    prs = Presentation(path)
    probs = []
    total_words = 0
    for i, s in enumerate(prs.slides, 1):
        title = None
        words = 0
        for sh in s.shapes:
            b = (sh.left, sh.top, sh.width or 0, sh.height or 0)
            if sh.top is not None and sh.top + (sh.height or 0) > Inches(5.3) and sh.shape_type != 13:
                probs.append((i, "below 5.25in: %s" % sh.name))
            if sh.is_placeholder and sh.placeholder_format.type in (1, 3):
                title = sh.text_frame.text.strip()
            if sh.has_text_frame:
                for p in sh.text_frame.paragraphs:
                    t = p.text.strip()
                    if not t:
                        continue
                    if EM in t:
                        probs.append((i, "em-dash: %r" % t[:60]))
                    if BANNED.search(t):
                        probs.append((i, "banned word: %r" % t[:70]))
                    for r in p.runs:
                        if r.font.size and r.font.size.pt < 10 and not sh.is_placeholder:
                            probs.append((i, "font %.1f pt: %r" % (r.font.size.pt, t[:40])))
                    if sh.is_placeholder and sh.placeholder_format.type == 2:
                        if t.endswith(".") and not re.search(r"\d\.$", t):
                            probs.append((i, "bullet ends with full stop: %r" % t[:60]))
                        if "?" in t:
                            probs.append((i, "question in bullet: %r" % t[:60]))
                        if len(t.split()) > 16:
                            probs.append((i, "bullet > 16 words: %r" % t[:60]))
                        if any(r.font.bold for r in p.runs):
                            probs.append((i, "bold in bullet: %r" % t[:60]))
                    if not (getattr(sh, "has_table", False)):
                        words += len(t.split())
            if getattr(sh, "has_table", False) and sh.has_table:
                for r in sh.table.rows:
                    for c in r.cells:
                        for p in c.text_frame.paragraphs:
                            for rn in p.runs:
                                if rn.font.size and rn.font.size.pt < 9:
                                    probs.append((i, "table font %.1f" % rn.font.size.pt))
        if title:
            if len(title) > 44 and i > 1:
                probs.append((i, "title > 44 chars (runs under logo): %r" % title))
            if title.endswith(".") or "?" in title:
                probs.append((i, "title punctuation: %r" % title))
            if len(title.split()) > 6 and i > 1:
                probs.append((i, "title > 6 words: %r" % title))
        if words > 90:
            probs.append((i, "%d words outside tables" % words))
        total_words += words
        if not s.has_notes_slide or not s.notes_slide.notes_text_frame.text.strip():
            if title and not title.lower().startswith(("appendix", "chapter")):
                probs.append((i, "no speaker notes"))
        # overlap check among non-line shapes
        shapes = [sh for sh in s.shapes if sh.shape_type not in (9, 10) and sh.width and sh.height
                  and sh.has_text_frame and sh.text_frame.text.strip()]
        for a in range(len(shapes)):
            for b in range(a + 1, len(shapes)):
                A, B = shapes[a], shapes[b]
                ox = min(A.left + A.width, B.left + B.width) - max(A.left, B.left)
                oy = min(A.top + A.height, B.top + B.height) - max(A.top, B.top)
                if ox > Inches(0.15) and oy > Inches(0.12):
                    # containers legitimately hold labels: skip if one is much larger
                    if A.width * A.height > 6 * B.width * B.height or B.width * B.height > 6 * A.width * A.height:
                        continue
                    probs.append((i, "overlap: %r / %r" % (A.text_frame.text[:25], B.text_frame.text[:25])))
    print("slides:", len(prs.slides), "| words outside tables:", total_words)
    for i, m in probs:
        print("slide %2d: %s" % (i, m))
    print(len(probs), "issues")


if __name__ == "__main__":
    main(sys.argv[1])

"""Assemble the supervisor deck from the per-chapter parts (author-style rebuild).

    python build_deck2.py            -> casper_supervisor_2026-09.pptx
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deck_lib2 import new_deck
from part_intro import build_intro, build_close, build_appendix_methods
from part_ch1 import build_ch1, build_appendix_ch1
from part_ch2 import build_ch2, build_appendix_ch2
from part_ch3 import build_ch3, build_appendix_ch3

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "casper_supervisor_2026-09.pptx")


import re
LABEL = re.compile(r"\s*\((?:[^()]*(?:chapter[ABC]_v2|paperC_lbr|tab:|fig:|sec:)[^()]*)\)")
GLYPHS = {"∨": "or", "★": "*", "₁": "1", "₀": "0", "≫": "far above"}
MATHY = re.compile(r"^[A-Za-z]$|[=(]|^cos|^(p|n|k|r)$|^\|")


def polish(prs):
    """House style post-pass: strip internal LaTeX labels; capitalise bullet starts."""
    for s in prs.slides:
        for sh in s.shapes:
            if getattr(sh, "has_table", False) and sh.has_table:
                for row in sh.table.rows:
                    for c in row.cells:
                        for p in c.text_frame.paragraphs:
                            for r in p.runs:
                                for g, rep in GLYPHS.items():
                                    if g in r.text:
                                        r.text = r.text.replace(g, rep)
                continue
            if not sh.has_text_frame:
                continue
            is_body = sh.is_placeholder and sh.placeholder_format.type == 2
            for p in sh.text_frame.paragraphs:
                runs = p.runs
                if not runs:
                    continue
                for r in runs:
                    if LABEL.search(r.text):
                        r.text = LABEL.sub("", r.text)
                    for g, rep in GLYPHS.items():
                        if g in r.text:
                            r.text = r.text.replace(g, rep)
                has_bullet = p._p.pPr is not None and p._p.pPr.find(
                    "{http://schemas.openxmlformats.org/drawingml/2006/main}buChar") is not None
                if is_body or has_bullet:
                    t = runs[0].text
                    first = t.split()[0] if t.split() else ""
                    if t and t[0].islower() and not MATHY.search(first):
                        runs[0].text = t[0].upper() + t[1:]


def main():
    prs = new_deck()
    build_intro(prs)
    build_ch1(prs)
    build_ch2(prs)
    build_ch3(prs)
    build_close(prs)
    build_appendix_methods(prs)
    build_appendix_ch1(prs)
    build_appendix_ch2(prs)
    build_appendix_ch3(prs)
    polish(prs)
    prs.save(OUT)
    print("saved", OUT, "slides:", len(prs.slides))


if __name__ == "__main__":
    main()

"""Pixel-based digitiser for Figure 2 of Hu & Yu, RecSys 2013.

Their results are FIGURE ONLY (bar charts, no scalar tables).  This script renders
page 4 (index 3) of the PDF at high resolution, locates each subplot's axis frame,
calibrates pixel-row -> value from the y tick LABEL centres (whose values are known
from the axis), and reads every bar's top pixel row.

Subplots: (a) MAP MovieLens, (b) MAP Netflix, (c) MAP Yahoo! Music, (d) RMSE MovieLens.
Series (MATLAB bar-group order == legend order):
    (a)(b)(c): FixQ (black), EIP (mid brown), Warm-MF (pale orange)
    (d):       FixQ (black), fMF, EIP, Warm-MF (pale orange)   [4 series]

Usage:  python scripts/baselines/huyu_figure_digitise.py --res 400
Writes: experiments/baselines/huyu2013/figure2_digitised.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image

ROOT = r"C:\dev\phd\casper"
PDF = os.path.join(ROOT, "external_literature", "papers",
                   "hu_yu_recsys2013_interview_process_learning.pdf")
OUT = os.path.join(ROOT, "experiments", "baselines", "huyu2013")

# Axis frames (page pixels at 400 dpi) and the y-tick-label values, top->bottom.
# Frames are auto-detected; the values come from reading the printed tick labels.
PANELS = {
    "a": dict(title="MAP on MovieLens", quad=(290, 730, 460, 980),
              ticks=[0.82, 0.80, 0.78, 0.76, 0.74, 0.72, 0.70],
              xlabels=[3, 5, 8, 10], series=["FixQ", "EIP", "Warm-MF"]),
    "b": dict(title="MAP on Netflix", quad=(290, 730, 1030, 1550),
              ticks=[0.84, 0.82, 0.80, 0.78, 0.76, 0.74, 0.72],
              xlabels=[3, 5, 8, 10], series=["FixQ", "EIP", "Warm-MF"]),
    "c": dict(title="MAP on Yahoo! Music", quad=(850, 1290, 460, 980),
              ticks=[0.92, 0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78],
              xlabels=[3, 4, 5, 6], series=["FixQ", "EIP", "Warm-MF"]),
    "d": dict(title="RMSE on MovieLens", quad=(850, 1290, 1030, 1550),
              ticks=[1.05, 1.00, 0.95, 0.90, 0.85, 0.80],
              xlabels=[3, 5, 8, 10], series=["FixQ", "fMF", "EIP", "Warm-MF"]),
}


def render(res: int) -> str:
    import pdfplumber
    os.makedirs(OUT, exist_ok=True)
    png = os.path.join(OUT, f"page4_{res}.png")
    if not os.path.exists(png):
        with pdfplumber.open(PDF) as pdf:
            pdf.pages[3].to_image(resolution=res).save(png)
    return png


def find_frame(dark: np.ndarray, quad):
    y0, y1, x0, x1 = quad
    sub = dark[y0:y1, x0:x1]
    rows = np.where(sub.sum(axis=1) > 0.9 * (x1 - x0))[0] + y0
    cols = np.where(sub.sum(axis=0) > 0.9 * (y1 - y0))[0] + x0
    yt = int(rows.min())
    xl, xr = int(cols.min()), int(cols.max())
    # Bottom frame: bar fills are painted over parts of it, so it is only ~2/3 dark.
    sub2 = dark[yt + 5:y1, xl:xr + 1]
    r2 = np.where(sub2.sum(axis=1) > 0.5 * (xr - xl))[0] + yt + 5
    yb = int(r2.max())
    return yt, yb, xl, xr


def tick_rows(grey: np.ndarray, yt, yb, xl, n_expected):
    """Centres of the y-tick label text groups, left of the axis frame."""
    pad = max(6, int(round(15 * (yb - yt) / 413.0)))
    wide = max(30, int(round(75 * (yb - yt) / 413.0)))
    strip = grey[yt - pad:yb + pad + 1, max(0, xl - wide):xl - max(2, pad // 4)]
    h = strip.sum(axis=1)
    rows = np.where(h > 0)[0] + yt - pad
    groups, cur = [], [rows[0]]
    for r in rows[1:]:
        if r - cur[-1] <= 4:
            cur.append(r)
        else:
            groups.append(cur)
            cur = [r]
    groups.append(cur)
    cent = [(float(np.mean(g)), len(g)) for g in groups]
    return cent


def calibrate(cent, ticks):
    """Fit row -> value from the detected label centres.

    One group may be merged with the rotated y-axis label (it is wider in rows);
    we fit a robust line on the clean groups and then check the residuals.
    """
    import numpy as _np
    sizes = _np.array([c[1] for c in cent])
    thr = 1.6 * _np.median(sizes)
    clean = [c for c in cent if c[1] <= thr]
    # assign clean centres to tick values by nearest-rank: the clean centres are a
    # subsequence of the full evenly spaced ladder; solve by least squares over the
    # best integer offsets derived from spacing.
    ys = np.array([c[0] for c in clean])
    d = np.diff(ys)
    step = float(np.median(d[d > 0.5 * np.median(d)])) if len(d) else 1.0
    idx = np.round((ys - ys[0]) / step).astype(int)
    vals = np.array(ticks)
    # values are top->bottom evenly spaced
    vstep = vals[1] - vals[0]
    v = vals[0] + idx * vstep
    A = np.vstack([ys, np.ones_like(ys)]).T
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    pred = A @ coef
    resid = float(np.max(np.abs(pred - v)))
    return coef, resid, step


def read_bars(a: np.ndarray, yt, yb, xl, xr, series, tol=26):
    """Return, per bar-group, the top pixel row of each series' bar."""
    interior = a[yt + 1:yb, xl + 1:xr]
    H, W, _ = interior.shape
    isdark = interior.max(axis=2) < 120
    sat = (interior.max(axis=2).astype(int) - interior.min(axis=2).astype(int)) > 25
    # colour of each saturated column = its median fill
    colcls = np.full(W, -1, dtype=int)
    # gather the distinct fills present
    px = interior[sat]
    fills = []
    if len(px):
        # quantise and take the frequent ones
        q = (px // 8) * 8
        uq, cnt = np.unique(q.reshape(-1, 3), axis=0, return_counts=True)
        order = np.argsort(-cnt)
        for i in order:
            c = uq[i]
            if cnt[i] < 200:
                break
            if all(np.abs(np.array(f, int) - c.astype(int)).max() > 20 for f in fills):
                fills.append(tuple(int(x) for x in c))
    # order fills dark -> pale (fMF/EIP/Warm-MF order in the legend)
    fills.sort(key=lambda c: sum(c))
    info = {"fills": fills}
    kind = np.full(W, "", dtype=object)
    top = np.full(W, -1)
    for x in range(W):
        col = interior[:, x]
        nblack = int(isdark[:, x].sum())
        best, bestn = None, 0
        for f in fills:
            n = int((np.abs(col.astype(int) - np.array(f)).max(axis=1) < tol).sum())
            if n > bestn:
                best, bestn = f, n
        # A bar is ANCHORED at the axis: we take the CONTIGUOUS run that reaches the
        # bottom of the interior.  This rejects the floating legend patches, which
        # can sit in the very same pixel columns as the right-most bar group.
        def anchored_top(mask):
            if not mask[-3:].any():
                return None
            y = H - 1
            while y >= 0 and not mask[y]:
                y -= 1
            while y >= 0 and mask[max(0, y - 2):y + 1].any():
                y -= 1
            return y + 1

        if bestn > 20:
            m = np.abs(col.astype(int) - np.array(best)).max(axis=1) < tol
            t = anchored_top(m)
            if t is not None:
                kind[x] = str(fills.index(best))
                top[x] = t
        elif nblack > 20:
            t = anchored_top(isdark[:, x])
            if t is not None:
                kind[x] = "black"
                top[x] = t
    # group contiguous columns of identical kind
    bars = []
    x = 0
    while x < W:
        if kind[x] == "":
            x += 1
            continue
        x2 = x
        while x2 + 1 < W and kind[x2 + 1] == kind[x]:
            x2 += 1
        if x2 - x >= 4:
            cols = np.arange(x, x2 + 1)
            core = cols[(cols > x + 1) & (cols < x2 - 1)]
            if len(core) == 0:
                core = cols
            t = float(np.median(top[core]))
            bars.append(dict(kind=kind[x], x0=int(x + xl + 1), x1=int(x2 + xl + 1),
                             top_row=t + yt + 1))
        x = x2 + 1
    info["bars"] = bars
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=400)
    args = ap.parse_args()
    png = render(args.res)
    a = np.asarray(Image.open(png).convert("RGB")).astype(int)
    dark = a.max(axis=2) < 120
    grey = a.max(axis=2) < 200

    out = {"resolution_dpi": args.res, "source_pdf": PDF, "panels": {}}
    scale = args.res / 400.0
    for key, P in PANELS.items():
        quad = tuple(int(round(q * scale)) for q in P["quad"])
        yt, yb, xl, xr = find_frame(dark, quad)
        cent = tick_rows(grey, yt, yb, xl, len(P["ticks"]))
        coef, resid, step = calibrate(cent, P["ticks"])
        val = lambda r: float(coef[0] * r + coef[1])
        px_per_unit = abs(1.0 / coef[0])
        info = read_bars(a, yt, yb, xl, xr, P["series"])
        bars = info["bars"]
        nser = len(P["series"])
        ngrp = len(P["xlabels"])
        rec = {"title": P["title"], "frame": [yt, yb, xl, xr],
               "tick_label_rows": cent, "calib_max_resid": resid,
               "px_per_unit": px_per_unit,
               "value_per_pixel": 1.0 / px_per_unit,
               "fills_dark_to_pale": info["fills"],
               "n_bars_found": len(bars), "bars_raw": [], "values": {}}
        for b in bars:
            rec["bars_raw"].append({**b, "value": round(val(b["top_row"]), 5)})
        if len(bars) == nser * ngrp:
            for gi in range(ngrp):
                grp = bars[gi * nser:(gi + 1) * nser]
                for si, s in enumerate(P["series"]):
                    rec["values"].setdefault(s, {})[str(P["xlabels"][gi])] = \
                        round(val(grp[si]["top_row"]), 4)
        out["panels"][key] = rec
        print(f"--- ({key}) {P['title']}  frame={yt,yb,xl,xr} "
              f"resid={resid:.5f} 1px={1.0/px_per_unit:.6f} bars={len(bars)} "
              f"fills={info['fills']}")
        for s in P["series"]:
            if s in rec["values"]:
                print(f"    {s:9s} " + "  ".join(
                    f"{k}:{v:.4f}" for k, v in rec["values"][s].items()))
    dst = os.path.join(OUT, f"figure2_digitised_{args.res}.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", dst)


if __name__ == "__main__":
    main()

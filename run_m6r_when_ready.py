#!/usr/bin/env python
"""Wait for the cold-on-train fit, then run the fair comparison. Nothing else touches the box."""
import json, os, subprocess, sys, time
R = r"C:\dev\phd\casper"; PY = r"C:\Program Files\Python312\python.exe"
LOG = os.path.join(R, "experiments", "battery")
ART = os.path.join(LOG, "coldtree_v2_train10000.json")
os.chdir(R)
for _ in range(2880):                                     # up to 24 h
    f = os.path.join(LOG, "coldtree_train10k_4t.out")
    if os.path.exists(f) and "[done]" in open(f, encoding="utf-8", errors="replace").read():
        break
    time.sleep(30)
else:
    sys.exit("[pipe] cold-on-train never finished -- ABORT")
d = json.load(open(ART, encoding="utf-8"))
print(f"[pipe] cold_train done: {d['n_nodes']} nodes, depth {d['depth_reached']}, "
      f"{d['seconds']/3600:.1f} h (this session)", flush=True)
print(f"[pipe] m6r START {time.strftime('%H:%M:%S')}", flush=True)
with open(os.path.join(LOG, "m6r_fair.out"), "w") as fh:
    rc = subprocess.call([PY, "src/instrument/m6r_fair_compare.py", "--i_have_author_goahead",
                          "--threads", "6"], stdout=fh, stderr=subprocess.STDOUT,
                         env=dict(os.environ))
print(f"[pipe] m6r rc={rc} {time.strftime('%H:%M:%S')}", flush=True)

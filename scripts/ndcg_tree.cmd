@echo off
REM NDCG-objective interview tree, STAGE 1 (build on the 10,000 canonical VALIDATION users).
REM Design sheet: docs/design/DS_NDCG_TREE.md. Label `ndcgtree` -- NOT `ndcgtree_d3`: the depth cap
REM was dropped on the author's redirect (2026-08-12), the min-leaf rule is the only stopping rule
REM and the realised depth is a RESULT, so a `d3` in the name would be actively misleading.
REM
REM --wall_hours 0 = NOTHING IS CUT. Growth is level-synchronous, so the partial tree on disk after
REM every completed level is exactly the tree a shallower request would have produced: this run can
REM be stopped at any point and still yields a complete, valid artifact, and it resumes
REM deterministically from its per-node CELF-heap sidecars.
REM
REM THREADS: the BLAS env vars must be exported HERE, before python starts. numpy is imported before
REM strategy_channel_suite's own os.environ lines run, so setting them in-process is the no-op
REM recorded on 2026-08-05. (Measured 2026-08-12: 8 threads buys nothing, 8.49s vs 8.51s per eval --
REM the bottleneck is not BLAS. Set explicitly anyway so the run is not at the mercy of a default.)
REM
REM The poetry venv on this box is EMPTY (no numpy); use the system 3.12.
cd /d C:\dev\phd\casper
set OMP_NUM_THREADS=8
set OPENBLAS_NUM_THREADS=8
set MKL_NUM_THREADS=8
"C:\Program Files\Python312\python.exe" src\instrument\ndcg_tree.py --label ndcgtree --wall_hours 0 --full_threads 1>> experiments\battery\ndcgtree.log 2>&1
echo [cmd] exit code %ERRORLEVEL% >> experiments\battery\ndcgtree.log

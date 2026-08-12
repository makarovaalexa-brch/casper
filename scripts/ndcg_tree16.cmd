@echo off
REM Budget 15 -> 16 + a POLICY-B static combined comparator. Label `ndcgtree16`,
REM comparator artifact `greedy_static_policyB_combined`. See src/instrument/ndcg_tree_extend16.py.
REM The tree is NOT regrown: every search resumes a banked CELF heap (copied to this job's own
REM paths -- HARD RULE 11(e)). TEST is not built, not loaded, not touched.
cd /d C:\dev\phd\casper
set OMP_NUM_THREADS=8
set OPENBLAS_NUM_THREADS=8
set MKL_NUM_THREADS=8
"C:\Program Files\Python312\python.exe" src\instrument\ndcg_tree_extend16.py --full_threads 1>> experiments\battery\ndcgtree16.log 2>&1
echo [cmd] exit code %ERRORLEVEL% >> experiments\battery\ndcgtree16.log

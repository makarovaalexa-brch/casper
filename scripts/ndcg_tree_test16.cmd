@echo off
REM THE TEST LOOK (author-authorised, ONE look). Label `ndcgtree16test`.
REM Extends the 15 leaf tails 15->16 from banked CELF heaps, re-measures validation, then routes the
REM 10,000 canonical TEST users through the FROZEN tree. The tree is NOT regrown.
REM Comparator = the POLICY-A-SELECTED combined order, labelled as such in every output.
cd /d C:\dev\phd\casper
set OMP_NUM_THREADS=8
set OPENBLAS_NUM_THREADS=8
set MKL_NUM_THREADS=8
"C:\Program Files\Python312\python.exe" src\instrument\ndcg_tree_test16.py --full_threads 1>> experiments\battery\ndcgtree16test.log 2>&1
echo [cmd] exit code %ERRORLEVEL% >> experiments\battery\ndcgtree16test.log

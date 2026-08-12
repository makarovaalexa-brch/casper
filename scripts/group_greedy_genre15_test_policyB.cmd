@echo off
REM A3 downstream -- genre split STAGE 2 RE-RUN against the REBUILT policy-B items-only comparator.
REM Stage-2 only: the group orders and the grouping rule come from groupgreedy_genre15.json and are
REM NOT re-searched. Own label, so the pre-registered artifact is untouched.
cd /d C:\dev\phd\casper
set OMP_NUM_THREADS=8
set OPENBLAS_NUM_THREADS=8
set MKL_NUM_THREADS=8
"C:\Program Files\Python312\python.exe" src\instrument\group_greedy_genre_test.py --global_from C:\dev\phd\casper\experiments\battery\greedy_static_policyB_items.json 1>> experiments\battery\groupgreedy_genre15_TEST_policyB.log 2>&1
echo [cmd] exit code %ERRORLEVEL% >> experiments\battery\groupgreedy_genre15_TEST_policyB.log

@echo off
REM Genre-split group-conditioned greedy, STAGE 2 -- ONE-SHOT TEST EVALUATION.
REM Plan pre-specified and committed BEFORE any test number existed: see
REM docs/design/DS_GROUP_GREEDY_GENRE.md section "Stage 2". Primary endpoint K=15 (pre-data change,
REM cost only). Two arms: ALL-GROUPS and FALLBACK-200. Paired percentile bootstrap, 10,000
REM resamples, seed 98765. ONE LOOK -- do not re-run with a variant.
cd /d C:\dev\phd\casper
"C:\Program Files\Python312\python.exe" src\instrument\group_greedy_genre_test.py 1>> experiments\battery\groupgreedy_genre15_TEST.log 2>&1
echo [cmd] exit code %ERRORLEVEL% >> experiments\battery\groupgreedy_genre15_TEST.log

@echo off
REM A4 -- rebuild the CONCEPTS-ONLY static greedy order under policy B, 16 picks, full 500-concept
REM answerable bank, all 10,000 canonical validation users.
cd /d C:\dev\phd\casper
set OMP_NUM_THREADS=8
set OPENBLAS_NUM_THREADS=8
set MKL_NUM_THREADS=8
"C:\Program Files\Python312\python.exe" src\instrument\greedy_static_policyB_channels.py --channel concepts --full_threads 1>> experiments\battery\greedy_static_policyB_concepts.log 2>&1
echo [cmd] exit code %ERRORLEVEL% >> experiments\battery\greedy_static_policyB_concepts.log

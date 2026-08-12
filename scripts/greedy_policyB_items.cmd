@echo off
REM A3 -- rebuild the ITEMS-ONLY static greedy order under policy B, 16 picks, full 500-item bank,
REM all 10,000 canonical validation users. See src/instrument/greedy_static_policyB_channels.py.
cd /d C:\dev\phd\casper
set OMP_NUM_THREADS=8
set OPENBLAS_NUM_THREADS=8
set MKL_NUM_THREADS=8
"C:\Program Files\Python312\python.exe" src\instrument\greedy_static_policyB_channels.py --channel items --full_threads 1>> experiments\battery\greedy_static_policyB_items.log 2>&1
echo [cmd] exit code %ERRORLEVEL% >> experiments\battery\greedy_static_policyB_items.log

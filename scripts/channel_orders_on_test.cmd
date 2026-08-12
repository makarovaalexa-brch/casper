@echo off
REM A3/A4 downstream -- the three policy-B static channel orders on the 10,000 canonical TEST users:
REM bank-vs-conditioning decomposition at equal budget + Act 4's channel comparison.
cd /d C:\dev\phd\casper
set OMP_NUM_THREADS=8
set OPENBLAS_NUM_THREADS=8
set MKL_NUM_THREADS=8
"C:\Program Files\Python312\python.exe" src\instrument\channel_orders_on_test.py --full_threads 1>> experiments\battery\channel_orders_policyB_test.log 2>&1
echo [cmd] exit code %ERRORLEVEL% >> experiments\battery\channel_orders_policyB_test.log

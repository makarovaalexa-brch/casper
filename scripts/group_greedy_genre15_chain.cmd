@echo off
REM Waits for STAGE 1 (groupgreedy_genre15.json) to land, then runs STAGE 2 -- the one-shot test
REM evaluation -- in the SAME process tree. Scheduled-task driven so the chain survives any session
REM ending. Stage 2 refuses to start unless stage 1 is citable and every group finished; it also
REM re-asserts the tower/concept pair against the stage-1 build.
cd /d C:\dev\phd\casper
:wait
if exist experiments\battery\groupgreedy_genre15.json goto go
timeout /t 20 /nobreak >nul
goto wait
:go
echo [chain] stage 1 artifact present; starting stage 2 >> experiments\battery\groupgreedy_genre15_TEST.log
"C:\Program Files\Python312\python.exe" src\instrument\group_greedy_genre_test.py 1>> experiments\battery\groupgreedy_genre15_TEST.log 2>&1
echo [cmd] stage2 exit code %ERRORLEVEL% >> experiments\battery\groupgreedy_genre15_TEST.log

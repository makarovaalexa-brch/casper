@echo off
REM Keep the Act-2 criterion sweep alive.
REM
REM WHY THIS EXISTS. `scripts/act2_pv_quartiles.py --score` was silently killed three times on
REM 2026-08-10 (15:27, 19:02, 19:33) with no traceback and no stderr, at 4.3 h, 3.0 h and 0.1 h after
REM launch, with 4-6 GB free each time -- so it is not the job running out of memory, it is the
REM process being reaped with the session that spawned it. The sweep is a ~30 h job, so it has to
REM outlive whatever is doing the reaping.
REM
REM Restarting is SAFE and idempotent: the state list is append-only, the criterion arrays are
REM contract-signed memmaps flushed every 25 states, and a resume refuses unless the banked state
REM list is a prefix of the run's. A restart therefore costs at most 25 states of work and can never
REM mix two contracts. When there is nothing left to do the script logs "nothing to do" and exits 0,
REM which is the loop's stop condition.
REM
REM Launch it OUT OF THE CALLER'S JOB OBJECT, or it inherits the same fate:
REM   Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
REM       CommandLine='cmd /c C:\dev\phd\casper\scripts\act2_pv_score_daemon.cmd';
REM       CurrentDirectory='C:\dev\phd\casper' }
cd /d C:\dev\phd\casper
:loop
"C:\Program Files\Python312\python.exe" scripts\act2_pv_quartiles.py --score --no_preflight >> experiments\baselines\interview\act2pv_score_daemon.out 2>> experiments\baselines\interview\act2pv_score_daemon.err
findstr /C:"nothing to do" experiments\baselines\interview\act2pv_score_daemon.out >nul && goto done
echo [daemon] respawning after exit code %ERRORLEVEL% >> experiments\baselines\interview\act2pv_score_daemon.out
timeout /t 10 /nobreak >nul
goto loop
:done
echo [daemon] sweep complete >> experiments\baselines\interview\act2pv_score_daemon.out

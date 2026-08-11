@echo off
REM Genre-split group-conditioned greedy (label groupgreedy_genre). See docs/design/DS_GROUP_GREEDY_GENRE.md.
REM
REM Driven by a Windows SCHEDULED TASK, not a background launcher: on this box every backgrounded
REM launch of a long job has been killed by a console interrupt delivered to the process group (the
REM .err file holds exactly two bytes, ^C). A scheduled task is the only thing that has survived.
REM
REM   schtasks /Create /TN casper_groupgreedy_genre /SC ONCE /ST <hh:mm> /TR "cmd /c C:\dev\phd\casper\scripts\group_greedy_genre.cmd"
REM   Set-ScheduledTask -TaskName casper_groupgreedy_genre -Settings (New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero))
REM       ^^ schtasks /Create defaults to DisallowStartIfOnBatteries=$true; without this the task sits
REM          QUEUED forever while reporting Last Result 0. Hit on 2026-08-10.
REM   schtasks /Run    /TN casper_groupgreedy_genre
REM   schtasks /Delete /TN casper_groupgreedy_genre /F
REM
REM The poetry venv on this box is EMPTY (no numpy); every launcher in scripts/ uses the system 3.12.
REM Restart is safe: groupgreedy_genre_partial.json is contract-signed and rewritten after every pick;
REM a completed group is skipped on resume. Single run, no respawn loop -- one heavy job at a time.
cd /d C:\dev\phd\casper
"C:\Program Files\Python312\python.exe" src\instrument\group_greedy_genre.py %* 1>> experiments\battery\groupgreedy_genre.log 2>&1
echo [cmd] exit code %ERRORLEVEL% >> experiments\battery\groupgreedy_genre.log

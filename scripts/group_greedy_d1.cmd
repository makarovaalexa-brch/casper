@echo off
REM Depth-1 group-conditioned greedy (label groupgreedy_d1). See docs/design/DS_GROUP_GREEDY_D1.md.
REM
REM Driven by a Windows SCHEDULED TASK, not a background launcher: on this box every backgrounded
REM launch of a long job has been killed by a console interrupt delivered to the process group (the
REM .err file holds exactly two bytes, ^C). A scheduled task is the only thing that has survived.
REM
REM   schtasks /Create /TN casper_groupgreedy_d1 /SC ONCE /ST <hh:mm> /TR "cmd /c C:\dev\phd\casper\scripts\group_greedy_d1.cmd"
REM   schtasks /Run    /TN casper_groupgreedy_d1
REM   schtasks /Delete /TN casper_groupgreedy_d1 /F
REM
REM Restart is safe: groupgreedy_d1_partial.json is contract-signed and rewritten after every pick;
REM a completed group is skipped on resume. Single run, no respawn loop -- one heavy job at a time.
cd /d C:\dev\phd\casper
"C:\Program Files\Python312\python.exe" src\instrument\group_greedy_d1.py %* 1>> experiments\battery\groupgreedy_d1.log 2>&1
echo [cmd] exit code %ERRORLEVEL% >> experiments\battery\groupgreedy_d1.log

@echo off
REM Genre-split group-conditioned greedy, STAGE 1 at L=15 (label groupgreedy_genre15).
REM See docs/design/DS_GROUP_GREEDY_GENRE.md. Reuses the four completed orders banked by the
REM abandoned 15-step run, but ONLY after asserting the contract field by field; searches the 14
REM missing groups. The poetry venv on this box is EMPTY (no numpy); use the system 3.12.
cd /d C:\dev\phd\casper
"C:\Program Files\Python312\python.exe" src\instrument\group_greedy_genre.py --label groupgreedy_genre15 --L 15 --import_banked groupgreedy_genre_ABANDONED_15step_partial.json 1>> experiments\battery\groupgreedy_genre15.log 2>&1
echo [cmd] exit code %ERRORLEVEL% >> experiments\battery\groupgreedy_genre15.log

r"""Power-state stamp for long runs on this box.

WHY THIS EXISTS. Twice a multi-hour fit has been destroyed by the laptop running out of battery,
not by anything in the code:

  2026-08-21 04:08  Critical Battery Trigger Met -> the arthouse fit crawled for   4.1 h
  2026-08-22 11:59  Critical Battery Trigger Met -> the cold-on-train fit froze for 53.5 h

Neither was a crash and neither lost work -- the partials and CELF heaps resumed cleanly -- but
between them they cost over two days of wall-clock on jobs whose whole cost is wall-clock. The AC
sleep policy is already `never`; the failure mode is simply that the machine was unplugged, and
nothing in the launch path noticed or said so.

`require_mains()` costs a few milliseconds at startup and STAMPS THE POWER STATE into the run's own
log. That stamp is what identified the 53.5 h freeze as a dead battery rather than a bug in the
fitter, which is the whole value here.

IT DOES NOT BLOCK -- author's ruling, 2026-08-25. It briefly refused to start on battery, and on
2026-08-24 that refusal killed an overnight distillation outright. That is a worse outcome than the
thing it was guarding against: a frozen run resumes cleanly from its partial and its CELF heaps and
loses only wall-clock, whereas a run that never starts loses the whole night. Never convert a slow
run into no run.
"""
from __future__ import annotations

import subprocess


def power_state():
    """(on_mains, percent, detail). `on_mains` is None when it cannot be determined."""
    ps = (
        "$b = Get-CimInstance -Namespace root\\wmi -ClassName BatteryStatus "
        "-ErrorAction SilentlyContinue | Select-Object -First 1; "
        "$w = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue | Select-Object -First 1; "
        "if ($null -eq $b -and $null -eq $w) { 'NOBATTERY' } "
        "else { $online = if ($null -ne $b) { [int]$b.PowerOnline } else { -1 }; "
        "$pct = if ($null -ne $w) { [int]$w.EstimatedChargeRemaining } else { -1 }; "
        "$st = if ($null -ne $w) { [int]$w.BatteryStatus } else { -1 }; "
        "\"$online|$pct|$st\" }"
    )
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                             capture_output=True, text=True, timeout=25).stdout.strip()
    except Exception as e:                                    # a probe failure is not a power failure
        return None, -1, f"probe failed ({type(e).__name__})"
    if not out:
        return None, -1, "probe returned nothing"
    if out == "NOBATTERY":
        return True, -1, "no battery present (desktop/VM) -- treated as mains"
    try:
        online, pct, st = (int(x) for x in out.split("|"))
    except ValueError:
        return None, -1, f"unparsable probe output {out!r}"
    # root\wmi BatteryStatus.PowerOnline is the reliable signal; Win32_Battery.BatteryStatus == 2
    # ("Unknown" in the MOF, in practice AC-connected) is the fallback when PowerOnline is absent.
    if online >= 0:
        return bool(online), pct, f"PowerOnline={online} charge={pct}% Win32_BatteryStatus={st}"
    return (st == 2), pct, f"Win32_BatteryStatus={st} charge={pct}% (PowerOnline unavailable)"


def require_mains(context, log=print, allow_battery=True, min_percent=95):
    """Record the power state. NEVER blocks -- author's ruling, 2026-08-25.

    This started life as a refusal and that was wrong: on 2026-08-24 it blocked an overnight
    distillation outright and cost a night, which is a worse outcome than the freeze it was guarding
    against. The freezes were survivable (partials and CELF heaps resume cleanly); a run that never
    starts is not. A guard that turns a slow run into no run is not a guard.

    What is kept is the DIAGNOSTIC, which cost nothing and was the genuinely useful half: stamping
    the power state into the run's own log is what identified the 53.5 h freeze as a dead battery
    rather than a bug in the fitter. `allow_battery` and `min_percent` are retained so existing call
    sites keep working; neither can cause a refusal.
    """
    on_mains, pct, detail = power_state()
    if on_mains is None:
        log(f"[power] {context}: power state UNKNOWN ({detail})")
    elif on_mains:
        log(f"[power] {context}: ON MAINS ({detail})")
    else:
        log(f"[power] {context}: ON BATTERY ({detail}) -- proceeding. Note: critical-battery sleep "
            f"froze runs for 4.1 h and 53.5 h (2026-08-21/22). Work resumes cleanly; wall-clock "
            f"does not.")
    return on_mains, pct


if __name__ == "__main__":
    print(power_state())
    require_mains("self-test")

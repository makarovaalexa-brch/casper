r"""Pre-flight guard for long runs on this box.

WHY THIS EXISTS. Twice a multi-hour fit has been destroyed by the laptop running out of battery,
not by anything in the code:

  2026-08-21 04:08  Critical Battery Trigger Met -> the arthouse fit crawled for   4.1 h
  2026-08-22 11:59  Critical Battery Trigger Met -> the cold-on-train fit froze for 53.5 h

Neither was a crash and neither lost work -- the partials and CELF heaps resumed cleanly -- but
between them they cost over two days of wall-clock on jobs whose whole cost is wall-clock. The AC
sleep policy is already `never`; the failure mode is simply that the machine was unplugged, and
nothing in the launch path noticed or said so.

`require_mains()` is a few milliseconds at startup. It refuses to begin a long run on battery unless
the caller explicitly opts in, and it prints the charge level either way so the log records the
power state the run started under.

DELIBERATELY NOT FAIL-CLOSED ON A FAILED PROBE. A desktop, a VM, a CI box or a machine whose WMI
provider is unavailable has no battery to report; blocking work because a probe failed would be a
worse bug than the one this prevents. Unknown power state => proceed, and say so.
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


def require_mains(context, log=print, allow_battery=False, min_percent=95):
    """Refuse to start a long run on battery. Returns the (on_mains, percent) it observed."""
    on_mains, pct, detail = power_state()
    if on_mains is None:
        log(f"[power] {context}: power state UNKNOWN ({detail}) -- proceeding, unguarded")
        return None, pct
    if on_mains:
        log(f"[power] {context}: ON MAINS ({detail})")
        return True, pct
    msg = (f"REFUSING TO START {context} ON BATTERY ({detail}). This box has lost 4.1 h and 53.5 h "
           f"of fitting to critical-battery sleep (2026-08-21, 2026-08-22). Plug it in, or pass "
           f"--allow_battery if you genuinely intend a run that may be suspended mid-node.")
    if not allow_battery:
        raise SystemExit(f"[power] {msg}")
    log(f"[power] {context}: ON BATTERY ({detail}) -- --allow_battery given, proceeding anyway")
    if 0 <= pct < min_percent:
        log(f"[power] WARNING: charge is {pct}%, below {min_percent}% -- a multi-hour run will not "
            f"survive this. Work is resumable (partials + CELF heaps), but wall-clock is not.")
    return False, pct


if __name__ == "__main__":
    print(power_state())
    require_mains("self-test")

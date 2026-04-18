"""Time zone policy for the engine.

All timestamps flowing through the engine are EST (UTC-5, fixed offset,
no DST). Rationale: financial logs need one unambiguous wall-clock
representation; a fixed offset avoids the twice-yearly discontinuity of
``America/New_York``. If a desk later wants DST-aware Eastern time,
swap ``EST`` for ``ZoneInfo("America/New_York")`` here — no other file
needs to change.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

EST: timezone = timezone(timedelta(hours=-5), name="EST")


def now_est() -> datetime:
    """Current wall-clock in EST."""
    return datetime.now(EST)


def require_est(v: datetime) -> datetime:
    """Reject anything that isn't a fixed-offset EST datetime."""
    if v.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    offset = v.utcoffset()
    if offset is None or offset != timedelta(hours=-5):
        raise ValueError("datetime must be EST (offset -05:00)")
    return v

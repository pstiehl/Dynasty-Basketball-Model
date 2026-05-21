"""Player name normalization for duplicate detection.

Different sources spell the same player differently:

    "T.J. McConnell" vs "TJ McConnell"
    "P.J. Tucker" vs "PJ Tucker"
    "Nicolas Claxton" vs "Nic Claxton"
    "Kelly Oubre" vs "Kelly Oubre Jr."
    "Wendell Carter" vs "Wendell Carter Jr."

We collapse all of those to one canonical key so sync resolution finds
the existing Player row instead of creating a duplicate.

Rules:
  1. Lowercase.
  2. Fold diacritics (e -> e).
  3. Strip generational suffixes: jr, sr, ii, iii, iv, v
     (with or without leading comma, with or without trailing period).
  4. Drop periods.
  5. Collapse internal whitespace.

Idempotent. Returns None for empty/None input.
"""
from __future__ import annotations
import re
import unicodedata


# Suffix matcher: either ", Jr" or " Jr", with optional trailing period.
# Anchored to end of string so we don't chop names like "Iverson".
_SUFFIX_RE = re.compile(
    r"(?:,\s*|\s+)(jr|sr|ii|iii|iv|v)\.?$",
    flags=re.IGNORECASE,
)

_WHITESPACE_RE = re.compile(r"\s+")


def normalize(name):
    """Suffix-stripped, lowercased, ascii-folded form. Idempotent."""
    if not name:
        return None

    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    s = s.strip().lower()

    # Strip suffixes repeatedly (e.g. "jr. iii" if anyone ever did that).
    while True:
        new = _SUFFIX_RE.sub("", s).strip()
        if new == s:
            break
        s = new

    s = s.replace(".", "")
    s = _WHITESPACE_RE.sub(" ", s).strip()

    return s or None

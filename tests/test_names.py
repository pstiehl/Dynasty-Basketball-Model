"""Tests for player name normalization.

Ported from Dynasty-Football-Model's tests/test_names.py with NBA-specific
example names.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dynasty_bball.names import normalize


def test_handles_none_and_empty():
    assert normalize(None) is None
    assert normalize("") is None
    assert normalize("   ") is None


def test_basic_lowercase_strip():
    assert normalize("LeBron James") == "lebron james"
    assert normalize("  Jayson Tatum  ") == "jayson tatum"


def test_strips_period_initials():
    assert normalize("T.J. McConnell") == "tj mcconnell"
    assert normalize("P.J. Tucker") == "pj tucker"


def test_strips_jr_sr_suffixes():
    assert normalize("Kelly Oubre Jr.") == "kelly oubre"
    assert normalize("Kelly Oubre, Jr.") == "kelly oubre"
    assert normalize("Wendell Carter Jr.") == "wendell carter"
    # Plain doesn't change.
    assert normalize("Kelly Oubre") == "kelly oubre"


def test_strips_roman_numeral_suffixes():
    assert normalize("Gary Trent II") == "gary trent"
    assert normalize("Tim Hardaway III") == "tim hardaway"


def test_does_not_strip_mid_name_numerals():
    # No suffix to strip — middle-of-name should be preserved.
    assert normalize("Iverson") == "iverson"


def test_folds_diacritics():
    assert normalize("Nikola Jokić") == "nikola jokic"
    assert normalize("Luka Dončić") == "luka doncic"


def test_idempotent():
    s = normalize("Kelly Oubre Jr.")
    assert normalize(s) == s

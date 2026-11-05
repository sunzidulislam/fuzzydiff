"""Tests of settings.py (python -I test_settings.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from settings import load_settings  # noqa: E402


def test_settings():
    s = load_settings()
    assert s.get("cfgCFG") == 7.0 and s.get("cfgSteps", int) == 28
    assert s.get("cfgEtaZero") == 0.05 and "cfgEtaZero" in s.provisional_used
    assert "cfgCFG" not in s.provisional_used
    try:
        load_settings(provisional=False).get("cfgEtaZero")
        assert False, "unset value must raise"
    except ValueError as e:
        assert "cfgEtaZero" in str(e)


def test_latex_wrappers_are_stripped():
    s = load_settings()
    assert s.get("cfgBackbonePlaceholder", default=3.0) == 3.0
    assert s.parse(r"\verify{0.25}") == 0.25 and s.parse(r"$0.5$") == 0.5


def test_blocks_and_tables():
    s = load_settings()
    assert s.get("cfgReadBlocks", list) == list(range(6, 19))
    assert s.get("hedge_centers", dict)["slightly"] == 0.15
    assert set(s.get("prototypes", dict)) >= {"red", "blue", "greenish-gray"}


def main():
    test_settings()
    test_latex_wrappers_are_stripped()
    test_blocks_and_tables()
    print("settings: pass")


if __name__ == "__main__":
    main()

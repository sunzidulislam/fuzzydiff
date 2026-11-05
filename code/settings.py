"""Typed access to the experiment settings.

configs/fuzzydiff_sd3.yaml is the single source for values reported in the paper; a value that is
still an unfilled placeholder ('[value]') comes from configs/provisional.yaml when provisional
settings are allowed, and every such use is recorded in `provisional_used`.
"""
from __future__ import annotations

import os
import re
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_config import ROOT, load  # noqa: E402

PROVISIONAL_PATH = os.path.join(ROOT, "configs", "provisional.yaml")
_MISSING = object()


class Settings:
    def __init__(self, base, provisional):
        self.base, self.provisional = base, provisional
        self.provisional_used = set()

    @staticmethod
    def parse(value, cast=float):
        s = str(value)
        for wrapper in (r"\\verify\{(.*)\}", r"\$(.*)\$"):
            m = re.fullmatch(wrapper, s.strip())
            if m:
                s = m.group(1)
        return cast(s.strip())

    def get(self, name, cast=float, default=_MISSING):
        value = self.base.get(name, _MISSING)
        if value is not _MISSING and "[" not in str(value):
            return self.parse(value, cast)
        if name in self.provisional:
            self.provisional_used.add(name)
            value = self.provisional[name]
            return value if cast in (list, dict) else self.parse(value, cast)
        if default is not _MISSING:
            return default
        raise ValueError(f"setting {name} has no value (placeholder: {value if value is not _MISSING else 'none'})")


def load_settings(provisional=True):
    extra = {}
    if provisional:
        with open(PROVISIONAL_PATH) as f:
            extra = yaml.safe_load(f)
    return Settings(load(), extra)

"""Measure every estimator the current program needs, on one clean prediction."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constraints import Evidence  # noqa: E402
from estimators import scene_lightness  # noqa: E402


def measure_scene(backend, x0, out, program, bin):
    """Scene-level evidence only (M1): no segmentation or attention."""
    return Evidence(x0=x0, scene={"scene_lightness": scene_lightness(x0)})

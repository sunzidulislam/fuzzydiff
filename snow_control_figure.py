"""Compatibility facade for fuzzydiff.experiments.snow.control_figure."""
import sys
import importlib
sys.modules[__name__] = importlib.import_module('fuzzydiff.experiments.snow.control_figure')

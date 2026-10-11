"""Compatibility facade for fuzzydiff.experiments.snow.study."""
import sys
import importlib
sys.modules[__name__] = importlib.import_module('fuzzydiff.experiments.snow.study')

"""Compatibility facade; implementation lives in fuzzydiff.experiments.comparison."""
import sys
import importlib
sys.modules[__name__] = importlib.import_module('fuzzydiff.experiments.comparison')

"""Compatibility facade for fuzzydiff.experiments.snow.gpu."""
import sys
import importlib
sys.modules[__name__] = importlib.import_module('fuzzydiff.experiments.snow.gpu')

"""Compatibility facade; implementation lives in fuzzydiff.reporting.timing."""
import sys
import importlib
sys.modules[__name__] = importlib.import_module('fuzzydiff.reporting.timing')

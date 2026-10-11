"""Compatibility facade for fuzzydiff.experiments.snow.ratings."""
if __name__ == '__main__':
    from fuzzydiff.cli.snow_ratings import main
    raise SystemExit(main())
else:
    import sys
    import importlib
    sys.modules[__name__] = importlib.import_module('fuzzydiff.experiments.snow.ratings')

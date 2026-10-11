"""Compatibility facade; implementation lives in fuzzydiff.experiments.benchmark."""
if __name__ == '__main__':
    import importlib
    from fuzzydiff import compatibility
    importlib.reload(compatibility)
    compatibility.reload_workflow('benchmark')
    from fuzzydiff.cli.benchmark import main
    raise SystemExit(main())
else:
    import sys
    import importlib
    sys.modules[__name__] = importlib.import_module('fuzzydiff.experiments.benchmark')

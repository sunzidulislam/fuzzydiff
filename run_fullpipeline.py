"""Compatibility facade; implementation lives in fuzzydiff.cli.full_pipeline."""
if __name__ == '__main__':
    import importlib
    from fuzzydiff import compatibility
    importlib.reload(compatibility)
    compatibility.reload_workflow('full')
    from fuzzydiff.cli.full_pipeline import main
    raise SystemExit(main())
else:
    import sys
    import importlib
    sys.modules[__name__] = importlib.import_module('fuzzydiff.cli.full_pipeline')

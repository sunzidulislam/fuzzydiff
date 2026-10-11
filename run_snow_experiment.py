"""Compatibility facade for fuzzydiff.cli.snow."""
if __name__ == '__main__':
    import importlib
    from fuzzydiff import compatibility
    importlib.reload(compatibility)
    compatibility.reload_workflow('snow')
    from fuzzydiff.cli.snow import main
    raise SystemExit(main())
else:
    import sys
    import importlib
    sys.modules[__name__] = importlib.import_module('fuzzydiff.cli.snow')

"""Compatibility entry point for SD 1.x generation."""
if __name__ == '__main__':
    import importlib
    from fuzzydiff import compatibility
    importlib.reload(compatibility)
    compatibility.reload_workflow('sd15')
    from fuzzydiff.cli.sd15 import main
    main()
else:
    import sys
    from fuzzydiff.models import sd15 as implementation
    sys.modules[__name__] = implementation

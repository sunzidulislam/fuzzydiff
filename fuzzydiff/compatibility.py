"""Refresh loaded package modules after git pull in a long-lived Kaggle kernel."""
import importlib
import sys


_COMMON = ('paths', 'notebook_runtime', 'models.registry', 'reporting.timing',
           'reporting.tables', 'models.session', 'models.sd15', 'models.archived',
           'models.sdxl', 'models.sdxl_turbo')
_WORKFLOWS = {
    'full': ('cli.full_pipeline', 'experiments.comparison'),
    'sd15': ('cli.full_pipeline', 'cli.sd15'),
    'benchmark': ('cli.benchmark', 'experiments.benchmark'),
    'generate': ('cli.full_pipeline', 'generation.requests', 'config',
                 'generation.runner', 'cli.generate'),
    'snow': ('experiments.snow.study', 'experiments.snow.gpu',
             'experiments.snow.artifacts', 'experiments.snow.ratings',
             'experiments.snow.control_source', 'experiments.snow.control',
             'experiments.snow.control_figure', 'cli.snow', 'cli.snow_control',
             'cli.snow_ratings'),
}


def reload_workflow(workflow: str) -> None:
    if workflow not in _WORKFLOWS:
        raise ValueError(f'Unknown workflow {workflow!r}.')
    # Reload only already-loaded modules: --help stays free of Torch imports.
    for name in _COMMON + _WORKFLOWS[workflow]:
        module = sys.modules.get('fuzzydiff.' + name)
        if module is not None:
            importlib.reload(module)

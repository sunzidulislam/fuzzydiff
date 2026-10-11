"""Explicit notebook loading policies; no model initialization or fuzzy mathematics."""
import ast
import hashlib
import json
import sys
import types
from itertools import count
from pathlib import Path
from types import CodeType
from typing import Iterable

from .paths import repository_root

_NAMESPACE_IDS = count()


def notebook_cells(path: Path) -> list[tuple[int, str]]:
    notebook = json.loads(Path(path).read_text(encoding='utf-8'))
    cells = []
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = cell['source']
        source = ''.join(source) if isinstance(source, list) else source
        source = '\n'.join(line for line in source.splitlines()
                           if not line.lstrip().startswith(('%pip ', '!pip ')))
        cells.append((index, source))
    return cells


def compile_notebook(path: Path) -> list[tuple[int, CodeType]]:
    return [(index, compile(source, f'{path}:cell-{index}', 'exec'))
            for index, source in notebook_cells(path)]


def load_prefix_definitions(path: Path, prefixes: tuple[str, ...], namespace: dict) -> dict:
    for _, source in notebook_cells(path):
        if source.startswith(prefixes):
            exec(compile(source, str(path), 'exec'), namespace)
    return namespace


def definition_code(path: Path) -> CodeType:
    body = []
    for _, source in notebook_cells(path):
        source = '\n'.join(line for line in source.splitlines()
                           if not line.lstrip().startswith(('!', '%')))
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                body.append(node)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                roots = ([name.name.split('.')[0] for name in node.names]
                         if isinstance(node, ast.Import) else [(node.module or '').split('.')[0]])
                if not set(roots) & {'clip', 'cv2', 'IPython', 'pandas'}:
                    body.append(node)
            elif isinstance(node, ast.Assign) and all(
                    isinstance(target, ast.Name) and target.id in
                    {'HEDGE_TARGETS', 'RELATION_VERBS', 'FUZZY_ATTRIBUTE_TARGETS'}
                    for target in node.targets):
                body.append(node)
    return compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                   str(path), 'exec')


def load_definitions(path: Path, compiler=None) -> dict:
    # Dataclasses resolves annotations through the defining module.
    name = '_benchmark_' + hashlib.sha256(str(path).encode()).hexdigest()[:12] + f'_{next(_NAMESPACE_IDS)}'
    module = types.ModuleType(name)
    sys.modules[name] = module
    try:
        exec((compiler or definition_code)(path), module.__dict__)
    except BaseException:
        release_definitions(module.__dict__)
        raise
    return module.__dict__


def release_definitions(namespace: dict) -> None:
    name = namespace.get('__name__', '')
    module = sys.modules.get(name)
    # Another owner may have registered the same name; release only this namespace.
    if name.startswith('_benchmark_') and module is not None and module.__dict__ is namespace:
        del sys.modules[name]
    namespace.clear()


def source_hashes(paths: Iterable[Path]) -> dict[str, str]:
    root = repository_root()
    return {Path(path).resolve().relative_to(root).as_posix():
            hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths}

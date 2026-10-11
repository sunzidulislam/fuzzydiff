"""Checkout paths, independent of caller working directory."""
from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def canonical_notebook() -> Path:
    return repository_root() / 'pipeline_fuzzy/fuzzydiff-fullpipeline.ipynb'

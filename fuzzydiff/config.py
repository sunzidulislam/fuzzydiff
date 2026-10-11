"""JSON request settings; explicit command flags override configuration."""
import json
from dataclasses import fields
from pathlib import Path

from .generation.requests import GenerationRequest


def load_request_config(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Configuration must be a JSON object.')
    return value


def merge_request_config(config: dict, explicit_flags: dict) -> GenerationRequest:
    for mapping in (config, explicit_flags):
        if not isinstance(mapping, dict):
            raise ValueError('Generation configuration must be an object.')
        if not isinstance(mapping.get('method_options', {}), dict):
            raise ValueError('method_options must be a JSON object.')
    allowed = {f.name for f in fields(GenerationRequest)}
    unknown = (set(config) | set(explicit_flags)) - allowed
    if unknown:
        raise ValueError('Unknown configuration fields: ' + ', '.join(sorted(unknown)))
    value = {**config, **explicit_flags}
    value['method_options'] = {**config.get('method_options', {}),
                               **explicit_flags.get('method_options', {})}
    try:
        return GenerationRequest(**value)
    except TypeError as error:
        raise ValueError(f'Invalid generation configuration: {error}') from error

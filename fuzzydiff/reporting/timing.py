"""Exclusive wall-clock accounting for a separate, instrumented inference pass."""
from collections import defaultdict
from contextlib import contextmanager, ExitStack
from functools import wraps
from time import perf_counter


class TimeLedger:
    def __init__(self, synchronize=lambda: None, clock=perf_counter):
        self.synchronize = synchronize
        self.clock = clock
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self.failures = defaultdict(int)
        self.stack = []

    @contextmanager
    def measure(self, category):
        self.synchronize()
        frame = [self.clock(), 0.0]
        self.stack.append(frame)
        self.calls[category] += 1
        try:
            yield
        except BaseException:
            self.failures[category] += 1
            raise
        finally:
            self.synchronize()
            elapsed = self.clock() - frame[0]
            self.stack.pop()
            self.seconds[category] += elapsed - frame[1]
            if self.stack:
                self.stack[-1][1] += elapsed


@contextmanager
def patch_method(owner, name, replacement):
    # Restore inherited methods by removing the temporary instance/class override.
    had_own = name in vars(owner)
    previous = vars(owner).get(name)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        if had_own:
            setattr(owner, name, previous)
        else:
            delattr(owner, name)


@contextmanager
def instrument_tokenizers(tokenizers, ledger):
    targets = {id(tokenizer) for tokenizer in tokenizers if tokenizer is not None}
    classes = {type(tokenizer) for tokenizer in tokenizers if tokenizer is not None}
    with ExitStack() as stack:
        for cls in classes:
            for name in ('__call__', 'encode', 'decode'):
                if not hasattr(cls, name):
                    continue
                original = getattr(cls, name)

                def wrap(original):
                    @wraps(original)
                    def timed(instance, *args, **kwargs):
                        if id(instance) not in targets:
                            return original(instance, *args, **kwargs)
                        with ledger.measure('tokenize'):
                            return original(instance, *args, **kwargs)
                    return timed

                stack.enter_context(patch_method(cls, name, wrap(original)))
        yield


@contextmanager
def instrument_pipelines(pipelines, ledger):
    tokenizers, seen = [], set()
    with ExitStack() as stack:
        for pipeline in pipelines:
            tokenizers.extend(getattr(pipeline, name, None)
                              for name in ('tokenizer', 'tokenizer_2'))
            for component, method, category in (
                    ('text_encoder', 'forward', 'text_encode'),
                    ('text_encoder_2', 'forward', 'text_encode'),
                    ('vae', 'encode', 'vae'), ('vae', 'decode', 'vae')):
                module = getattr(pipeline, component, None)
                if module is None or (id(module), method) in seen:
                    continue
                seen.add((id(module), method))
                original = getattr(module, method)

                def wrap(original, category):
                    @wraps(original)
                    def timed(*args, **kwargs):
                        with ledger.measure(category):
                            return original(*args, **kwargs)
                    return timed

                stack.enter_context(patch_method(module, method, wrap(original, category)))
        stack.enter_context(instrument_tokenizers(tokenizers, ledger))
        yield

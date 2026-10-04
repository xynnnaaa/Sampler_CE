"""Nested wall-clock timings shared by the sampler and its engine.

Inclusive times describe phases; exclusive times form a non-overlapping
breakdown. Instrument batches, not individual tuples, to limit overhead.
"""
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from time import perf_counter


class SamplingTimings:
    def __init__(self):
        self.reset()

    def reset(self):
        self.inclusive = defaultdict(float)
        self.exclusive = defaultdict(float)
        self.calls = defaultdict(int)
        self.counters = defaultdict(int)
        self._stack = []

    @contextmanager
    def span(self, name):
        frame = [perf_counter(), 0.0]
        self._stack.append(frame)
        try:
            yield
        finally:
            elapsed = perf_counter() - frame[0]
            self._stack.pop()
            self.inclusive[name] += elapsed
            self.exclusive[name] += max(0.0, elapsed - frame[1])
            self.calls[name] += 1
            if self._stack:
                self._stack[-1][1] += elapsed

    def snapshot(self):
        return tuple(dict(values) for values in
                     (self.inclusive, self.exclusive, self.calls, self.counters))

    def report(self, label, elapsed, before=None):
        previous = before or ({}, {}, {}, {})
        current = self.snapshot()
        inclusive, exclusive, calls, counters = (
            {key: value - old.get(key, 0) for key, value in new.items()}
            for new, old in zip(current, previous)
        )
        denominator = elapsed or 1.0
        lines = [f"    [Timing] {label}: wall={elapsed:.4f}s",
                 "      Inclusive phases (nested; do not sum):"]
        for name in sorted(inclusive):
            if calls.get(name, 0):
                seconds = inclusive[name]
                lines.append(f"        {name}: {seconds:.4f}s "
                             f"({100 * seconds / denominator:.2f}%), calls={calls[name]}")
        lines.append("      Exclusive breakdown (non-overlapping, sorted by time):")
        lines.append("        Parent entries below contain only their own Python/control time.")
        for name, seconds in sorted(exclusive.items(), key=lambda item: item[1], reverse=True):
            if calls.get(name, 0):
                lines.append(f"        {name}: {seconds:.4f}s "
                             f"({100 * seconds / denominator:.2f}%)")
        accounted = sum(exclusive.values())
        lines.append(f"        outside_spans/reporting: {max(0.0, elapsed - accounted):.4f}s "
                     f"({100 * max(0.0, elapsed - accounted) / denominator:.2f}%)")
        lines.append("      Counters: " + ", ".join(
            f"{key}={value}" for key, value in sorted(counters.items()) if value))
        # execute is client-observed driver time, not PostgreSQL server-only time.
        db_seconds = sum(value for name, value in exclusive.items()
                         if name.endswith('.execute') or name.endswith('.fetchall'))
        lines.append(f"      DB execute+fetchall: {db_seconds:.4f}s "
                     f"({100 * db_seconds / denominator:.2f}%); "
                     f"Python/control/logging: {max(0.0, elapsed - db_seconds):.4f}s "
                     f"({100 * max(0.0, elapsed - db_seconds) / denominator:.2f}%)")
        print("\n".join(lines), flush=True)

    @contextmanager
    def round(self, label):
        before = self.snapshot()
        start = perf_counter()
        try:
            yield
        finally:
            self.report(label, perf_counter() - start, before)


def timed(name):
    def decorate(func):
        @wraps(func)
        def wrapped(self, *args, **kwargs):
            with self.timings.span(name):
                return func(self, *args, **kwargs)
        return wrapped
    return decorate


def timed_template(func):
    @wraps(func)
    def wrapped(self, template_id, template_data):
        self.timings.reset()
        start = perf_counter()
        status = 'failed'
        try:
            with self.timings.span('template.control'):
                result = func(self, template_id, template_data)
            status = 'complete' if result else 'empty'
            return result
        finally:
            self.timings.report(
                f"Template {template_id!r} [{status}]", perf_counter() - start)
    return wrapped

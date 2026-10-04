"""Bit-exact reference comparisons and real multiprocess publication checks."""
from contextlib import redirect_stdout
import io
import json
import multiprocessing as mp
from pathlib import Path
import random
import re
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predicate_cache import SharedPredicateCache
from sampling_timing import SamplingTimings
from template_annotations import SharedMemoryBudget, TemplateAnnotations


class Graph:
    def __init__(self, aliases=None):
        self.aliases = aliases or {'a': {'real_name': 't'}}

    def nodes(self, data=False):
        return list(self.aliases.items())


class Connection:
    def __init__(self, rows, marker=None, pause=None, fail=False, stats=None):
        self.rows = rows
        self.statements = []
        self.marker = marker
        self.pause = pause
        self.fail = fail
        self.stats = stats

    def cursor(self, name=None):
        return Cursor(self, name)

    @staticmethod
    def matches(predicate, row):
        if predicate == 'flag = 1':
            return row.get('flag') == 1
        if predicate == "tag = 'Brand#35'":
            return row.get('tag') == 'Brand#35'
        index = int(predicate[1:])
        modulus = index % 7 + 2
        return row['id'] % modulus == index % modulus


class Cursor:
    def __init__(self, conn, name):
        self.conn = conn
        self.name = name
        self.position = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, sql):
        self.conn.statements.append(sql)
        if not self.name:
            return
        if self.conn.marker:
            with open(self.conn.marker, 'a') as stream:
                stream.write('scan\n')
        cases = re.findall(r"CASE WHEN \((.*?)\) THEN B'([01]+)' ELSE B'([01]+)' END", sql)
        base = re.search(r"B'([01]+)'", sql)[1]
        data = sorted(self.conn.rows, key=lambda r: r['id']) if 'ORDER BY id' in sql else self.conn.rows
        self.rows = []
        for row in data:
            mask = int(base, 2)
            for predicate, yes, no in cases:
                if self.conn.matches(predicate, row):
                    mask |= int(yes, 2)
            self.rows.append((row['id'], format(mask, f'0{len(base)}b')))

    def fetchone(self):
        if self.conn.stats is not None:
            return self.conn.stats
        ids = [row['id'] for row in self.conn.rows]
        return len(ids), min(ids) if ids else None, max(ids) if ids else None

    def fetchmany(self, size):
        if self.conn.fail:
            raise RuntimeError('injected cache build failure')
        if self.conn.pause:
            self.conn.pause.set()
            time.sleep(30)
        rows = self.rows[self.position:self.position + size]
        self.position += len(rows)
        return rows


def concurrent_builder(directory, marker, start, output):
    conn = Connection([{'id': i} for i in (4, 1, 3, 2)], marker=marker)
    cache = SharedPredicateCache(directory, 'parallel', {'dbname': 'test'},
                                 {'t': {i: f'p{i}' for i in range(70)}}, SamplingTimings(), batch_size=2)
    start.wait(10)
    with redirect_stdout(io.StringIO()):
        manifest = cache.ensure(conn, 't')
    output.put((manifest['count'], cache.timings.counters['predicate_cache.tables_built']))


def interrupted_builder(directory, ready):
    conn = Connection([{'id': i} for i in (4, 1, 3, 2)], pause=ready)
    cache = SharedPredicateCache(directory, 'crash', {}, {'t': {0: 'p0'}}, SamplingTimings())
    with redirect_stdout(io.StringIO()):
        cache.ensure(conn, 't')


class SharedCacheChecks(unittest.TestCase):
    def cache(self, directory, predicates, timings=None, run='test', **kwargs):
        return SharedPredicateCache(directory, run, {}, {'t': predicates},
                                    timings or SamplingTimings(), batch_size=2, **kwargs)

    def annotate(self, conn, instances, predicates, cache=None, graph=None, budget=None):
        annotations = TemplateAnnotations(SamplingTimings(), batch_size=2, predicate_cache=cache,
                                          compose_batch_size=3, budget=budget)
        with redirect_stdout(io.StringIO()):
            annotations.prepare(conn, graph or Graph(), instances, {'t': predicates})
        return annotations

    def test_exact_masks_dense_sparse_null_literals_and_qid_boundaries(self):
        predicates = {i: f'p{i}' for i in range(70)}
        predicates[70] = 'flag = 1'
        predicates[71] = "tag = 'Brand#35'"
        for ids in ((8, 5, 7, 6), (100, -5, 5000000000)):
            rows = [{'id': i, 'flag': None if i % 2 else 1,
                     'tag': 'Brand#35' if i % 3 else 'brand#35'} for i in ids]
            with tempfile.TemporaryDirectory() as directory:
                cache = self.cache(directory, predicates)
                for width in (1, 63, 64, 65, 200, 1042):
                    instances = [{'a': -1 if q % 5 == 0 else q % 72} for q in range(width)]
                    original = self.annotate(Connection(rows), instances, predicates)
                    changed = self.annotate(Connection(rows), instances, predicates, cache)
                    self.assertEqual(changed.aliases['a'].buffer, original.aliases['a'].buffer)
                    self.assertEqual(changed.lookup_many('a', list(ids) + [999]),
                                     original.lookup_many('a', list(ids) + [999]))
                    changed.release()
                    original.release()
                self.assertEqual(cache.timings.counters['predicate_cache.scans'], 2)

    def test_templates_reorder_qids_and_share_table_aliases(self):
        predicates = {0: 'p0', 1: 'p1', 2: 'p2'}
        graph = Graph({'a': {'real_name': 't'}, 'b': {'real_name': 't'}, 'c': {'real_name': 't'}})
        with tempfile.TemporaryDirectory() as directory:
            rows = [{'id': i} for i in (3, 1, 2)]
            cache = self.cache(directory, predicates)
            conn = Connection(rows)
            for instances in ([{'a': 0, 'b': 1, 'c': -1}, {'a': 2, 'b': -1, 'c': -1}],
                              [{'a': 2, 'b': 0, 'c': -1}, {'a': 0, 'b': 1, 'c': -1}]):
                actual = self.annotate(conn, instances, predicates, cache, graph)
                reference = self.annotate(Connection(rows), instances, predicates, graph=graph)
                for alias in ('a', 'b', 'c'):
                    self.assertEqual(actual.lookup_many(alias, [1, 2, 3, 999]),
                                     reference.lookup_many(alias, [1, 2, 3, 999]))
                actual.release()
            self.assertEqual(sum('COUNT(*)' in sql for sql in conn.statements), 1)
            self.assertEqual(cache.timings.counters['predicate_cache.scans'], 1)
            # A second worker reuses files with a completely separate cache object.
            other = self.cache(directory, predicates)
            other_conn = Connection(rows)
            other.ensure(other_conn, 't')
            self.assertEqual(other_conn.statements, [])

    def test_empty_and_constant_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = self.cache(directory, {0: 'p0'})
            annotations = self.annotate(Connection([]), [{'a': 0}], {0: 'p0'}, cache)
            self.assertEqual(annotations.nbytes, 0)
            self.assertEqual(annotations.lookup_many('a', [1]), {'1': 0})
            conn = Connection([])
            self.annotate(conn, [{'a': -1}], {0: 'p0'}, cache)
            self.assertEqual(conn.statements, [])

    def test_dense_duplicate_rejected_across_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = self.cache(directory, {0: 'p0'})
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'unique'):
                cache.ensure(Connection([{'id': i} for i in (3, 1, 3)], stats=(3, 1, 3)), 't')
            self.assertFalse(list(cache.root.glob('*/manifest.json')))
            self.assertFalse(list(cache.root.glob('*.building.*')))

    def test_failure_releases_budget_and_retry_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = self.cache(directory, {0: 'p0'})
            budget = SharedMemoryBudget(Path(directory) / 'budget.json', 1000)
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                self.annotate(Connection([{'id': 1}], fail=True), [{'a': 0}], {0: 'p0'}, cache, budget=budget)
            self.assertFalse(budget.reserved)
            self.assertEqual(json.loads((cache.root/'capacity.json').read_text())['entries'], {})
            self.annotate(Connection([{'id': 1}]), [{'a': 0}], {0: 'p0'}, cache)

    def test_new_run_does_not_reuse_old_data(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = self.cache(directory, {0: 'flag = 1'}, run='a'), self.cache(directory, {0: 'flag = 1'}, run='b')
            first = self.annotate(Connection([{'id': 1, 'flag': 1}]), [{'a': 0}], {0: 'flag = 1'}, a)
            second = self.annotate(Connection([{'id': 1, 'flag': 0}]), [{'a': 0}], {0: 'flag = 1'}, b)
            self.assertEqual(first.lookup_many('a', [1]), {'1': 1})
            self.assertEqual(second.lookup_many('a', [1]), {'1': 0})

    def test_capacity_bound_and_truncated_file_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = self.cache(directory, {0: 'p0'}, max_bytes=7)
            with redirect_stdout(io.StringIO()), self.assertRaises(MemoryError):
                cache.ensure(Connection([{'id': 1}]), 't')
        with tempfile.TemporaryDirectory() as directory:
            cache = self.cache(directory, {0: 'p0'})
            with redirect_stdout(io.StringIO()):
                cache.ensure(Connection([{'id': 1}]), 't')
            next(cache.root.glob('*/block_0.bin')).write_bytes(b'')
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                self.cache(directory, {0: 'p0'}).ensure(Connection([]), 't')

    def test_concurrent_workers_build_once(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = str(Path(directory)/'scans.txt')
            start, output = mp.Event(), mp.Queue()
            processes = [mp.Process(target=concurrent_builder, args=(directory, marker, start, output)) for _ in range(4)]
            try:
                for process in processes:
                    process.start()
                start.set()
                results = [output.get(timeout=15) for _ in processes]
                for process in processes:
                    process.join(10)
                    self.assertEqual(process.exitcode, 0)
                self.assertEqual(sum(built for count, built in results), 1)
                self.assertEqual(Path(marker).read_text().splitlines(), ['scan', 'scan'])
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(5)

    def test_killed_builder_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            ready = mp.Event()
            process = mp.Process(target=interrupted_builder, args=(directory, ready))
            process.start()
            try:
                self.assertTrue(ready.wait(10))
                process.terminate()
                process.join(5)
                cache = self.cache(directory, {0: 'p0'}, run='crash')
                self.annotate(Connection([{'id': i} for i in (4, 1, 3, 2)]), [{'a': 0}], {0: 'p0'}, cache)
                self.assertEqual(len(list(cache.root.glob('*/manifest.json'))), 1)
                self.assertFalse(list(cache.root.glob('*.building.*')))
            finally:
                if process.is_alive():
                    process.terminate()
                process.join(5)


if __name__ == '__main__':
    unittest.main()

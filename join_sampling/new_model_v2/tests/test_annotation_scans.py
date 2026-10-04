"""Regression checks for unordered dense writes and worker-local metadata reuse."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sampling_timing import SamplingTimings
from template_annotations import TemplateAnnotations


class Graph:
    def nodes(self, data=False):
        return [('a', {'real_name': 't'})]


class Cursor:
    def __init__(self, connection, name=None):
        self.connection = connection
        self.name = name
        self.position = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, statement):
        self.connection.statements.append(statement)
        if self.name:
            ids = self.connection.ids
            if 'ORDER BY id' in statement:
                ids = sorted(ids)
            width = int(re.search(r'bit varying\((\d+)\)', statement)[1])
            # QID 0 has the odd-ID predicate; all remaining QIDs are unfiltered.
            base = ((1 << width) - 1) ^ 1
            self.rows = [(i, format(base | (i % 2), f'0{width}b')) for i in ids]

    def fetchone(self):
        if self.connection.stats is not None:
            return self.connection.stats
        ids = self.connection.ids
        return len(ids), min(ids) if ids else None, max(ids) if ids else None

    def fetchmany(self, size):
        result = self.rows[self.position:self.position + size]
        self.position += len(result)
        return result


class Connection:
    def __init__(self, ids, stats=None):
        self.ids = ids
        self.stats = stats
        self.statements = []

    def cursor(self, name=None):
        return Cursor(self, name)


class AnnotationScans(unittest.TestCase):
    def prepare(self, conn, cache=None, width=3):
        annotations = TemplateAnnotations(SamplingTimings(), batch_size=2,
                                          table_stats_cache=cache)
        instances = [{'a': 0}] + [{'a': -1} for _ in range(width - 1)]
        with redirect_stdout(io.StringIO()):
            annotations.prepare(conn, Graph(), instances, {'t': {0: 'id % 2 = 1'}})
        return annotations

    def test_dense_unordered_batches_and_word_boundaries(self):
        for width in (1, 64, 65, 200):
            for ids in ([4, 1, 3, 2], [-1, -4, -2, -3],
                        [5000000003, 5000000000, 5000000002, 5000000001]):
                conn = Connection(ids)
                annotations = self.prepare(conn, width=width)
                self.assertNotIn('ORDER BY', conn.statements[-1])
                base = ((1 << width) - 1) ^ 1
                self.assertEqual(annotations.lookup_many('a', ids),
                                 {str(i): base | (i % 2) for i in ids})
                self.assertEqual(annotations.timings.counters[
                    'annotations.validation_scratch_bytes'], 1)
                annotations.release()

    def test_sparse_keeps_ordered_scan(self):
        conn = Connection([100, -5, 3])
        annotations = self.prepare(conn)
        self.assertIn('ORDER BY id', conn.statements[-1])
        self.assertEqual(list(annotations.aliases['a'].ids), [-5, 3, 100])
        self.assertEqual(annotations.lookup_many('a', conn.ids),
                         {'100': 6, '-5': 7, '3': 7})

    def test_cache_survives_release_and_new_template(self):
        conn = Connection([3, 1, 2])
        cache = {}
        first = self.prepare(conn, cache)
        first.release()
        self.assertEqual(cache, {'t': (3, 1, 3)})
        second = self.prepare(conn, cache, width=65)
        self.assertEqual(sum('COUNT(*)' in sql for sql in conn.statements), 1)
        self.assertEqual(second.timings.counters['annotations.stats.cache_hits'], 1)
        self.assertEqual(second.lookup_many('a', [2]), {'2': (1 << 65) - 2})
        # A new worker/connection gets its own cache by default.
        other = Connection([10, 11])
        self.prepare(other)
        self.assertEqual(sum('COUNT(*)' in sql for sql in other.statements), 1)

    def test_unordered_duplicate_id_rejected_and_released(self):
        conn = Connection([3, 1, 3], stats=(3, 1, 3))
        annotations = TemplateAnnotations(SamplingTimings(), batch_size=2)
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'unique'):
            annotations.prepare(conn, Graph(), [{'a': 0}], {'t': {0: 'id % 2 = 1'}})
        self.assertEqual(annotations.aliases, {})
        self.assertEqual(annotations.nbytes, 0)

    def test_cached_stats_detect_count_and_range_changes(self):
        for ids, error in (([1, 2], 'row count'), ([1, 2, 4], 'ID range'),
                           ([1, 2, 3, 4], 'row count')):
            with self.subTest(ids=ids), self.assertRaisesRegex(RuntimeError, error):
                self.prepare(Connection(ids), {'t': (3, 1, 3)})

    def test_empty_table(self):
        conn = Connection([])
        annotations = self.prepare(conn)
        self.assertEqual(annotations.nbytes, 0)
        self.assertNotIn('ORDER BY', conn.statements[-1])


if __name__ == '__main__':
    unittest.main()

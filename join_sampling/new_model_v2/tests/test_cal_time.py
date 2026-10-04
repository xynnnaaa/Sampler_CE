"""Parsing tests for complete, nested, partial and failed timing reports."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cal_time import main, parse_log, summarize


REPORT = '''    [Timing] Template ('a', 'b') [complete]: wall=100.0000s
      Inclusive phases (nested; do not sum):
        predicate_cache.build: 40.0000s (40.00%), calls=1
        template.control: 100.0000s (100.00%), calls=1
      Exclusive breakdown (non-overlapping, sorted by time):
        predicate_cache.table_lock_wait: 30.0000s (30.00%)
        predicate_cache.capacity_lock_wait: 5.0000s (5.00%)
        annotations.budget_wait: 10.0000s (10.00%)
        predicate_cache.build: 35.0000s (35.00%)
        work: 20.0000s (20.00%)
      Counters: samples.selected=100
      DB execute+fetch: 0.0000s (0.00%); Python/control/logging: 100.0000s (100.00%)
'''
HEADER = '''Initializing JoinSampler
Parsed templates across all queries in 12.00s.
Worker 0 assigned 1 templates.
Predicate cache run: example, directory: /tmp/cache
'''


class CalTimeChecks(unittest.TestCase):
    def test_nested_waits_and_bitmap_not_double_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'worker.log'
            bitmap = REPORT.replace("Template ('a', 'b') [complete]", 'Bitmap 1/1')
            path.write_text(HEADER + bitmap + REPORT + 'Worker 0 finished in 112.50s\n')
            result = summarize([parse_log(path)])
            self.assertEqual(result['reported'], 1)
            self.assertEqual(result['template_seconds'], 100)
            self.assertEqual(result['net_seconds'], 65)
            self.assertEqual(result['active_seconds'], 55)
            self.assertEqual(result['cache_build_seconds'], 40)
            self.assertEqual(result['cache_build_active_seconds'], 35)
            self.assertEqual(result['online_seconds'], 20)
            self.assertTrue(result['complete'])

    def test_unfinished_report_excluded_but_failed_cost_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'worker.log'
            partial = REPORT.split('      Counters:')[0]
            path.write_text(HEADER.replace('assigned 1', 'assigned 3') +
                            REPORT.replace('[complete]', '[failed]') + partial)
            worker = parse_log(path)
            result = summarize([worker])
            self.assertEqual(result['reported'], 1)
            self.assertEqual(result['statuses'], {'failed': 1})
            self.assertEqual(result['template_seconds'], 100)
            self.assertFalse(result['complete'])
            self.assertTrue(worker.warnings)

    def test_appended_runs_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'worker.log'
            path.write_text(HEADER + REPORT + HEADER + REPORT)
            with self.assertRaisesRegex(ValueError, '多次运行'):
                parse_log(path)

    def test_cli_json_csv_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'worker.log'
            path.write_text(HEADER + REPORT)
            with redirect_stdout(io.StringIO()):
                status = main([str(path), '--json', str(Path(directory)/'result.json'),
                               '--csv', str(Path(directory)/'result.csv'), '--top', '0'])
            self.assertEqual(status, 0)
            self.assertTrue((Path(directory)/'result.json').exists())
            self.assertEqual(len((Path(directory)/'result.csv').read_text().splitlines()), 2)


if __name__ == '__main__':
    unittest.main()

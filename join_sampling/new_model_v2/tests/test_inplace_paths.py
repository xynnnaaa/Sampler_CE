"""Check ownership, shared neighbors and survivor order during in-place updates."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wander_join import WanderJoinEngine


class InplacePaths(unittest.TestCase):
    def test_private_paths_reused_shared_neighbor_unchanged(self):
        engine = WanderJoinEngine(None, None)
        root = {'a.id': '1', 'a.key': '7'}
        paths = [{'vals': root.copy(), 'acc_bmp': mask, 'alive': True}
                 for mask in (0b111, 0b010)]
        # Missing parent and missing-neighbor paths are excluded in place order.
        missing_parent = {'vals': {'a.id': '3'}, 'acc_bmp': 7, 'alive': True}
        missing_neighbor = {'vals': {'a.id': '4', 'a.key': '99'}, 'acc_bmp': 7, 'alive': True}
        active = [paths[0], missing_parent, paths[1], missing_neighbor]
        neighbor = {'b.id': '10', 'b.fk': '7', 'b.key': '20'}
        original_neighbor = deepcopy(neighbor)
        original_vals = [p['vals'] for p in paths]
        engine._batch_fetch_neighbors = lambda *args: ({'7': [neighbor]}, 0)
        engine._batch_lookup_qid_bitmaps = lambda *args: ({'10': 0b101}, 0)
        step = dict(alias='b', real_name='child', parent='a',
                    join_condition='a.key=b.fk', sels=['b.id', 'b.fk', 'b.key'])
        with patch('wander_join.random.choice', return_value=neighbor) as choose:
            result = engine.extend_paths_one_step(active, step, {}, {}, '')
        self.assertEqual(choose.call_count, 2)
        self.assertEqual(len(result), 2)
        for i in range(2):
            self.assertIs(result[i], paths[i])
            self.assertIs(result[i]['vals'], original_vals[i])
        self.assertEqual([p['acc_bmp'] for p in result], [0b101, 0])
        self.assertEqual(neighbor, original_neighbor)
        self.assertNotIn('b.id', root)
        result[0]['vals']['b.key'] = 'changed'
        self.assertEqual(result[1]['vals']['b.key'], '20')
        self.assertEqual(neighbor['b.key'], '20')
        self.assertFalse(missing_parent['alive'])
        self.assertFalse(missing_neighbor['alive'])


if __name__ == '__main__':
    unittest.main()

import importlib.util
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / 'scripts/export_bf16.py'
SPEC = importlib.util.spec_from_file_location('checkpoint_export', PATH)
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


class CheckpointExportTests(unittest.TestCase):
    def test_shards_cover_every_tensor_once_with_expected_bf16_sizes(self):
        layout = {'large': {'shape': [12]}, 'small_a': {'shape': [2]}, 'small_b': {'shape': [3]}}
        groups = EXPORT.shard_groups(layout, 10)
        self.assertEqual(groups, [['large'], ['small_a', 'small_b']])
        self.assertEqual([name for group in groups for name in group], sorted(layout))

    def test_exactly_fitting_tensor_is_not_split(self):
        layout = {'a': {'shape': [2, 3]}, 'b': {'shape': [1]}}
        self.assertEqual(EXPORT.shard_groups(layout, 12), [['a'], ['b']])


if __name__ == '__main__':
    unittest.main()

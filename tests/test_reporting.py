import copy
import json
import tempfile
import unittest
from pathlib import Path

from tool_rl.reporting import aggregate_evaluation, build_report, check_recorded, read_jsonl, training_statistics

ROOT = Path(__file__).resolve().parents[1]


class ReportingTests(unittest.TestCase):
    def test_shipped_log_counts_and_explicit_aggregate_fallback(self):
        report = build_report(ROOT, verify=True)
        self.assertEqual(report['experiments']['E1']['evidence'], 'aggregate_only')
        self.assertEqual(report['experiments']['E2']['evidence'], 'per_example_logs')
        self.assertEqual(report['experiments']['E3']['evidence'], 'aggregate_only')
        self.assertEqual([report['experiments']['E2']['results'][b]['correct'] for b in
                          ('gsm8k', 'math500', 'humaneval', 'mbpp')], [104, 70, 17, 27])
        self.assertEqual(report['score_changes_pp']['E2_to_E3'],
                         {'gsm8k': 3.0, 'math500': 2.0, 'humaneval': 6.0, 'mbpp': 6.0})
        self.assertEqual(report['experiments']['E2']['sources'][0]['path'], 'results/logs/E2.jsonl')

    def test_missing_raw_logs_cannot_be_claimed_as_recomputed(self):
        with self.assertRaisesRegex(ValueError, 'E1.*unavailable'):
            build_report(ROOT, require_raw=True)

    def test_duplicate_examples_are_rejected(self):
        row = read_jsonl(ROOT/'results/logs/E2.jsonl')[0]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            aggregate_evaluation([row, row])

    def test_nonboolean_correctness_is_rejected(self):
        row = copy.deepcopy(read_jsonl(ROOT/'results/logs/E2.jsonl')[0])
        row['score']['correct'] = 1
        with self.assertRaisesRegex(ValueError, 'boolean'):
            aggregate_evaluation([row])

    def test_recorded_score_mismatch_is_detected(self):
        expected = json.loads((ROOT/'results/retained-aggregates.json').read_text())['E2']['results']
        actual = copy.deepcopy(expected)
        actual['gsm8k']['correct'] -= 1
        with self.assertRaisesRegex(ValueError, 'correct'):
            check_recorded(actual, expected)

    def test_completed_updates_have_four_samples_and_real_advantage_signal(self):
        values = training_statistics(ROOT/'results/logs', 25)
        self.assertEqual(values['rollouts'], 200)
        self.assertEqual(values['distinct_prompts'], 50)
        self.assertEqual(values['reward_groups'], 50)
        self.assertEqual(values['mixed_reward_groups'], 24)
        self.assertEqual(values['reward_distribution'], {'0.0': 137, '1.0': 63})
        self.assertEqual(values['rank0_nonzero_gradient_events'], 25)

    def test_failed_next_update_excluded_and_incomplete_groups_rejected(self):
        rows = [{'step': 0, 'id': 'train/0', 'domain': 'math', 'reward': r,
                 'tool_metrics': {'requested_calls': 0, 'executed_calls': 0}}
                for r in (0.0, 1.0, 0.0, 1.0)]
        rows.append(dict(rows[0], step=1, id='unfinished'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'training-rollouts.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            self.assertEqual(training_statistics(directory, 1)['rollouts'], 4)
            self.assertEqual(training_statistics(directory, 1)['mixed_reward_groups'], 1)
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                training_statistics(directory, 2)


if __name__ == '__main__':
    unittest.main()

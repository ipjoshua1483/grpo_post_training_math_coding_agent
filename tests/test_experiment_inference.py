import json
import tempfile
import unittest
from pathlib import Path

from tool_rl.experiment_inference import build_requests, validate_requests
from tool_rl.inference import BASE_MODEL, BASE_REVISION

ROOT = Path(__file__).resolve().parents[1]


class ExperimentInferenceTests(unittest.TestCase):
    def test_e1_selects_base_model_and_fewshot_configs_for_both_domains(self):
        requests = build_requests(ROOT, 'E1', gpus=8)
        self.assertEqual([r['domain'] for r in requests], ['math', 'code'])
        for request in requests:
            self.assertEqual((request['model'], request['revision']), (BASE_MODEL, BASE_REVISION))
            self.assertEqual(request['prompt_variant'], 'optional_tool')
            self.assertIn('--nproc-per-node=8', request['command'])
            self.assertIn('fewshot', request['config'])

    def test_e2_and_e3_use_identical_evaluation_settings_and_plans(self):
        before, after = build_requests(ROOT, 'E2'), build_requests(ROOT, 'E3')
        for base, trained in zip(before, after):
            for field in ('config', 'plan', 'prompt_variant'):
                self.assertEqual(base[field], trained[field])
            self.assertEqual(base['model'], BASE_MODEL)
            self.assertEqual(trained['model'], str(ROOT/'models/grpo-bf16'))
            self.assertNotIn('torch.distributed.run', trained['command'])

    def test_base_experiments_cannot_accidentally_load_a_checkpoint(self):
        with self.assertRaisesRegex(ValueError, 'E3-only'):
            build_requests(ROOT, 'E1', checkpoint='trained/model')

    def test_plan_id_and_policy_changes_are_rejected(self):
        controls = json.loads((ROOT/'repro/eval-controls.json').read_text())
        config = json.loads((ROOT/'configs/eval-compact-math.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            requests = build_requests(ROOT, 'E2', domain='math', plan_dir=directory)
            plan = {'domain': 'math', 'prompt_variant': 'compact_math',
                    'selected_ids': controls['math']['selected_ids'], 'policy': controls['math']['policy']}
            path = Path(requests[0]['plan'])
            path.write_text(json.dumps(plan))
            validate_requests(ROOT, requests)
            plan['selected_ids'] = plan['selected_ids'][:-1]
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'Plan IDs'):
                validate_requests(ROOT, requests)
            plan['selected_ids'] = controls['math']['selected_ids']
            plan['policy'] = dict(plan['policy'], temperature=1.0)
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'policy'):
                validate_requests(ROOT, requests)


if __name__ == '__main__':
    unittest.main()

import json
import unittest
from pathlib import Path

from tool_rl.prompt_variants import make_math_training_prompt, make_coding_training_prompt
from tool_rl.train import validate_config

ROOT = Path(__file__).resolve().parents[1]


class FrozenPromptTests(unittest.TestCase):
    def test_exports_match_all_four_runtime_prefixes(self):
        for family in ('fewshot', 'compact'):
            for domain in ('math', 'code'):
                with self.subTest(family=family, domain=domain):
                    export = json.loads((ROOT/f'prompts/{family}-{domain}.json').read_text())
                    config = json.loads((ROOT/f'configs/eval-{family}-{domain}.json').read_text())
                    builder = make_math_training_prompt if domain == 'math' else make_coding_training_prompt
                    prompt = builder({'domain': domain, 'prompt': 'Public example'},
                                     config['max_execute_calls'], prompt_variant=config['prompt_variant'])
                    self.assertEqual(prompt[:-1], export['prefix_messages'])
                    self.assertEqual(config['prompt_variant'], export['prompt_variant'])
                    self.assertEqual(len(prompt[:-1]), 7 if family == 'fewshot' else 1)

    def test_grpo_uses_the_same_compact_conditions_as_evaluation(self):
        training = json.loads((ROOT/'configs/grpo.json').read_text())
        for domain in ('math', 'code'):
            config = json.loads((ROOT/f'configs/eval-compact-{domain}.json').read_text())
            self.assertEqual(training['domain_prompt_variants'][domain], config['prompt_variant'])
            self.assertEqual(training['domain_policies'][domain],
                             {key: config[key] for key in ('max_execute_calls', 'max_invalid_calls', 'max_agent_turns')})

    def test_four_reward_groups_require_the_intended_training_world_size(self):
        config = json.loads((ROOT/'configs/grpo.json').read_text())
        validate_config(config, world_size=8)
        with self.assertRaisesRegex(ValueError, 'complete prompt groups'):
            validate_config(config, world_size=1)


if __name__ == '__main__':
    unittest.main()

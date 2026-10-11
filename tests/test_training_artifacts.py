import json
import tempfile
import unittest
from pathlib import Path

from tool_rl.train import checkpoint_step, resolve_resume_checkpoint, trim_logs_for_resume


class TrainingArtifactTests(unittest.TestCase):


    def test_latest_resume_checkpoint_and_log_rollback(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            (root / "checkpoint-5").mkdir()
            (root / "checkpoint-12").mkdir()
            self.assertEqual(checkpoint_step(resolve_resume_checkpoint(root, "latest")), 12)
            metrics = root / "metrics.jsonl"
            metrics.write_text("".join(json.dumps({"step": step, "reward": 0}) + "\n" for step in (1, 11, 12, 13)), encoding="utf-8")
            trim_logs_for_resume(root, 12)
            retained = [json.loads(line)["step"] for line in metrics.read_text().splitlines()]
            self.assertEqual(retained, [1, 11])

    def test_resume_cannot_escape_run_directory(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            output = root / "run"
            output.mkdir()
            outside = root / "checkpoint-4"
            outside.mkdir()
            with self.assertRaisesRegex(ValueError, "directly inside"):
                resolve_resume_checkpoint(output, outside)


if __name__ == "__main__":
    unittest.main()

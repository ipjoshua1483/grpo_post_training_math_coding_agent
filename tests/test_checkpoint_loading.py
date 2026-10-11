import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tool_rl.inference import BASE_MODEL, BASE_REVISION, resolve_inference_checkpoint, validate_checkpoint_folder


def checkpoint_fixture(folder, shards=False):
    for filename, value in [('config.json', {'model_type': 'qwen2'}),
                            ('tokenizer_config.json', {}), ('tokenizer.json', {})]:
        (folder/filename).write_text(json.dumps(value))
    filename = 'model-00001-of-00001.safetensors' if shards else 'model.safetensors'
    (folder/filename).write_bytes(b'fixture')
    if shards:
        (folder/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'model.weight': filename}}))


class CheckpointLoadingTests(unittest.TestCase):
    def test_local_export_is_resolved_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            checkpoint_fixture(folder, shards=True)
            with patch.dict('sys.modules', {'huggingface_hub': None}):
                result = resolve_inference_checkpoint(str(folder))
            self.assertEqual(result['path'], str(folder.resolve()))
            self.assertEqual(result['source'], 'local')

    def test_published_checkpoint_downloads_its_own_revision_and_weight_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            checkpoint_fixture(folder, shards=True)
            api, download = Mock(), Mock(return_value=directory)
            api.model_info.return_value = types.SimpleNamespace(sha='a'*40)
            hub = types.SimpleNamespace(HfApi=Mock(return_value=api), snapshot_download=download)
            with patch.dict('sys.modules', {'huggingface_hub': hub}):
                result = resolve_inference_checkpoint('author/trained-qwen', 'release')
            api.model_info.assert_called_once_with('author/trained-qwen', revision='release')
            self.assertEqual(download.call_args.kwargs['repo_id'], 'author/trained-qwen')
            self.assertEqual(download.call_args.kwargs['revision'], 'a'*40)
            self.assertIn('model*.safetensors', download.call_args.kwargs['allow_patterns'])
            self.assertNotIn('optimizer.pt', download.call_args.kwargs['allow_patterns'])
            self.assertEqual(result['revision'], 'a'*40)

    def test_default_base_model_uses_recorded_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_fixture(Path(directory))
            api = Mock()
            api.model_info.return_value = types.SimpleNamespace(sha=BASE_REVISION)
            hub = types.SimpleNamespace(HfApi=Mock(return_value=api), snapshot_download=Mock(return_value=directory))
            with patch.dict('sys.modules', {'huggingface_hub': hub}):
                resolve_inference_checkpoint(BASE_MODEL)
            api.model_info.assert_called_once_with(BASE_MODEL, revision=BASE_REVISION)

    def test_incomplete_export_fails_without_falling_back_to_base(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            checkpoint_fixture(folder, shards=True)
            (folder/'model-00001-of-00001.safetensors').unlink()
            with self.assertRaisesRegex(ValueError, 'missing weight shard'):
                validate_checkpoint_folder(folder)

    def test_missing_local_path_does_not_trigger_hub_download(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict('sys.modules', {'huggingface_hub': None}):
                with self.assertRaises(FileNotFoundError):
                    resolve_inference_checkpoint(str(Path(directory)/'missing'))

    def test_git_lfs_pointer_explains_how_to_download_actual_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            checkpoint_fixture(folder)
            (folder/'model.safetensors').write_text('version https://git-lfs.github.com/spec/v1\noid sha256:'+'a'*64+'\nsize 123456\n')
            with self.assertRaisesRegex(ValueError, 'git lfs pull'):
                validate_checkpoint_folder(folder)


if __name__ == '__main__':
    unittest.main()

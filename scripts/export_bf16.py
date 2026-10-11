"""Export a single-file FP32 safetensors checkpoint for BF16 inference on CPU."""
import argparse
import gc
import hashlib
import json
import math
import shutil
import struct
from pathlib import Path

BASE_MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
BASE_REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                   'special_tokens_map.json', 'added_tokens.json', 'vocab.json',
                   'merges.txt', 'tokenizer.model', 'generation_config.json')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_layout(path):
    with Path(path).open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        if not 0 < length <= 16 * 1024 * 1024:
            raise ValueError('Unsupported safetensors header size')
        header = json.loads(stream.read(length))
    tensors = {name: value for name, value in header.items() if name != '__metadata__'}
    if not tensors or any(value['dtype'] != 'F32' for value in tensors.values()):
        raise ValueError('This exporter expects a nonempty FP32 model.safetensors file')
    return tensors


def shard_groups(layout, limit):
    groups, current, size = [], [], 0
    for name, value in sorted(layout.items()):
        tensor_size = math.prod(value['shape']) * 2
        if current and size + tensor_size > limit:
            groups.append(current)
            current, size = [], 0
        current.append(name)
        size += tensor_size
    if current:
        groups.append(current)
    return groups


def export_checkpoint(checkpoint, output, *, max_shard_mb=1024, threads=4,
                      base_model=BASE_MODEL, base_revision=BASE_REVISION):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if output == checkpoint or output.is_relative_to(checkpoint):
        raise ValueError('Export directory must be separate from the source checkpoint')
    if max_shard_mb < 1 or threads < 1:
        raise ValueError('Shard size and CPU threads must be positive')
    if output.exists():
        raise ValueError('Export directory already exists; choose a fresh directory')
    source = checkpoint / 'model.safetensors'
    for name in ('model.safetensors', 'config.json', 'tokenizer_config.json'):
        if not (checkpoint / name).is_file():
            raise ValueError(f'Checkpoint is missing {name}')
    if not any((checkpoint / name).is_file() for name in ('tokenizer.json', 'vocab.json', 'tokenizer.model')):
        raise ValueError('Checkpoint is missing tokenizer vocabulary')
    layout = read_layout(source)
    groups = shard_groups(layout, max_shard_mb * 1024 * 1024)
    torch.set_num_threads(threads)
    output.mkdir(parents=True)
    weight_map, shard_files = {}, []
    verified = 0
    with torch.inference_mode(), safe_open(source, framework='pt', device='cpu') as original:
        for index, names in enumerate(groups, 1):
            filename = (f'model-{index:05d}-of-{len(groups):05d}.safetensors'
                        if len(groups) > 1 else 'model.safetensors')
            path = output / filename
            converted = {name: original.get_tensor(name).to(torch.bfloat16).contiguous() for name in names}
            if any(not bool(torch.isfinite(tensor).all()) for tensor in converted.values()):
                raise ValueError('Non-finite values found in the BF16 export')
            save_file(converted, str(path), metadata={'format': 'pt'})
            del converted
            gc.collect()
            with safe_open(path, framework='pt', device='cpu') as exported:
                if set(exported.keys()) != set(names):
                    raise ValueError('Exported tensor names differ from the source')
                for name in names:
                    actual = exported.get_tensor(name)
                    expected = original.get_tensor(name).to(torch.bfloat16)
                    if actual.dtype != torch.bfloat16 or not torch.equal(actual, expected):
                        raise ValueError(f'BF16 conversion verification failed: {name}')
                    verified += 1
                    del actual, expected
            weight_map.update({name: filename for name in names})
            shard_files.append(path)
            print(f'Exported and verified shard {index}/{len(groups)} ({len(names)} tensors)', flush=True)
    total_size = sum(math.prod(value['shape']) * 2 for value in layout.values())
    if len(groups) > 1:
        (output / 'model.safetensors.index.json').write_text(json.dumps({
            'metadata': {'total_size': total_size}, 'weight_map': weight_map,
        }, indent=2) + '\n', encoding='utf-8')
    config = json.loads((checkpoint / 'config.json').read_text())
    config['dtype'] = 'bfloat16'
    if 'torch_dtype' in config:
        config['torch_dtype'] = 'bfloat16'
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    for filename in TOKENIZER_FILES:
        if (checkpoint / filename).is_file():
            shutil.copy2(checkpoint / filename, output / filename)
    training_metadata = checkpoint / 'tool_rl_training.json'
    origin = (json.loads(training_metadata.read_text())['origin'] if training_metadata.exists()
              else {'model': base_model, 'revision': base_revision})
    step = int(checkpoint.name.split('-')[-1]) if checkpoint.name.startswith('checkpoint-') and checkpoint.name.split('-')[-1].isdigit() else None
    (output / 'tool_rl_training.json').write_text(json.dumps({
        'origin': origin, 'checkpoint_step': step, 'inference_dtype': 'bfloat16',
    }, indent=2) + '\n', encoding='utf-8')
    card = f'''---
library_name: transformers
base_model: {base_model}
tags:
- grpo
- math
- code
---
# Tool-Augmented RLVR — BF16 inference checkpoint

Qwen2.5-1.5B-Instruct after joint math/coding GRPO. Checkpoint step: {step}.
This package contains BF16 inference weights, tokenizer, chat template, and configuration.
Every exported tensor was verified against the BF16 conversion of the original FP32 checkpoint.

Use the companion code repository's restricted Python tool and compact prompts:

```bash
python infer.py --model /path/to/this/package --domain math --prompt compact --problem "Compute 23 * 17."
```

The companion inference runtime requires a CUDA GPU and Docker.
Optimizer, scheduler, RNG state, and training data are excluded.
Base-model use remains subject to the upstream model's terms.
'''
    (output / 'README.md').write_text(card, encoding='utf-8')
    print('Computing source and export file hashes...', flush=True)
    manifest = {
        'format': 'inference_only_bf16', 'checkpoint_step': step,
        'source_dtype': 'float32', 'dtype': 'bfloat16',
        'source_weights_sha256': sha256(source),
        'tensors': len(layout), 'verified_tensors': verified,
        'tensor_data_bytes': total_size,
        'verification': 'Every saved tensor equals source.to(torch.bfloat16) on CPU; no model generation run',
        'files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)}
                  for p in sorted(output.iterdir()) if p.is_file()},
    }
    (output / 'export-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    manifest['package_bytes'] = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    print(json.dumps({'output': str(output), 'verified_tensors': verified,
                      'package_gb': manifest['package_bytes'] / 1e9}, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-shard-mb', type=int, default=1024)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--base-model', default=BASE_MODEL)
    parser.add_argument('--base-revision', default=BASE_REVISION)
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output, max_shard_mb=args.max_shard_mb,
                      threads=args.threads, base_model=args.base_model, base_revision=args.base_revision)


if __name__ == '__main__':
    main()

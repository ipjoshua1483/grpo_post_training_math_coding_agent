"""Run the three reported conditions on the same frozen math/coding datasets."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from .inference import resolve_inference_checkpoint
from .io import write_json

ROOT = Path(__file__).resolve().parents[2]
POLICY_KEYS = ('seed', 'max_completion_length', 'max_prompt_tokens', 'max_execute_calls',
               'max_invalid_calls', 'max_agent_turns', 'temperature', 'top_p', 'enable_thinking')


def build_requests(root, experiment, *, domain='all', gpus=1, output='runs/inference',
                   data_dir=None, plan_dir=None, checkpoint=None, revision=None):
    root = Path(root).resolve()
    experiments = json.loads((root/'configs/experiments.json').read_text())
    if experiment not in experiments:
        raise ValueError('Experiment must be E1, E2, or E3')
    if domain not in {'math', 'code', 'all'} or type(gpus) is not int or gpus < 1:
        raise ValueError('Choose domain=math/code/all and a positive GPU count')
    selected = experiments[experiment]
    if selected['model']['kind'] == 'base':
        if checkpoint is not None or revision is not None:
            raise ValueError('E1/E2 use the fixed base model; checkpoint overrides are E3-only')
        model, model_revision = selected['model']['id'], selected['model']['revision']
    else:
        model = str(checkpoint or root/selected['model']['path'])
        if checkpoint is not None and not Path(checkpoint).is_absolute() and (root/checkpoint).is_dir():
            model = str((root/checkpoint).resolve())
        model_revision = revision
    domains = ('math', 'code') if domain == 'all' else (domain,)
    prefix = [sys.executable]
    if gpus > 1:
        prefix += ['-m', 'torch.distributed.run', '--standalone', f'--nproc-per-node={gpus}']
    requests = []
    data_path = Path(data_dir or root/'data/frozen')
    plan_root = Path(plan_dir or root/'eval-plans')
    if not data_path.is_absolute():
        data_path = root/data_path
    if not plan_root.is_absolute():
        plan_root = root/plan_root
    for current in domains:
        settings = selected['domains'][current]
        config = root/settings['config']
        plan = plan_root/settings['plan']
        destination = Path(output)/current
        if not destination.is_absolute():
            destination = root/destination
        command = prefix + [str(root/'evaluate.py'), '--model', model,
                           '--data-dir', str(data_path),
                           '--plan', str(plan), '--config', str(config),
                           '--prompt-variant', settings['prompt_variant'], '--output', str(destination)]
        if model_revision:
            command += ['--revision', model_revision]
        requests.append({'domain': current, 'config': str(config), 'plan': str(plan),
                         'prompt_variant': settings['prompt_variant'], 'model': model,
                         'revision': model_revision, 'output': str(destination), 'command': command})
    return requests


def validate_requests(root, requests):
    controls = json.loads((Path(root)/'repro/eval-controls.json').read_text())
    for request in requests:
        path = Path(request['plan'])
        if not path.is_file():
            raise ValueError('Missing frozen plans; run python scripts/prepare_eval.py first')
        plan = json.loads(path.read_text())
        config = json.loads(Path(request['config']).read_text())
        if plan.get('domain') != request['domain'] or plan.get('prompt_variant') != request['prompt_variant']:
            raise ValueError('Frozen plan does not match the selected experiment/domain')
        if plan.get('selected_ids') != controls[request['domain']]['selected_ids']:
            raise ValueError('Plan IDs differ from the fixed reported evaluation set')
        if any(plan['policy'].get(key) != config.get(key) for key in POLICY_KEYS):
            raise ValueError('Plan policy differs from the experiment configuration')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, choices=('E1', 'E2', 'E3'))
    parser.add_argument('--domain', choices=('math', 'code', 'all'), default='all')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data-dir', default=str(ROOT/'data/frozen'))
    parser.add_argument('--plan-dir', default=str(ROOT/'eval-plans'))
    parser.add_argument('--checkpoint', help='E3 override: local checkpoint folder or Hugging Face model ID')
    parser.add_argument('--revision', help='E3 remote checkpoint revision')
    parser.add_argument('--dry-run', action='store_true', help='Validate IDs/settings and print commands without loading a model')
    args = parser.parse_args()
    try:
        requests = build_requests(ROOT, args.experiment, domain=args.domain, gpus=args.gpus,
                                  output=args.output, data_dir=args.data_dir, plan_dir=args.plan_dir,
                                  checkpoint=args.checkpoint, revision=args.revision)
        validate_requests(ROOT, requests)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps({'experiment': args.experiment, 'requests': requests}, indent=2))
        return
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT/output
    if output.exists():
        parser.error('Output directory already exists; choose a fresh experiment directory')
    source = None
    if args.experiment == 'E3':
        source = resolve_inference_checkpoint(requests[0]['model'], args.revision)
        requests = build_requests(ROOT, args.experiment, domain=args.domain, gpus=args.gpus,
                                  output=args.output, data_dir=args.data_dir, plan_dir=args.plan_dir,
                                  checkpoint=source['path'])
    output.mkdir(parents=True)
    manifest = {'experiment': args.experiment, 'status': 'running', 'gpus': args.gpus,
                'checkpoint_source': source, 'requests': requests,
                'config_sha256': {r['domain']: hashlib.sha256(Path(r['config']).read_bytes()).hexdigest() for r in requests}}
    write_json(output/'experiment.json', manifest)
    environment = os.environ.copy()
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
                'GROUP_RANK', 'ROLE_RANK', 'ROLE_WORLD_SIZE', 'LOCAL_WORLD_SIZE'):
        environment.pop(key, None)
    try:
        for request in requests:
            print(f"{args.experiment}: dataset inference for {request['domain']}", flush=True)
            subprocess.run(request['command'], cwd=ROOT, env=environment, check=True)
        manifest['status'] = 'complete'
    except BaseException:
        manifest['status'] = 'failed'
        raise
    finally:
        write_json(output/'experiment.json', manifest)


if __name__ == '__main__':
    main()

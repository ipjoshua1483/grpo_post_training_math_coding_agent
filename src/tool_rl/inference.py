"""Single-problem inference using the same prompt/controller paths as evaluation."""
import argparse
import json
from pathlib import Path

from .coding_environment import CodingToolEnvironment
from .coding_rewards import extract_code_submission
from .coding_sandbox import CodingSandbox
from .environment import ToolEnvironment
from .io import write_json
from .math_answers import extract_math_answer
from .prompt_variants import (OPTIONAL_TOOL, COMPACT_MATH, COMPACT_CODE,
    make_math_training_prompt, make_coding_training_prompt)
from .protocol import final_answer
from .rollout import run_agent_episode, run_native_tool_episode
from .runtime import HuggingFaceTurns, NativeToolHuggingFaceTurns, load_runtime
from .sandbox import DockerSandbox

BASE_MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
BASE_REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
CHECKPOINT_DOWNLOAD_FILES = (
    'config.json', 'generation_config.json', 'model*.safetensors',
    'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json',
    'tokenizer.model', 'vocab.json', 'merges.txt', 'special_tokens_map.json',
    'added_tokens.json', 'chat_template*.jinja', 'tool_rl_training.json',
    'export-manifest.json',
)


def validate_checkpoint_folder(folder):
    """Require a complete Qwen checkpoint; never substitute untrained weights."""
    folder = Path(folder)
    for filename in ('config.json', 'tokenizer_config.json'):
        if not (folder / filename).is_file():
            raise ValueError(f'Checkpoint is missing {filename}')
    if json.loads((folder / 'config.json').read_text())['model_type'] != 'qwen2':
        raise ValueError('Inference expects a Qwen2.5 checkpoint (model_type=qwen2)')
    if not (folder / 'tokenizer.json').is_file() and not ((folder / 'vocab.json').is_file() and (folder / 'merges.txt').is_file()):
        raise ValueError('Checkpoint is missing tokenizer vocabulary')
    index = folder / 'model.safetensors.index.json'
    if index.is_file():
        weight_map = json.loads(index.read_text()).get('weight_map')
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError('Checkpoint has an invalid safetensors index')
        for filename in set(weight_map.values()):
            path = Path(filename)
            if path.is_absolute() or '..' in path.parts or path.suffix != '.safetensors':
                raise ValueError('Checkpoint index must reference relative safetensors files')
            if not (folder / path).is_file():
                raise ValueError(f'Checkpoint is missing weight shard {filename}')
    elif not (folder / 'model.safetensors').is_file():
        raise ValueError('Checkpoint is missing model.safetensors or its shard index')


def resolve_inference_checkpoint(model_id, revision=None):
    """Resolve a local folder or download/cache a pinned Hub model snapshot."""
    candidate = Path(model_id).expanduser()
    if candidate.is_dir():
        folder, resolved, kind = candidate.resolve(), None, 'local'
    else:
        if candidate.is_absolute() or str(model_id).startswith(('.', '~')):
            raise FileNotFoundError(f'Checkpoint directory not found: {model_id}')
        from huggingface_hub import HfApi, snapshot_download
        requested = revision or (BASE_REVISION if model_id == BASE_MODEL else None)
        resolved = HfApi().model_info(model_id, revision=requested).sha
        folder = Path(snapshot_download(
            repo_id=model_id, revision=resolved, allow_patterns=list(CHECKPOINT_DOWNLOAD_FILES),
        ))
        kind = 'huggingface_hub'
    validate_checkpoint_folder(folder)
    return {'path': str(folder), 'model': model_id, 'revision': resolved, 'source': kind}


def run_with_backend(problem, policy, backend, environment, prompt_variant):
    """The caller supplies a real or fake backend; no private grader data is accepted."""
    if set(problem) - {'id', 'domain', 'prompt', 'benchmark', 'entry_point', 'required_symbols', 'source_split'}:
        raise ValueError('Inference accepts public problem fields only; do not supply answers or tests')
    if problem['domain'] == 'math':
        prompt = make_math_training_prompt(problem, policy['max_execute_calls'], prompt_variant=prompt_variant)
    elif problem['domain'] == 'code':
        prompt = make_coding_training_prompt(problem, policy['max_execute_calls'], prompt_variant=prompt_variant)
    else:
        raise ValueError('domain must be math or code')
    native = problem['domain'] == 'math' and prompt_variant == COMPACT_MATH
    kwargs = {k:policy[k] for k in ('max_execute_calls','max_invalid_calls','max_agent_turns')}
    if not native: kwargs['entry_point'] = problem.get('entry_point')
    trajectory = (run_native_tool_episode if native else run_agent_episode)(prompt, backend, environment, **kwargs)
    if problem['domain'] == 'math':
        extracted = extract_math_answer(trajectory['messages'])
        candidate, method = extracted['answer'], extracted['method']
        format_ok = final_answer(trajectory['messages']) is not None
    else:
        candidate, format_ok, method = extract_code_submission(trajectory['messages'])
    return {
        'domain':problem['domain'], 'prompt_variant':prompt_variant,
        'candidate':candidate, 'format_ok':format_ok, 'extraction_method':method,
        'correctness':None, 'grading_note':'Open-ended inference: no gold answer or hidden tests supplied.',
        'tool_metrics':trajectory['tool_metrics'], 'generated_tokens':trajectory['generated_tokens'],
        'truncated':trajectory['truncated'], 'error':trajectory['error'],
        'messages':trajectory['messages'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', choices=('math','code'), required=True)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--problem'); inputs.add_argument('--problem-file')
    parser.add_argument('--entry-point', help='Exact requested function name for code problems')
    parser.add_argument('--prompt', choices=('fewshot','compact'), default='compact')
    parser.add_argument('--model', default=BASE_MODEL, help='Local checkpoint folder or Hugging Face model ID; Hub weights are downloaded and cached')
    parser.add_argument('--revision', help='Remote model commit; defaults to the recorded revision for the base model')
    parser.add_argument('--max-new-tokens', type=int, help='Total completion-context budget, including observations')
    parser.add_argument('--output', help='Optional JSON trajectory file')
    args=parser.parse_args()
    if args.domain=='code' and (not args.entry_point or not args.entry_point.isidentifier()):
        parser.error('Code inference requires --entry-point with a valid Python name')
    text=Path(args.problem_file).read_text() if args.problem_file else args.problem
    if not text.strip(): parser.error('Problem must be nonempty')
    variant=OPTIONAL_TOOL if args.prompt=='fewshot' else (COMPACT_MATH if args.domain=='math' else COMPACT_CODE)
    policy={
        'seed':42, 'max_completion_length':1536 if args.domain=='math' else 2048,
        'max_prompt_tokens':4096, 'max_execute_calls':2 if args.domain=='math' else 4,
        'max_invalid_calls':2, 'max_agent_turns':6 if args.domain=='math' else 7,
        'temperature':0.0, 'top_p':0.95, 'enable_thinking':False,
    }
    if args.max_new_tokens is not None:
        if args.max_new_tokens<1: parser.error('--max-new-tokens must be positive')
        policy['max_completion_length']=args.max_new_tokens
    problem={'id':f'inference/{args.domain}', 'domain':args.domain, 'prompt':text}
    if args.domain=='code':problem.update(entry_point=args.entry_point, required_symbols=[args.entry_point])
    sandbox= DockerSandbox() if args.domain=='math' else CodingSandbox()
    sandbox_info=sandbox.preflight()
    checkpoint=resolve_inference_checkpoint(args.model,args.revision)
    model,tokenizer,runtime=load_runtime(checkpoint['path'],seed=policy['seed'])
    runtime.update(model=args.model,revision=checkpoint['revision'],checkpoint_source=checkpoint['source'])
    if not (Path(checkpoint['path'])/'tool_rl_training.json').is_file():
        runtime['origin']={'model':args.model,'revision':checkpoint['revision']}
    backend=(NativeToolHuggingFaceTurns if variant==COMPACT_MATH else HuggingFaceTurns)(model,tokenizer,policy)
    environment=(ToolEnvironment(sandbox,policy['max_execute_calls']) if args.domain=='math'
                 else CodingToolEnvironment(problem,sandbox,policy['max_execute_calls']))
    result=run_with_backend(problem,policy,backend,environment,variant)
    result.update(policy=policy,runtime=runtime,sandbox=sandbox_info)
    if args.output:write_json(args.output,result)
    print(result['candidate'] or '(No final candidate extracted)')
    print(json.dumps({k:result[k] for k in ('tool_metrics','truncated','error')},indent=2))


if __name__=='__main__':main()

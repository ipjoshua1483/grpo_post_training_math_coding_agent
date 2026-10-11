"""Recompute retained log metrics and explicitly label aggregate-only conditions."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .evaluate import summarize
from .coding_evaluate import summarize_coding

BENCHMARKS = ('gsm8k', 'math500', 'humaneval', 'mbpp')
CONDITIONS = {'E1': 'Few-shot base', 'E2': 'Compact base', 'E3': 'Compact + GRPO step 25'}
FIELDS = ('condition', 'benchmark', 'n', 'correct', 'score', 'score_name',
          'tool_call_rate', 'valid_tool_call_rate', 'mean_calls_per_problem',
          'mean_executed_calls', 'tool_errors_per_problem', 'invalid_calls_per_problem',
          'timeouts_per_problem', 'verifier_timeouts', 'mean_response_tokens',
          'truncation_rate', 'format_rate', 'evidence')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def display_path(path, root):
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return Path(path).name


def aggregate_evaluation(rows):
    """Use the same summarizers as evaluation; never combine rank and merged logs."""
    if not rows:
        raise ValueError('Evaluation log is empty')
    ids = [(row['benchmark'], row['id']) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate example IDs; use merged metrics or disjoint shards, not both')
    for row in rows:
        if row['benchmark'] not in BENCHMARKS or type(row['score']['correct']) is not bool:
            raise ValueError('Unsupported benchmark or non-boolean correctness')
        if type(row['generated_tokens']) is not int or row['generated_tokens'] < 0:
            raise ValueError('generated_tokens must be a nonnegative integer')
        for value in row['tool_metrics'].values():
            if type(value) is not int or value < 0:
                raise ValueError('Tool counters must be nonnegative integers')
    math_rows = [r for r in rows if r['benchmark'] in {'gsm8k', 'math500'}]
    code_rows = [r for r in rows if r['benchmark'] in {'humaneval', 'mbpp'}]
    combined = {}
    if math_rows: combined.update(summarize(math_rows))
    if code_rows: combined.update(summarize_coding(code_rows))
    for benchmark, values in combined.items():
        group = [r for r in rows if r['benchmark'] == benchmark]
        values['correct'] = sum(r['score']['correct'] for r in group)
        values['score_name'] = 'accuracy' if benchmark in {'gsm8k', 'math500'} else 'pass_at_1'
        values['score'] = values[values['score_name']]
        values['mean_executed_calls'] = sum(r['tool_metrics']['executed_calls'] for r in group) / len(group)
        values['timeouts_per_problem'] = values.get('timeouts_per_problem', values.get('tool_timeouts_per_problem'))
        values['verifier_timeouts'] = sum(r['score'].get('verification', {}).get('timed_out') is True for r in group)
    return combined


def validate_aggregate(values):
    if set(values) != set(BENCHMARKS):
        raise ValueError('Each condition must contain all four benchmarks')
    for benchmark, row in values.items():
        n, correct, score = row['n'], row['correct'], row['score']
        if type(n) is not int or type(correct) is not int or not 0 <= correct <= n or n < 1:
            raise ValueError(f'{benchmark}: invalid numerator/denominator')
        if not math.isclose(score, correct/n, abs_tol=1e-12):
            raise ValueError(f'{benchmark}: score does not match correct/n')


def check_recorded(actual, expected):
    for benchmark in BENCHMARKS:
        for key in ('n', 'correct', 'score', 'tool_call_rate', 'valid_tool_call_rate', 'mean_response_tokens'):
            if key not in expected[benchmark]: continue
            a, b = actual[benchmark].get(key), expected[benchmark][key]
            if a is None or b is None:
                if a != b: raise ValueError(f'{benchmark}.{key}: retained values disagree')
            elif not math.isclose(a, b, abs_tol=1e-9):
                raise ValueError(f'{benchmark}.{key}: {a} differs from retained {b}')


def training_statistics(logs, checkpoint_step, generations=4):
    path = Path(logs) / 'training-rollouts.jsonl'
    if not path.exists(): return {'available': False}
    rows = [r for r in read_jsonl(path) if r['step'] < checkpoint_step]
    if not rows: raise ValueError('No completed-update training rollouts')
    groups = defaultdict(list)
    for row in rows:
        if row['reward'] not in (0.0, 1.0): raise ValueError('Reported rewards must be binary')
        groups[(row['step'], row['id'])].append(row['reward'])
    if any(len(values) != generations for values in groups.values()):
        raise ValueError('Incomplete GRPO reward groups in the retained completed updates')
    by_domain = {}
    for domain in sorted({r['domain'] for r in rows}):
        group = [r for r in rows if r['domain'] == domain]
        executed = [r for r in group if r['tool_metrics']['executed_calls'] > 0]
        direct = [r for r in group if r['tool_metrics']['executed_calls'] == 0]
        by_domain[domain] = {
            'n': len(group), 'mean_reward': sum(r['reward'] for r in group)/len(group),
            'tool_call_rate': sum(r['tool_metrics']['requested_calls'] > 0 for r in group)/len(group),
            'execution_rate': len(executed)/len(group),
            'reward_when_executed': sum(r['reward'] for r in executed)/len(executed) if executed else None,
            'reward_without_execution': sum(r['reward'] for r in direct)/len(direct) if direct else None,
        }
    result = {
        'available': True, 'checkpoint_step': checkpoint_step,
        'rollouts': len(rows), 'distinct_prompts': len({r['id'] for r in rows}),
        'reward_distribution': dict(Counter(str(float(r['reward'])) for r in rows)),
        'mean_reward': sum(r['reward'] for r in rows)/len(rows),
        'reward_groups': len(groups),
        'mixed_reward_groups': sum(len(set(v)) > 1 for v in groups.values()),
        'by_domain': by_domain, 'rollout_log_sha256': fingerprint(path),
        'scope': 'step < checkpoint_step; failed-next-update rollouts excluded',
    }
    metrics_path = Path(logs)/'training-metrics.jsonl'
    if metrics_path.exists():
        metrics = [r for r in read_jsonl(metrics_path) if r['step'] <= checkpoint_step]
        result['optimizer_steps_logged'] = len({r['step'] for r in metrics})
        result['last_metrics'] = {k:metrics[-1].get(k) for k in ('step', 'reward', 'kl', 'entropy', 'completions/mean_length', 'grad_norm')}
        result['logged_step_time_s'] = sum(r.get('step_time',0) for r in metrics)
    gradient_path = Path(logs)/'gradient-audit.jsonl'
    if gradient_path.exists():
        gradients = [r for r in read_jsonl(gradient_path) if r['step'] <= checkpoint_step]
        result['rank0_gradient_events'] = len(gradients)
        result['rank0_nonzero_gradient_events'] = sum(r['gradient_norm'] > 0 for r in gradients)
    update_path = Path(logs)/'update-audit.jsonl'
    if update_path.exists():
        updates = [r for r in read_jsonl(update_path) if r['step'] <= checkpoint_step]
        result['last_parameter_sample_delta_l2'] = updates[-1]['parameter_sample_delta_l2'] if updates else None
        result['rank0_peak_allocated_gib'] = max(r.get('gpu_peak_allocated_bytes',0) for r in updates)/2**30 if updates else None
    return result


def build_report(root, logs_dir=None, run_inputs=None, require_raw=False, verify=False):
    root = Path(root)
    logs = Path(logs_dir) if logs_dir else root/'results/logs'
    retained_path = root/'results/retained-aggregates.json'
    retained = json.loads(retained_path.read_text())
    controls = json.loads((root/'repro/eval-controls.json').read_text())
    expected_ids = set(controls['math']['selected_ids']) | set(controls['code']['selected_ids'])
    experiments = {}
    for condition, name in CONDITIONS.items():
        paths = list((run_inputs or {}).get(condition, []))
        if not paths and (logs/f'{condition}.jsonl').exists(): paths = [logs/f'{condition}.jsonl']
        if paths:
            files = [Path(p)/'metrics.jsonl' if Path(p).is_dir() else Path(p) for p in paths]
            rows = [r for p in files for r in read_jsonl(p)]
            if {r['id'] for r in rows} != expected_ids:
                raise ValueError(f'{condition}: log IDs differ from the fixed reported evaluation set')
            values = aggregate_evaluation(rows)
            evidence = 'per_example_logs'
            sources = [{'path':display_path(p, root), 'sha256':fingerprint(p)} for p in files]
        else:
            if require_raw: raise ValueError(f'{condition}: raw per-example logs unavailable; aggregate-only evidence retained')
            values = retained[condition]['results']
            evidence = 'aggregate_only'
            sources = [{'path':display_path(retained_path, root), 'sha256':fingerprint(retained_path)}]
        validate_aggregate(values)
        if verify: check_recorded(values, retained[condition]['results'])
        experiments[condition] = {'name':name, 'evidence':evidence, 'sources':sources, 'results':values}
    deltas = {}
    for earlier, later in [('E1','E2'),('E2','E3')]:
        deltas[f'{earlier}_to_{later}'] = {b:round(100*(experiments[later]['results'][b]['score']-experiments[earlier]['results'][b]['score']),10) for b in BENCHMARKS}
    training_run = json.loads((root/'repro/training-run.json').read_text())
    return {'experiments':experiments, 'score_changes_pp':deltas,
            'training':training_statistics(logs, training_run['checkpoint_step'], training_run['config']['num_generations']),
            'interpretation':'Point estimates only; aggregate-only conditions cannot support paired per-example significance tests.'}


def write_report(report, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    rows = []
    for condition, experiment in report['experiments'].items():
        for benchmark in BENCHMARKS:
            v=experiment['results'][benchmark]
            row={k:v.get(k) for k in FIELDS}
            row.update(condition=condition,benchmark=benchmark,evidence=experiment['evidence'],score_name='accuracy' if benchmark in {'gsm8k','math500'} else 'pass_at_1')
            row['timeouts_per_problem']=v.get('timeouts_per_problem',v.get('tool_timeouts_per_problem'))
            rows.append(row)
    with (output/'scores.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
    lines=['| Benchmark | n | E1: few-shot | E2: compact | E3: GRPO | E3 − E2 |','|---|---:|---:|---:|---:|---:|']
    for b in BENCHMARKS:
        values=[report['experiments'][c]['results'][b] for c in CONDITIONS]
        lines.append(f"| {b} | {values[0]['n']} | "+' | '.join(f"{v['correct']}/{v['n']} ({100*v['score']:.1f}%)" for v in values)+f" | {report['score_changes_pp']['E2_to_E3'][b]:+.1f} pp |")
    lines += ['', '| Experiment | Evidence |','|---|---|']
    lines += [f"| {c} | {e['evidence']} |" for c,e in report['experiments'].items()]
    lines += ['',report['interpretation']]
    (output/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')




def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--logs-dir')
    parser.add_argument('--run',action='append',default=[],metavar='EXPERIMENT=PATH',help='Merged metrics file or run directory; repeat for math/code')
    parser.add_argument('--output',default='results/generated')
    parser.add_argument('--require-raw',action='store_true')
    parser.add_argument('--check-recorded',action='store_true')
    args=parser.parse_args();overrides=defaultdict(list)
    for value in args.run:
        condition,sep,path=value.partition('=')
        if not sep or condition not in CONDITIONS: parser.error('--run requires E1, E2, or E3 followed by =PATH')
        overrides[condition].append(path)
    report=build_report(args.root,args.logs_dir,overrides,args.require_raw,args.check_recorded)
    write_report(report,args.output)
    print((Path(args.output)/'comparison.md').read_text())
    print(json.dumps({'output':args.output,'training':report['training']},indent=2))


if __name__=='__main__':main()

"""Build fresh evaluator plans while preserving the recorded IDs and data hashes."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tool_rl.experiment_data import freeze_plan
from tool_rl.io import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',default='data/frozen')
    parser.add_argument('--output-dir',default='eval-plans')
    args=parser.parse_args()
    controls=json.loads((ROOT/'repro/eval-controls.json').read_text())
    output=Path(args.output_dir)
    items=[(family,domain) for family in ('fewshot','compact') for domain in ('math','code')]
    if any((output/f'{family}-{domain}.json').exists() for family,domain in items):
        parser.error('Frozen plans already exist; choose a fresh output directory')
    plans=[]
    for family,domain in items:
        config=json.loads((ROOT/f'configs/eval-{family}-{domain}.json').read_text())
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            value=freeze_plan(args.data_dir,config,Path(temporary)/'plan.json')
        recorded=controls[domain]
        if set(value['selected_ids'])!=set(recorded['selected_ids']):
            raise ValueError('Selected IDs differ from the recorded comparison')
        if value['files']!=recorded['files'] or value['dataset_revisions']!=recorded['dataset_revisions']:
            raise ValueError('Dataset files/revisions differ from the recorded comparison')
        value['selected_ids']=recorded['selected_ids']
        plans.append((output/f'{family}-{domain}.json',value))
    for path,value in plans:
        write_json(path,value);print(f"{path}: {value['counts']} {value['prompt_variant']}")


if __name__=='__main__':main()

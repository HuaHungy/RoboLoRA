"""Compare baseline100 and routed-model success rates under one eval protocol."""

import argparse
import json
from pathlib import Path

from common import TASKS, write_json


def load_result(run_dir, task, step, seed, episodes):
    path = (run_dir / 'evaluations' / task /
            f'step_{step}_seed_{seed}_n{episodes}' / 'result.json')
    if not path.is_file():
        raise FileNotFoundError(f'Missing evaluation result: {path}')
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-run', type=Path, required=True)
    parser.add_argument('--router-run', type=Path, required=True)
    parser.add_argument('--baseline-step', type=int, default=5000)
    parser.add_argument('--router-step', type=int, default=5000)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    baseline_run = args.baseline_run.resolve()
    router_run = args.router_run.resolve()
    rows = []
    for task in TASKS:
        baseline = load_result(baseline_run, task, args.baseline_step, args.seed, args.episodes)
        router = load_result(router_run, task, args.router_step, args.seed, args.episodes)
        rows.append({
            'task': task,
            'baseline_successes': baseline['successes'],
            'baseline_episodes': baseline['episodes'],
            'baseline_success_rate': baseline['success_rate'],
            'router_successes': router['successes'],
            'router_episodes': router['episodes'],
            'router_success_rate': router['success_rate'],
            'router_minus_baseline': router['success_rate'] - baseline['success_rate'],
        })
    baseline_successes = sum(row['baseline_successes'] for row in rows)
    router_successes = sum(row['router_successes'] for row in rows)
    total = sum(row['baseline_episodes'] for row in rows)
    report = {
        'protocol': {'episodes_per_task': args.episodes, 'seed': args.seed, 'instruction_type': 'unseen'},
        'data_warning': (
            'baseline uses the 5x20=100-episode subset; the existing routed checkpoint used '
            'the five complete 50-episode sources (250 episodes), so this is not a strict data-matched ablation.'
        ),
        'per_task': rows,
        'overall': {
            'baseline_successes': baseline_successes,
            'router_successes': router_successes,
            'episodes': total,
            'baseline_success_rate': baseline_successes / total,
            'router_success_rate': router_successes / total,
            'router_minus_baseline': (router_successes - baseline_successes) / total,
        },
    }
    output = baseline_run / 'comparison_vs_router.json'
    write_json(output, report)
    lines = [
        '# Standard π0 baseline100 vs cluster Router', '',
        '> ' + report['data_warning'], '',
        '| Task | Baseline | Router | Router - baseline |',
        '|---|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['baseline_successes']}/{row['baseline_episodes']} "
            f"({row['baseline_success_rate']:.1%}) | {row['router_successes']}/{row['router_episodes']} "
            f"({row['router_success_rate']:.1%}) | {row['router_minus_baseline']:+.1%} |")
    overall = report['overall']
    lines.extend([
        f"| **Overall** | **{overall['baseline_successes']}/{total} ({overall['baseline_success_rate']:.1%})** "
        f"| **{overall['router_successes']}/{total} ({overall['router_success_rate']:.1%})** "
        f"| **{overall['router_minus_baseline']:+.1%}** |", '',
    ])
    markdown = baseline_run / 'comparison_vs_router.md'
    markdown.write_text('\n'.join(lines))
    print(markdown.read_text())
    print(f'JSON: {output}')
    print(f'Markdown: {markdown}')


if __name__ == '__main__':
    main()

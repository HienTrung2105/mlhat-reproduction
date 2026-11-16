"""Summarize completed EXP-02 runs and optionally publish saved evidence to W&B."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics

from experiments.reproduce_exp2 import StreamResult, _wandb_series


def main() -> None:
    """Write seed summaries and curves, or upload measured records to a common project.

    Args:
        CLI options select the result directory and optional W&B destination.

    Returns:
        None. Writes summary CSV, PNG curves and updated records containing run URLs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default='artifacts/exp2_hyperplane')
    parser.add_argument('--wandb-project')
    parser.add_argument('--wandb-entity')
    args = parser.parse_args()
    directory = Path(args.directory)
    records = []
    for path in sorted(directory.glob('*_seed*.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        if record['result']['n_samples'] != 50000:
            raise ValueError(f'Incomplete stream: {path}')
        if args.wandb_project and not record.get('wandb_url'):
            import wandb

            snapshot = directory / 'source_snapshot.zip'
            if not snapshot.exists():
                raise ValueError('Missing source_snapshot.zip for measured source provenance')
            with wandb.init(
                project=args.wandb_project, entity=args.wandb_entity,
                name=f"EXP-02-{record['dataset']}-{record['model']}-seed-{record['seed']}",
                job_type='reproduction',
                config={**{k: v for k, v in record.items() if k != 'result'},
                        'logging_mode': 'upload_completed_local_measurement',
                        'source_snapshot_sha256': hashlib.sha256(snapshot.read_bytes()).hexdigest()},
            ) as run:
                for step, values in _wandb_series(StreamResult(**record['result'])):
                    run.log(values, step=step)
                run.summary.update({key: record['result'][key] for key in
                                    ('n_samples', 'example_f1', 'hamming_loss', 'subset_accuracy')})
                run.summary['duration_seconds'] = record['duration_seconds_excluding_download']
                run.summary['structural_events_instrumented'] = record['model'] != 'BR+HAT'
                if record['model'] != 'BR+HAT':
                    for action in ('START_BACKGROUND', 'PROMOTE_BACKGROUND', 'DISCARD_BACKGROUND'):
                        run.summary[action.lower()] = sum(
                            e['action'] == action for e in record['result']['drift_events'])
                artifact = wandb.Artifact(f"exp2-{record['dataset']}-{record['model'].replace('+','-')}-seed{record['seed']}", type='experiment-evidence')
                artifact.add_file(str(path), name='measurement.json')
                artifact.add_file(str(snapshot), name='source_snapshot.zip')
                run.log_artifact(artifact)
                record['wandb_url'] = run.url
            path.write_text(json.dumps(record, indent=2), encoding='utf-8')
        records.append(record)
    if not records:
        raise SystemExit('No completed EXP-02 measurements yet')
    groups = sorted({(r['dataset'], r['model']) for r in records})
    with (directory / 'run_registry.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dataset', 'model', 'seed', 'n_samples', 'example_f1', 'hamming_loss',
                         'subset_accuracy', 'seconds', 'start_background', 'promote_background',
                         'discard_background', 'wandb_url', 'git_dirty'])
        for record in records:
            result = record['result']
            counts = [sum(e['action'] == action for e in result['drift_events']) for action in
                      ('START_BACKGROUND', 'PROMOTE_BACKGROUND', 'DISCARD_BACKGROUND')]
            if record['model'] == 'BR+HAT':
                counts = ['', '', '']  # River's internal actions are not instrumented here.
            writer.writerow([record['dataset'], record['model'], record['seed'], result['n_samples'],
                             result['example_f1'], result['hamming_loss'], result['subset_accuracy'],
                             record['duration_seconds_excluding_download'], *counts,
                             record.get('wandb_url', ''), record.get('git_dirty')])
    with (directory / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dataset', 'model', 'completed_seeds', 'metric', 'mean', 'sample_sd'])
        for dataset, model in groups:
            group = [r for r in records if (r['dataset'], r['model']) == (dataset, model)]
            for metric in ('example_f1', 'hamming_loss', 'subset_accuracy'):
                values = [r['result'][metric] for r in group]
                writer.writerow([dataset, model, ','.join(str(r['seed']) for r in group), metric,
                                 statistics.mean(values), statistics.stdev(values) if len(values) > 1 else ''])
    cache = directory / 'matplotlib-cache'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache.resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for dataset in sorted({r['dataset'] for r in records}):
        for curve in ('rolling_f1', 'faded_f1'):
            fig, ax = plt.subplots(figsize=(10, 4.5), layout='constrained')
            for model in sorted({r['model'] for r in records if r['dataset'] == dataset}):
                group = [r for r in records if (r['dataset'], r['model']) == (dataset, model)]
                points = group[0]['result'][curve]
                xs = [p['sample_index'] for p in points]
                ys = [statistics.mean(r['result'][curve][i]['example_f1'] for r in group) for i in range(len(xs))]
                ax.plot(xs, ys, label=f'{model} (n={len(group)} seeds)')
            for position in (12500, 25000, 37500):
                ax.axvline(position, color='gray', linestyle='--', linewidth=0.8)
            definition = 'window 1000; step 250' if curve == 'rolling_f1' else 'fading alpha=0.995; step 50'
            ax.set(title=f'{dataset}: {definition}', xlabel='Samples processed (one-based)', ylabel='Example F1', ylim=(0, 1))
            ax.legend()
            fig.savefig(directory / f'{dataset}_{curve}.png', dpi=180)
            plt.close(fig)
    print(f'Summarized {len(records)}/18 completed runs in {directory}')


if __name__ == '__main__':
    main()

"""Summarize local runs by validation loss, without using test results for ranking."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metrics', nargs='+', type=Path, help='Paths to metrics.json files')
    parser.add_argument('--output', type=Path, default=Path('outputs/comparison.csv'))
    args = parser.parse_args()
    rows = []
    for path in args.metrics:
        run = json.loads(path.read_text())
        c = run['config']
        rows.append({'run': path.parent.name, 'hidden_units': str(c['hidden_units']),
                     'learning_rate': c['learning_rate'], 'batch_size': c['batch_size'],
                     'best_epoch': run['best_epoch'],
                     'validation_loss': run['validation']['loss'],
                     'validation_accuracy': run['validation']['accuracy']})
    rows.sort(key=lambda row: row['validation_loss'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {args.output}; ranked by validation loss, not test accuracy.')


if __name__ == '__main__':
    main()

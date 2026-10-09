"""Verify and summarize archived traces; does not invoke any hardware backend."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'data' / 'historical'


def numbers(value):
    result = []
    for part in str(value).split(','):
        if '-' in part:
            a, b = map(int, part.split('-'))
            result.extend(range(a, b + 1))
        else:
            result.append(int(part))
    return result


def positive(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (ValueError, TypeError):
        return None


def summarize(rows, spec, config, budgets):
    agg_key = 'cells' if spec['backend'] == 'cpu' else 'batch'
    agg_config = 'cells_list' if spec['backend'] == 'cpu' else 'batches'
    expected = set(itertools.product(numbers(config['rb_list']),
                                    numbers(config['layers_list']),
                                    numbers(config['mcs_list']),
                                    numbers(config[agg_config])))
    seen = set()
    valid = []
    failed = 0
    invalid = 0
    for row in rows:
        key = tuple(int(row[k]) for k in ['rb', 'layers', 'mcs', agg_key])
        if key in seen or key not in expected:
            raise ValueError('Duplicate or unexpected configuration: ' + str(key))
        seen.add(key)
        if row['success'].lower() != 'true':
            failed += 1
            continue
        lat = positive(row.get(spec['latency_column']))
        energy = positive(row.get(spec['energy_column']))
        if lat is None or energy is None:
            invalid += 1
            continue
        valid.append((lat, energy * spec['energy_to_nj']))
    denominator = len(expected)
    return {
        'id': spec['id'], 'expected_configurations': denominator,
        'recorded_rows': len(rows), 'missing_configurations': len(expected - seen),
        'failed_rows': failed, 'invalid_success_rows': invalid,
        'valid_latency_energy_rows': len(valid),
        'deadline_us': spec['deadline_us'],
        'latency_column': spec['latency_column'], 'energy_column': spec['energy_column'],
        'joint_coverage': [
            {'energy_budget_nj_per_bit': b,
             'feasible_configurations': sum(l <= spec['deadline_us'] and e <= b for l, e in valid),
             'fraction': sum(l <= spec['deadline_us'] and e <= b for l, e in valid) / denominator}
            for b in budgets],
    }


def replay(root=ROOT, budgets=(10, 100, 1000, 10000, 100000, 1000000)):
    if not budgets or any(not math.isfinite(b) or b <= 0 for b in budgets):
        raise ValueError('Budgets must be finite positive nJ/bit values')
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    reports = []
    for spec in manifest['datasets']:
        folder = root / spec['id']
        for name, field in [('trace.csv', 'trace_sha256'), ('metadata.json', 'metadata_sha256')]:
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != spec[field]:
                raise ValueError('Checksum mismatch: ' + str(folder / name))
        with (folder / 'trace.csv').open(encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != spec['rows']:
            raise ValueError('Row count mismatch: ' + spec['id'])
        config = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))['config']
        reports.append(summarize(rows, spec, config, budgets))
    return {'scope': 'Historical per-dataset coverage, not final-paper reproduction or matched hardware comparison.',
            'datasets': reports}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT)
    parser.add_argument('--budgets', default='10,100,1000,10000,100000,1000000')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    report = replay(args.data, tuple(float(x) for x in args.budgets.split(',')))
    text = json.dumps(report, indent=2) + '\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding='utf-8')
    else:
        print(text, end='')

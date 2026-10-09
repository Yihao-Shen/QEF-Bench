"""Lossless metric import for legacy GPU scans; does not run benchmarks."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def number(value):
    if value is None or str(value).strip().lower() in ('', 'none', 'nan', 'null'):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_run(trace, metadata):
    meta = json.loads(metadata.read_text(encoding='utf-8-sig'))
    config = meta['config']
    iterations = number(config.get('num_iterations'))
    findings = []
    records = []
    seen = set()
    with trace.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'rb', 'layers', 'mcs', 'batch', 'success', 'p99_999_us', 'net_nj_per_bit'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Trace is missing required columns')
        for line, row in enumerate(reader, 2):
            workload = {k: number(row.get(k)) for k in ('rb', 'layers', 'mcs', 'k_bits', 'n_bits', 'mod_order', 'code_rate')}
            workload.update(symbols=config.get('num_symbols'), mcs_table=config.get('mcs_table'), iterations=iterations)
            batch = number(row.get('batch'))
            key = (workload['rb'], workload['layers'], workload['mcs'], batch)
            if key in seen:
                findings.append({'line': line, 'issue': 'duplicate_workload_batch'})
            seen.add(key)
            energy = number(row.get('net_nj_per_bit'))
            latency = number(row.get('p99_999_us'))
            success = row.get('success', '').lower() == 'true'
            if success and (energy is None or latency is None):
                findings.append({'line': line, 'issue': 'successful_row_missing_metric'})
            if energy is not None and energy < 0:
                findings.append({'line': line, 'issue': 'negative_net_energy'})
            if latency is not None and latency < 0:
                findings.append({'line': line, 'issue': 'negative_latency'})
            records.append({
                'source_line': line,
                'workload': workload,
                'batch': batch,
                'num_streams': config.get('num_streams'),
                'run_success': success,
                'latency': {'p99_999_us': latency, 'source_requested': config.get('latency_source'),
                            'clock_requested': config.get('latency_clock'), 'include_h2d': config.get('include_h2d'),
                            'samples_requested': config.get('lat_samples'), 'discard_requested': config.get('lat_discard')},
                'energy': {'net_nj_per_bit': energy, 'gross_nj_per_bit': number(row.get('abs_nj_per_bit')),
                           'net_nj_per_bit_per_configured_iteration': energy / iterations if energy is not None and iterations and iterations > 0 else None,
                           'boundary': 'GPU device (legacy NVML benchmark)',
                           'measurement_mode': 'separate throughput loop; inspect benchmark version'},
                'throughput_mbps': number(row.get('throughput_mbps')),
            })
    if meta.get('total_rows') != len(records):
        findings.append({'issue': 'metadata_trace_row_count_mismatch', 'metadata_rows': meta.get('total_rows')})
    if meta.get('successful_rows') != sum(r['run_success'] for r in records):
        findings.append({'issue': 'metadata_trace_success_count_mismatch'})
    if not iterations or iterations <= 0:
        findings.append({'issue': 'missing_or_invalid_iteration_count'})
    if config.get('include_h2d'):
        findings.append({'issue': 'h2d_latency_enabled_verify_energy_boundary_separately'})
    findings.append({'issue': 'legacy_summary_selection_not_used', 'reason': 'Preserve every MCS configuration in the trace'})
    audit = {'trace_sha256': sha256(trace), 'metadata_sha256': sha256(metadata),
             'rows': len(records), 'successful_rows': sum(r['run_success'] for r in records),
             'rb_values': sorted({r['workload']['rb'] for r in records if r['workload']['rb'] is not None}),
             'findings': findings,
             'limitations': ['Run success does not certify decoded-bit correctness.',
                            'Reported metrics do not establish raw-sample provenance.',
                            'No paper-suite coverage or QEF denominator is inferred.']}
    return records, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', required=True, type=Path)
    parser.add_argument('--metadata', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    records, audit = import_run(args.trace, args.metadata)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ('measurements.jsonl', 'audit.json'):
        if (args.out / name).exists():
            raise FileExistsError(f'Refusing to overwrite {args.out / name}')
    with (args.out / 'measurements.jsonl').open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, allow_nan=False) + '\n')
    (args.out / 'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'rows': audit['rows'], 'successful_rows': audit['successful_rows'], 'findings': len(audit['findings'])}))


if __name__ == '__main__':
    main()

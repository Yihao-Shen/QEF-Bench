"""Plan or run legacy CPU/GPU scans using the paper workload table.

Default: write a plan only. --execute explicitly starts measurements.
Never changes existing result directories or historical measurements.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'paper-workloads.json'


def make_plan(args):
    data = json.loads(CONFIG.read_text(encoding='utf-8'))
    suite = data[args.suite]
    degrees = [int(v) for v in args.aggregation.split(',')] if args.aggregation else suite['aggregation']
    if not degrees or len(set(degrees)) != len(degrees) or any(c not in suite['aggregation'] for c in degrees):
        raise ValueError('Aggregation must be a unique subset of the configured range')
    if args.samples <= 0:
        raise ValueError('samples must be positive')
    if args.platform == 'gpu' and args.samples <= 50:
        raise ValueError('GPU sample count must exceed the 50 discarded warm-up samples')
    if args.platform == 'gpu' and (not args.bench_script or not args.latency_source):
        raise ValueError('GPU requires --bench-script and explicit --latency-source')
    if args.platform == 'cpu' and (not args.benchmark or not args.cpus or not args.energy_scope):
        raise ValueError('CPU requires --benchmark, --cpus and --energy-scope')
    csv = lambda values: ','.join(map(str, values))
    jobs = []
    for iters in suite['iterations']:
        for c in degrees:
            out = args.out.resolve() / f'{args.suite}_iter{iters}_c{c}'
            command = [args.python, str(args.scan_script.resolve()),
                       '--rb-list', csv(suite['rb']), '--layers-list', csv(suite['layers']),
                       '--mcs-list', csv(suite['mcs']), '--mcs-table', str(data['mcs_table']),
                       '--out-dir', str(out), '--latency-limit-us', str(suite['qos_us']),
                       '--idle-mode', 'per-point']
            if args.platform == 'gpu':
                command += ['--bench-script', str(args.bench_script.resolve()), '--python-bin', args.python,
                            '--num-symbols', str(suite['symbols']), '--num-iterations', str(iters),
                            '--batches', str(c), '--num-streams', '1', '--k-mode', 'auto',
                            '--mode', 'both', '--lat-samples', str(args.samples), '--lat-discard', '50',
                            '--latency-clock', 'both', '--latency-source', args.latency_source,
                            '--gpu', str(args.gpu), '--prefer-net-energy', '--idle-per-batch-sec', '30']
                if args.include_h2d:
                    command += ['--include-h2d']
            else:
                command += ['--benchmark', str(args.benchmark.resolve()), '--symbols', str(suite['symbols']),
                            '--iters', str(iters), '--cells-list', str(c), '--reps', str(args.samples),
                            '--latency-metric', 'p99_999', '--energy-metric', 'pkg',
                            '--energy-scope', args.energy_scope, '--cpus', args.cpus,
                            '--threads', str(args.threads), '--decoder', args.decoder,
                            '--dematcher', args.decoder, '--idle-seconds', '30']
            jobs.append({'iterations': iters, 'aggregation': c, 'out_dir': str(out), 'argv': command})
    return {'schema_version': 1, 'platform': args.platform, 'suite': args.suite,
            'config_sha256': hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            'workload_config': suite, 'selected_aggregation': degrees,
            'nominal_grid_points': len(suite['rb'])*len(suite['mcs'])*len(suite['layers'])*len(jobs),
            'notes': ['Counts describe the requested grid, not certified valid transport blocks.',
                      'Analyze trace rows, not the legacy MCS-selected summary.',
                      'GPU batch is the legacy adapter mapping for concurrent TB instances; streams stay at one.',
                      'H2D selection affects GPU latency only in the legacy benchmark; energy uses its separate throughput loop.',
                      'Clock locking and machine isolation must be configured separately.',
                      'Arm external power acquisition is not implemented by this launcher.'],
            'jobs': jobs}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--platform', required=True, choices=['cpu','gpu'])
    p.add_argument('--suite', required=True, choices=['urllc','embb'])
    p.add_argument('--scan-script', required=True, type=Path)
    p.add_argument('--bench-script', type=Path)
    p.add_argument('--benchmark', type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--aggregation', help='Optional subset, e.g. 1,2,4,8,16,32; default is every integer in the table range')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--samples', type=int, default=300000)
    p.add_argument('--latency-source', choices=['host_wall','gpu_event'])
    p.add_argument('--include-h2d', action='store_true')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--cpus')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--energy-scope', choices=['sum','socket0','socket1'])
    p.add_argument('--decoder', choices=['generic','avx2','avx512','neon'], default='avx512')
    p.add_argument('--execute', action='store_true')
    return p


def main():
    args = parser().parse_args()
    plan = make_plan(args)
    if args.threads < 1:
        raise ValueError('threads must be positive')
    if args.out.exists():
        raise FileExistsError('Choose a new output directory; existing data are never overwritten')
    if args.execute:
        for path in [args.scan_script, args.bench_script if args.platform == 'gpu' else args.benchmark]:
            if not path.is_file():
                raise FileNotFoundError(path)
    args.out.mkdir(parents=True)
    (args.out/'plan.json').write_text(json.dumps(plan, indent=2),encoding='utf-8')
    print(f"{len(plan['jobs'])} jobs, {plan['nominal_grid_points']} requested grid points. execute={args.execute}")
    if args.execute:
        with (args.out/'execution.jsonl').open('x',encoding='utf-8') as log:
            for job in plan['jobs']:
                completed = subprocess.run(job['argv'])
                log.write(json.dumps({'iterations':job['iterations'],'aggregation':job['aggregation'],'returncode':completed.returncode})+'\n')
                log.flush()
                if completed.returncode:
                    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()

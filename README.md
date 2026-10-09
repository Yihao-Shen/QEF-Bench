# QEF-Bench

Source code and experiment tooling for studying RAN compute energy under latency
constraints. This initial source release preserves CPU/GPU LDPC measurements and
Arm PHY module benchmarks, with explicit workload plans for URLLC and eMBB.

## Contents

| Directory | Contents |
| --- | --- |
| `benchmarks/cpu` | OCUDU LDPC timing/energy C++ sources and scan driver |
| `benchmarks/gpu` | pyAerial LDPC latency/energy benchmark and scan drivers |
| `benchmarks/arm` | OCUDU PHY source overlay and module commands |
| `configs` | Workload table for new measurements |
| `tools` | Workload planner and loss-preserving GPU trace importer |

Start with [platform setup](docs/setup.md) and the
[measurement protocol](docs/measurement-protocol.md).
The planner and importer need Python 3.10+ with no third-party Python packages.
Hardware backends have separate dependencies.

## Release status

This is a source release, not a claim of independently reproduced paper results.
Planner/importer unit tests and Python syntax checks are provided. Hardware
experiments and C++/CUDA builds were not rerun for this release. Six historical trace sets are available in [the data archive](data/historical).
Their replay computes per-dataset coverage with explicit boundaries; it does not
reproduce all final paper figures. Arm power acquisition is not bundled.
The importer produces intermediate measurement records.

## Replay archived measurements

The [historical archive](data/historical) includes 468 CPU/GPU configuration rows,
metadata, integrity hashes and a hardware-free coverage replay:

```sh
python tools/replay_historical.py --out results/historical-coverage.json
```

See the archive README for selection, measurement units and known differences
from the current workload table.

## Audit a legacy GPU run

Requires Python 3.10+ and the original `gpu_tbs_grid_trace.csv` and
`gpu_tbs_grid_summary.json` from the same run:

```sh
python tools/import_gpu_trace.py --trace /path/to/gpu_tbs_grid_trace.csv --metadata /path/to/gpu_tbs_grid_summary.json --out /path/to/audit
```

The importer retains every workload row, including failures. It does not use the
legacy `best_points` selection, which can remove MCS configurations. Energy is
preserved as reported in nJ/bit; iteration-normalized energy is a separate field.
No missing measurements are silently replaced by zero and negative net energy
is flagged. Latency and energy boundaries remain separately identified.

Outputs are `measurements.jsonl` and `audit.json`. They are an intermediate
format, not a claim that the input reproduces the paper's workload suite or QEF.
Batch and stream counts are preserved separately and are not assumed to be
equivalent to cells or aggregation degree without adapter-specific validation.

## Run the paper workload table

`configs/paper-workloads.json` defines URLLC (4 symbols, RB 6/12/20/25/37/50,
MCS 0–9, one layer, 2/5 iterations, 125 us, aggregation 1–4) and eMBB
(12 symbols, RB 75/100/135/170/200/273, MCS 10–28, layers 1/2/4,
5/10 iterations, 1000 us, aggregation 1–32). MCS table 1 follows the
existing experiment setup. All integer aggregation degrees are included
by default. Optional subsets are recorded explicitly in the plan.

Use the preserved original scan scripts; this launcher supplies their
parameters explicitly instead of changing historical defaults and results.
Example on the GPU host, from the Aerial source directory:

```sh
python /path/to/QEF-Bench/tools/run_paper_suite.py \
  --platform gpu --suite urllc \
  --scan-script /path/to/QEF-Bench/benchmarks/gpu/scan_ldpc_gpu_real_tbs.py \
  --bench-script /path/to/QEF-Bench/benchmarks/gpu/ldpc_bench_3.4_tbs_latency.py \
  --latency-source host_wall --include-h2d \
  --gpu 0 --aggregation 1,2,4 --out /path/to/new-plan
```

This example explicitly includes H2D in latency. Choose `gpu_event` and omit
`--include-h2d` for the device timing mode instead; neither mode is silently
inferred from the paper. The legacy energy measurement remains a separate
throughput measurement without per-call H2D copies.

CPU example, from the OCUDU source directory:

```sh
python /path/to/QEF-Bench/tools/run_paper_suite.py \
  --platform cpu --suite embb \
  --scan-script /path/to/QEF-Bench/benchmarks/cpu/scan_ldpc_cpu_real_tbs.py \
  --benchmark build_user/tests/benchmarks/phy/upper/channel_coding/ldpc/ldpc_decoder_energy_benchmark_cb_cells \
  --cpus 0 --threads 1 --energy-scope socket0 --decoder avx512 \
  --aggregation 1 --out /path/to/new-cpu-plan
```

Both commands only write `plan.json`. To measure, select an unused output
directory and add `--execute`. Set CPU affinity, package selection, GPU index,
clock policy and permissions for the actual machine. The launcher never uses
sudo or changes device clocks automatically. Each iteration/aggregation pair
gets its own output directory. Hardware execution has not been validated yet.
The full nominal grids contain 480 URLLC and 21888 eMBB operating points per
platform; actual transport-block validity must be checked by the adapters.
Use full trace files for analysis; the underlying legacy summary can still
select among MCS values and is not the complete QEF workload set.

Tests: `python -B -m unittest discover -s tools -p 'test_*.py'`.

## License and citation

Original contributions use [Apache-2.0](LICENSE). OCUDU-derived sources retain
BSD-3-Clause-Open-MPI; see [third-party notices](THIRD_PARTY_NOTICES.md).
Use [CITATION.cff](CITATION.cff) for the software citation. Contributions and
measurement corrections are welcome through issues and pull requests.

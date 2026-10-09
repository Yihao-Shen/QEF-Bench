# Historical measurement archive

This archive contains six complete per-configuration traces (468 rows) referenced
by the original CPU/GPU four-case plotting scripts in the author's plot workspace.
Selection follows those input mappings, not energy or latency rankings. The two
CPU eMBB inputs were derived summary files; they are excluded from this first
archive pending a separate provenance check. Other repeated runs and exploratory
microbenchmarks are not included. No claim of a complete experiment archive is made.

## Included runs

| Dataset | Rows | Iterations | Layers | H2D in latency |
| --- | ---: | ---: | ---: | --- |
| cpu-urllc-2iter | 60 | 2 | 1 | n/a |
| cpu-urllc-5iter | 60 | 5 | 1 | n/a |
| gpu-urllc-2iter | 60 | 2 | 1 | yes |
| gpu-urllc-5iter | 60 | 5 | 1 | no |
| gpu-embb-5iter-l4 | 114 | 5 | 4 | yes |
| gpu-embb-10iter-l4 | 114 | 10 | 4 | yes |

Each folder has a trace CSV and sanitized run metadata. These are scanner-level
measurement summaries, not the underlying individual latency samples or power
sensor time series. Exact device model, clock state and source revision are not
independently recorded in these summary files. Folder names alone are insufficient
to certify those facts. The four GPU datasets must not be treated as a controlled
iteration comparison because their sample counts and H2D settings differ.

Historical URLLC uses RB 6/12/18/25/37/50, MCS 0-9, four symbols, one cell/batch.
Historical eMBB uses RB 75/100/135/170/200/273, MCS 10-28, twelve symbols,
four layers and one batch. These are not the full current paper workload table;
in particular its URLLC 20-RB point is absent here. Original config thresholds
are retained, even where they differ from the analysis deadlines below.

## Recompute without hardware

From the repository root, using Python 3.10+:

```sh
python tools/replay_historical.py --out results/historical-coverage.json
python -B -m unittest discover -s tools -p 'test_*.py'
```

The replay checks SHA-256 hashes, row counts and configuration uniqueness. It
uses **p99.999 latency** for all six datasets, with 125 us for URLLC and 1000 us
for eMBB. CPU energy is exactly `net_pkg_j_per_bit * 1e9`; GPU energy is exactly
`net_nj_per_bit`. No division by iteration count, interpolation, percentile
fallback or gross/net substitution is applied. Budget values are illustrative
analysis thresholds in nJ/bit, not hardware specifications or calibrated targets.
Use `--budgets 100,1000,10000` to choose other thresholds.

Coverage denominator is the Cartesian workload grid recorded in each run's config,
not just successful rows. Missing, failed, nonfinite and nonpositive measurements
cannot earn feasibility credit. Duplicate or unexpected configurations stop the
analysis. One eMBB 10-iteration GPU point reports `CUDA_ERROR_ILLEGAL_ADDRESS`;
its failed row and sanitized diagnostic remain included. A scanner success flag
is not independent verification of bit-correct decoding.

This replay intentionally does not recreate the original plot transformations:
some plotting code interpolated missing values, used p90 on CPU or alternate GPU
percentiles, and mixed fallback energy fields. Those operations are not presented
as original measurements here. The output is historical per-dataset coverage,
not reproduction of final paper figures or a matched CPU-versus-GPU comparison.
See ../../docs/measurement-protocol.md for measurement-boundary details.

## Provenance, privacy and license

`manifest.json` records original filenames, source hashes and public file hashes.
All measurement cell strings, row order and success flags are unchanged. CPU
executable paths and absolute diagnostic paths are shortened; config path fields
are reduced to basenames. Config and run metadata are retained; legacy selected
`best_points` are not exported. The original local files are untouched.

The authors' released data are provided under the repository's Apache-2.0 license.
Cite QEF-Bench using the root CITATION.cff and identify the dataset IDs and commit
used in your analysis. No new hardware experiment was performed for this release.

# Measurement and interpretation

The workload table describes requested configurations. Save the generated plan,
dependency revisions, actual valid transport blocks, affinity, clock policy and
measurement settings with each new run. Do not relabel historical runs as the new
table. CPU cells and GPU batches are backend mappings; equivalence of processing
scope must be checked before comparison.

## Latency and energy

CPU scans request the package energy metric and p99.999 latency. Check that
package selection and permissions match the system. Record sample counts and
interpret extreme percentiles accordingly.

GPU host-wall and CUDA-event latency are distinct boundaries. H2D inclusion applies
to the latency loop only. Energy comes from a separate throughput loop without
per-call H2D copies. The original `auto` mode prefers raw host-wall statistics;
trimmed statistics are separate diagnostics. New plans require an explicit clock
source. GPU energy uses the benchmark's bit count (`calls * batch * K`), which is
not an independent count of successfully decoded payload bits.

The legacy benchmark can clamp idle-subtracted energy to zero. Keep gross energy,
idle baseline and net energy when available; zero is not evidence of free compute.
Iteration-normalized energy is a separate quantity from energy per bit. Arm module
sources provide timing; external power acquisition is outside this release.

## Coverage analysis

Use full trace rows, including failures. Legacy best-point summaries select among
MCS configurations and cannot be used as the full QEF denominator. Define the
configuration universe and aggregation policy before computing the fraction that
meets both the deadline and a stated energy budget. Missing/invalid measurements
must be reported explicitly. `import_gpu_trace.py` preserves rows and flags issues;
it does not certify successful decoding or compute the final coverage metric.

Comparisons across platforms require aligned processing scope, payload accounting
and energy boundaries. Device rated power is not measured energy per bit.

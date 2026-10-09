# Platform setup

## CPU / OCUDU

Obtain OCUDU from https://gitlab.com/ocudu/ocudu and install dependencies according
to its own build instructions. The recovered CPU sources came from a tree based
on commit `189e83bf943fafe876e8bec9b1e5f6cc2e4bf0df` with local modifications.
Use a separate checkout and preserve your own changes.

Copy the two C++ files and `qef-targets.cmake` from `benchmarks/cpu` to
`tests/benchmarks/phy/upper/channel_coding/ldpc/` inside OCUDU. Add
`include(${CMAKE_CURRENT_LIST_DIR}/qef-targets.cmake)` to that directory's
`CMakeLists.txt`. Enable OCUDU benchmark builds using its documented CMake
configuration, then build the `ldpc_decoder_energy_benchmark_cb_cells` and
`ldpc_decoder_single_cb_scan` targets. This overlay has not been rebuilt in this
release; upstream API changes or other historical local changes may require adaptation.

The scan driver invokes the built executable. Configure Linux CPU affinity,
RAPL/powercap access and the actual energy package. Run the executable and scanner
with `--help` for available options. Do not assume socket0 or avx512 fits every CPU.

## GPU / pyAerial

Install NVIDIA Aerial CUDA-Accelerated RAN with working pyAerial, a compatible
CUDA driver/runtime, NumPy, CuPy and `cuda.bindings.runtime`. Follow the upstream
SDK instructions for compatible versions; these dependencies are not bundled.
The historical source environment used Aerial commit
`3bf76a43dceb493b00f2ee75fdfbb87038eab7c6`.

Run the scripts inside that configured environment. The primary pair is
`scan_ldpc_gpu_real_tbs.py` and `ldpc_bench_3.4_tbs_latency.py`.
`ldpc_scan_k_iters_energy_gpu.py` is an additional preserved exploratory scanner.
Inspect its own arguments before using it; the paper-suite launcher uses the
primary pair only. Device permissions and NVML power sampling must work on the
selected GPU. No clock changes or hardware execution are part of installation.

## Arm / portable modules

`benchmarks/arm/source` is an OCUDU source overlay, not a standalone project.
Merge the matching source files into a separate compatible checkout and merge
its LDPC CMake additions into the existing CMake file; do not replace unrelated
targets. Build OCUDU benchmarks for your Arm toolchain. See
[the module commands](../benchmarks/arm/README.md). No board binaries or external
power-meter integration are shipped. The exact base revision is unknown.

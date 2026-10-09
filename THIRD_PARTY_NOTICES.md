# Licensing and provenance

Original QEF-Bench Python tooling, experiment drivers and documentation are
licensed under Apache-2.0. Copyright 2026 QEF-Bench contributors.

OCUDU-derived C++ sources and CMake files in `benchmarks/cpu` and
`benchmarks/arm/source` retain their original notices and are distributed under
the BSD-3-Clause-Open-MPI license in `third_party/licenses/OCUDU.txt`.
They include local benchmark modifications; this is not an unmodified upstream release.
The recovered CPU source tree identifies upstream commit
`189e83bf943fafe876e8bec9b1e5f6cc2e4bf0df` of
https://gitlab.com/ocudu/ocudu. The Arm overlay's exact base commit is unrecorded;
porting may be required. These license terms take precedence over the root license.

GPU scripts use NVIDIA Aerial pyAerial. The source environment identifies
Aerial commit `3bf76a43dceb493b00f2ee75fdfbb87038eab7c6`.
The upstream license and dependency notice are preserved in
`third_party/licenses/Aerial.txt`. Aerial, CUDA, containers and their dependencies
must be obtained separately under their applicable terms. No SDK or container
binaries are redistributed. Existing per-file notices are retained.

`source-manifest.json` records SHA-256 hashes of the preserved benchmark files.
No machine credentials, private machine inventory or historical experiment data
are included.

# Arm / portable PHY source overlay

Only source files are distributed. Build in a compatible OCUDU checkout and copy
the resulting binaries to the target board. External power acquisition is not
included. The exact original upstream revision was not recorded.

## Modified or Added Benchmarks

- `dft_processor_benchmark`: added `-F`, `-N`, and `-B` filters for fixed FFT implementation, size, and batch size.
- `channel_equalizer_benchmark`: added `-n`, `-r`, and `-l` filters for PRB, RX ports, and TX layers.
- `ldpc_decoder_benchmark`: added `-B` and `-K` filters for base graph and codeblock length.
- `ldpc_rate_dematcher_benchmark`: new standalone LDPC rate-dematching benchmark.
- `modulation_chain_benchmark`: added `-m`, `-Q`, and `-N` filters for soft demodulation-only testing.
- `pusch_processor_benchmark`: added `-l`, `-i`, and `-S` for layer selection, LDPC iteration count, and disabling early stop.

## Running

The following are preserved historical command examples. Set `CORE` to the
appropriate core on your own board; implementation support depends on your build.
Run from the directory containing the built executables.

## Recommended RX Module Commands

FFT, 512-point and 4096-point:

```bash
taskset -c $CORE ./dft_processor_benchmark -R 10000 -F fftw -N 512 -B 500
taskset -c $CORE ./dft_processor_benchmark -R 10000 -F fftw -N 4096 -B 100
```

ZF equalizer:

```bash
taskset -c $CORE ./channel_equalizer_benchmark -R 10000 -T ZF -l 1 -r 1 -n 275
taskset -c $CORE ./channel_equalizer_benchmark -R 10000 -T ZF -l 2 -r 2 -n 275
taskset -c $CORE ./channel_equalizer_benchmark -R 10000 -T ZF -l 4 -r 4 -n 275
```

MMSE equalizer:

```bash
taskset -c $CORE ./channel_equalizer_benchmark -R 10000 -T MMSE -l 1 -r 1 -n 275
taskset -c $CORE ./channel_equalizer_benchmark -R 10000 -T MMSE -l 2 -r 2 -n 275
taskset -c $CORE ./channel_equalizer_benchmark -R 10000 -T MMSE -l 4 -r 4 -n 275
```

Soft demodulation:

```bash
taskset -c $CORE ./modulation_chain_benchmark -R 10000 -m demapper -Q 256QAM -N 45864
taskset -c $CORE ./modulation_chain_benchmark -R 10000 -m demapper -Q 256QAM -N 91728
```

LDPC rate dematching:

```bash
taskset -c $CORE ./ldpc_rate_dematcher_benchmark -R 10000 -T neon -B 1 -L 384 -E 9216 -Q 256QAM -v 0
taskset -c $CORE ./ldpc_rate_dematcher_benchmark -R 10000 -T neon -B 1 -L 384 -E 25344 -Q 256QAM -v 0
```

LDPC decoding:

```bash
taskset -c $CORE ./ldpc_decoder_benchmark -R 10000 -T neon -I 5 -B 1 -L 384 -K 9216
taskset -c $CORE ./ldpc_decoder_benchmark -R 10000 -T neon -I 5 -B 1 -L 384 -K 25344
```

PUSCH RX chain, fixed 5 LDPC iterations and early-stop disabled:

```bash
taskset -c $CORE ./pusch_processor_benchmark -m throughput_total -R 20 -B 5 -T 1 -i 5 -S -P scs15_5MHz_qpsk_rv0_1port_1layer -l 1
taskset -c $CORE ./pusch_processor_benchmark -m throughput_total -R 10 -B 2 -T 1 -i 5 -S -P scs30_100MHz_256qam_rv0_4port_nlayer -l 1
taskset -c $CORE ./pusch_processor_benchmark -m throughput_total -R 10 -B 2 -T 1 -i 5 -S -P scs30_100MHz_256qam_rv0_4port_nlayer -l 2
```

## x86 Notes

Build these sources inside the same OCUDU tree on x86. Use the same command shapes, but replace ARM-specific implementation selectors:

- LDPC decoder: use `-T avx2` or `-T avx512`.
- LDPC rate dematcher: use `-T avx2` or `-T avx512`.
- Keep `taskset -c $CORE` and `-T 1` for single-core comparisons.


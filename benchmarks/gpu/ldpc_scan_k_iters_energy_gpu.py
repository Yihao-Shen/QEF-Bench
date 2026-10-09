#!/usr/bin/env python3
# ldpc_scan_k_iters_energy_gpu.py
#
# Scan Aerial LDPC decoder on GPU over:
#   - BG in {1,2}
#   - Zc set (=> K)
#   - max iterations
#   - code rate (fixed or sweep)
#
# Measure per-point:
#   - decode latency stats (CUDA event based)
#   - throughput (Mbps)
#   - GPU energy (J), optional idle-baseline net energy (J)
#   - energy efficiency (J/bit)

import argparse
import bisect
import csv
import math
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cupy as cp
import numpy as np
from aerial.phy5g.ldpc import LdpcDecoder


ZC_FULL_UNION = sorted(
    set(
        [2, 4, 8, 16, 32, 64, 128, 256]
        + [3, 6, 12, 24, 48, 96, 192, 384]
        + [5, 10, 20, 40, 80, 160, 320]
        + [7, 14, 28, 56, 112, 224]
        + [9, 18, 36, 72, 144, 288]
        + [11, 22, 44, 88, 176, 352]
        + [13, 26, 52, 104, 208]
        + [15, 30, 60, 120, 240]
    )
)

ZC_POW2 = [2, 4, 8, 16, 32, 64, 128, 256, 384]


def parse_int_list(s: str) -> List[int]:
    xs: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            xs.append(int(part))
    if not xs:
        raise ValueError("Empty int list")
    return xs


def parse_float_list(s: str) -> List[float]:
    xs: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            xs.append(float(part))
    if not xs:
        raise ValueError("Empty float list")
    return xs


def bg_cols(bg: int) -> int:
    if bg == 1:
        return 22
    if bg == 2:
        return 10
    raise ValueError("BG must be 1 or 2")


def get_nshort(bg: int) -> int:
    if bg == 1:
        return 66
    if bg == 2:
        return 50
    raise ValueError("BG must be 1 or 2")


def zc_to_k(bg: int, zc: int) -> int:
    return bg_cols(bg) * int(zc)


def choose_input_length(bg: int, zc: int, k_bits: int, code_rate: float) -> int:
    min_input = int(k_bits + 2 * zc)
    max_input = int(get_nshort(bg) * zc)
    by_rate = int(math.ceil(k_bits / code_rate)) if code_rate > 0 else min_input
    e = max(min_input, by_rate)
    e = min(e, max_input)
    return int(e)


def choose_input_length_fixed(bg: int, zc: int, k_bits: int, rm_len_fixed: int) -> int:
    min_input = int(k_bits + 2 * zc)
    max_input = int(get_nshort(bg) * zc)
    e = max(min_input, int(rm_len_fixed))
    e = min(e, max_input)
    return int(e)


def percentile_sorted(xs_sorted: List[float], p: float) -> float:
    if not xs_sorted:
        return float("nan")
    k = int(round((p / 100.0) * (len(xs_sorted) - 1)))
    k = max(0, min(k, len(xs_sorted) - 1))
    return float(xs_sorted[k])


def integrate_energy_with_endpoints_j(samples: List[tuple[int, float]], t0_ns: int, t1_ns: int) -> float:
    if t1_ns <= t0_ns:
        return 0.0
    if not samples:
        return 0.0

    pts = sorted((int(t), float(p)) for (t, p) in samples)
    if len(pts) == 1:
        return max(0.0, pts[0][1]) * ((t1_ns - t0_ns) * 1e-9)

    ts = [t for t, _ in pts]

    def _linear_power(tq: int, ta: int, pa: float, tb: int, pb: float) -> float:
        if tb == ta:
            return max(0.0, pa)
        ratio = (tq - ta) / (tb - ta)
        return max(0.0, pa + ratio * (pb - pa))

    def power_at_ns(tq: int) -> float:
        if tq <= pts[0][0]:
            ta, pa = pts[0]
            tb, pb = pts[1]
            return _linear_power(tq, ta, pa, tb, pb)
        if tq >= pts[-1][0]:
            ta, pa = pts[-2]
            tb, pb = pts[-1]
            return _linear_power(tq, ta, pa, tb, pb)

        i = bisect.bisect_right(ts, tq)
        ta, pa = pts[i - 1]
        tb, pb = pts[i]
        return _linear_power(tq, ta, pa, tb, pb)

    inside = [(t, p) for (t, p) in pts if t0_ns < t < t1_ns]
    xs = [(t0_ns, power_at_ns(t0_ns)), *inside, (t1_ns, power_at_ns(t1_ns))]
    if len(xs) < 2:
        return 0.0

    e = 0.0
    for (ta, pa), (tb, pb) in zip(xs[:-1], xs[1:]):
        dt = (tb - ta) * 1e-9
        if dt > 0:
            e += 0.5 * (pa + pb) * dt
    return e


@dataclass
class LatencyStats:
    n: int
    mean_us: float
    min_us: float
    p1_us: float
    p5_us: float
    p20_us: float
    p50_us: float
    p95_us: float
    p99_us: float
    p99_999_us: float
    max_us: float


class GpuEnergyMeter:
    def __init__(self, gpu_index: int, sample_interval_ms: float = 10.0):
        self.gpu_index = int(gpu_index)
        self.sample_interval_s = max(0.001, float(sample_interval_ms) / 1000.0)

        self.available = False
        self.method = "none"
        self._nvml = None
        self._handle = None

        self._running = False
        self._samples: List[tuple[int, float]] = []
        self._window_t0_ns = 0
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()

        self._counter_start_mj = 0
        self._has_total_energy_counter = False

        self._init_backend()

    def _init_backend(self) -> None:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)

            # Prefer HW total energy counter when available.
            try:
                _ = pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
                self._has_total_energy_counter = True
                self.available = True
                self.method = "nvml_total_energy"
                return
            except Exception:
                pass

            # Fallback to NVML power sampling integration.
            _ = pynvml.nvmlDeviceGetPowerUsage(self._handle)
            self.available = True
            self.method = "nvml_power_integral"
            return
        except Exception:
            pass

        # Last fallback: nvidia-smi power polling integration.
        p = self._read_power_w_nvidia_smi()
        if p is not None:
            self.available = True
            self.method = "nvidia_smi_power_integral"

    def _read_power_w_nvml(self) -> Optional[float]:
        if self._nvml is None or self._handle is None:
            return None
        try:
            mw = self._nvml.nvmlDeviceGetPowerUsage(self._handle)
            return float(mw) / 1000.0
        except Exception:
            return None

    def _read_power_w_nvidia_smi(self) -> Optional[float]:
        cmd = [
            "nvidia-smi",
            "-i",
            str(self.gpu_index),
            "--query-gpu=power.draw",
            "--format=csv,noheader,nounits",
        ]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
            if not out:
                return None
            line = out.splitlines()[0].strip()
            return float(line)
        except Exception:
            return None

    def _read_power_w(self) -> Optional[float]:
        if self.method in ("nvml_total_energy", "nvml_power_integral"):
            p = self._read_power_w_nvml()
            if p is not None:
                return p
        return self._read_power_w_nvidia_smi()

    def _sampler_loop(self) -> None:
        while not self._stop_evt.is_set():
            now_ns = time.perf_counter_ns()
            p_w = self._read_power_w()
            if p_w is not None:
                with self._lock:
                    self._samples.append((now_ns, float(p_w)))
            self._stop_evt.wait(self.sample_interval_s)

    def start(self) -> None:
        if not self.available:
            return
        if self._running:
            return
        self._running = True

        if self.method == "nvml_total_energy" and self._nvml is not None and self._handle is not None:
            self._counter_start_mj = int(self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle))
            return

        self._window_t0_ns = time.perf_counter_ns()
        with self._lock:
            self._samples = []
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._sampler_loop, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        if not self.available or not self._running:
            return 0.0

        self._running = False

        if self.method == "nvml_total_energy" and self._nvml is not None and self._handle is not None:
            end_mj = int(self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle))
            delta_mj = max(0, end_mj - self._counter_start_mj)
            return float(delta_mj) * 1e-3

        t1_ns = time.perf_counter_ns()
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sample_interval_s * 4.0))

        with self._lock:
            samples = list(self._samples)
        return float(integrate_energy_with_endpoints_j(samples, self._window_t0_ns, t1_ns))

    def measure_idle_power_w(self, idle_seconds: int) -> float:
        if not self.available or idle_seconds <= 0:
            return 0.0
        self.start()
        time.sleep(float(idle_seconds))
        energy_j = self.stop()
        return energy_j / float(idle_seconds) if idle_seconds > 0 else 0.0

    def close(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass


def compute_latency_stats_us(lat_us: List[float]) -> LatencyStats:
    xs = sorted(lat_us)
    n = len(xs)
    if n == 0:
        return LatencyStats(
            0,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )

    return LatencyStats(
        n=n,
        mean_us=float(np.mean(xs)),
        min_us=float(xs[0]),
        p1_us=percentile_sorted(xs, 1),
        p5_us=percentile_sorted(xs, 5),
        p20_us=percentile_sorted(xs, 20),
        p50_us=percentile_sorted(xs, 50),
        p95_us=percentile_sorted(xs, 95),
        p99_us=percentile_sorted(xs, 99),
        p99_999_us=percentile_sorted(xs, 99.999),
        max_us=float(xs[-1]),
    )


def measure_one_point(
    decoder: LdpcDecoder,
    stream: cp.cuda.Stream,
    llr: cp.ndarray,
    k_bits: int,
    code_rate: float,
    rv: int,
    rm_len: int,
    batch: int,
    warmup: int,
    reps: int,
    energy_meter: GpuEnergyMeter,
) -> Dict[str, Any]:
    input_llrs = [llr]
    tb_sizes = [int(k_bits)] * batch
    code_rates = [float(code_rate)] * batch
    redundancy_versions = [int(rv)] * batch
    rate_match_lengths = [int(rm_len)] * batch

    for _ in range(int(warmup)):
        decoder.decode(
            input_llrs=input_llrs,
            tb_sizes=tb_sizes,
            code_rates=code_rates,
            redundancy_versions=redundancy_versions,
            rate_match_lengths=rate_match_lengths,
        )
    stream.synchronize()

    ev_start = cp.cuda.Event()
    ev_stop = cp.cuda.Event()
    lat_us: List[float] = []

    energy_meter.start()
    t0_all = time.perf_counter()

    for _ in range(int(reps)):
        ev_start.record(stream)
        decoder.decode(
            input_llrs=input_llrs,
            tb_sizes=tb_sizes,
            code_rates=code_rates,
            redundancy_versions=redundancy_versions,
            rate_match_lengths=rate_match_lengths,
        )
        ev_stop.record(stream)
        ev_stop.synchronize()
        ms = cp.cuda.get_elapsed_time(ev_start, ev_stop)
        lat_us.append(float(ms) * 1000.0)

    stream.synchronize()
    wall_s = float(time.perf_counter() - t0_all)
    run_energy_j = float(energy_meter.stop())

    st = compute_latency_stats_us(lat_us)
    return {
        "n": st.n,
        "mean_us": st.mean_us,
        "min_us": st.min_us,
        "p1_us": st.p1_us,
        "p5_us": st.p5_us,
        "p20_us": st.p20_us,
        "p50_us": st.p50_us,
        "p95_us": st.p95_us,
        "p99_us": st.p99_us,
        "p99_999_us": st.p99_999_us,
        "max_us": st.max_us,
        "wall_s": wall_s,
        "run_gpu_j": run_energy_j,
    }


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU LDPC scan: K x iters with latency and energy")

    p.add_argument("--bg", type=str, default="both", choices=["1", "2", "both"], help="Base graph(s) to scan")
    p.add_argument("--zc-mode", type=str, default="full", choices=["full", "pow2", "custom"], help="Zc mode")
    p.add_argument("--zc-list", type=str, default="", help="Comma-separated Zc list if --zc-mode=custom")
    p.add_argument("--iters-list", type=str, default="1,2,3,4,5,6,8,10,12", help="Comma-separated iters list")

    p.add_argument("--rate-control-mode", type=str, default="fixed", choices=["fixed", "sweep"])
    p.add_argument("--rate-fixed", type=float, default=0.50)
    p.add_argument(
        "--rate-list",
        type=str,
        default="0.1171875,0.25,0.33203125,0.40,0.50,0.60,0.650390625,0.66,0.75,0.83,0.92578125",
    )

    p.add_argument("--rm-len-mode", type=str, default="from_rate", choices=["from_rate", "fixed"])
    p.add_argument("--rm-len-fixed", type=int, default=28800)

    p.add_argument("--rv", type=int, default=0)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--batch-list", type=str, default="", help="Comma-separated batch list; overrides --batch when set")
    p.add_argument("--throughput-mode", type=int, default=1)
    p.add_argument("--gpu", type=int, default=0)

    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=30)

    p.add_argument("--idle", type=int, default=0, help="Idle seconds for baseline power measurement")
    p.add_argument("--energy-sample-ms", type=float, default=10.0, help="Sampling interval if counter unavailable")
    p.add_argument(
        "--energy-eff-mode",
        type=str,
        default="run",
        choices=["run", "raw_net", "net"],
        help="Energy-efficiency metric: run=total run energy, raw_net=run-idle*wall, net=max(raw_net,0)",
    )

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-csv", type=str, default="scan_k_iters_gpu_latency_energy.csv")
    p.add_argument("--silent", action="store_true")

    return p.parse_args()


def main() -> None:
    args = get_args()

    if args.reps <= 0:
        raise ValueError("--reps must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.rate_fixed <= 0.0 or args.rate_fixed > 1.0:
        raise ValueError("--rate-fixed must be in (0,1]")
    if args.rm_len_fixed <= 0:
        raise ValueError("--rm-len-fixed must be > 0")

    if args.zc_mode == "full":
        zc_values = list(ZC_FULL_UNION)
    elif args.zc_mode == "pow2":
        zc_values = list(ZC_POW2)
    else:
        zc_values = parse_int_list(args.zc_list)

    iters_values = parse_int_list(args.iters_list)

    if args.rate_control_mode == "fixed":
        rate_values = [float(args.rate_fixed)]
    else:
        rate_values = parse_float_list(args.rate_list)

    bgs = [2, 1] if args.bg == "both" else [int(args.bg)]

    if not zc_values or not iters_values or not rate_values:
        raise ValueError("zc/iters/rate list cannot be empty")

    for r in rate_values:
        if r <= 0.0 or r > 1.0:
            raise ValueError("All code rates must be in (0,1]")

    cp.random.seed(int(args.seed))
    cp.cuda.Device(int(args.gpu)).use()
    stream = cp.cuda.Stream(non_blocking=False)

    throughput_mode = bool(int(args.throughput_mode))
    if args.batch_list.strip():
        batch_values = parse_int_list(args.batch_list)
    else:
        batch_values = [int(args.batch)]

    if any(b <= 0 for b in batch_values):
        raise ValueError("All batch sizes must be > 0")

    energy_meter = GpuEnergyMeter(gpu_index=int(args.gpu), sample_interval_ms=float(args.energy_sample_ms))
    idle_gpu_w = energy_meter.measure_idle_power_w(int(args.idle)) if energy_meter.available else 0.0

    if args.out_csv:
        out_dir = os.path.dirname(os.path.abspath(args.out_csv))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    total_pts = len(bgs) * len(zc_values) * len(iters_values) * len(rate_values) * len(batch_values)
    idx = 0

    if not args.silent:
        print("==== GPU LDPC Scan: K x iters (code_rate as control) ====", flush=True)
        print(
            f"BG count={len(bgs)} | Zc count={len(zc_values)} | iters count={len(iters_values)} | rates count={len(rate_values)}",
            flush=True,
        )
        print(f"gpu={args.gpu} | batch_count={len(batch_values)} | throughput_mode={throughput_mode}", flush=True)
        print(f"batches={batch_values}", flush=True)
        print(f"rm_len_mode={args.rm_len_mode}", flush=True)
        if energy_meter.available:
            print(
                f"energy_method={energy_meter.method} | idle_seconds={args.idle} | idle_gpu_w={idle_gpu_w:.6f}",
                flush=True,
            )
        else:
            print("energy_method=unavailable (install pynvml or ensure nvidia-smi is available)", flush=True)
        print("", flush=True)

    rows: List[Dict[str, Any]] = []

    llr_fixed_cache: Dict[tuple[int, int], cp.ndarray] = {}

    try:
        for bg in bgs:
            for zc in zc_values:
                k_bits = zc_to_k(bg, zc)

                for iters in iters_values:
                    decoder = LdpcDecoder(
                        cuda_stream=stream.ptr,
                        num_iterations=int(iters),
                        throughput_mode=throughput_mode,
                    )
                    for code_rate in rate_values:
                        for batch in batch_values:
                            idx += 1

                            if args.rm_len_mode == "fixed":
                                rm_len = choose_input_length_fixed(bg, zc, k_bits, int(args.rm_len_fixed))
                                cache_key = (int(rm_len), int(batch))
                                if cache_key not in llr_fixed_cache:
                                    with stream:
                                        llr_fixed_cache[cache_key] = cp.random.standard_normal((rm_len, batch), dtype=cp.float32)
                                    stream.synchronize()
                                llr = llr_fixed_cache[cache_key]
                            else:
                                rm_len = choose_input_length(bg, zc, k_bits, float(code_rate))
                                with stream:
                                    llr = cp.random.standard_normal((rm_len, batch), dtype=cp.float32)
                                stream.synchronize()

                            point = measure_one_point(
                                decoder=decoder,
                                stream=stream,
                                llr=llr,
                                k_bits=int(k_bits),
                                code_rate=float(code_rate),
                                rv=int(args.rv),
                                rm_len=int(rm_len),
                                batch=batch,
                                warmup=int(args.warmup),
                                reps=int(args.reps),
                                energy_meter=energy_meter,
                            )

                            wall_s = float(point["wall_s"])
                            run_energy_j = float(point["run_gpu_j"])

                            raw_net_gpu_j = run_energy_j - idle_gpu_w * wall_s
                            net_gpu_j = max(0.0, raw_net_gpu_j)

                            total_bits = int(k_bits) * int(point["n"]) * int(batch)
                            run_gpu_j_per_bit = run_energy_j / total_bits if total_bits > 0 else 0.0
                            raw_net_gpu_j_per_bit = raw_net_gpu_j / total_bits if total_bits > 0 else 0.0
                            net_gpu_j_per_bit = net_gpu_j / total_bits if total_bits > 0 else 5E-9

                            if args.energy_eff_mode == "run":
                                eff_j_per_bit = run_gpu_j_per_bit
                            elif args.energy_eff_mode == "raw_net":
                                eff_j_per_bit = raw_net_gpu_j_per_bit
                            else:
                                eff_j_per_bit = net_gpu_j_per_bit

                            p50_us = float(point["p50_us"])
                            mean_us = float(point["mean_us"])
                            thr_p50_mbps = (float(k_bits) * float(batch) / p50_us) if p50_us > 0.0 else 0.0
                            thr_mean_mbps = (float(k_bits) * float(batch) / mean_us) if mean_us > 0.0 else 0.0

                            row = {
                                "BG": int(bg),
                                "Zc": int(zc),
                                "K": int(k_bits),
                                "iters": int(iters),
                                "code_rate": float(code_rate),
                                "rm_len": int(rm_len),
                                "batch": int(batch),
                                "thr_mode": int(throughput_mode),
                                "n": int(point["n"]),
                                "p1_us": float(point["p1_us"]),
                                "p5_us": float(point["p5_us"]),
                                "p20_us": float(point["p20_us"]),
                                "p50_us": p50_us,
                                "p95_us": float(point["p95_us"]),
                                "p99_us": float(point["p99_us"]),
                                "p99_999_us": float(point["p99_999_us"]),
                                "mean_us": mean_us,
                                "min_us": float(point["min_us"]),
                                "max_us": float(point["max_us"]),
                                "wall_s": wall_s,
                                "success": int(point["n"]),
                                "avg_iters_success": float(iters),
                                "thr_p50_mbps": thr_p50_mbps,
                                "thr_mean_mbps": thr_mean_mbps,
                                "energy_available": int(1 if energy_meter.available else 0),
                                "energy_method": energy_meter.method,
                                "run_gpu_j": run_energy_j,
                                "idle_gpu_w": idle_gpu_w,
                                "run_gpu_j_per_bit": run_gpu_j_per_bit,
                                "raw_net_gpu_j": raw_net_gpu_j,
                                "net_gpu_j": net_gpu_j,
                                "raw_net_gpu_j_per_bit": raw_net_gpu_j_per_bit,
                                "net_gpu_j_per_bit": net_gpu_j_per_bit,
                                "eff_mode": args.energy_eff_mode,
                                "eff_j_per_bit": eff_j_per_bit,
                            }
                            rows.append(row)

                            if (not args.silent) and (idx == 1 or (idx % 50) == 0):
                                print(
                                    f"[{idx}/{total_pts}] BG{bg} Zc={zc} K={k_bits} iters={iters} batch={batch} rate={code_rate:.6f} "
                                    f"E={rm_len} -> min={float(point['min_us']):.2f}us p1={float(point['p1_us']):.2f}us p5={float(point['p5_us']):.2f}us "
                                    f"p20={float(point['p20_us']):.2f}us p50={p50_us:.2f}us p99.999={float(point['p99_999_us']):.2f}us thr={thr_p50_mbps:.2f}Mbps "
                                    f"run/raw/net Jbit={run_gpu_j_per_bit:.6e}/{raw_net_gpu_j_per_bit:.6e}/{net_gpu_j_per_bit:.6e} "
                                    f"eff[{args.energy_eff_mode}]={eff_j_per_bit:.6e}",
                                    flush=True,
                                )
    finally:
        energy_meter.close()

    if args.out_csv:
        fieldnames = [
            "BG",
            "Zc",
            "K",
            "iters",
            "code_rate",
            "rm_len",
            "batch",
            "thr_mode",
            "n",
            "p1_us",
            "p5_us",
            "p20_us",
            "p50_us",
            "p95_us",
            "p99_us",
            "p99_999_us",
            "mean_us",
            "min_us",
            "max_us",
            "wall_s",
            "success",
            "avg_iters_success",
            "thr_p50_mbps",
            "thr_mean_mbps",
            "energy_available",
            "energy_method",
            "run_gpu_j",
            "idle_gpu_w",
            "run_gpu_j_per_bit",
            "raw_net_gpu_j",
            "net_gpu_j",
            "raw_net_gpu_j_per_bit",
            "net_gpu_j_per_bit",
            "eff_mode",
            "eff_j_per_bit",
        ]

        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        if not args.silent:
            print("", flush=True)
            print(f"[OK] Wrote: {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# ldpc_bench_v3_3_multi_aligned.py
# 
# 对齐 v0.3.2.1a2 完整输出逻辑的多流版本

import os
import time
import json
import argparse
import bisect
from collections import Counter

import numpy as np
import cupy as cp
from cuda.bindings import runtime as cudart
from aerial.phy5g.ldpc import LdpcDecoder

# ----------------------------
# NVML & Helpers (与你版本完全一致)
# ----------------------------
USE_NVML = True
try:
    from pynvml import * # noqa
except Exception:
    USE_NVML = False

def percentile_sorted(xs_sorted, p):
    if not xs_sorted: return float("nan")
    k = max(0, min(int(round((p / 100.0) * (len(xs_sorted) - 1))), len(xs_sorted) - 1))
    return float(xs_sorted[k])

def latency_stats_from_sorted(lat_sorted):
    n = len(lat_sorted)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "p5": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "p99_9": float("nan"),
            "p99_99": float("nan"),
            "p99_999": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": n,
        "mean": float(np.mean(lat_sorted)),
        "p5": percentile_sorted(lat_sorted, 5),
        "p50": percentile_sorted(lat_sorted, 50),
        "p90": percentile_sorted(lat_sorted, 90),
        "p95": percentile_sorted(lat_sorted, 95),
        "p99": percentile_sorted(lat_sorted, 99),
        "p99_9": percentile_sorted(lat_sorted, 99.9),
        "p99_99": percentile_sorted(lat_sorted, 99.99),
        "p99_999": percentile_sorted(lat_sorted, 99.999),
        "max": float(lat_sorted[-1]),
    }

def print_latency_stats(tag, stats):
    print(
        f"[Latency:{tag}] n={stats['n']} mean={stats['mean']:.2f} p5={stats['p5']:.2f} p50={stats['p50']:.2f} "
        f"p99={stats['p99']:.2f} p99.9={stats['p99_9']:.2f} p99.99={stats['p99_99']:.2f} "
        f"p99.999={stats['p99_999']:.2f} max={stats['max']:.2f}"
    )

def integrate_abs_energy_j(samples, t0_ns, t1_ns):
    if t1_ns <= t0_ns:
        return 0.0, 0

    pts = sorted((int(t), float(p)) for (t, p, *_rest) in samples)
    if not pts:
        return 0.0, 0

    ts = [t for t, _ in pts]

    def _linear_power(tq, ta, pa, tb, pb):
        if tb == ta:
            return max(0.0, float(pa))
        ratio = (tq - ta) / (tb - ta)
        return max(0.0, float(pa) + ratio * (float(pb) - float(pa)))

    def power_at_ns(tq):
        n = len(pts)
        if n == 1:
            return max(0.0, pts[0][1])

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
        return 0.0, len(xs)

    e = 0.0
    for (ta, pa), (tb, pb) in zip(xs[:-1], xs[1:]):
        dt = (tb - ta) * 1e-9
        if dt > 0:
            e += 0.5 * (pa + pb) * dt
    return e, len(xs)

def summarize_nvml(samples, t0_ns, t1_ns):
    xs = [s for s in samples if t0_ns <= s[0] <= t1_ns]
    if not xs: return None
    power, ugpu, umem, vram, smclk, memclk = [np.asarray([s[i] for s in xs], dtype=np.float32) for i in range(1, 7)]
    pstates = [s[7] for s in xs]
    pstate_mode = Counter(pstates).most_common(1)[0][0]
    def q(a, p): return float(np.percentile(a, p))
    return {
        "n": len(xs), "power_mean": float(np.mean(power)), "power_p95": q(power, 95),
        "ugpu_mean": float(np.mean(ugpu)), "ugpu_p50": q(ugpu, 50), "ugpu_p95": q(ugpu, 95),
        "umem_mean": float(np.mean(umem)), "umem_p50": q(umem, 50), "umem_p95": q(umem, 95),
        "vram_mean_mib": float(np.mean(vram)), "vram_p95_mib": q(vram, 95),
        "smclk_p50": q(smclk, 50), "smclk_p95": q(smclk, 95),
        "memclk_p50": q(memclk, 50), "memclk_p95": q(memclk, 95),
        "pstate_mode": pstate_mode,
    }

def cupy_mem_info_gib():
    free_b, total_b = cp.cuda.runtime.memGetInfo()
    return (total_b - free_b) / (1024**3), total_b / (1024**3)

# ----------------------------
# NVML Sampler (对齐 v0.3.2.1a2)
# ----------------------------
class NvmlProcSampler:
    def __init__(self, gpu_index=0, interval_ms=10, qmax=500_000):
        self.gpu_index, self.interval_s, self.qmax = gpu_index, interval_ms/1000.0, qmax
        self._proc, self._stop_evt, self._q = None, None, None

    @staticmethod
    def _worker(gpu_index, interval_s, stop_evt, q):
        if not USE_NVML: return
        try:
            nvmlInit()
            h = nvmlDeviceGetHandleByIndex(int(gpu_index))
            next_t = time.perf_counter()
            while not stop_evt.is_set():
                t_ns = time.perf_counter_ns()
                p = float(nvmlDeviceGetPowerUsage(h)) / 1000.0
                u = nvmlDeviceGetUtilizationRates(h)
                m = nvmlDeviceGetMemoryInfo(h)
                try: sc, mc = nvmlDeviceGetClockInfo(h, NVML_CLOCK_SM), nvmlDeviceGetClockInfo(h, NVML_CLOCK_MEM)
                except: sc, mc = -1, -1
                try: ps = f"P{int(nvmlDeviceGetPerformanceState(h))}"
                except: ps = "P?"
                try: q.put_nowait((t_ns, p, u.gpu, u.memory, m.used//1048576, sc, mc, ps))
                except: pass
                next_t += interval_s
                sleep_s = next_t - time.perf_counter()
                if sleep_s > 0: time.sleep(sleep_s)
        finally:
            try: nvmlShutdown()
            except: pass

    def start(self):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self._stop_evt, self._q = ctx.Event(), ctx.Queue(maxsize=self.qmax)
        self._proc = ctx.Process(target=NvmlProcSampler._worker, args=(self.gpu_index, self.interval_s, self._stop_evt, self._q), daemon=True)
        self._proc.start()

    def stop_and_drain(self):
        if self._stop_evt: self._stop_evt.set()
        if self._proc: self._proc.join(timeout=5.0)
        samples = []
        if self._q:
            while not self._q.empty(): samples.append(self._q.get_nowait())
        return samples

# ----------------------------
# Args (保留所有原始参数，新增 num_streams)
# ----------------------------
def get_args():
    p = argparse.ArgumentParser(description="Aerial LDPC Multi-Stream Bench (v0.3.3)")
    p.add_argument("--K", type=int, default=8448)
    p.add_argument("--code_rate", type=float, default=340/1024)
    p.add_argument("--rv", type=int, default=0)
    p.add_argument("--num_iterations", type=int, default=5)
    p.add_argument("--num_prb", type=int, default=100)
    p.add_argument("--num_symbols", type=int, default=14)
    p.add_argument("--start_sym", type=int, default=0)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--mod_order", type=int, default=2)
    p.add_argument("--batches", type=str, default="1,2,4,8,16,32,64,128,256,512,1024,2048")
    p.add_argument("--num_streams", type=int, default=1)
    p.add_argument("--mode", type=str, default="both", choices=["latency", "throughput", "both"])
    p.add_argument("--throughput_mode", type=int, default=1)
    p.add_argument("--reuse_decoder", type=int, default=0)
    p.add_argument("--warmup_iters", type=int, default=50)
    p.add_argument("--lat_samples", type=int, default=200000)
    p.add_argument("--lat_discard", type=int, default=50)
    p.add_argument("--lat_sync_every", type=int, default=50)
    p.add_argument("--lat_trim_us", type=float, default=1000.0)
    p.add_argument("--latency-clock", type=str, default="both", choices=["gpu_event", "host_wall", "both"],
                   help="Latency timing clock: gpu_event=CUDA event, host_wall=perf_counter around decode+sync, both=collect both")
    p.add_argument("--include-h2d", type=int, default=0, help="If 1, include per-iteration host->device copy in latency measurement")
    p.add_argument("--target_sec", type=float, default=20.0)
    p.add_argument("--yield_every_n_calls", type=int, default=0)
    p.add_argument("--yield_sleep_s", type=float, default=0)
    p.add_argument("--steady_sec", type=float, default=0.0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--nvml_interval_ms", type=int, default=10)
    p.add_argument("--nvml_qmax", type=int, default=500000)
    p.add_argument("--idle_per_batch_sec", type=float, default=30)
    p.add_argument("--idle_baseline", type=str, default="P20", choices=["P20", "P50", "MEAN"])
    p.add_argument("--idle_sync_grace_ms", type=int, default=50)
    p.add_argument("--min_samples_energy", type=int, default=800)
    p.add_argument("--min_samples_ratio", type=float, default=0.70)
    p.add_argument("--out_dir", type=str, default="bench_out_v3_3_multi")
    p.add_argument("--save_latency_npy", type=int, default=1)
    p.add_argument("--save_json", type=int, default=1)
    return p.parse_args()

def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 物理参数解析
    dmrs_sym = [0,0,1,0,0,0,0,0,0,0,0,1,0,0]
    num_data_sym = (np.array(dmrs_sym[args.start_sym:args.start_sym + args.num_symbols]) == 0).sum()
    N = int(num_data_sym * args.num_prb * 12 * args.num_layers * args.mod_order)
    K, code_rate, rv, iters = args.K, args.code_rate, args.rv, args.num_iterations
    THROUGHPUT_MODE, REUSE_DECODER = bool(args.throughput_mode), bool(args.reuse_decoder)
    BATCH_LIST = [int(x.strip()) for x in args.batches.split(",") if x.strip()]
    
    expected_nvml_samples = int(round(args.target_sec * 1000.0 / args.nvml_interval_ms))
    required_in_window = max(int(args.min_samples_energy), int(round(args.min_samples_ratio * expected_nvml_samples)))

    # CUDA 环境
    cudart.cudaSetDevice(args.gpu)
    _, main_stream = cudart.cudaStreamCreate()
    _, ev_start = cudart.cudaEventCreate(); _, ev_stop = cudart.cudaEventCreate()

    print("==== Aerial Multi-Stream Throughput+Latency+Power+Util Benchmark (v0.3.3) ====")
    print(f"GPU={args.gpu} | Streams={args.num_streams}")
    print(f"K={K}, code_rate={code_rate:.6f}, N={N}, iters={iters}")
    print(f"mode={args.mode} | throughput_mode={THROUGHPUT_MODE} | reuse_decoder={REUSE_DECODER}")
    print(f"latency_clock={args.latency_clock}")
    print(f"include_h2d={bool(args.include_h2d)}")
    print(f"target_sec={args.target_sec:.2f} | required_in_window={required_in_window}")
    print(f"batches={BATCH_LIST}\n")

    # 创建多流资源
    extra_streams = [cudart.cudaStreamCreate()[1] for _ in range(args.num_streams)]
    all_results = []

    for batch in BATCH_LIST:
        llr = cp.random.standard_normal((N, batch), dtype=cp.float32)
        host_llr = None
        latency_ext_stream = None
        if bool(args.include_h2d):
            host_llr = np.random.standard_normal((N, batch)).astype(np.float32)
            llr = cp.empty((N, batch), dtype=cp.float32)
            llr.set(host_llr)
            try:
                latency_ext_stream = cp.cuda.ExternalStream(int(extra_streams[0]))
            except Exception:
                latency_ext_stream = None
        params = {"input_llrs": [llr], "tb_sizes": [K]*batch, "code_rates": [code_rate]*batch,
                  "redundancy_versions": [rv]*batch, "rate_match_lengths": [N]*batch}
        
        decoders = [LdpcDecoder(cuda_stream=extra_streams[i], num_iterations=iters, throughput_mode=THROUGHPUT_MODE) for i in range(args.num_streams)]

        # Warmup
        for _ in range(int(args.warmup_iters)): decoders[0].decode(**params)
        cudart.cudaDeviceSynchronize()

        print(f"--- batch={batch} ---")
        used_gib, total_gib = cupy_mem_info_gib()
        print(f"[Mem] CuPy visible used: {used_gib:.2f} GiB / {total_gib:.2f} GiB")

        res = {"batch": batch, "K": K, "N": N, "code_rate": code_rate, "iters": iters, "throughput_mode": THROUGHPUT_MODE}

        # 1) Latency (与原始版本逻辑一致)
        if args.mode in ("latency", "both"):
            use_event = args.latency_clock in ("gpu_event", "both")
            use_host = args.latency_clock in ("host_wall", "both")
            lat_event_us = []
            lat_host_us = []
            for i in range(args.lat_samples):
                if use_event:
                    cudart.cudaEventRecord(ev_start, extra_streams[0])
                host_t0_ns = time.perf_counter_ns() if use_host else 0

                if bool(args.include_h2d):
                    if latency_ext_stream is not None:
                        with latency_ext_stream:
                            llr.set(host_llr)
                    else:
                        llr.set(host_llr)

                decoders[0].decode(**params)
                if use_event:
                    cudart.cudaEventRecord(ev_stop, extra_streams[0])
                    cudart.cudaEventSynchronize(ev_stop)
                    _, ms = cudart.cudaEventElapsedTime(ev_start, ev_stop)
                    event_us = ms * 1000.0
                else:
                    cudart.cudaStreamSynchronize(extra_streams[0])
                    event_us = None

                if use_host:
                    host_t1_ns = time.perf_counter_ns()
                    host_us = (host_t1_ns - host_t0_ns) / 1000.0
                else:
                    host_us = None

                if i >= args.lat_discard:
                    if event_us is not None:
                        lat_event_us.append(event_us)
                    if host_us is not None:
                        lat_host_us.append(host_us)

            if use_event:
                lat_event_sorted = sorted(lat_event_us)
                event_raw = latency_stats_from_sorted(lat_event_sorted)
                print_latency_stats("RAW-GPU_EVENT", event_raw)
                res["lat_raw_us_gpu_event"] = event_raw

                lat_event_trim = [x for x in lat_event_sorted if x <= args.lat_trim_us]
                event_trim = latency_stats_from_sorted(lat_event_trim)
                print(
                    f"[Latency:TRIM-GPU_EVENT] (<= {args.lat_trim_us} us): "
                    f"kept={len(lat_event_trim)}/{len(lat_event_sorted)}"
                )
                print_latency_stats("TRIM-GPU_EVENT", event_trim)
                event_trim["thr"] = args.lat_trim_us
                event_trim["kept"] = len(lat_event_trim)
                res["lat_trim_us_gpu_event"] = event_trim
                res["lat_raw_us"] = event_raw
                res["lat_trim_us"] = event_trim

            if use_host:
                lat_host_sorted = sorted(lat_host_us)
                host_raw = latency_stats_from_sorted(lat_host_sorted)
                print_latency_stats("RAW-HOST_WALL", host_raw)
                res["lat_raw_us_host_wall"] = host_raw

                lat_host_trim = [x for x in lat_host_sorted if x <= args.lat_trim_us]
                host_trim = latency_stats_from_sorted(lat_host_trim)
                print(
                    f"[Latency:TRIM-HOST_WALL] (<= {args.lat_trim_us} us): "
                    f"kept={len(lat_host_trim)}/{len(lat_host_sorted)}"
                )
                print_latency_stats("TRIM-HOST_WALL", host_trim)
                host_trim["thr"] = args.lat_trim_us
                host_trim["kept"] = len(lat_host_trim)
                res["lat_trim_us_host_wall"] = host_trim

                if not use_event:
                    res["lat_raw_us"] = host_raw
                    res["lat_trim_us"] = host_trim

            if args.save_latency_npy:
                if use_event:
                    np.save(os.path.join(args.out_dir, f"lat_us_gpu_event_batch{batch}.npy"), np.asarray(lat_event_us, dtype=np.float32))
                if use_host:
                    np.save(os.path.join(args.out_dir, f"lat_us_host_wall_batch{batch}.npy"), np.asarray(lat_host_us, dtype=np.float32))

        # 2) Throughput (多流异步流水线)
        if args.mode in ("throughput", "both"):
            if args.steady_sec > 0:
                ts = time.perf_counter()
                while (time.perf_counter() - ts) < args.steady_sec: decoders[0].decode(**params)
                cudart.cudaDeviceSynchronize()

            sampler = NvmlProcSampler(args.gpu, args.nvml_interval_ms, args.nvml_qmax)
            sampler.start()
            
            t0_ns = time.perf_counter_ns(); t0 = time.perf_counter(); n_calls = 0
            while (time.perf_counter() - t0) < args.target_sec:
                decoders[n_calls % args.num_streams].decode(**params)
                n_calls += 1
                if args.yield_every_n_calls > 0 and (n_calls % args.yield_every_n_calls == 0):
                    time.sleep(args.yield_sleep_s)
            
            for s in extra_streams: cudart.cudaStreamSynchronize(s)
            t1_ns = time.perf_counter_ns(); dt_s = (t1_ns - t0_ns) * 1e-9
            
            total_bits = n_calls * batch * K
            thr_Mbps = (total_bits / dt_s) / 1e6
            print(f"[Throughput] info Mb/s={thr_Mbps:.2f} (window={dt_s:.2f}s, calls={n_calls})")

            # NVML 统计 (复刻原始输出)
            thr_samples = sampler.stop_and_drain()
            if USE_NVML and thr_samples:
                nvml_stats = summarize_nvml(thr_samples, t0_ns, t1_ns)
                inwin_n = len([s for s in thr_samples if t0_ns <= s[0] <= t1_ns])
                coverage = inwin_n / expected_nvml_samples
                print(f"[NVML] coverage={coverage:.3f} | ugpu% mean/p50={nvml_stats['ugpu_mean']:.1f}/{nvml_stats['ugpu_p50']:.1f} | power W mean={nvml_stats['power_mean']:.1f}")
                
                E_abs_j, _ = integrate_abs_energy_j(thr_samples, t0_ns, t1_ns)
                abs_nj = (E_abs_j / total_bits) * 1e9
                print(f"[AbsEnergy] P_avg={E_abs_j/dt_s:.2f} W | {abs_nj:.2f} nJ/bit")
                
                res["throughput"] = {"thr_Mbps": thr_Mbps, "calls": n_calls, "dt_s": dt_s}
                res["nvml"] = {"stats": nvml_stats, "coverage": coverage}
                res["abs_energy"] = {"E_abs_j": E_abs_j, "abs_nJ_per_bit": abs_nj}

        # 3) Idle Baseline & NetEnergy (复刻原始逻辑)
        if USE_NVML and args.idle_per_batch_sec > 0 and args.mode in ("throughput", "both"):
            idle_sampler = NvmlProcSampler(args.gpu, args.nvml_interval_ms)
            idle_sampler.start(); time.sleep(args.idle_per_batch_sec)
            idle_p = [s[1] for s in idle_sampler.stop_and_drain()]
            if idle_p:
                p20, p50, pmean = percentile_sorted(sorted(idle_p), 20), percentile_sorted(sorted(idle_p), 50), np.mean(idle_p)
                p_idle = {"P20": p20, "P50": p50, "MEAN": pmean}[args.idle_baseline]
                e_net = E_abs_j - p_idle * dt_s
                print(f"[Idle@batch] P20={p20:.2f} W | MEAN={pmean:.2f} W")
                print(f"[NetEnergy:{args.idle_baseline}] P_idle={p_idle:.2f} W | {(e_net/total_bits)*1e9:.2f} nJ/bit")
                res["net_energy"] = {"net_nJ_per_bit": (e_net/total_bits)*1e9, "P_idle": p_idle}

        all_results.append(res)
        print()

    # JSON 摘要保存 (复刻原始保存逻辑)
    if args.save_json:
        with open(os.path.join(args.out_dir, "summary_multi.json"), "w") as f:
            json.dump({"results": all_results, "args": vars(args)}, f, indent=2)

    for s in extra_streams: cudart.cudaStreamDestroy(s)

if __name__ == "__main__": main()
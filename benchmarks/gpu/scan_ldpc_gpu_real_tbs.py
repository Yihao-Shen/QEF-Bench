#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


MCS_TABLES: Dict[int, Dict[int, Tuple[int, float]]] = {
    1: {
        0: (2, 120.0), 1: (2, 157.0), 2: (2, 193.0), 3: (2, 251.0),
        4: (2, 308.0), 5: (2, 379.0), 6: (2, 449.0), 7: (2, 526.0),
        8: (2, 602.0), 9: (2, 679.0), 10: (4, 340.0), 11: (4, 378.0),
        12: (4, 434.0), 13: (4, 490.0), 14: (4, 553.0), 15: (4, 616.0),
        16: (4, 658.0), 17: (6, 438.0), 18: (6, 466.0), 19: (6, 517.0),
        20: (6, 567.0), 21: (6, 616.0), 22: (6, 666.0), 23: (6, 719.0),
        24: (6, 772.0), 25: (6, 822.0), 26: (6, 873.0), 27: (6, 910.0),
        28: (6, 948.0),
    },
    2: {
        0: (2, 120.0), 1: (2, 193.0), 2: (2, 308.0), 3: (2, 449.0),
        4: (2, 602.0), 5: (4, 378.0), 6: (4, 434.0), 7: (4, 490.0),
        8: (4, 553.0), 9: (4, 616.0), 10: (4, 658.0), 11: (6, 466.0),
        12: (6, 517.0), 13: (6, 567.0), 14: (6, 616.0), 15: (6, 666.0),
        16: (6, 719.0), 17: (6, 772.0), 18: (6, 822.0), 19: (6, 873.0),
        20: (8, 682.5), 21: (8, 711.0), 22: (8, 754.0), 23: (8, 797.0),
        24: (8, 841.0), 25: (8, 885.0), 26: (8, 916.5), 27: (8, 948.0),
    },
    3: {
        0: (2, 30.0), 1: (2, 40.0), 2: (2, 50.0), 3: (2, 64.0),
        4: (2, 78.0), 5: (2, 99.0), 6: (2, 120.0), 7: (2, 157.0),
        8: (2, 193.0), 9: (2, 251.0), 10: (2, 308.0), 11: (2, 379.0),
        12: (2, 449.0), 13: (2, 526.0), 14: (2, 602.0), 15: (4, 340.0),
        16: (4, 378.0), 17: (4, 434.0), 18: (4, 490.0), 19: (4, 553.0),
        20: (4, 616.0), 21: (6, 438.0), 22: (6, 466.0), 23: (6, 517.0),
        24: (6, 567.0), 25: (6, 616.0), 26: (6, 666.0), 27: (6, 719.0),
        28: (6, 772.0),
    },
}


@dataclass
class GpuScanPoint:
    rb: int
    layers: int
    mcs: int
    mod_order: int
    code_rate: float
    k_bits: int
    n_bits: int
    batch: int
    throughput_mbps: Optional[float]
    abs_nj_per_bit: Optional[float]
    net_nj_per_bit: Optional[float]
    p50_us: Optional[float]
    p99_us: Optional[float]
    p99_9_us: Optional[float]
    p99_99_us: Optional[float]
    p99_999_us: Optional[float]
    power_mean_w: Optional[float]
    ugpu_mean: Optional[float]
    success: bool
    command: str
    error: str


def parse_int_list(expr: str) -> List[int]:
    values: List[int] = []
    for token in expr.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(token))
    ordered_unique: List[int] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered_unique.append(value)
    return ordered_unique


def row_to_point(row: Dict[str, str]) -> GpuScanPoint:
    def to_i(name: str) -> int:
        return int(row.get(name, "0") or 0)

    def to_f(name: str) -> Optional[float]:
        value = row.get(name, "")
        if value is None or value == "":
            return None
        return float(value)

    return GpuScanPoint(
        rb=to_i("rb"),
        layers=to_i("layers"),
        mcs=to_i("mcs"),
        mod_order=to_i("mod_order"),
        code_rate=float(row.get("code_rate", "0") or 0),
        k_bits=to_i("k_bits"),
        n_bits=to_i("n_bits"),
        batch=to_i("batch"),
        throughput_mbps=to_f("throughput_mbps"),
        abs_nj_per_bit=to_f("abs_nj_per_bit"),
        net_nj_per_bit=to_f("net_nj_per_bit"),
        p50_us=to_f("p50_us"),
        p99_us=to_f("p99_us"),
        p99_9_us=to_f("p99_9_us"),
        p99_99_us=to_f("p99_99_us"),
        p99_999_us=to_f("p99_999_us"),
        power_mean_w=to_f("power_mean_w"),
        ugpu_mean=to_f("ugpu_mean"),
        success=(row.get("success", "False").strip().lower() in {"1", "true", "yes"}),
        command=row.get("command", ""),
        error=row.get("error", ""),
    )


def load_resume(trace_csv: Path) -> Tuple[List[GpuScanPoint], Set[Tuple[int, int, int, int]]]:
    if not trace_csv.exists():
        return [], set()
    points: List[GpuScanPoint] = []
    completed: Set[Tuple[int, int, int, int]] = set()
    with trace_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                point = row_to_point(row)
            except Exception:
                continue
            points.append(point)
            if point.success:
                completed.add((point.rb, point.layers, point.mcs, point.batch))
    return points, completed


def parse_summary_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    return results if isinstance(results, list) else []


def to_point_from_result(
    rb: int,
    layers: int,
    mcs: int,
    mod_order: int,
    code_rate: float,
    result: dict,
    command: str,
    latency_source: str,
) -> GpuScanPoint:
    throughput = result.get("throughput", {}) or {}
    nvml = result.get("nvml", {}) or {}
    nvml_stats = nvml.get("stats", {}) or {}
    abs_energy = result.get("abs_energy", {}) or {}
    net_energy = result.get("net_energy", {}) or {}
    lat_default = result.get("lat_raw_us", {}) or {}
    lat_gpu_event = result.get("lat_raw_us_gpu_event", {}) or {}
    lat_host_wall = result.get("lat_raw_us_host_wall", {}) or {}

    if latency_source == "host_wall":
        lat = lat_host_wall or lat_default
    elif latency_source == "gpu_event":
        lat = lat_gpu_event or lat_default
    else:
        lat = lat_host_wall or lat_default or lat_gpu_event

    def pick_latency_value(candidates: List[str]) -> Optional[float]:
        for key in candidates:
            value = _safe_float(lat.get(key))
            if value is not None:
                return value
        return None

    return GpuScanPoint(
        rb=rb,
        layers=layers,
        mcs=mcs,
        mod_order=mod_order,
        code_rate=code_rate,
        k_bits=int(result.get("K", 0) or 0),
        n_bits=int(result.get("N", 0) or 0),
        batch=int(result.get("batch", 0) or 0),
        throughput_mbps=_safe_float(throughput.get("thr_Mbps")),
        abs_nj_per_bit=_safe_float(abs_energy.get("abs_nJ_per_bit")),
        net_nj_per_bit=_safe_float(net_energy.get("net_nJ_per_bit")),
        p50_us=pick_latency_value(["p50"]),
        p99_us=pick_latency_value(["p99"]),
        p99_9_us=pick_latency_value(["p99_9", "p99.9"]),
        p99_99_us=pick_latency_value(["p99_99", "p99.99"]),
        p99_999_us=pick_latency_value(["p99_999", "p99.999"]),
        power_mean_w=_safe_float(nvml_stats.get("power_mean")),
        ugpu_mean=_safe_float(nvml_stats.get("ugpu_mean")),
        success=True,
        command=command,
        error="",
    )


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def read_power_w_nvidia_smi(gpu_index: int) -> Optional[float]:
    cmd = [
        "nvidia-smi",
        "-i",
        str(gpu_index),
        "--query-gpu=power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        if not out:
            return None
        return float(out.splitlines()[0].strip())
    except Exception:
        return None


def measure_idle_power_once(gpu_index: int, idle_seconds: float, sample_interval_s: float) -> Optional[float]:
    if idle_seconds <= 0:
        return None
    t0 = time.perf_counter()
    samples: List[float] = []
    interval = max(0.02, sample_interval_s)
    while (time.perf_counter() - t0) < idle_seconds:
        p = read_power_w_nvidia_smi(gpu_index)
        if p is not None:
            samples.append(p)
        time.sleep(interval)
    if not samples:
        return None
    return float(sum(samples) / len(samples))


def write_csv(path: Path, points: Sequence[GpuScanPoint]) -> None:
    if not points:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(points[0]).keys()))
        writer.writeheader()
        for point in points:
            writer.writerow(asdict(point))


def save_outputs(
    out_dir: Path,
    args: argparse.Namespace,
    points: Sequence[GpuScanPoint],
    latency_limit: Optional[float],
) -> Tuple[Path, Path, Path]:
    trace_csv = out_dir / "gpu_tbs_grid_trace.csv"
    write_csv(trace_csv, points)

    summary_rows = choose_best(points, latency_limit_us=latency_limit, prefer_net=args.prefer_net_energy)
    summary_csv = out_dir / "gpu_tbs_grid_summary.csv"
    if summary_rows:
        keys: List[str] = []
        for row in summary_rows:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary_rows)

    meta_json = out_dir / "gpu_tbs_grid_summary.json"
    with meta_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "run_metadata": {
                    "idle_mode": args.idle_mode,
                    "global_idle_power_w": getattr(args, "_global_idle_power_w", None),
                    "pause_between_points_seconds": args.pause_between_points_seconds,
                },
                "total_rows": len(points),
                "successful_rows": sum(1 for p in points if p.success),
                "trace_csv": str(trace_csv),
                "summary_csv": str(summary_csv),
                "best_points": summary_rows,
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=2,
        )

    return trace_csv, summary_csv, meta_json


def choose_best(points: Sequence[GpuScanPoint], latency_limit_us: Optional[float], prefer_net: bool) -> List[dict]:
    groups: Dict[Tuple[int, int, int], List[GpuScanPoint]] = {}
    for point in points:
        groups.setdefault((point.rb, point.layers, point.batch), []).append(point)

    rows: List[dict] = []
    for (rb, layers, batch), candidates in sorted(groups.items()):
        valid = [c for c in candidates if c.success and c.throughput_mbps and c.throughput_mbps > 0]
        if latency_limit_us is not None:
            valid = [c for c in valid if c.p99_999_us is not None and c.p99_999_us <= latency_limit_us]
        if not valid:
            rows.append({"rb": rb, "layers": layers, "batch": batch, "selected": False, "reason": "no_valid_point"})
            continue

        def energy_value(p: GpuScanPoint) -> float:
            if prefer_net and p.net_nj_per_bit is not None and p.net_nj_per_bit > 0:
                return p.net_nj_per_bit
            if p.abs_nj_per_bit is not None and p.abs_nj_per_bit > 0:
                return p.abs_nj_per_bit
            return float("inf")

        valid.sort(key=energy_value)
        best = valid[0]
        row = asdict(best)
        row["selected"] = True
        row["prefer_net_energy"] = prefer_net
        rows.append(row)
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Scan GPU LDPC over RB x layers x MCS by invoking ldpc_bench_3.3.py")
    p.add_argument("--bench-script", default="ldpc_bench_3.3.py", help="Path to ldpc_bench_3.3.py")
    p.add_argument("--python-bin", default=sys.executable, help="Python executable to run bench script (default: current interpreter)")
    p.add_argument("--out-dir", default="scan_out_gpu_tbs", help="Output directory")

    p.add_argument("--rb-list", default="10,20,50,100,273")
    p.add_argument("--layers-list", default="1,2,4")
    p.add_argument("--mcs-list", default="0-27")
    p.add_argument("--mcs-table", type=int, choices=[1, 2, 3], default=1)

    p.add_argument("--K", type=int, default=8448, help="Fixed value for K when --k-mode=fixed")
    p.add_argument("--k-mode", choices=["auto", "fixed"], default="auto", help="auto: derive K per point from N*code_rate; fixed: always use --K")
    p.add_argument("--rv", type=int, default=0)
    p.add_argument("--num-iterations", "--num_iterations", dest="num_iterations", type=int, default=5)
    p.add_argument("--num-symbols", type=int, default=14)
    p.add_argument("--start-sym", type=int, default=0)
    p.add_argument("--batches", default="1")
    p.add_argument("--num-streams", type=int, default=1)
    p.add_argument("--mode", choices=["latency", "throughput", "both"], default="both")
    p.add_argument("--target-sec", type=float, default=8.0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--warmup-iters", type=int, default=50)
    p.add_argument("--lat-samples", type=int, default=10000)
    p.add_argument("--lat-discard", type=int, default=50)
    p.add_argument("--lat-trim-us", type=float, default=1000.0)
    p.add_argument("--latency-clock", choices=["gpu_event", "host_wall", "both"], default="both", help="Forwarded to benchmark script")
    p.add_argument("--latency-source", choices=["auto", "host_wall", "gpu_event"], default="auto", help="Which latency block from summary_multi.json to use in scan trace")
    p.add_argument("--include-h2d", action="store_true", help="Include per-iteration H2D copy in benchmark latency measurement")
    p.add_argument("--idle-per-batch-sec", type=float, default=30)
    p.add_argument("--idle-mode", choices=["once", "per-point"], default="once", help="once: measure idle once before scan and reuse; per-point: let bench measure idle every point")
    p.add_argument("--nvml-interval-ms", type=int, default=10)

    p.add_argument("--latency-limit-us", type=float, default=-1.0)
    p.add_argument("--prefer-net-energy", action="store_true")
    p.add_argument("--point-timeout-seconds", type=float, default=0.0)
    p.add_argument("--pause-between-points-seconds", type=float, default=0.0)
    p.add_argument("--checkpoint-every", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--bench-live-output", action="store_true", help="Stream benchmark stdout/stderr to console in real time")
    return p


def run_bench_command(
    cmd: Sequence[str],
    timeout_s: float,
    live_output: bool,
    log_path: Path,
) -> Tuple[int, str]:
    timeout = timeout_s if timeout_s > 0 else None
    if not live_output:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        combined = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        if combined:
            log_path.write_text(combined + "\n", encoding="utf-8")
        return proc.returncode, combined

    lines: List[str] = []
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    lines.append(line)
                    print(line, end="", flush=True)
            return_code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
            combined = "".join(lines).strip()
            if combined:
                log_path.write_text(combined + "\n", encoding="utf-8")
            raise

    combined = "".join(lines).strip()
    if combined:
        log_path.write_text(combined + "\n", encoding="utf-8")
    return return_code, combined


def compute_n_bits(num_prb: int, num_symbols: int, start_sym: int, layers: int, mod_order: int) -> int:
    dmrs_sym = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    start = max(0, min(start_sym, len(dmrs_sym)))
    end = max(start, min(len(dmrs_sym), start + num_symbols))
    data_syms = sum(1 for x in dmrs_sym[start:end] if x == 0)
    return int(data_syms * num_prb * 12 * layers * mod_order)


def choose_k_bits(args: argparse.Namespace, n_bits: int, code_rate: float) -> int:
    if args.k_mode == "fixed":
        return int(args.K)
    k_bits = int(n_bits * code_rate)
    k_bits = max(24, k_bits)
    if k_bits >= n_bits:
        k_bits = max(24, n_bits - 1)
    return k_bits


def main() -> int:
    args = build_parser().parse_args()

    bench_script = Path(args.bench_script)
    if not bench_script.exists():
        raise FileNotFoundError(f"bench script not found: {bench_script}")

    out_dir = Path(args.out_dir)
    points_dir = out_dir / "points"
    out_dir.mkdir(parents=True, exist_ok=True)
    points_dir.mkdir(parents=True, exist_ok=True)

    mcs_table = MCS_TABLES[args.mcs_table]
    rb_list = parse_int_list(args.rb_list)
    layers_list = parse_int_list(args.layers_list)
    mcs_list = parse_int_list(args.mcs_list)
    batch_list = parse_int_list(args.batches)
    if not all(m in mcs_table for m in mcs_list):
        bad = [m for m in mcs_list if m not in mcs_table]
        raise ValueError(f"MCS not in table {args.mcs_table}: {bad}")

    trace_csv = out_dir / "gpu_tbs_grid_trace.csv"
    points: List[GpuScanPoint] = []
    completed: Set[Tuple[int, int, int, int]] = set()
    if args.resume:
        points, completed = load_resume(trace_csv)
        print(f"[resume] loaded {len(points)} rows from {trace_csv}", flush=True)

    total = len(rb_list) * len(layers_list) * len(mcs_list)
    idx = 0
    done_points = 0
    checkpoint_every = max(1, int(args.checkpoint_every))
    latency_limit = None if args.latency_limit_us < 0 else args.latency_limit_us
    args._global_idle_power_w = None

    if args.mode in ("throughput", "both") and args.idle_mode == "once" and args.idle_per_batch_sec > 0:
        sample_interval = max(0.05, float(args.nvml_interval_ms) / 1000.0)
        print(f"[idle] measuring once for {args.idle_per_batch_sec:.2f}s on gpu={args.gpu}", flush=True)
        idle_power = measure_idle_power_once(args.gpu, float(args.idle_per_batch_sec), sample_interval)
        args._global_idle_power_w = idle_power
        if idle_power is None:
            print("[idle] warning: failed to measure global idle power, net_nj_per_bit will follow benchmark/raw values", flush=True)
        else:
            print(f"[idle] global idle power = {idle_power:.3f} W", flush=True)

    try:
        for rb in rb_list:
            for layers in layers_list:
                for mcs in mcs_list:
                    idx += 1
                    mod_order, code_rate_1024 = mcs_table[mcs]
                    code_rate = code_rate_1024 / 1024.0
                    n_bits = compute_n_bits(rb, args.num_symbols, args.start_sym, layers, mod_order)
                    k_bits = choose_k_bits(args, n_bits, code_rate)
                    point_out = points_dir / f"rb{rb}_ly{layers}_mcs{mcs}"
                    point_out.mkdir(parents=True, exist_ok=True)

                    per_key_done = all((rb, layers, mcs, b) in completed for b in batch_list)
                    if per_key_done:
                        print(f"[{idx}/{total}] skip existing rb={rb} layers={layers} mcs={mcs}", flush=True)
                        continue

                    cmd = [
                        args.python_bin,
                        str(bench_script),
                        "--K", str(k_bits),
                        "--code_rate", str(code_rate),
                        "--rv", str(args.rv),
                        "--num_iterations", str(args.num_iterations),
                        "--num_prb", str(rb),
                        "--num_symbols", str(args.num_symbols),
                        "--start_sym", str(args.start_sym),
                        "--num_layers", str(layers),
                        "--mod_order", str(mod_order),
                        "--batches", args.batches,
                        "--num_streams", str(args.num_streams),
                        "--mode", args.mode,
                        "--target_sec", str(args.target_sec),
                        "--gpu", str(args.gpu),
                        "--warmup_iters", str(args.warmup_iters),
                        "--lat_samples", str(args.lat_samples),
                        "--lat_discard", str(args.lat_discard),
                        "--lat_trim_us", str(args.lat_trim_us),
                        "--latency-clock", str(args.latency_clock),
                        "--include-h2d", "1" if args.include_h2d else "0",
                        "--idle_per_batch_sec", str(args.idle_per_batch_sec if args.idle_mode == "per-point" else 0.0),
                        "--nvml_interval_ms", str(args.nvml_interval_ms),
                        "--out_dir", str(point_out),
                        "--save_json", "1",
                        "--save_latency_npy", "0",
                    ]
                    command_text = " ".join(cmd)

                    if k_bits >= n_bits:
                        reason = f"invalid_tb_setting_precheck: K({k_bits}) >= N({n_bits})"
                        for batch in batch_list:
                            points.append(
                                GpuScanPoint(
                                    rb=rb,
                                    layers=layers,
                                    mcs=mcs,
                                    mod_order=mod_order,
                                    code_rate=code_rate,
                                    k_bits=k_bits,
                                    n_bits=n_bits,
                                    batch=batch,
                                    throughput_mbps=None,
                                    abs_nj_per_bit=None,
                                    net_nj_per_bit=None,
                                    p50_us=None,
                                    p99_us=None,
                                    p99_9_us=None,
                                    p99_99_us=None,
                                    p99_999_us=None,
                                    power_mean_w=None,
                                    ugpu_mean=None,
                                    success=False,
                                    command=command_text,
                                    error=reason,
                                )
                            )
                            completed.add((rb, layers, mcs, batch))
                        done_points += 1
                        if done_points % checkpoint_every == 0:
                            save_outputs(out_dir, args, points, latency_limit)
                        if args.pause_between_points_seconds > 0:
                            time.sleep(args.pause_between_points_seconds)
                        continue

                    print(
                        f"[{idx}/{total}] run rb={rb} layers={layers} mcs={mcs} K={k_bits} N={n_bits}",
                        flush=True,
                    )

                    try:
                        timeout_s = max(0.0, float(args.point_timeout_seconds))
                        bench_log = point_out / "bench_stdout_stderr.log"
                        returncode, bench_output = run_bench_command(
                            cmd,
                            timeout_s=timeout_s,
                            live_output=bool(args.bench_live_output),
                            log_path=bench_log,
                        )
                        if returncode != 0:
                            error = (bench_output or f"benchmark failed with return code {returncode}")[:4000]
                            for batch in batch_list:
                                points.append(
                                    GpuScanPoint(
                                        rb=rb,
                                        layers=layers,
                                        mcs=mcs,
                                        mod_order=mod_order,
                                        code_rate=code_rate,
                                        k_bits=k_bits,
                                        n_bits=n_bits,
                                        batch=batch,
                                        throughput_mbps=None,
                                        abs_nj_per_bit=None,
                                        net_nj_per_bit=None,
                                        p50_us=None,
                                        p99_us=None,
                                        p99_9_us=None,
                                        p99_99_us=None,
                                        p99_999_us=None,
                                        power_mean_w=None,
                                        ugpu_mean=None,
                                        success=False,
                                        command=command_text,
                                        error=error,
                                    )
                                )
                                completed.add((rb, layers, mcs, batch))
                            done_points += 1
                            if done_points % checkpoint_every == 0:
                                save_outputs(out_dir, args, points, latency_limit)
                            if args.pause_between_points_seconds > 0:
                                time.sleep(args.pause_between_points_seconds)
                            continue
                    except subprocess.TimeoutExpired as exc:
                        error = f"timeout after {args.point_timeout_seconds}s\n{str(exc)}"
                        for batch in batch_list:
                            points.append(
                                GpuScanPoint(
                                    rb=rb,
                                    layers=layers,
                                    mcs=mcs,
                                    mod_order=mod_order,
                                    code_rate=code_rate,
                                    k_bits=k_bits,
                                    n_bits=n_bits,
                                    batch=batch,
                                    throughput_mbps=None,
                                    abs_nj_per_bit=None,
                                    net_nj_per_bit=None,
                                    p50_us=None,
                                    p99_us=None,
                                    p99_9_us=None,
                                    p99_99_us=None,
                                    p99_999_us=None,
                                    power_mean_w=None,
                                    ugpu_mean=None,
                                    success=False,
                                    command=command_text,
                                    error=error,
                                )
                            )
                            completed.add((rb, layers, mcs, batch))
                        done_points += 1
                        if done_points % checkpoint_every == 0:
                            save_outputs(out_dir, args, points, latency_limit)
                        if args.pause_between_points_seconds > 0:
                            time.sleep(args.pause_between_points_seconds)
                        continue

                    summary_json = point_out / "summary_multi.json"
                    if not summary_json.exists():
                        for batch in batch_list:
                            points.append(
                                GpuScanPoint(
                                    rb=rb,
                                    layers=layers,
                                    mcs=mcs,
                                    mod_order=mod_order,
                                    code_rate=code_rate,
                                    k_bits=k_bits,
                                    n_bits=n_bits,
                                    batch=batch,
                                    throughput_mbps=None,
                                    abs_nj_per_bit=None,
                                    net_nj_per_bit=None,
                                    p50_us=None,
                                    p99_us=None,
                                    p99_9_us=None,
                                    p99_99_us=None,
                                    p99_999_us=None,
                                    power_mean_w=None,
                                    ugpu_mean=None,
                                    success=False,
                                    command=command_text,
                                    error="summary_multi.json missing",
                                )
                            )
                            completed.add((rb, layers, mcs, batch))
                        done_points += 1
                        if done_points % checkpoint_every == 0:
                            save_outputs(out_dir, args, points, latency_limit)
                        if args.pause_between_points_seconds > 0:
                            time.sleep(args.pause_between_points_seconds)
                        continue

                    results = parse_summary_json(summary_json)
                    got_batches: Set[int] = set()
                    for item in results:
                        point = to_point_from_result(
                            rb,
                            layers,
                            mcs,
                            mod_order,
                            code_rate,
                            item,
                            command_text,
                            args.latency_source,
                        )
                        if point.k_bits <= 0:
                            point.k_bits = k_bits
                        if point.n_bits <= 0:
                            point.n_bits = n_bits
                        if args.idle_mode == "once" and args._global_idle_power_w is not None and point.throughput_mbps and point.throughput_mbps > 0:
                            if point.abs_nj_per_bit is not None:
                                idle_nj_per_bit = args._global_idle_power_w * 1e3 / point.throughput_mbps
                                point.net_nj_per_bit = max(0.0, point.abs_nj_per_bit - idle_nj_per_bit)
                        points.append(point)
                        got_batches.add(point.batch)
                        completed.add((rb, layers, mcs, point.batch))

                    for batch in batch_list:
                        if batch not in got_batches:
                            points.append(
                                GpuScanPoint(
                                    rb=rb,
                                    layers=layers,
                                    mcs=mcs,
                                    mod_order=mod_order,
                                    code_rate=code_rate,
                                    k_bits=k_bits,
                                    n_bits=n_bits,
                                    batch=batch,
                                    throughput_mbps=None,
                                    abs_nj_per_bit=None,
                                    net_nj_per_bit=None,
                                    p50_us=None,
                                    p99_us=None,
                                    p99_9_us=None,
                                    p99_99_us=None,
                                    p99_999_us=None,
                                    power_mean_w=None,
                                    ugpu_mean=None,
                                    success=False,
                                    command=command_text,
                                    error="batch missing in summary_multi.json",
                                )
                            )
                            completed.add((rb, layers, mcs, batch))

                    done_points += 1
                    if done_points % checkpoint_every == 0:
                        save_outputs(out_dir, args, points, latency_limit)
                    if args.pause_between_points_seconds > 0:
                        time.sleep(args.pause_between_points_seconds)

    except KeyboardInterrupt:
        trace_csv, summary_csv, meta_json = save_outputs(out_dir, args, points, latency_limit)
        print(f"[interrupt] partial results saved: trace={trace_csv} summary={summary_csv} json={meta_json}", flush=True)
        return 130

    trace_csv, summary_csv, meta_json = save_outputs(out_dir, args, points, latency_limit)
    print(f"Done. trace={trace_csv} summary={summary_csv} json={meta_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

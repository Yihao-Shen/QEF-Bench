#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


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


MOD_BY_ORDER = {
    1: "BPSK",
    2: "QPSK",
    4: "QAM16",
    6: "QAM64",
    8: "QAM256",
}


def normalize_backend(name: str) -> str:
    lowered = (name or "").strip().lower()
    if lowered == "avx256":
        return "avx2"
    return lowered


@dataclass
class ScanPoint:
    cells: int
    threads_used: int
    nof_cb: Optional[int]
    rb: int
    layers: int
    mcs: int
    mod_order: int
    mod: str
    code_rate: float
    tbs_bits: Optional[int]
    p50_us: Optional[float]
    p75_us: Optional[float]
    p90_us: Optional[float]
    p99_us: Optional[float]
    p99_9_us: Optional[float]
    p99_99_us: Optional[float]
    p99_999_us: Optional[float]
    worst_us: Optional[float]
    latency_thr_mbps: Optional[float]
    wall_thr_mbps: Optional[float]
    avg_iters: Optional[float]
    work_pkg_w: Optional[float]
    work_pkg_dram_w: Optional[float]
    idle_pkg_w: Optional[float]
    idle_pkg_dram_w: Optional[float]
    gross_pkg_j_per_bit: Optional[float]
    gross_pkg_dram_j_per_bit: Optional[float]
    net_pkg_j_per_bit: Optional[float]
    net_pkg_dram_j_per_bit: Optional[float]
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
            if end < start:
                raise ValueError(f"invalid range: {token}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(token))
    return sorted(set(values))


def format_cpu_list(cpus: Sequence[int]) -> str:
    return ",".join(str(cpu) for cpu in cpus)


def _read_int_file(path: str) -> Optional[int]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except Exception:
        return None


def _read_text_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None


def _rapl_diff(start_uj: int, end_uj: int, max_uj: int) -> int:
    if end_uj >= start_uj:
        return end_uj - start_uj
    if max_uj <= 0:
        return 0
    return (max_uj - start_uj) + end_uj


def _scope_values(scope: str, pkg0: int, dram0: int, pkg1: int, dram1: int) -> Tuple[int, int]:
    if scope == "socket0":
        return pkg0, dram0
    if scope == "socket1":
        return pkg1, dram1
    return pkg0 + pkg1, dram0 + dram1


def _discover_rapl_domains() -> Dict[str, Tuple[str, str]]:
    fallback = {
        "pkg0": (
            "/sys/class/powercap/intel-rapl:0/energy_uj",
            "/sys/class/powercap/intel-rapl:0/max_energy_range_uj",
        ),
        "dram0": (
            "/sys/class/powercap/intel-rapl:0:0/energy_uj",
            "/sys/class/powercap/intel-rapl:0:0/max_energy_range_uj",
        ),
        "pkg1": (
            "/sys/class/powercap/intel-rapl:1/energy_uj",
            "/sys/class/powercap/intel-rapl:1/max_energy_range_uj",
        ),
        "dram1": (
            "/sys/class/powercap/intel-rapl:1:0/energy_uj",
            "/sys/class/powercap/intel-rapl:1:0/max_energy_range_uj",
        ),
    }

    root = Path("/sys/class/powercap")
    if not root.exists():
        return fallback

    discovered: Dict[str, Tuple[str, str]] = {}
    for domain in sorted(root.glob("intel-rapl:*")):
        domain_name = domain.name
        if domain_name.count(":") != 1:
            continue

        name = _read_text_file(str(domain / "name"))
        if not name:
            continue

        match = re.fullmatch(r"package-(\d+)", name)
        if not match:
            continue

        package_id = int(match.group(1))
        pkg_key = f"pkg{package_id}"
        pkg_energy = domain / "energy_uj"
        pkg_max = domain / "max_energy_range_uj"
        discovered[pkg_key] = (str(pkg_energy), str(pkg_max))

        dram_key = f"dram{package_id}"
        for subdomain in sorted(root.glob(f"{domain_name}:*")):
            sub_name = _read_text_file(str(subdomain / "name"))
            if sub_name != "dram":
                continue
            dram_energy = subdomain / "energy_uj"
            dram_max = subdomain / "max_energy_range_uj"
            discovered[dram_key] = (str(dram_energy), str(dram_max))
            break

    merged = dict(fallback)
    merged.update(discovered)
    return merged


def _read_rapl_snapshot() -> Tuple[Dict[str, Optional[int]], Dict[str, Optional[int]]]:
    domains = _discover_rapl_domains()
    values: Dict[str, Optional[int]] = {}
    maxv: Dict[str, Optional[int]] = {}
    for name, (energy_path, max_path) in domains.items():
        values[name] = _read_int_file(energy_path)
        maxv[name] = _read_int_file(max_path)
    return values, maxv


def _scoped_energy_from_snapshots(
    scope: str,
    start: Dict[str, Optional[int]],
    end: Dict[str, Optional[int]],
    maxv: Dict[str, Optional[int]],
) -> Tuple[Optional[float], Optional[float]]:
    def domain_energy(name: str) -> int:
        s = start.get(name)
        e = end.get(name)
        m = maxv.get(name)
        if s is None or e is None or m is None:
            return 0
        return _rapl_diff(s, e, m)

    pkg0 = domain_energy("pkg0")
    dram0 = domain_energy("dram0")
    pkg1 = domain_energy("pkg1")
    dram1 = domain_energy("dram1")

    if scope == "socket0" and pkg0 <= 0 and dram0 <= 0:
        return None, None
    if scope == "socket1" and pkg1 <= 0 and dram1 <= 0:
        return None, None
    if scope == "sum" and (pkg0 + dram0 + pkg1 + dram1) <= 0:
        return None, None

    scoped_pkg, scoped_dram = _scope_values(scope, pkg0, dram0, pkg1, dram1)
    return scoped_pkg * 1e-6, (scoped_pkg + scoped_dram) * 1e-6


def measure_idle_power_once(idle_seconds: int, energy_scope: str) -> Tuple[Optional[float], Optional[float]]:
    if idle_seconds <= 0:
        return None, None

    start, maxv = _read_rapl_snapshot()

    if all(value is None for value in start.values()):
        return None, None

    time.sleep(idle_seconds)

    end, _ = _read_rapl_snapshot()
    scoped_pkg_j, scoped_pkg_dram_j = _scoped_energy_from_snapshots(energy_scope, start, end, maxv)
    if scoped_pkg_j is None or scoped_pkg_dram_j is None:
        return None, None

    idle_pkg_w = scoped_pkg_j / idle_seconds
    idle_pkg_dram_w = scoped_pkg_dram_j / idle_seconds
    return idle_pkg_w, idle_pkg_dram_w


def _to_optional_int(value: str) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(float(value))


def _to_optional_float(value: str) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def row_to_scan_point(row: Dict[str, str]) -> ScanPoint:
    return ScanPoint(
        cells=int(row.get("cells", "0") or 0),
        threads_used=int(row.get("threads_used", "0") or 0),
        nof_cb=_to_optional_int(row.get("nof_cb", "")),
        rb=int(row.get("rb", "0") or 0),
        layers=int(row.get("layers", "0") or 0),
        mcs=int(row.get("mcs", "0") or 0),
        mod_order=int(row.get("mod_order", "0") or 0),
        mod=row.get("mod", ""),
        code_rate=float(row.get("code_rate", "0") or 0.0),
        tbs_bits=_to_optional_int(row.get("tbs_bits", "")),
        p50_us=_to_optional_float(row.get("p50_us", "")),
        p75_us=_to_optional_float(row.get("p75_us", "")),
        p90_us=_to_optional_float(row.get("p90_us", "")),
        p99_us=_to_optional_float(row.get("p99_us", "")),
        p99_9_us=_to_optional_float(row.get("p99_9_us", "")),
        p99_99_us=_to_optional_float(row.get("p99_99_us", "")),
        p99_999_us=_to_optional_float(row.get("p99_999_us", "")),
        worst_us=_to_optional_float(row.get("worst_us", "")),
        latency_thr_mbps=_to_optional_float(row.get("latency_thr_mbps", "")),
        wall_thr_mbps=_to_optional_float(row.get("wall_thr_mbps", "")),
        avg_iters=_to_optional_float(row.get("avg_iters", "")),
        work_pkg_w=_to_optional_float(row.get("work_pkg_w", "")),
        work_pkg_dram_w=_to_optional_float(row.get("work_pkg_dram_w", "")),
        idle_pkg_w=_to_optional_float(row.get("idle_pkg_w", "")),
        idle_pkg_dram_w=_to_optional_float(row.get("idle_pkg_dram_w", "")),
        gross_pkg_j_per_bit=_to_optional_float(row.get("gross_pkg_j_per_bit", "")),
        gross_pkg_dram_j_per_bit=_to_optional_float(row.get("gross_pkg_dram_j_per_bit", "")),
        net_pkg_j_per_bit=_to_optional_float(row.get("net_pkg_j_per_bit", "")),
        net_pkg_dram_j_per_bit=_to_optional_float(row.get("net_pkg_dram_j_per_bit", "")),
        success=(row.get("success", "False").strip().lower() in {"1", "true", "yes"}),
        command=row.get("command", ""),
        error=row.get("error", ""),
    )


def load_resume_points(trace_csv: Path) -> Tuple[List[ScanPoint], set]:
    if not trace_csv.exists():
        return [], set()
    loaded: List[ScanPoint] = []
    completed_keys = set()
    with trace_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                point = row_to_scan_point(row)
            except Exception:
                continue
            loaded.append(point)
            completed_keys.add((point.cells, point.rb, point.layers, point.mcs))
    return loaded, completed_keys


def parse_benchmark_output(text: str) -> Dict[str, Optional[float]]:
    parsed: Dict[str, Optional[float]] = {
        "tbs_bits": None,
        "nof_cb": None,
        "p50_us": None,
        "p75_us": None,
        "p90_us": None,
        "p99_us": None,
        "p99_9_us": None,
        "p99_99_us": None,
        "p99_999_us": None,
        "worst_us": None,
        "latency_thr_mbps": None,
        "wall_thr_mbps": None,
        "avg_iters": None,
        "work_pkg_w": None,
        "work_pkg_dram_w": None,
        "idle_pkg_w": None,
        "idle_pkg_dram_w": None,
        "gross_pkg_j_per_bit": None,
        "gross_pkg_dram_j_per_bit": None,
        "net_pkg_j_per_bit": None,
        "net_pkg_dram_j_per_bit": None,
    }

    tbs_m = re.search(r"tbs_bits=(\d+)", text)
    if tbs_m:
        parsed["tbs_bits"] = float(tbs_m.group(1))

    seg_m = re.search(r"\bseg=(\d+)\b", text)
    if seg_m:
        parsed["nof_cb"] = float(seg_m.group(1))

    latency_rows: Dict[str, List[str]] = {}
    for line in text.splitlines():
        row_match = re.match(r"^\s*(packet|cb)\s*\|", line)
        if not row_match:
            continue
        cols = [c.strip() for c in line.split("|") if c.strip()]
        if len(cols) >= 8:
            latency_rows[row_match.group(1)] = cols

    cols = latency_rows.get("packet") or latency_rows.get("cb")
    if cols is not None:
        parsed["p50_us"] = float(cols[1])
        parsed["p75_us"] = float(cols[2])
        parsed["p90_us"] = float(cols[3])
        parsed["p99_us"] = float(cols[4])
        parsed["p99_9_us"] = float(cols[5])
        parsed["p99_99_us"] = float(cols[6])
        if len(cols) >= 9:
            parsed["p99_999_us"] = float(cols[7])
            parsed["worst_us"] = float(cols[8])
        else:
            parsed["p99_999_us"] = float(cols[7])
            parsed["worst_us"] = float(cols[7])

    lat_thr = re.search(r"Latency\s+thr\s*:\s*([0-9.eE+-]+)\s*Mb/s", text)
    if lat_thr:
        parsed["latency_thr_mbps"] = float(lat_thr.group(1))

    wall_thr = re.search(r"Throughput\s*:\s*([0-9.eE+-]+)\s*Mb/s\s*\(TBS bits\)", text)
    if wall_thr:
        parsed["wall_thr_mbps"] = float(wall_thr.group(1))

    avg_iters = re.search(r"Avg\s+iters\s*:\s*([0-9.eE+-]+)", text)
    if avg_iters:
        parsed["avg_iters"] = float(avg_iters.group(1))

    eff_pkg = re.search(r"Efficiency\s*:\s*([0-9.eE+-]+)\s*J/bit", text)
    if eff_pkg:
        parsed["net_pkg_j_per_bit"] = float(eff_pkg.group(1))
    else:
        net_pkg = re.search(r"Net\s+pkg\s*:\s*[0-9.eE+-]+\s*J\s*\(([0-9.eE+-]+)\s*J/bit\)", text)
        if net_pkg:
            parsed["net_pkg_j_per_bit"] = float(net_pkg.group(1))

    eff_pkg_dram = re.search(r"Efficiency\+dram\s*:\s*([0-9.eE+-]+)\s*J/bit", text)
    if eff_pkg_dram:
        parsed["net_pkg_dram_j_per_bit"] = float(eff_pkg_dram.group(1))
    else:
        net_pkg_dram = re.search(r"Net\s+pkg\+dram\s*:\s*[0-9.eE+-]+\s*J\s*\(([0-9.eE+-]+)\s*J/bit\)", text)
        if net_pkg_dram:
            parsed["net_pkg_dram_j_per_bit"] = float(net_pkg_dram.group(1))

    work_pkg = re.search(r"(?:Work power|Selected\s+pkg\s*:\s*[0-9.eE+-]+\s*J\s*\()\s*([0-9.eE+-]+)\s*W", text)
    if work_pkg:
        parsed["work_pkg_w"] = float(work_pkg.group(1))

    work_pkg_dram = re.search(r"(?:Work power\+dram|Selected\s+pkg\+dram\s*:\s*[0-9.eE+-]+\s*J\s*\()\s*([0-9.eE+-]+)\s*W", text)
    if work_pkg_dram:
        parsed["work_pkg_dram_w"] = float(work_pkg_dram.group(1))

    idle_pkg = re.search(r"Idle\s+pkg(?:\s*\([^)]*\))?\s*:\s*([0-9.eE+-]+)\s*W", text)
    if idle_pkg:
        parsed["idle_pkg_w"] = float(idle_pkg.group(1))

    idle_pkg_dram = re.search(r"Idle\s+pkg\+dram\s*:\s*([0-9.eE+-]+)\s*W", text)
    if idle_pkg_dram:
        parsed["idle_pkg_dram_w"] = float(idle_pkg_dram.group(1))

    thr_mbps_for_gross = parsed["wall_thr_mbps"] if parsed["wall_thr_mbps"] is not None else parsed["latency_thr_mbps"]
    if thr_mbps_for_gross and thr_mbps_for_gross > 0:
        if parsed["work_pkg_w"] is not None:
            parsed["gross_pkg_j_per_bit"] = parsed["work_pkg_w"] / (thr_mbps_for_gross * 1e6)
        if parsed["work_pkg_dram_w"] is not None:
            parsed["gross_pkg_dram_j_per_bit"] = parsed["work_pkg_dram_w"] / (thr_mbps_for_gross * 1e6)

    return parsed


def latency_value(point: ScanPoint, metric: str) -> Optional[float]:
    if metric == "p99_999":
        return point.p99_999_us
    if metric == "p99_99":
        return point.p99_99_us
    if metric == "worst":
        return point.worst_us
    return point.p99_us


def _fmt_opt(value: Optional[float], fmt: str) -> str:
    if value is None:
        return "n/a"
    return format(value, fmt)


def run_point(
    benchmark: str,
    cells: int,
    threads_used: int,
    nof_cb: Optional[int],
    cpus_arg: str,
    rb: int,
    layers: int,
    mcs: int,
    mod_order: int,
    code_rate_1024: float,
    args: argparse.Namespace,
) -> ScanPoint:
    mod = MOD_BY_ORDER[mod_order]
    code_rate = code_rate_1024 / 1024.0

    per_point_idle = args.idle_seconds if args.idle_mode == "per-point" else 0

    cmd = [
        benchmark,
        "--pure",
        "--decoder", args.decoder,
        "--dematcher", args.dematcher,
        "--threads", str(threads_used),
        "--cells", str(cells),
        "--reps", str(args.reps),
        "--iters", str(args.iters),
        "--warmup", str(args.warmup),
        "--idle", str(per_point_idle),
        "--energy-scope", args.energy_scope,
        "--rb", str(rb),
        "--symbols", str(args.symbols),
        "--layers", str(layers),
        "--mod", mod,
        "--rate", str(code_rate),
        "--rv", str(args.rv),
    ]
    if cpus_arg:
        cmd.extend(["--cpus", cpus_arg])
    if args.schedule:
        cmd.extend(["--schedule", args.schedule])
    if args.crc:
        cmd.append("--crc")
    if args.force:
        cmd.append("--force")
    if args.early_syndrome:
        cmd.append("--early-syndrome")
    if args.decode_only:
        cmd.append("--decode-only")

    command_str = " ".join(cmd)
    timeout_s = max(0.0, float(args.point_timeout_seconds))
    heartbeat_s = max(0.0, float(args.heartbeat_seconds))
    energy_start, energy_max = _read_rapl_snapshot()
    try:
        start_ts = time.monotonic()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        last_heartbeat = start_ts

        while proc.poll() is None:
            now = time.monotonic()
            elapsed = now - start_ts
            if timeout_s > 0 and elapsed > timeout_s:
                proc.kill()
                stdout, stderr = proc.communicate()
                error_text = (
                    f"timeout after {timeout_s:.1f}s\n"
                    f"stdout:\n{(stdout or '').strip()}\n"
                    f"stderr:\n{(stderr or '').strip()}"
                )
                return ScanPoint(
                    threads_used=threads_used,
                    nof_cb=nof_cb,
                    rb=rb,
                    cells=cells,
                    layers=layers,
                    mcs=mcs,
                    mod_order=mod_order,
                    mod=mod,
                    code_rate=code_rate,
                    tbs_bits=None,
                    p50_us=None,
                    p75_us=None,
                    p90_us=None,
                    p99_us=None,
                    p99_9_us=None,
                    p99_99_us=None,
                    p99_999_us=None,
                    worst_us=None,
                    latency_thr_mbps=None,
                    wall_thr_mbps=None,
                    avg_iters=None,
                    work_pkg_w=None,
                    work_pkg_dram_w=None,
                    idle_pkg_w=None,
                    idle_pkg_dram_w=None,
                    gross_pkg_j_per_bit=None,
                    gross_pkg_dram_j_per_bit=None,
                    net_pkg_j_per_bit=None,
                    net_pkg_dram_j_per_bit=None,
                    success=False,
                    command=command_str,
                    error=error_text[:4000],
                )

            if heartbeat_s > 0 and (now - last_heartbeat) >= heartbeat_s:
                print(
                    f"  [heartbeat] running point cells={cells} threads={threads_used} rb={rb} layers={layers} mcs={mcs} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
                last_heartbeat = now

            time.sleep(0.2)

        stdout, stderr = proc.communicate()
    except Exception as exc:
        return ScanPoint(
            threads_used=threads_used,
            nof_cb=nof_cb,
            rb=rb,
            cells=cells,
            layers=layers,
            mcs=mcs,
            mod_order=mod_order,
            mod=mod,
            code_rate=code_rate,
            tbs_bits=None,
            p50_us=None,
            p75_us=None,
            p90_us=None,
            p99_us=None,
            p99_9_us=None,
            p99_99_us=None,
            p99_999_us=None,
            worst_us=None,
            latency_thr_mbps=None,
            wall_thr_mbps=None,
            avg_iters=None,
            work_pkg_w=None,
            work_pkg_dram_w=None,
            idle_pkg_w=None,
            idle_pkg_dram_w=None,
            gross_pkg_j_per_bit=None,
            gross_pkg_dram_j_per_bit=None,
            net_pkg_j_per_bit=None,
            net_pkg_dram_j_per_bit=None,
            success=False,
            command=command_str,
            error=str(exc),
        )

    elapsed_s = max(0.0, time.monotonic() - start_ts)
    energy_end, _ = _read_rapl_snapshot()

    output = (stdout or "") + "\n" + (stderr or "")
    parsed = parse_benchmark_output(output)
    scoped_pkg_j, scoped_pkg_dram_j = _scoped_energy_from_snapshots(args.energy_scope, energy_start, energy_end, energy_max)
    if elapsed_s > 0 and scoped_pkg_j is not None:
        parsed["work_pkg_w"] = scoped_pkg_j / elapsed_s
    if elapsed_s > 0 and scoped_pkg_dram_j is not None:
        parsed["work_pkg_dram_w"] = scoped_pkg_dram_j / elapsed_s

    total_bits = None
    if parsed["tbs_bits"] is not None:
        total_bits = float(parsed["tbs_bits"]) * float(args.reps) * float(cells)
    elif parsed["wall_thr_mbps"] is not None and elapsed_s > 0:
        total_bits = float(parsed["wall_thr_mbps"]) * 1e6 * elapsed_s

    if total_bits is not None and total_bits > 0:
        if scoped_pkg_j is not None:
            parsed["gross_pkg_j_per_bit"] = scoped_pkg_j / total_bits
        if scoped_pkg_dram_j is not None:
            parsed["gross_pkg_dram_j_per_bit"] = scoped_pkg_dram_j / total_bits

    success = proc.returncode == 0 and parsed["p50_us"] is not None
    failure_prefix = "" if proc.returncode == 0 else f"returncode={proc.returncode}\n"

    return ScanPoint(
        threads_used=threads_used,
        nof_cb=nof_cb if nof_cb is not None else (int(parsed["nof_cb"]) if parsed["nof_cb"] is not None else None),
        rb=rb,
        cells=cells,
        layers=layers,
        mcs=mcs,
        mod_order=mod_order,
        mod=mod,
        code_rate=code_rate,
        tbs_bits=int(parsed["tbs_bits"]) if parsed["tbs_bits"] is not None else None,
        p50_us=parsed["p50_us"],
        p75_us=parsed["p75_us"],
        p90_us=parsed["p90_us"],
        p99_us=parsed["p99_us"],
        p99_9_us=parsed["p99_9_us"],
        p99_99_us=parsed["p99_99_us"],
        p99_999_us=parsed["p99_999_us"],
        worst_us=parsed["worst_us"],
        latency_thr_mbps=parsed["latency_thr_mbps"],
        wall_thr_mbps=parsed["wall_thr_mbps"],
        avg_iters=parsed["avg_iters"],
        work_pkg_w=parsed["work_pkg_w"],
        work_pkg_dram_w=parsed["work_pkg_dram_w"],
        idle_pkg_w=parsed["idle_pkg_w"],
        idle_pkg_dram_w=parsed["idle_pkg_dram_w"],
        gross_pkg_j_per_bit=parsed["gross_pkg_j_per_bit"],
        gross_pkg_dram_j_per_bit=parsed["gross_pkg_dram_j_per_bit"],
        net_pkg_j_per_bit=parsed["net_pkg_j_per_bit"],
        net_pkg_dram_j_per_bit=parsed["net_pkg_dram_j_per_bit"],
        success=success,
        command=command_str,
        error="" if success else (failure_prefix + output.strip())[:4000],
    )


def probe_nof_cb(
    benchmark: str,
    cpus_arg: str,
    rb: int,
    layers: int,
    mod: str,
    code_rate: float,
    args: argparse.Namespace,
) -> Optional[int]:
    cmd = [
        benchmark,
        "--pure",
        "--decoder", args.decoder,
        "--dematcher", args.dematcher,
        "--threads", "1",
        "--cells", "1",
        "--reps", "1",
        "--iters", str(args.iters),
        "--warmup", "0",
        "--idle", "0",
        "--energy-scope", args.energy_scope,
        "--rb", str(rb),
        "--symbols", str(args.symbols),
        "--layers", str(layers),
        "--mod", mod,
        "--rate", str(code_rate),
        "--rv", str(args.rv),
    ]
    if cpus_arg:
        cmd.extend(["--cpus", cpus_arg])
    if args.schedule:
        cmd.extend(["--schedule", args.schedule])
    if args.crc:
        cmd.append("--crc")
    if args.force:
        cmd.append("--force")
    if args.early_syndrome:
        cmd.append("--early-syndrome")
    if args.decode_only:
        cmd.append("--decode-only")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception:
        return None

    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    parsed = parse_benchmark_output(output)
    if parsed["nof_cb"] is None:
        return None
    return int(parsed["nof_cb"])


def write_csv(path: Path, points: Sequence[ScanPoint]) -> None:
    if not points:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(points[0]).keys()))
        writer.writeheader()
        for point in points:
            writer.writerow(asdict(point))


def save_outputs(
    *,
    out_dir: Path,
    points: Sequence[ScanPoint],
    args: argparse.Namespace,
    latency_limit: Optional[float],
    run_metadata: Optional[Dict[str, object]] = None,
) -> Tuple[Path, Path, Path]:
    trace_csv = out_dir / "cpu_tbs_grid_trace.csv"
    write_csv(trace_csv, points)

    summary_rows = choose_best(points, args.latency_metric, latency_limit, args.energy_metric)
    summary_csv = out_dir / "cpu_tbs_grid_summary.csv"
    if summary_rows:
        summary_fieldnames: List[str] = []
        for row in summary_rows:
            for key in row.keys():
                if key not in summary_fieldnames:
                    summary_fieldnames.append(key)
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary_rows)

    summary_json = out_dir / "cpu_tbs_grid_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "run_metadata": run_metadata or {},
                "total_points": len(points),
                "successful_points": sum(1 for point in points if point.success),
                "best_points": summary_rows,
                "trace_csv": str(trace_csv),
                "summary_csv": str(summary_csv),
            },
            f,
            indent=2,
        )

    return trace_csv, summary_csv, summary_json


def choose_best(
    points: Sequence[ScanPoint],
    latency_metric: str,
    latency_limit_us: Optional[float],
    energy_metric: str,
) -> List[dict]:
    groups: Dict[Tuple[int, int, int, int], List[ScanPoint]] = {}
    for point in points:
        groups.setdefault((point.cells, point.rb, point.layers, point.mcs), []).append(point)

    rows: List[dict] = []
    for (cells, rb, layers, mcs), candidates in sorted(groups.items()):
        ok = []
        for point in candidates:
            if not point.success:
                continue
            lv = latency_value(point, latency_metric)
            if lv is None:
                continue
            if latency_limit_us is not None and lv > latency_limit_us:
                continue
            em = point.net_pkg_j_per_bit if energy_metric == "pkg" else point.net_pkg_dram_j_per_bit
            if em is None or em <= 0:
                continue
            ok.append((em, point))

        if not ok:
            rows.append({
                "cells": cells,
                "rb": rb,
                "layers": layers,
                "mcs": mcs,
                "selected": False,
                "reason": "no_valid_point_under_latency_constraint",
            })
            continue

        ok.sort(key=lambda item: item[0])
        best = ok[0][1]
        row = asdict(best)
        row["selected"] = True
        row["energy_metric"] = energy_metric
        row["latency_metric"] = latency_metric
        row["latency_limit_us"] = latency_limit_us
        row["qee_score"] = 1.0 / ok[0][0]
        rows.append(row)

    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan real-load LDPC CPU decode points over RB x layers x MCS and export latency/throughput/energy.",
    )
    parser.add_argument("--benchmark", required=True, help="Path to ldpc_decoder_energy_benchmark binary")
    parser.add_argument("--out-dir", default="scan_out_tbs_cpu", help="Output directory")

    parser.add_argument("--rb-list", default="10,20,50,100,273", help="RB list/range, e.g. 10,20,50 or 10-100")
    parser.add_argument("--layers-list", default="1,2,4", help="Layer list/range")
    parser.add_argument("--mcs-list", default="0-27", help="MCS list/range")
    parser.add_argument("--mcs-table", type=int, choices=[1, 2, 3], default=1, help="MCS table index")

    parser.add_argument("--threads", type=int, default=1, help="Number of CB worker threads")
    parser.add_argument("--auto-threads", action="store_true", help="Auto set threads=cells*nof_cb by probing each point")
    parser.add_argument("--auto-threads-max", type=int, default=16, help="Cap for auto-threads; <=0 disables cap")
    parser.add_argument("--cells-list", default="", help="Concurrent TBs list/range, mapped to benchmark --cells; e.g. 1,2,4,8")
    parser.add_argument("--cpus", default="", help="CPU list for benchmark, e.g. 0-15,32-47")
    parser.add_argument("--symbols", type=int, default=14)
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--idle-seconds", type=int, default=2)
    parser.add_argument("--idle-mode", choices=["once", "per-point"], default="once", help="Idle baseline mode: once=measure once and reuse; per-point=benchmark measures idle every point")
    parser.add_argument("--rv", type=int, default=0)
    parser.add_argument("--energy-scope", choices=["sum", "socket0", "socket1"], default="sum")
    parser.add_argument("--decoder", default="generic")
    parser.add_argument("--dematcher", default="generic")
    parser.add_argument("--schedule", default="", help="Optional scheduler mode passed to benchmark, e.g. dynamic or static-cb")

    parser.add_argument("--crc", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--early-syndrome", action="store_true")
    parser.add_argument("--decode-only", action="store_true")

    parser.add_argument("--latency-metric", choices=["p99", "p99_99", "p99_999", "worst"], default="p99_999")
    parser.add_argument("--latency-limit-us", type=float, default=1000.0, help="Latency guard in microseconds; use <0 to disable")
    parser.add_argument("--energy-metric", choices=["pkg", "pkg_dram"], default="pkg")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save partial trace/summary every N completed points")
    parser.add_argument("--point-timeout-seconds", type=float, default=0.0, help="Timeout for each point; <=0 disables timeout")
    parser.add_argument("--heartbeat-seconds", type=float, default=0.0, help="Heartbeat interval while a point is running; <=0 disables")
    parser.add_argument("--pause-between-points-seconds", type=float, default=0.0, help="Optional pause between points; <=0 disables")
    parser.add_argument("--resume", action="store_true", help="Resume from existing trace CSV in out-dir and skip completed points")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    original_decoder = args.decoder
    original_dematcher = args.dematcher
    args.decoder = normalize_backend(args.decoder)
    args.dematcher = normalize_backend(args.dematcher)
    if original_decoder != args.decoder:
        print(f"[warn] normalized decoder backend '{original_decoder}' -> '{args.decoder}'", flush=True)
    if original_dematcher != args.dematcher:
        print(f"[warn] normalized dematcher backend '{original_dematcher}' -> '{args.dematcher}'", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mcs_table = MCS_TABLES[args.mcs_table]
    cells_list = parse_int_list(args.cells_list) if args.cells_list else [1]
    rb_list = parse_int_list(args.rb_list)
    layers_list = parse_int_list(args.layers_list)
    mcs_list = parse_int_list(args.mcs_list)

    unknown_mcs = [m for m in mcs_list if m not in mcs_table]
    if unknown_mcs:
        raise ValueError(f"MCS not present in table {args.mcs_table}: {unknown_mcs}")

    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"benchmark not found: {benchmark_path}")
    if not benchmark_path.is_file():
        raise ValueError(f"benchmark is not a file: {benchmark_path}")
    if not os.access(benchmark_path, os.X_OK):
        raise PermissionError(
            f"benchmark is not executable: {benchmark_path} (pass built binary, not source .cpp)"
        )
    benchmark = str(benchmark_path)
    trace_csv = out_dir / "cpu_tbs_grid_trace.csv"
    points: List[ScanPoint] = []
    completed_keys = set()
    if args.resume:
        points, completed_keys = load_resume_points(trace_csv)
        print(f"[resume] loaded {len(points)} points from {trace_csv}", flush=True)
    probe_cache: Dict[Tuple[int, int, int], Optional[int]] = {}

    global_idle_pkg_w: Optional[float] = None
    global_idle_pkg_dram_w: Optional[float] = None
    if args.idle_mode == "once" and args.idle_seconds > 0:
        print(f"[info] measuring global idle once for {args.idle_seconds}s (scope={args.energy_scope})", flush=True)
        global_idle_pkg_w, global_idle_pkg_dram_w = measure_idle_power_once(args.idle_seconds, args.energy_scope)
        if global_idle_pkg_w is None or global_idle_pkg_dram_w is None:
            print("[warn] global idle measurement unavailable; net energy may stay unchanged", flush=True)
        else:
            print(
                f"[info] global idle baseline: pkg={global_idle_pkg_w:.6f} W, pkg+dram={global_idle_pkg_dram_w:.6f} W",
                flush=True,
            )

    run_metadata: Dict[str, object] = {
        "idle_mode": args.idle_mode,
        "global_idle_pkg_w": global_idle_pkg_w,
        "global_idle_pkg_dram_w": global_idle_pkg_dram_w,
        "pause_between_points_seconds": args.pause_between_points_seconds,
    }

    detected_cpus = os.cpu_count() or 1
    preferred_auto_pool = parse_int_list("16-31,0-15,32-47")
    auto_cpu_pool = [cpu for cpu in preferred_auto_pool if cpu < detected_cpus]
    if not auto_cpu_pool:
        auto_cpu_pool = list(range(detected_cpus))
    auto_cpu_mode = args.auto_threads and not args.cpus
    if auto_cpu_mode:
        print("[info] auto-threads: auto affinity enabled with cpu pool 16-31,0-15,32-47 (prefer 16-31)", flush=True)
    auto_threads_max = int(args.auto_threads_max)

    if auto_cpu_mode:
        available_cores = len(auto_cpu_pool)
    elif args.cpus:
        try:
            available_cores = len(parse_int_list(args.cpus))
        except Exception:
            available_cores = os.cpu_count() or 1
    else:
        available_cores = os.cpu_count() or 1

    total = len(cells_list) * len(rb_list) * len(layers_list) * len(mcs_list)
    latency_limit = None if args.latency_limit_us < 0 else args.latency_limit_us
    checkpoint_every = max(1, args.checkpoint_every)
    job_start = time.perf_counter()
    run_done = 0
    progress_index = 0
    try:
        for cells in cells_list:
            for rb in rb_list:
                for layers in layers_list:
                    for mcs in mcs_list:
                        mod_order, code_rate_1024 = mcs_table[mcs]
                        if mod_order not in MOD_BY_ORDER:
                            continue
                        mod = MOD_BY_ORDER[mod_order]
                        code_rate = code_rate_1024 / 1024.0

                        progress_index += 1
                        point_key = (cells, rb, layers, mcs)
                        if point_key in completed_keys:
                            print(f"[{progress_index}/{total}] skip existing cells={cells} rb={rb} layers={layers} mcs={mcs}", flush=True)
                            continue

                        nof_cb: Optional[int] = None
                        threads_used = args.threads
                        cpus_arg = args.cpus
                        if args.auto_threads:
                            if auto_cpu_mode:
                                if threads_used <= len(auto_cpu_pool):
                                    cpus_arg = format_cpu_list(auto_cpu_pool[:threads_used])
                                else:
                                    cpus_arg = format_cpu_list(auto_cpu_pool)
                            key = (rb, layers, mcs)
                            if key not in probe_cache:
                                probe_cache[key] = probe_nof_cb(
                                    benchmark=benchmark,
                                    cpus_arg=cpus_arg,
                                    rb=rb,
                                    layers=layers,
                                    mod=mod,
                                    code_rate=code_rate,
                                    args=args,
                                )
                            nof_cb = probe_cache[key]
                            if nof_cb is not None and nof_cb > 0:
                                threads_used = cells * nof_cb
                                if auto_threads_max > 0 and threads_used > auto_threads_max:
                                    threads_used = auto_threads_max
                                if auto_cpu_mode:
                                    if threads_used <= len(auto_cpu_pool):
                                        cpus_arg = format_cpu_list(auto_cpu_pool[:threads_used])
                                    else:
                                        cpus_arg = format_cpu_list(auto_cpu_pool)
                            else:
                                print(f"[warn] probe nof_cb failed at rb={rb} layers={layers} mcs={mcs}, fallback threads={threads_used}", flush=True)

                        if threads_used > available_cores:
                            print(
                                f"[warn] threads={threads_used} > available_cores={available_cores} "
                                f"(cells={cells}, rb={rb}, layers={layers}, mcs={mcs})",
                                flush=True,
                            )
                        if auto_cpu_mode and threads_used > len(auto_cpu_pool):
                            print(
                                f"[warn] threads={threads_used} exceeds auto cpu pool size={len(auto_cpu_pool)}; affinity will wrap",
                                flush=True,
                            )

                        nof_cb_msg = "?" if nof_cb is None else str(nof_cb)
                        print(
                            f"[{progress_index}/{total}] start cells={cells} threads={threads_used} nof_cb={nof_cb_msg} "
                            f"rb={rb} layers={layers} mcs={mcs}",
                            flush=True,
                        )

                        point_start = time.perf_counter()
                        point = run_point(
                            benchmark=benchmark,
                            cells=cells,
                            threads_used=threads_used,
                            nof_cb=nof_cb,
                            cpus_arg=cpus_arg,
                            rb=rb,
                            layers=layers,
                            mcs=mcs,
                            mod_order=mod_order,
                            code_rate_1024=code_rate_1024,
                            args=args,
                        )
                        point_elapsed = time.perf_counter() - point_start

                        if args.idle_mode == "once" and point.success:
                            if global_idle_pkg_w is not None:
                                point.idle_pkg_w = global_idle_pkg_w
                            if global_idle_pkg_dram_w is not None:
                                point.idle_pkg_dram_w = global_idle_pkg_dram_w

                            if point.wall_thr_mbps and point.wall_thr_mbps > 0:
                                throughput_bps = point.wall_thr_mbps * 1e6
                                if point.work_pkg_w is not None and global_idle_pkg_w is not None:
                                    point.net_pkg_j_per_bit = max(0.0, point.work_pkg_w - global_idle_pkg_w) / throughput_bps
                                if point.work_pkg_dram_w is not None and global_idle_pkg_dram_w is not None:
                                    point.net_pkg_dram_j_per_bit = max(0.0, point.work_pkg_dram_w - global_idle_pkg_dram_w) / throughput_bps

                        points.append(point)
                        completed_keys.add(point_key)
                        run_done += 1

                        elapsed = time.perf_counter() - job_start
                        finished = len(completed_keys)
                        avg = elapsed / max(1, run_done)
                        eta = avg * max(0, total - finished)
                        p99_9_text = "n/a" if point.p99_9_us is None else f"{point.p99_9_us:.3f}us"
                        p99_99_text = "n/a" if point.p99_99_us is None else f"{point.p99_99_us:.3f}us"
                        p99_999_text = "n/a" if point.p99_999_us is None else f"{point.p99_999_us:.3f}us"
                        idle_pkg_text = _fmt_opt(point.idle_pkg_w, ".3f")
                        idle_pkg_dram_text = _fmt_opt(point.idle_pkg_dram_w, ".3f")
                        eff_pkg_text = _fmt_opt(point.net_pkg_j_per_bit, ".3e")
                        eff_pkg_dram_text = _fmt_opt(point.net_pkg_dram_j_per_bit, ".3e")
                        status = "ok" if point.success else "fail"
                        print(
                            f"[{progress_index}/{total}] done={status} point={point_elapsed:.2f}s total={elapsed:.1f}s "
                            f"eta={eta/60:.1f}m p99.9={p99_9_text} p99.99={p99_99_text} p99.999={p99_999_text} "
                            f"idleW={idle_pkg_text} idleW+dram={idle_pkg_dram_text} "
                            f"effJ/bit={eff_pkg_text} effJ/bit+dram={eff_pkg_dram_text}",
                            flush=True,
                        )

                        if run_done % checkpoint_every == 0:
                            trace_csv, summary_csv, summary_json = save_outputs(
                                out_dir=out_dir,
                                points=points,
                                args=args,
                                latency_limit=latency_limit,
                                run_metadata=run_metadata,
                            )
                            print(
                                f"[checkpoint] saved {len(completed_keys)}/{total} -> trace={trace_csv} summary={summary_csv}",
                                flush=True,
                            )

                        if args.pause_between_points_seconds > 0:
                            time.sleep(args.pause_between_points_seconds)
    except KeyboardInterrupt:
        trace_csv, summary_csv, summary_json = save_outputs(
            out_dir=out_dir,
            points=points,
            args=args,
            latency_limit=latency_limit,
            run_metadata=run_metadata,
        )
        print(
            f"[interrupt] partial results saved: trace={trace_csv} summary={summary_csv} json={summary_json}",
            flush=True,
        )
        return 130

    trace_csv, summary_csv, summary_json = save_outputs(
        out_dir=out_dir,
        points=points,
        args=args,
        latency_limit=latency_limit,
        run_metadata=run_metadata,
    )

    print(f"Done. trace={trace_csv} summary={summary_csv} json={summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

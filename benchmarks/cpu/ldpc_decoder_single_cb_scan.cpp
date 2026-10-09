/*
 *
 * Copyright 2021-2026 Software Radio Systems Limited
 *
 * By using this file, you agree to the terms and conditions set
 * forth in the LICENSE file which can be found at the top level of
 * the distribution.
 *
 */

/// \file
/// \brief LDPC decoder single-codeblock CPU scan benchmark.

#include "ocudu/phy/upper/channel_coding/channel_coding_factories.h"
#include "ocudu/support/ocudu_test.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <sched.h>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace ocudu;
using namespace ocudu::ldpc;

namespace {

static const std::array<unsigned, 51> ZC_FULL_UNION = {
    2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,  18,  20,
    22, 24, 26, 28, 30, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72,  80,  88,
    96, 104, 112, 120, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288,
    320, 352, 384};

static const std::array<unsigned, 9> ZC_POW2 = {2, 4, 8, 16, 32, 64, 128, 256, 384};

struct options {
  std::string decoder_type = "generic";
  std::string bg = "both";
  std::string zc_mode = "full";
  std::string zc_list;
  std::string iters_list = "1,2,3,4,5,6,8,10,12";
  std::string rate_control_mode = "fixed";
  float       rate_fixed = 0.5F;
  std::string rate_list = "0.1171875,0.25,0.33203125,0.40,0.50,0.60,0.650390625,0.66,0.75,0.83,0.92578125";
  std::string rm_len_mode = "from_rate";
  unsigned    rm_len_fixed = 28800;
  std::string energy_scope = "sum";
  unsigned    idle_seconds = 0;
  unsigned    reps = 200;
  unsigned    warmup = 30;
  unsigned    nof_threads = 1;
  std::string threads_list;
  std::string cpu_list;
  unsigned    nof_filler_bits = 0;
  bool        use_crc = false;
  unsigned    seed = 0;
  std::string out_csv;
  bool        silent = false;
};

struct rapl_domain {
  std::string energy_path;
  std::string max_path;
  bool        ok = false;
  uint64_t    max_uj = 0;
};

struct rapl_snapshot {
  uint64_t pkg0 = 0;
  uint64_t dram0 = 0;
  uint64_t pkg1 = 0;
  uint64_t dram1 = 0;
};

struct rapl_energy {
  uint64_t pkg0 = 0;
  uint64_t dram0 = 0;
  uint64_t pkg1 = 0;
  uint64_t dram1 = 0;
};

struct rapl_topology {
  rapl_domain pkg0;
  rapl_domain dram0;
  rapl_domain pkg1;
  rapl_domain dram1;
  bool        ok_any = false;
};

enum class rapl_scope { sum, socket0, socket1 };

struct latency_stats {
  unsigned n = 0;
  double   mean_us = 0.0;
  double   min_us = 0.0;
  double   p1_us = 0.0;
  double   p5_us = 0.0;
  double   p20_us = 0.0;
  double   p50_us = 0.0;
  double   p95_us = 0.0;
  double   p99_us = 0.0;
  double   p99_9_us = 0.0;
  double   p99_99_us = 0.0;
  double   p99_999_us = 0.0;
  double   max_us = 0.0;
};

static void usage(const char* prog)
{
  std::cout << "Usage: " << prog << " [options]\n";
  std::cout << "  -T, --decoder      Decoder type (generic,avx2,avx512,neon) [Default generic]\n";
  std::cout << "      --bg           Base graph: 1, 2, both [Default both]\n";
  std::cout << "      --zc-mode      full, pow2, custom [Default full]\n";
  std::cout << "      --zc-list      Comma-separated Zc list when --zc-mode=custom\n";
  std::cout << "  -I, --iters-list   Comma-separated max-iteration list [Default 1,2,3,4,5,6,8,10,12]\n";
  std::cout << "      --rate-control-mode  fixed or sweep [Default fixed]\n";
  std::cout << "      --rate-fixed    Fixed code rate [Default 0.5]\n";
  std::cout << "      --rate-list     Comma-separated rates when mode=sweep\n";
  std::cout << "      --rm-len-mode   from_rate or fixed [Default from_rate]\n";
  std::cout << "      --rm-len-fixed  Fixed input length E when mode=fixed [Default 28800]\n";
  std::cout << "  -E, --energy-scope  Energy scope: sum, socket0, socket1 [Default sum]\n";
  std::cout << "  -x, --idle          Idle seconds for net energy baseline [Default 0]\n";
  std::cout << "  -R, --reps         Repetitions per point [Default 200]\n";
  std::cout << "  -w, --warmup       Warmup repetitions per point [Default 30]\n";
  std::cout << "  -n, --threads      Number of worker threads [Default 1]\n";
  std::cout << "      --threads-list Comma-separated thread list for scan, e.g. 1,2,4,8\n";
  std::cout << "  -c, --cpu-list     CPU list for thread pinning, e.g. 0-7,16-23\n";
  std::cout << "  -f, --filler       Number of filler bits [Default 0]\n";
  std::cout << "  -C, --crc          Enable CRC early stop [Default off]\n";
  std::cout << "  -S, --seed         RNG seed [Default 0]\n";
  std::cout << "  -o, --out-csv      Output CSV path [Default none]\n";
  std::cout << "  -s, --silent       Silent mode [Default off]\n";
  std::cout << "  -h, --help         Show this message\n";
}

static bool read_uj(const std::string& path, uint64_t& out)
{
  std::ifstream file(path);
  if (!file.is_open()) {
    return false;
  }
  std::string value;
  std::getline(file, value);
  if (value.empty()) {
    return false;
  }
  char* endp = nullptr;
  unsigned long long v = std::strtoull(value.c_str(), &endp, 10);
  if (!endp || endp == value.c_str()) {
    return false;
  }
  out = static_cast<uint64_t>(v);
  return true;
}

static rapl_domain make_domain(const std::string& energy_path, const std::string& max_path)
{
  rapl_domain d;
  d.energy_path = energy_path;
  d.max_path = max_path;
  uint64_t max_uj = 0;
  uint64_t sample_uj = 0;
  if (read_uj(max_path, max_uj) && max_uj > 0 && read_uj(energy_path, sample_uj)) {
    d.max_uj = max_uj;
    d.ok = true;
  }
  return d;
}

static rapl_topology detect_rapl()
{
  rapl_topology t;
  t.pkg0 = make_domain("/sys/class/powercap/intel-rapl:0/energy_uj",
                       "/sys/class/powercap/intel-rapl:0/max_energy_range_uj");
  t.dram0 = make_domain("/sys/class/powercap/intel-rapl:0:0/energy_uj",
                        "/sys/class/powercap/intel-rapl:0:0/max_energy_range_uj");
  t.pkg1 = make_domain("/sys/class/powercap/intel-rapl:1/energy_uj",
                       "/sys/class/powercap/intel-rapl:1/max_energy_range_uj");
  t.dram1 = make_domain("/sys/class/powercap/intel-rapl:1:0/energy_uj",
                        "/sys/class/powercap/intel-rapl:1:0/max_energy_range_uj");
  t.ok_any = t.pkg0.ok || t.dram0.ok || t.pkg1.ok || t.dram1.ok;
  return t;
}

static rapl_snapshot read_rapl(const rapl_topology& t)
{
  rapl_snapshot s;
  if (t.pkg0.ok) {
    read_uj(t.pkg0.energy_path, s.pkg0);
  }
  if (t.dram0.ok) {
    read_uj(t.dram0.energy_path, s.dram0);
  }
  if (t.pkg1.ok) {
    read_uj(t.pkg1.energy_path, s.pkg1);
  }
  if (t.dram1.ok) {
    read_uj(t.dram1.energy_path, s.dram1);
  }
  return s;
}

static uint64_t diff_uj(uint64_t start, uint64_t end, uint64_t max_uj)
{
  if (end >= start) {
    return end - start;
  }
  if (max_uj == 0) {
    return 0;
  }
  return (max_uj - start) + end;
}

static rapl_energy compute_rapl_energy(const rapl_topology& t, const rapl_snapshot& s0, const rapl_snapshot& s1)
{
  rapl_energy e;
  if (t.pkg0.ok) {
    e.pkg0 = diff_uj(s0.pkg0, s1.pkg0, t.pkg0.max_uj);
  }
  if (t.dram0.ok) {
    e.dram0 = diff_uj(s0.dram0, s1.dram0, t.dram0.max_uj);
  }
  if (t.pkg1.ok) {
    e.pkg1 = diff_uj(s0.pkg1, s1.pkg1, t.pkg1.max_uj);
  }
  if (t.dram1.ok) {
    e.dram1 = diff_uj(s0.dram1, s1.dram1, t.dram1.max_uj);
  }
  return e;
}

static rapl_scope parse_scope(const std::string& s)
{
  if (s == "socket0") {
    return rapl_scope::socket0;
  }
  if (s == "socket1") {
    return rapl_scope::socket1;
  }
  return rapl_scope::sum;
}

static void scope_energy(const rapl_energy& e, rapl_scope scope, uint64_t& pkg, uint64_t& dram)
{
  switch (scope) {
    case rapl_scope::socket0:
      pkg = e.pkg0;
      dram = e.dram0;
      break;
    case rapl_scope::socket1:
      pkg = e.pkg1;
      dram = e.dram1;
      break;
    case rapl_scope::sum:
    default:
      pkg = e.pkg0 + e.pkg1;
      dram = e.dram0 + e.dram1;
      break;
  }
}

static std::vector<unsigned> parse_uint_list(const std::string& text)
{
  std::vector<unsigned> out;
  std::stringstream     ss(text);
  std::string           token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) {
      continue;
    }
    unsigned v = static_cast<unsigned>(std::strtoul(token.c_str(), nullptr, 10));
    if (v > 0) {
      out.push_back(v);
    }
  }
  return out;
}

static std::vector<float> parse_float_list(const std::string& text)
{
  std::vector<float> out;
  std::stringstream  ss(text);
  std::string        token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) {
      continue;
    }
    float v = std::strtof(token.c_str(), nullptr);
    if (v > 0.0F && v <= 1.0F) {
      out.push_back(v);
    }
  }
  return out;
}

static std::vector<unsigned> build_zc_values(const options& opt)
{
  if (opt.zc_mode == "pow2") {
    return std::vector<unsigned>(ZC_POW2.begin(), ZC_POW2.end());
  }
  if (opt.zc_mode == "custom") {
    return parse_uint_list(opt.zc_list);
  }
  return std::vector<unsigned>(ZC_FULL_UNION.begin(), ZC_FULL_UNION.end());
}

static std::vector<unsigned> build_bg_values(const options& opt)
{
  if (opt.bg == "both") {
    return {2U, 1U};
  }
  if (opt.bg == "1") {
    return {1U};
  }
  if (opt.bg == "2") {
    return {2U};
  }
  return {};
}

static std::vector<float> build_rate_values(const options& opt)
{
  if (opt.rate_control_mode == "sweep") {
    return parse_float_list(opt.rate_list);
  }
  return {opt.rate_fixed};
}

static std::vector<int> parse_cpu_list(const std::string& text)
{
  std::vector<int> cpus;
  if (text.empty()) {
    return cpus;
  }

  std::stringstream ss(text);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) {
      continue;
    }

    size_t dash_pos = token.find('-');
    if (dash_pos == std::string::npos) {
      int cpu = std::stoi(token);
      if (cpu >= 0) {
        cpus.push_back(cpu);
      }
      continue;
    }

    int start_cpu = std::stoi(token.substr(0, dash_pos));
    int end_cpu = std::stoi(token.substr(dash_pos + 1));
    if (start_cpu > end_cpu) {
      std::swap(start_cpu, end_cpu);
    }
    for (int cpu = start_cpu; cpu <= end_cpu; ++cpu) {
      if (cpu >= 0) {
        cpus.push_back(cpu);
      }
    }
  }

  std::vector<int> unique_cpus;
  std::unordered_set<int> seen;
  for (int cpu : cpus) {
    if (seen.insert(cpu).second) {
      unique_cpus.push_back(cpu);
    }
  }
  return unique_cpus;
}

static int default_cpu_for_thread(unsigned tid)
{
  if (tid <= 15U) {
    return static_cast<int>(tid);
  }
  return 32 + static_cast<int>(tid - 16U);
}

static bool set_thread_affinity(const std::vector<int>& cpu_list, unsigned tid, bool verbose)
{
  int cpu = -1;
  if (!cpu_list.empty()) {
    cpu = cpu_list[tid % cpu_list.size()];
  } else {
    cpu = default_cpu_for_thread(tid);
  }
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(cpu, &mask);
  if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
    if (verbose) {
      std::cerr << "[WARN] sched_setaffinity failed for tid=" << tid << " cpu=" << cpu
                << " err=" << std::strerror(errno) << "\n";
    }
    return false;
  }
  return true;
}

static std::vector<unsigned> build_thread_values(const options& opt)
{
  if (opt.threads_list.empty()) {
    return {opt.nof_threads};
  }

  std::vector<unsigned> values = parse_uint_list(opt.threads_list);
  std::vector<unsigned> unique_values;
  std::unordered_set<unsigned> seen;
  for (unsigned v : values) {
    if (v > 0 && seen.insert(v).second) {
      unique_values.push_back(v);
    }
  }
  return unique_values;
}

static void parse_args(int argc, char** argv, options& opt)
{
  static struct option long_opts[] = {{"decoder", required_argument, nullptr, 'T'},
                                       {"bg", required_argument, nullptr, 'b'},
                                       {"zc-mode", required_argument, nullptr, 'm'},
                                       {"zc-list", required_argument, nullptr, 'z'},
                                       {"iters-list", required_argument, nullptr, 'I'},
                                       {"rate-control-mode", required_argument, nullptr, 'k'},
                                       {"rate-fixed", required_argument, nullptr, 'r'},
                                       {"rate-list", required_argument, nullptr, 'l'},
                                       {"rm-len-mode", required_argument, nullptr, 'M'},
                                       {"rm-len-fixed", required_argument, nullptr, 'F'},
                                       {"energy-scope", required_argument, nullptr, 'E'},
                                       {"idle", required_argument, nullptr, 'x'},
                                       {"reps", required_argument, nullptr, 'R'},
                                       {"warmup", required_argument, nullptr, 'w'},
                                       {"threads", required_argument, nullptr, 'n'},
                                       {"threads-list", required_argument, nullptr, 'N'},
                                       {"cpu-list", required_argument, nullptr, 'c'},
                                       {"filler", required_argument, nullptr, 'f'},
                                       {"crc", no_argument, nullptr, 'C'},
                                       {"seed", required_argument, nullptr, 'S'},
                                       {"out-csv", required_argument, nullptr, 'o'},
                                       {"silent", no_argument, nullptr, 's'},
                                       {"help", no_argument, nullptr, 'h'},
                                       {nullptr, 0, nullptr, 0}};

  int c;
  while ((c = getopt_long(argc, argv, "T:b:m:z:I:k:r:l:M:F:E:x:R:w:n:N:c:f:CS:o:sh", long_opts, nullptr)) != -1) {
    switch (c) {
      case 'T':
        opt.decoder_type = optarg;
        break;
      case 'b':
        opt.bg = optarg;
        break;
      case 'm':
        opt.zc_mode = optarg;
        break;
      case 'z':
        opt.zc_list = optarg;
        break;
      case 'I':
        opt.iters_list = optarg;
        break;
      case 'k':
        opt.rate_control_mode = optarg;
        break;
      case 'r':
        opt.rate_fixed = std::strtof(optarg, nullptr);
        break;
      case 'l':
        opt.rate_list = optarg;
        break;
      case 'M':
        opt.rm_len_mode = optarg;
        break;
      case 'F':
        opt.rm_len_fixed = static_cast<unsigned>(std::strtoul(optarg, nullptr, 10));
        break;
      case 'E':
        opt.energy_scope = optarg;
        break;
      case 'x':
        opt.idle_seconds = static_cast<unsigned>(std::strtoul(optarg, nullptr, 10));
        break;
      case 'R':
        opt.reps = static_cast<unsigned>(std::strtoul(optarg, nullptr, 10));
        break;
      case 'w':
        opt.warmup = static_cast<unsigned>(std::strtoul(optarg, nullptr, 10));
        break;
      case 'n':
        opt.nof_threads = static_cast<unsigned>(std::strtoul(optarg, nullptr, 10));
        break;
      case 'N':
        opt.threads_list = optarg;
        break;
      case 'c':
        opt.cpu_list = optarg;
        break;
      case 'f':
        opt.nof_filler_bits = static_cast<unsigned>(std::strtoul(optarg, nullptr, 10));
        break;
      case 'C':
        opt.use_crc = true;
        break;
      case 'S':
        opt.seed = static_cast<unsigned>(std::strtoul(optarg, nullptr, 10));
        break;
      case 'o':
        opt.out_csv = optarg;
        break;
      case 's':
        opt.silent = true;
        break;
      case 'h':
      default:
        usage(argv[0]);
        std::exit(0);
    }
  }
}

static unsigned get_k(unsigned bg, unsigned zc)
{
  return ((bg == 1) ? 22U : 10U) * zc;
}

static unsigned get_nshort(unsigned bg)
{
  return (bg == 1) ? 66U : 50U;
}

static bool is_valid_zc(unsigned zc)
{
  return std::find(ZC_FULL_UNION.begin(), ZC_FULL_UNION.end(), zc) != ZC_FULL_UNION.end();
}

static unsigned choose_input_length(unsigned bg, unsigned zc, unsigned k_bits, float code_rate)
{
  unsigned min_input = k_bits + 2U * zc;
  unsigned max_input = get_nshort(bg) * zc;

  double   inv_rate = (code_rate > 0.0F) ? (1.0 / static_cast<double>(code_rate)) : 2.0;
  unsigned by_rate = static_cast<unsigned>(std::ceil(static_cast<double>(k_bits) * inv_rate));

  unsigned e = std::max(min_input, by_rate);
  e = std::min(e, max_input);
  return e;
}

static unsigned choose_input_length_fixed(unsigned bg, unsigned zc, unsigned k_bits, unsigned rm_len_fixed)
{
  unsigned min_input = k_bits + 2U * zc;
  unsigned max_input = get_nshort(bg) * zc;
  unsigned e = std::max(min_input, rm_len_fixed);
  e = std::min(e, max_input);
  return e;
}

static double percentile_us(const std::vector<uint64_t>& sorted_ns, double p)
{
  if (sorted_ns.empty()) {
    return 0.0;
  }
  size_t idx = static_cast<size_t>(p * static_cast<double>(sorted_ns.size()));
  idx = std::min(idx, sorted_ns.size() - 1);
  return static_cast<double>(sorted_ns[idx]) * 1e-3;
}

static latency_stats compute_stats(std::vector<uint64_t> ns)
{
  latency_stats st;
  if (ns.empty()) {
    return st;
  }

  std::sort(ns.begin(), ns.end());
  st.n = static_cast<unsigned>(ns.size());
  st.min_us = static_cast<double>(ns.front()) * 1e-3;
  st.max_us = static_cast<double>(ns.back()) * 1e-3;
  st.p1_us = percentile_us(ns, 0.01);
  st.p5_us = percentile_us(ns, 0.05);
  st.p20_us = percentile_us(ns, 0.20);
  st.p50_us = percentile_us(ns, 0.50);
  st.p95_us = percentile_us(ns, 0.95);
  st.p99_us = percentile_us(ns, 0.99);
  st.p99_9_us = percentile_us(ns, 0.999);
  st.p99_99_us = percentile_us(ns, 0.9999);
  st.p99_999_us = percentile_us(ns, 0.99999);
  st.mean_us = static_cast<double>(std::accumulate(ns.begin(), ns.end(), 0ULL)) * 1e-3 / static_cast<double>(ns.size());
  return st;
}

struct result_row {
  unsigned     bg = 0;
  unsigned     zc = 0;
  unsigned     k_bits = 0;
  unsigned     rm_len = 0;
  unsigned     max_iters = 0;
  unsigned     nof_threads = 1;
  unsigned     reps = 0;
  unsigned     warmup = 0;
  float        code_rate = 0.0F;
  unsigned     n = 0;
  unsigned     batch = 1;
  unsigned     thr_mode = 0;
  unsigned     success = 0;
  double       avg_iters_success = 0.0;
  double       wall_s = 0.0;
  bool         rapl_available = false;
  double       run_pkg_j = 0.0;
  double       run_pkg_dram_j = 0.0;
  double       idle_pkg_w = 0.0;
  double       idle_pkg_dram_w = 0.0;
  double       raw_net_pkg_j = 0.0;
  double       raw_net_pkg_dram_j = 0.0;
  double       net_pkg_j = 0.0;
  double       net_pkg_dram_j = 0.0;
  double       raw_net_pkg_j_per_bit = 0.0;
  double       raw_net_pkg_dram_j_per_bit = 0.0;
  double       net_pkg_j_per_bit = 0.0;
  double       net_pkg_dram_j_per_bit = 0.0;
  latency_stats lat;
};

static result_row run_one_point(const options&                    opt,
                                const rapl_topology&              rapl,
                                rapl_scope                        scope,
                                double                            idle_pkg_w_baseline,
                                double                            idle_pkg_dram_w_baseline,
                                unsigned                          bg,
                                unsigned                          zc,
                                unsigned                          iters,
                                unsigned                          nof_threads,
                                const std::vector<int>&           cpu_list,
                                float                             code_rate,
                                std::shared_ptr<ldpc_decoder_factory> decoder_factory,
                                std::mt19937&                     rng)
{
  result_row row;
  row.bg = bg;
  row.zc = zc;
  row.max_iters = iters;
  row.nof_threads = std::max(1U, nof_threads);
  row.reps = opt.reps;
  row.warmup = opt.warmup;
  row.code_rate = code_rate;
  row.n = opt.reps;
  row.batch = 1;
  row.thr_mode = 0;

  row.k_bits = get_k(bg, zc);
  row.rm_len = (opt.rm_len_mode == "fixed") ? choose_input_length_fixed(bg, zc, row.k_bits, opt.rm_len_fixed)
                                              : choose_input_length(bg, zc, row.k_bits, code_rate);

  ocudu::ldpc_decoder::configuration cfg_dec = {
      .base_graph = (bg == 1) ? ldpc_base_graph_type::BG1 : ldpc_base_graph_type::BG2,
      .lifting_size = static_cast<lifting_size_t>(zc),
      .nof_filler_bits = opt.nof_filler_bits,
      .nof_crc_bits = 16,
      .max_iterations = iters,
  };

  std::uniform_int_distribution<int> llr_gen(-10, 10);
  std::vector<log_likelihood_ratio>  input_llr(row.rm_len);
  for (auto& v : input_llr) {
    int sample = llr_gen(rng);
    if (sample == 0) {
      sample = 1;
    }
    v = static_cast<int8_t>(sample);
  }

  const unsigned active_threads = std::max(1U, row.nof_threads);

  std::vector<std::unique_ptr<ldpc_decoder>> decoders;
  decoders.reserve(active_threads);
  for (unsigned tid = 0; tid < active_threads; ++tid) {
    std::unique_ptr<ldpc_decoder> decoder = decoder_factory->create();
    TESTASSERT(decoder);
    decoders.emplace_back(std::move(decoder));
  }

  std::vector<std::unique_ptr<crc_calculator>> crcs(active_threads);
  if (opt.use_crc) {
    auto crc_factory = create_crc_calculator_factory_sw("auto");
    for (unsigned tid = 0; tid < active_threads; ++tid) {
      crcs[tid] = crc_factory->create(crc_generator_poly::CRC16);
    }
  }

  for (unsigned tid = 0; tid < active_threads; ++tid) {
    dynamic_bit_buffer warmup_output(row.k_bits);
    for (unsigned i = 0; i < opt.warmup; ++i) {
      (void)decoders[tid]->decode(warmup_output, input_llr, crcs[tid].get(), cfg_dec);
    }
  }

  rapl_snapshot energy_start = rapl.ok_any ? read_rapl(rapl) : rapl_snapshot{};
  auto          t0_all = std::chrono::high_resolution_clock::now();

  struct worker_result {
    std::vector<uint64_t> lat_ns;
    unsigned              success = 0;
    uint64_t              sum_iters = 0;
  };

  std::vector<worker_result> worker_results(active_threads);
  std::vector<std::thread> threads;
  threads.reserve(active_threads);

  for (unsigned tid = 0; tid < active_threads; ++tid) {
    threads.emplace_back([&, tid]() {
      (void)set_thread_affinity(cpu_list, tid, !opt.silent);
      worker_result& wr = worker_results[tid];
      wr.lat_ns.reserve((opt.reps + active_threads - 1) / active_threads);
      dynamic_bit_buffer output(row.k_bits);

      for (unsigned rep = tid; rep < opt.reps; rep += active_threads) {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::optional<unsigned> used_iters = decoders[tid]->decode(output, input_llr, crcs[tid].get(), cfg_dec);
        auto t1 = std::chrono::high_resolution_clock::now();
        wr.lat_ns.push_back(
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()));
        if (used_iters.has_value()) {
          wr.success++;
          wr.sum_iters += *used_iters;
        }
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  auto t1_all = std::chrono::high_resolution_clock::now();
  rapl_snapshot energy_end = rapl.ok_any ? read_rapl(rapl) : rapl_snapshot{};
  row.wall_s = std::chrono::duration_cast<std::chrono::duration<double>>(t1_all - t0_all).count();

  std::vector<uint64_t> lat_ns;
  lat_ns.reserve(opt.reps);
  uint64_t sum_iters = 0;
  for (const worker_result& wr : worker_results) {
    row.success += wr.success;
    sum_iters += wr.sum_iters;
    lat_ns.insert(lat_ns.end(), wr.lat_ns.begin(), wr.lat_ns.end());
  }

  row.avg_iters_success = (row.success > 0) ? (static_cast<double>(sum_iters) / static_cast<double>(row.success)) : 0.0;
  row.lat = compute_stats(std::move(lat_ns));

  row.rapl_available = rapl.ok_any;
  if (rapl.ok_any) {
    rapl_energy run_energy = compute_rapl_energy(rapl, energy_start, energy_end);

    uint64_t run_pkg_uj = 0;
    uint64_t run_dram_uj = 0;
    scope_energy(run_energy, scope, run_pkg_uj, run_dram_uj);

    row.run_pkg_j = static_cast<double>(run_pkg_uj) * 1e-6;
    row.run_pkg_dram_j = static_cast<double>(run_pkg_uj + run_dram_uj) * 1e-6;

    row.idle_pkg_w = idle_pkg_w_baseline;
    row.idle_pkg_dram_w = idle_pkg_dram_w_baseline;

    row.raw_net_pkg_j = row.run_pkg_j - row.idle_pkg_w * row.wall_s;
    row.raw_net_pkg_dram_j = row.run_pkg_dram_j - row.idle_pkg_dram_w * row.wall_s;

    row.net_pkg_j = std::max(0.0, row.raw_net_pkg_j);
    row.net_pkg_dram_j = std::max(0.0, row.raw_net_pkg_dram_j);

    uint64_t total_bits = static_cast<uint64_t>(row.k_bits) * static_cast<uint64_t>(row.n);
    row.raw_net_pkg_j_per_bit = (total_bits > 0) ? (row.raw_net_pkg_j / static_cast<double>(total_bits)) : 0.0;
    row.raw_net_pkg_dram_j_per_bit =
      (total_bits > 0) ? (row.raw_net_pkg_dram_j / static_cast<double>(total_bits)) : 0.0;
    row.net_pkg_j_per_bit = (total_bits > 0 && row.net_pkg_j > 0.0) ? (row.net_pkg_j / static_cast<double>(total_bits)) : 0.0;
    row.net_pkg_dram_j_per_bit =
        (total_bits > 0 && row.net_pkg_dram_j > 0.0) ? (row.net_pkg_dram_j / static_cast<double>(total_bits)) : 0.0;
  }

  return row;
}

} // namespace

int main(int argc, char** argv)
{
  options opt;
  parse_args(argc, argv, opt);

  std::vector<unsigned> bg_values = build_bg_values(opt);
  if (bg_values.empty()) {
    std::cerr << "Invalid --bg, must be 1, 2 or both.\n";
    return 1;
  }
  if (opt.reps == 0) {
    std::cerr << "Invalid --reps, must be > 0.\n";
    return 1;
  }
  if (opt.zc_mode != "full" && opt.zc_mode != "pow2" && opt.zc_mode != "custom") {
    std::cerr << "Invalid --zc-mode, must be full, pow2 or custom.\n";
    return 1;
  }
  if (opt.rate_control_mode != "fixed" && opt.rate_control_mode != "sweep") {
    std::cerr << "Invalid --rate-control-mode, must be fixed or sweep.\n";
    return 1;
  }
  if (opt.rm_len_mode != "from_rate" && opt.rm_len_mode != "fixed") {
    std::cerr << "Invalid --rm-len-mode, must be from_rate or fixed.\n";
    return 1;
  }
  if (opt.rate_fixed <= 0.0F || opt.rate_fixed > 1.0F) {
    std::cerr << "Invalid --rate-fixed, must be in (0,1].\n";
    return 1;
  }
  if (opt.rm_len_fixed == 0) {
    std::cerr << "Invalid --rm-len-fixed, must be > 0.\n";
    return 1;
  }
  if (opt.nof_threads == 0) {
    std::cerr << "Invalid --threads, must be > 0.\n";
    return 1;
  }

  if (opt.energy_scope != "sum" && opt.energy_scope != "socket0" && opt.energy_scope != "socket1") {
    std::cerr << "Invalid --energy-scope, must be sum, socket0 or socket1.\n";
    return 1;
  }

  std::vector<unsigned> zc_values = build_zc_values(opt);
  std::vector<unsigned> iters_values = parse_uint_list(opt.iters_list);
  std::vector<float> rate_values = build_rate_values(opt);
  std::vector<unsigned> thread_values = build_thread_values(opt);
  std::vector<int> cpu_list_values = parse_cpu_list(opt.cpu_list);

  if (zc_values.empty() || iters_values.empty() || rate_values.empty() || thread_values.empty()) {
    std::cerr << "Invalid --zc-list, --iters-list or --threads-list.\n";
    return 1;
  }
  for (unsigned zc : zc_values) {
    if (!is_valid_zc(zc)) {
      std::cerr << "Invalid Zc=" << zc << ". Supported lifting sizes are 3GPP set (e.g. 48, 52, 56 ...).\n";
      return 1;
    }
  }
  if (!opt.cpu_list.empty() && cpu_list_values.empty()) {
    std::cerr << "Invalid --cpu-list.\n";
    return 1;
  }

  ldpc_decoder_factory::ldpc_decoder_factory_configuration dec_cfg = {
      .force_decoding = false,
      .early_stop_syndrome = false,
  };
  std::shared_ptr<ldpc_decoder_factory> decoder_factory = create_ldpc_decoder_factory_sw(opt.decoder_type, dec_cfg);
  TESTASSERT(decoder_factory);

  rapl_topology rapl = detect_rapl();
  rapl_scope    scope = parse_scope(opt.energy_scope);

  double idle_pkg_w_baseline = 0.0;
  double idle_pkg_dram_w_baseline = 0.0;
  if (rapl.ok_any && opt.idle_seconds > 0) {
    rapl_snapshot idle_start = read_rapl(rapl);
    std::this_thread::sleep_for(std::chrono::seconds(opt.idle_seconds));
    rapl_snapshot idle_end = read_rapl(rapl);
    rapl_energy idle_energy = compute_rapl_energy(rapl, idle_start, idle_end);
    uint64_t idle_pkg_uj = 0;
    uint64_t idle_dram_uj = 0;
    scope_energy(idle_energy, scope, idle_pkg_uj, idle_dram_uj);
    idle_pkg_w_baseline = static_cast<double>(idle_pkg_uj) * 1e-6 / opt.idle_seconds;
    idle_pkg_dram_w_baseline = static_cast<double>(idle_pkg_uj + idle_dram_uj) * 1e-6 / opt.idle_seconds;
  }

  std::mt19937 rng(opt.seed);
  std::vector<result_row> rows;
  rows.reserve(bg_values.size() * zc_values.size() * iters_values.size() * rate_values.size() * thread_values.size());

  unsigned total_pts =
      static_cast<unsigned>(bg_values.size() * zc_values.size() * iters_values.size() * rate_values.size() * thread_values.size());
  unsigned idx = 0;

  if (!opt.silent) {
    std::cout << "==== CPU LDPC Scan: K x iters x threads (code_rate as control) ====\n";
    std::cout << "BG count=" << bg_values.size() << " | Zc count=" << zc_values.size() << " | iters count="
          << iters_values.size() << " | rates count=" << rate_values.size() << " | thread count=" << thread_values.size()
          << "\n";
    if (!cpu_list_values.empty()) {
      std::cout << "cpu_list=" << opt.cpu_list << "\n";
    } else {
      std::cout << "cpu_list=default_policy(0-15,32-47 then wrap)\n";
    }
    std::cout << "rm_len_mode=" << opt.rm_len_mode << "\n\n";
    if (rapl.ok_any) {
      std::cout << "energy_scope=" << opt.energy_scope << " | idle_seconds=" << opt.idle_seconds
                << " | idle_measured_once=on"
                << " | idle_pkg_w=" << idle_pkg_w_baseline
                << " | idle_pkg_dram_w=" << idle_pkg_dram_w_baseline << "\n\n";
    } else {
      std::cout << "energy_scope=" << opt.energy_scope << " | RAPL unavailable\n\n";
    }
  }

  for (unsigned bg : bg_values) {
    for (unsigned zc : zc_values) {
      for (unsigned iters : iters_values) {
        for (float code_rate : rate_values) {
          for (unsigned threads : thread_values) {
            ++idx;
            result_row row = run_one_point(opt,
                                           rapl,
                                           scope,
                                           idle_pkg_w_baseline,
                                           idle_pkg_dram_w_baseline,
                                           bg,
                                           zc,
                                           iters,
                                           threads,
                                           cpu_list_values,
                                           code_rate,
                                           decoder_factory,
                                           rng);
            rows.push_back(row);

            if (!opt.silent && (idx == 1 || (idx % 50) == 0)) {
              double p50_thr_mbps = (row.lat.p50_us > 0.0) ? (static_cast<double>(row.k_bits) / row.lat.p50_us) : 0.0;
              double total_thr_mbps = (row.wall_s > 0.0)
                                          ? (static_cast<double>(row.k_bits) * static_cast<double>(row.n) / row.wall_s / 1e6)
                                          : 0.0;
              std::cout << "[" << idx << "/" << total_pts << "] BG" << row.bg << " Zc=" << row.zc
                        << " K=" << row.k_bits << " iters=" << row.max_iters << " threads=" << row.nof_threads
                        << " rate=" << row.code_rate << " E=" << row.rm_len
                        << " -> min=" << row.lat.min_us << "us"
                        << " p1=" << row.lat.p1_us << "us"
                        << " p5=" << row.lat.p5_us << "us"
                        << " p20=" << row.lat.p20_us << "us"
                        << " p50=" << row.lat.p50_us << "us"
                        << " thr_p50=" << p50_thr_mbps << "Mbps"
                        << " thr_total=" << total_thr_mbps << "Mbps"
                        << " succ=" << row.success << "/" << row.reps;
              if (row.rapl_available) {
                std::cout << " raw_net_pkg_J/bit=" << row.raw_net_pkg_j_per_bit
                          << " net_pkg_J/bit=" << row.net_pkg_j_per_bit;
              }
              std::cout << "\n";
            }
          }
        }
      }
    }
  }

  if (!opt.out_csv.empty()) {
    std::ofstream csv(opt.out_csv);
    if (!csv.is_open()) {
      std::cerr << "Cannot open CSV file: " << opt.out_csv << "\n";
      return 1;
    }

            csv << "BG,Zc,K,iters,threads,code_rate,rm_len,batch,thr_mode,n,p1_us,p5_us,p20_us,p50_us,p95_us,p99_us,p99_9_us,p99_99_us,p99_999_us,mean_us,min_us,max_us,wall_s,"
              "success,avg_iters_success,thr_p50_mbps,thr_mean_mbps,thr_total_mbps,rapl_available,run_pkg_j,run_pkg_dram_j,idle_pkg_w," 
              "idle_pkg_dram_w,raw_net_pkg_j,raw_net_pkg_dram_j,net_pkg_j,net_pkg_dram_j,raw_net_pkg_j_per_bit," 
              "raw_net_pkg_dram_j_per_bit,net_pkg_j_per_bit,net_pkg_dram_j_per_bit\n";

    for (const result_row& row : rows) {
      double thr_p50 = (row.lat.p50_us > 0.0) ? (static_cast<double>(row.k_bits) / row.lat.p50_us) : 0.0;
      double thr_mean = (row.lat.mean_us > 0.0) ? (static_cast<double>(row.k_bits) / row.lat.mean_us) : 0.0;
      double thr_total =
          (row.wall_s > 0.0) ? (static_cast<double>(row.k_bits) * static_cast<double>(row.n) / row.wall_s / 1e6) : 0.0;
        csv << row.bg << ',' << row.zc << ',' << row.k_bits << ',' << row.max_iters << ',' << row.nof_threads << ',' << row.code_rate << ','
          << row.rm_len << ',' << row.batch << ',' << row.thr_mode << ',' << row.n << ',' << row.lat.p1_us << ','
          << row.lat.p5_us << ',' << row.lat.p20_us << ',' << row.lat.p50_us << ','
          << row.lat.p95_us << ',' << row.lat.p99_us << ',' << row.lat.p99_9_us << ',' << row.lat.p99_99_us << ',' << row.lat.p99_999_us << ','
          << row.lat.mean_us << ',' << row.lat.min_us << ','
          << row.lat.max_us << ',' << row.wall_s << ',' << row.success << ',' << row.avg_iters_success << ','
          << thr_p50 << ',' << thr_mean << ',' << thr_total << ',' << (row.rapl_available ? 1 : 0) << ',' << row.run_pkg_j << ','
          << row.run_pkg_dram_j << ',' << row.idle_pkg_w << ',' << row.idle_pkg_dram_w << ',' << row.raw_net_pkg_j << ','
          << row.raw_net_pkg_dram_j << ',' << row.net_pkg_j << ',' << row.net_pkg_dram_j << ','
          << row.raw_net_pkg_j_per_bit << ',' << row.raw_net_pkg_dram_j_per_bit << ',' << row.net_pkg_j_per_bit << ','
          << row.net_pkg_dram_j_per_bit << '\n';
    }
  }

  return 0;
}

/*
 * LDPC decoder latency + energy benchmark (OCUDU)
 */

#include "ocudu/ocudulog/ocudulog.h"
#include "ocudu/ocuduvec/bit.h"
#include "ocudu/phy/upper/channel_coding/channel_coding_factories.h"
#include "ocudu/phy/upper/channel_coding/ldpc/ldpc_encoder_buffer.h"
#include "ocudu/phy/upper/channel_coding/ldpc/ldpc_segmenter_buffer.h"
#include "ocudu/ran/sch/modulation_scheme.h"
#include "ocudu/support/benchmark_utils.h"
#include "ocudu/support/ocudu_test.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <mutex>
#include <numeric>
#include <random>
#include <sched.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace ocudu;
using namespace ocudu::ldpc;

namespace {
struct options {
  std::string decoder_type          = "generic";
  std::string dematcher_type        = "generic";
  bool        dematcher_type_set    = false;
  std::string energy_scope          = "sum";
  std::string cpu_list;
  unsigned    nof_repetitions        = 1000;
  unsigned    max_iterations         = 6;
  unsigned    nof_threads            = 1;
  unsigned    nof_cells              = 1;
  unsigned    nof_rb                 = 50;
  unsigned    nof_symbols            = 12;
  unsigned    nof_layers             = 1;
  unsigned    rv                     = 0;
  unsigned    tbs_bits               = 0;
  unsigned    warmup                 = 0;
  unsigned    sample_stride          = 0;
  unsigned    idle_seconds           = 0;
  bool        use_crc                = false;
  bool        force_decoding         = false;
  bool        early_stop_syndrome    = false;
  bool        decode_only            = false;
  bool        pure_output            = false;
  bool        silent                 = false;
  float       code_rate              = 0.5F;
  modulation_scheme mod              = modulation_scheme::QPSK;
  int         base_graph             = 0; // 0 auto, 1 BG1, 2 BG2
};

static std::mt19937 rgen(0);
static std::uniform_int_distribution<uint8_t> byte_gen(0, 255);

static void usage(const char* prog)
{
  fmt::print("Usage: {} [options]\n", prog);
  fmt::print("  -T, --decoder        Decoder type (generic,avx2,avx512,neon) [Default {}]\n", "generic");
  fmt::print("  -D, --dematcher      Dematcher type (generic,avx2,avx512,neon) [Default {}]\n", "generic");
  fmt::print("  -R, --reps           Repetitions [Default {}]\n", 1000);
  fmt::print("  -I, --iters          Max iterations [Default {}]\n", 6);
  fmt::print("  -n, --threads        Number of worker threads [Default {}]\n", 1);
  fmt::print("  -z, --cells          Concurrent TBs in one batch [Default {}]\n", 1);
  fmt::print("  -c, --cpus           CPU list, e.g. 0-7,16-23 [Default auto]\n");
  fmt::print("  -r, --rate           Code rate (0-1) [Default {}]\n", 0.5);
  fmt::print("  -b, --bg             Base graph: 0 auto, 1 BG1, 2 BG2 [Default auto]\n");
  fmt::print("  -B, --rb             Number of PRBs [Default {}]\n", 50);
  fmt::print("  -S, --symbols        Number of OFDM symbols [Default {}]\n", 12);
  fmt::print("  -L, --layers         Number of layers [Default {}]\n", 1);
  fmt::print("  -M, --mod            Modulation (BPSK,QPSK,QAM16,QAM64,QAM256) [Default QPSK]\n");
  fmt::print("  -t, --tbs            Transport block size in bits (overrides rate) [Default auto]\n");
  fmt::print("  -v, --rv             Redundancy version [Default {}]\n", 0);
  fmt::print("  -w, --warmup         Warmup trials (not measured) [Default {}]\n", 0);
  fmt::print("  -q, --sample         Sample stride for avg sampled latency [Default {}]\n", 0);
  fmt::print("  -x, --idle           Idle seconds for net energy [Default {}]\n", 0);
  fmt::print("  -C, --crc            Enable CRC early stop [Default off]\n");
  fmt::print("  -f, --force          Force decoding (ignore CRC/syndrome) [Default off]\n");
  fmt::print("  -e, --early-syndrome Enable syndrome early stop [Default off]\n");
  fmt::print("  -o, --decode-only    Decode only (exclude rate-dematch timing) [Default off]\n");
  fmt::print("  -p, --pure           Pure output (minimal summary) [Default off]\n");
  fmt::print("  -E, --energy-scope   Energy scope (sum,socket0,socket1) [Default sum]\n");
  fmt::print("  -s, --silent         Silent mode (no tables) [Default off]\n");
  fmt::print("  -h, --help           Show this message\n");
}

static bool parse_modulation(const std::string& s, modulation_scheme& out)
{
  if (s == "BPSK") {
    out = modulation_scheme::BPSK;
  } else if (s == "QPSK") {
    out = modulation_scheme::QPSK;
  } else if (s == "QAM16") {
    out = modulation_scheme::QAM16;
  } else if (s == "QAM64") {
    out = modulation_scheme::QAM64;
  } else if (s == "QAM256") {
    out = modulation_scheme::QAM256;
  } else {
    return false;
  }
  return true;
}

static std::vector<int> parse_cpu_list(const std::string& s)
{
  std::vector<int> cpus;
  if (s.empty()) {
    return cpus;
  }
  size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == ',')) {
      ++i;
    }
    if (i >= s.size()) {
      break;
    }
    char* endp = nullptr;
    long a = std::strtol(&s[i], &endp, 10);
    if (endp == &s[i] || a < 0) {
      return {};
    }
    i = static_cast<size_t>(endp - s.data());
    long b = a;
    if (i < s.size() && s[i] == '-') {
      ++i;
      b = std::strtol(&s[i], &endp, 10);
      if (endp == &s[i] || b < a) {
        return {};
      }
      i = static_cast<size_t>(endp - s.data());
    }
    for (long v = a; v <= b; ++v) {
      if (std::find(cpus.begin(), cpus.end(), static_cast<int>(v)) == cpus.end()) {
        cpus.push_back(static_cast<int>(v));
      }
    }
  }
  return cpus;
}

static std::vector<int> default_cpu_list(unsigned nof_threads)
{
  long nproc = sysconf(_SC_NPROCESSORS_CONF);
  std::vector<int> cpus;
  if (nproc <= 0) {
    return cpus;
  }
  unsigned count = std::min<unsigned>(static_cast<unsigned>(nproc), nof_threads);
  cpus.reserve(count);
  for (unsigned i = 0; i < count; ++i) {
    cpus.push_back(static_cast<int>(i));
  }
  return cpus;
}

static void set_thread_affinity(const std::vector<int>& cpus, unsigned tid)
{
  if (cpus.empty()) {
    return;
  }
  int cpu = cpus[tid % cpus.size()];
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  (void)sched_setaffinity(0, sizeof(set), &set);
}

static crc_generator_poly choose_crc_poly(unsigned nof_crc_bits)
{
  if (nof_crc_bits == 16) {
    return crc_generator_poly::CRC16;
  }
  if (nof_crc_bits == 24) {
    return crc_generator_poly::CRC24B;
  }
  return crc_generator_poly::CRC16;
}

static void parse_args(int argc, char** argv, options& opt)
{
  static struct option long_opts[] = {
      {"decoder", required_argument, nullptr, 'T'},
      {"dematcher", required_argument, nullptr, 'D'},
      {"reps", required_argument, nullptr, 'R'},
      {"iters", required_argument, nullptr, 'I'},
      {"threads", required_argument, nullptr, 'n'},
      {"cells", required_argument, nullptr, 'z'},
      {"cpus", required_argument, nullptr, 'c'},
      {"rate", required_argument, nullptr, 'r'},
      {"bg", required_argument, nullptr, 'b'},
      {"rb", required_argument, nullptr, 'B'},
      {"symbols", required_argument, nullptr, 'S'},
      {"layers", required_argument, nullptr, 'L'},
      {"mod", required_argument, nullptr, 'M'},
      {"tbs", required_argument, nullptr, 't'},
      {"rv", required_argument, nullptr, 'v'},
        {"warmup", required_argument, nullptr, 'w'},
        {"sample", required_argument, nullptr, 'q'},
        {"idle", required_argument, nullptr, 'x'},
      {"crc", no_argument, nullptr, 'C'},
      {"force", no_argument, nullptr, 'f'},
      {"early-syndrome", no_argument, nullptr, 'e'},
      {"decode-only", no_argument, nullptr, 'o'},
      {"pure", no_argument, nullptr, 'p'},
        {"energy-scope", required_argument, nullptr, 'E'},
      {"silent", no_argument, nullptr, 's'},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, 0, nullptr, 0}};

  int optch = 0;
      while ((optch = getopt_long(argc, argv, "T:D:R:I:n:z:c:r:b:B:S:L:M:t:v:w:q:x:CfeopE:sh", long_opts, nullptr)) != -1) {
    switch (optch) {
      case 'T':
        opt.decoder_type = optarg;
        break;
      case 'D':
        opt.dematcher_type = optarg;
        opt.dematcher_type_set = true;
        break;
      case 'R':
        opt.nof_repetitions = std::strtoul(optarg, nullptr, 10);
        break;
      case 'I':
        opt.max_iterations = std::strtoul(optarg, nullptr, 10);
        break;
      case 'n':
        opt.nof_threads = std::strtoul(optarg, nullptr, 10);
        break;
      case 'z':
        opt.nof_cells = std::strtoul(optarg, nullptr, 10);
        break;
      case 'c':
        opt.cpu_list = optarg;
        break;
      case 'r':
        opt.code_rate = std::strtof(optarg, nullptr);
        break;
      case 'b':
        opt.base_graph = std::strtol(optarg, nullptr, 10);
        break;
      case 'B':
        opt.nof_rb = std::strtoul(optarg, nullptr, 10);
        break;
      case 'S':
        opt.nof_symbols = std::strtoul(optarg, nullptr, 10);
        break;
      case 'L':
        opt.nof_layers = std::strtoul(optarg, nullptr, 10);
        break;
      case 'M': {
        modulation_scheme mod = modulation_scheme::QPSK;
        if (!parse_modulation(optarg, mod)) {
          usage(argv[0]);
          std::exit(1);
        }
        opt.mod = mod;
      } break;
      case 't':
        opt.tbs_bits = std::strtoul(optarg, nullptr, 10);
        break;
      case 'v':
        opt.rv = std::strtoul(optarg, nullptr, 10);
        break;
      case 'w':
        opt.warmup = std::strtoul(optarg, nullptr, 10);
        break;
      case 'q':
        opt.sample_stride = std::strtoul(optarg, nullptr, 10);
        break;
      case 'x':
        opt.idle_seconds = std::strtoul(optarg, nullptr, 10);
        break;
      case 'C':
        opt.use_crc = true;
        break;
      case 'f':
        opt.force_decoding = true;
        break;
      case 'e':
        opt.early_stop_syndrome = true;
        break;
      case 'o':
        opt.decode_only = true;
        break;
      case 'p':
        opt.pure_output = true;
        break;
      case 'E':
        opt.energy_scope = optarg;
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

struct prepared_codeword {
  std::vector<log_likelihood_ratio> llr;
  std::vector<described_rx_codeblock> codeblocks;
  unsigned                           tbs_bits = 0;
  unsigned                           cw_length = 0;
  float                              effective_rate = 0.0F;
};

static prepared_codeword prepare_codeword(const options& opt,
                                          ldpc_segmenter_tx& segmenter_tx,
                                          ldpc_segmenter_rx& segmenter_rx,
                                          ldpc_rate_matcher& rate_matcher,
                                          ldpc_encoder& encoder)
{
  const unsigned nof_re = opt.nof_rb * 12 * opt.nof_symbols;
  const unsigned nof_ch_symbols = nof_re * opt.nof_layers;

  unsigned tbs_bits = opt.tbs_bits;
  if (tbs_bits == 0) {
    float bits = static_cast<float>(nof_ch_symbols) * static_cast<float>(get_bits_per_symbol(opt.mod)) * opt.code_rate;
    tbs_bits = static_cast<unsigned>(bits);
    tbs_bits = (tbs_bits / 8U) * 8U;
  }
  TESTASSERT(tbs_bits > 0, "Invalid TBS bits");

  float effective_rate = static_cast<float>(tbs_bits) /
                         (static_cast<float>(nof_ch_symbols) * static_cast<float>(get_bits_per_symbol(opt.mod)));

  unsigned tbs_bytes = tbs_bits / 8;

  ldpc_base_graph_type bg = ldpc_base_graph_type::BG1;
  if (opt.base_graph == 1) {
    bg = ldpc_base_graph_type::BG1;
  } else if (opt.base_graph == 2) {
    bg = ldpc_base_graph_type::BG2;
  } else {
    if ((tbs_bytes <= 36) || ((tbs_bytes <= 478) && (effective_rate <= 0.67F)) || (effective_rate <= 0.25F)) {
      bg = ldpc_base_graph_type::BG2;
    }
  }

  segmenter_config cfg_seg = {
      .base_graph = bg,
      .rv = opt.rv,
      .mod = opt.mod,
      .Nref = 0,
      .nof_layers = opt.nof_layers,
      .nof_ch_symbols = nof_ch_symbols};

  std::vector<uint8_t> transport_block(tbs_bytes);
  std::generate(transport_block.begin(), transport_block.end(), []() { return byte_gen(rgen); });

  const ldpc_segmenter_buffer& segment_buffer = segmenter_tx.new_transmission(transport_block, cfg_seg);
  unsigned cw_length = segment_buffer.get_cw_length().value();

  std::vector<uint8_t> codeword_tx(cw_length);
  span<uint8_t>        codeword_view(codeword_tx);

  for (unsigned i_cb = 0, nof_cb = segment_buffer.get_nof_codeblocks(); i_cb != nof_cb; ++i_cb) {
    codeblock_metadata metadata = segment_buffer.get_cb_metadata(i_cb);

    dynamic_bit_buffer message_packed(segment_buffer.get_segment_length().value());
    segment_buffer.read_codeblock(message_packed, transport_block, i_cb);

    ldpc_encoder::configuration cfg_enc = {
        .base_graph = metadata.tb_common.base_graph,
        .lifting_size = metadata.tb_common.lifting_size,
        .Nref = 0,
    };
    const ldpc_encoder_buffer& rm_buffer = encoder.encode(message_packed, cfg_enc);

    unsigned           rm_length = segment_buffer.get_rm_length(i_cb);
    dynamic_bit_buffer output_buffer(rm_length);
    rate_matcher.rate_match(output_buffer, rm_buffer, metadata);

    ocuduvec::bit_unpack(codeword_view.first(rm_length), output_buffer);
    codeword_view = codeword_view.last(codeword_view.size() - rm_length);
  }

  TESTASSERT(codeword_view.empty(), "Codeword view should be empty.");

  std::vector<log_likelihood_ratio> llr(codeword_tx.size());
  std::transform(codeword_tx.begin(), codeword_tx.end(), llr.begin(), [](uint8_t b) {
    return log_likelihood_ratio::copysign(10, 1 - 2 * b);
  });

  static_vector<described_rx_codeblock, MAX_NOF_SEGMENTS> cb_views;
  segmenter_rx.segment(cb_views, llr, tbs_bits, cfg_seg);

  std::vector<described_rx_codeblock> codeblocks(cb_views.begin(), cb_views.end());

  prepared_codeword out;
  out.llr = std::move(llr);
  out.codeblocks = std::move(codeblocks);
  out.tbs_bits = tbs_bits;
  out.cw_length = cw_length;
  out.effective_rate = effective_rate;
  return out;
}

struct percentiles {
  double p50;
  double p75;
  double p90;
  double p99;
  double p999;
  double p9999;
  double p99999;
  double worst;
};

static percentiles compute_latency_us(const std::vector<uint64_t>& sorted_ns)
{
  const size_t n = sorted_ns.size();
  auto idx = [n](double p) {
    size_t i = static_cast<size_t>(n * p);
    return (i >= n) ? (n - 1) : i;
  };

  auto ns_to_us = [](uint64_t ns) { return static_cast<double>(ns) * 1e-3; };

  return {
      ns_to_us(sorted_ns[idx(0.5)]),
      ns_to_us(sorted_ns[idx(0.75)]),
      ns_to_us(sorted_ns[idx(0.9)]),
      ns_to_us(sorted_ns[idx(0.99)]),
      ns_to_us(sorted_ns[idx(0.999)]),
      ns_to_us(sorted_ns[idx(0.9999)]),
      ns_to_us(sorted_ns[idx(0.99999)]),
      ns_to_us(sorted_ns.back())};
}

static percentiles compute_throughput_mbps(const std::vector<uint64_t>& sorted_ns, unsigned size_bits)
{
  const size_t n = sorted_ns.size();
  auto idx = [n](double p) {
    size_t i = static_cast<size_t>(n * p);
    return (i >= n) ? (n - 1) : i;
  };
  auto to_mbps = [size_bits](uint64_t ns) {
    ns = std::max<uint64_t>(ns, 1);
    return static_cast<double>(size_bits) * 1000.0 / static_cast<double>(ns);
  };

  return {
      to_mbps(sorted_ns[idx(0.5)]),
      to_mbps(sorted_ns[idx(0.75)]),
      to_mbps(sorted_ns[idx(0.9)]),
      to_mbps(sorted_ns[idx(0.99)]),
      to_mbps(sorted_ns[idx(0.999)]),
      to_mbps(sorted_ns[idx(0.9999)]),
      to_mbps(sorted_ns[idx(0.99999)]),
      to_mbps(sorted_ns.back())};
}

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
  if (read_uj(max_path, max_uj) && max_uj > 0) {
    d.max_uj = max_uj;
    d.ok = true;
  }
  return d;
}

struct rapl_topology {
  rapl_domain pkg0;
  rapl_domain dram0;
  rapl_domain pkg1;
  rapl_domain dram1;
  bool ok_any = false;
};

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

struct rapl_energy {
  uint64_t pkg0 = 0;
  uint64_t dram0 = 0;
  uint64_t pkg1 = 0;
  uint64_t dram1 = 0;
};

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

enum class rapl_scope { sum, socket0, socket1 };

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

static void print_table(const std::string& title,
                        const std::string& units,
                        const std::string& descr,
                        const percentiles& p)
{
  fmt::print("\"{}\" performance for 1 profile. All values are in {}.\n", title, units);
  fmt::print(" {:<24}|{:^8}|{:^8}|{:^8}|{:^8}|{:^8}|{:^8}|{:^8}|{:^8}|\n",
             "Percentiles:",
             "50th",
             "75th",
             "90th",
             "99th",
             "99.9th",
             "99.99th",
             "99.999th",
             "Worst");
  fmt::print(" {:<24}|{:>8.3f}|{:>8.3f}|{:>8.3f}|{:>8.3f}|{:>8.3f}|{:>8.3f}|{:>8.3f}|{:>8.3f}|\n",
             descr,
             p.p50,
             p.p75,
             p.p90,
             p.p99,
             p.p999,
             p.p9999,
             p.p99999,
             p.worst);
}

struct worker_result {
  uint64_t total_iters = 0;
  uint64_t total_blocks = 0;
};

static void atomic_max(std::atomic<uint64_t>& target, uint64_t value)
{
  uint64_t prev = target.load(std::memory_order_relaxed);
  while (prev < value && !target.compare_exchange_weak(prev, value, std::memory_order_release, std::memory_order_relaxed)) {
  }
}

} // namespace

int main(int argc, char** argv)
{
  options opt;
  parse_args(argc, argv, opt);

  if (!opt.dematcher_type_set) {
    opt.dematcher_type = opt.decoder_type;
  }

  if (opt.nof_threads == 0) {
    fmt::print("Invalid thread count.\n");
    return 1;
  }
  if (opt.nof_cells == 0) {
    fmt::print("Invalid cell count.\n");
    return 1;
  }
  if (opt.nof_repetitions == 0) {
    fmt::print("Invalid repetitions.\n");
    return 1;
  }
  if (opt.sample_stride != 0 && opt.sample_stride < opt.nof_cells) {
    fmt::print("Sample stride should be >= cells to avoid bias.\n");
  }

  std::vector<int> cpus = parse_cpu_list(opt.cpu_list);
  if (cpus.empty()) {
    cpus = default_cpu_list(opt.nof_threads);
  }
  if (cpus.empty()) {
    fmt::print("No CPUs available for affinity.\n");
    return 1;
  }

  ldpc_decoder_factory::ldpc_decoder_factory_configuration dec_cfg = {
      .force_decoding = opt.force_decoding,
      .early_stop_syndrome = opt.early_stop_syndrome,
  };

  std::shared_ptr<ldpc_decoder_factory> decoder_factory = create_ldpc_decoder_factory_sw(opt.decoder_type, dec_cfg);
  TESTASSERT(decoder_factory);
  std::shared_ptr<ldpc_rate_dematcher_factory> dematcher_factory =
      create_ldpc_rate_dematcher_factory_sw(opt.dematcher_type);
  TESTASSERT(dematcher_factory);

  std::shared_ptr<ldpc_encoder_factory> encoder_factory = create_ldpc_encoder_factory_sw("generic");
  TESTASSERT(encoder_factory);
  std::shared_ptr<ldpc_rate_matcher_factory> rate_matcher_factory = create_ldpc_rate_matcher_factory_sw();
  TESTASSERT(rate_matcher_factory);
  std::shared_ptr<ldpc_segmenter_tx_factory> segmenter_tx_factory =
      create_ldpc_segmenter_tx_factory_sw(create_crc_calculator_factory_sw("auto"));
  TESTASSERT(segmenter_tx_factory);
  std::shared_ptr<ldpc_segmenter_rx_factory> segmenter_rx_factory = create_ldpc_segmenter_rx_factory_sw();
  TESTASSERT(segmenter_rx_factory);

  std::unique_ptr<ldpc_encoder> encoder = encoder_factory->create();
  std::unique_ptr<ldpc_rate_matcher> rate_matcher = rate_matcher_factory->create();
  std::unique_ptr<ldpc_segmenter_tx> segmenter_tx = segmenter_tx_factory->create();
  std::unique_ptr<ldpc_segmenter_rx> segmenter_rx = segmenter_rx_factory->create();

  prepared_codeword prepared = prepare_codeword(opt, *segmenter_tx, *segmenter_rx, *rate_matcher, *encoder);

  unsigned tbs_bits = prepared.tbs_bits;
  unsigned cw_length = prepared.cw_length;

  const codeblock_metadata& meta0 = prepared.codeblocks.front().second;

  std::unique_ptr<crc_calculator> crc;
  if (opt.use_crc && meta0.cb_specific.nof_crc_bits > 0) {
    auto crc_factory = create_crc_calculator_factory_sw("auto");
    crc = crc_factory->create(choose_crc_poly(meta0.cb_specific.nof_crc_bits));
  }

  rapl_topology rapl = detect_rapl();
  rapl_scope scope = parse_scope(opt.energy_scope);

  rapl_snapshot idle_start = {};
  rapl_snapshot idle_end = {};
  if (opt.idle_seconds > 0 && rapl.ok_any) {
    idle_start = read_rapl(rapl);
    std::this_thread::sleep_for(std::chrono::seconds(opt.idle_seconds));
    idle_end = read_rapl(rapl);
  }

  rapl_snapshot energy_start = rapl.ok_any ? read_rapl(rapl) : rapl_snapshot{};
  auto wall_start = std::chrono::high_resolution_clock::now();

  std::vector<worker_result> results(opt.nof_threads);

  std::vector<uint64_t> lat_ns;
  std::vector<uint64_t> sample_ns;
  lat_ns.reserve(opt.nof_repetitions);
  if (opt.sample_stride > 0) {
    sample_ns.reserve((opt.nof_repetitions + opt.sample_stride - 1) / opt.sample_stride);
  }

  std::vector<std::vector<log_likelihood_ratio>> warmup_dematched;
  if (opt.warmup > 0 && opt.decode_only) {
    warmup_dematched.resize(prepared.codeblocks.size());
    auto dematcher = dematcher_factory->create();
    for (size_t i = 0; i < prepared.codeblocks.size(); ++i) {
      const auto& cb = prepared.codeblocks[i];
      warmup_dematched[i].resize(cb.second.cb_specific.full_length);
      dematcher->rate_dematch(warmup_dematched[i], cb.first, true, cb.second);
    }
  }

  if (opt.warmup > 0) {
    for (unsigned w = 0; w < opt.warmup; ++w) {
      for (size_t i = 0; i < prepared.codeblocks.size(); ++i) {
        const auto& cb = prepared.codeblocks[i];
        const codeblock_metadata& meta = cb.second;
        auto decoder = decoder_factory->create();

        std::vector<log_likelihood_ratio> dematched;
        span<log_likelihood_ratio> dematched_view;
        if (opt.decode_only) {
          dematched_view = warmup_dematched[i];
        } else {
          dematched.resize(meta.cb_specific.full_length);
          auto dematcher = dematcher_factory->create();
          dematcher->rate_dematch(dematched, cb.first, true, meta);
          dematched_view = dematched;
        }

        ldpc_decoder::configuration cfg_dec = {
            .base_graph = meta.tb_common.base_graph,
            .lifting_size = meta.tb_common.lifting_size,
            .nof_filler_bits = meta.cb_specific.nof_filler_bits,
            .nof_crc_bits = meta.cb_specific.nof_crc_bits,
            .max_iterations = opt.max_iterations,
        };
        dynamic_bit_buffer out_bits(meta.cb_specific.full_length);
        decoder->decode(out_bits, dematched_view, nullptr, cfg_dec);
        do_not_optimize(out_bits);
      }
    }
  }

  unsigned completed_reps = 0;
  const unsigned nof_cb = static_cast<unsigned>(prepared.codeblocks.size());
  while (completed_reps < opt.nof_repetitions) {
    unsigned batch_cells = std::min(opt.nof_cells, opt.nof_repetitions - completed_reps);
    const unsigned active_tasks = batch_cells * nof_cb;
    std::atomic<unsigned> next_task{0};
    std::vector<std::atomic<uint64_t>> tb_max_elapsed_ns(batch_cells);
    for (auto& value : tb_max_elapsed_ns) {
      value.store(0, std::memory_order_relaxed);
    }

    std::vector<std::thread> batch_threads;
    batch_threads.reserve(opt.nof_threads);
    auto batch_t0 = std::chrono::high_resolution_clock::now();

    for (unsigned tid = 0; tid < opt.nof_threads; ++tid) {
      batch_threads.emplace_back([&, tid]() {
        set_thread_affinity(cpus, tid);

        auto local_decoder = decoder_factory->create();
        auto local_dematcher = dematcher_factory->create();
        std::unique_ptr<crc_calculator> local_crc =
            (crc != nullptr) ? create_crc_calculator_factory_sw("auto")->create(choose_crc_poly(meta0.cb_specific.nof_crc_bits))
                             : nullptr;

        std::vector<std::vector<log_likelihood_ratio>> dematched(nof_cb);
        std::vector<dynamic_bit_buffer> decoded;
        decoded.reserve(nof_cb);

        for (size_t i = 0; i < nof_cb; ++i) {
          const codeblock_metadata& meta = prepared.codeblocks[i].second;
          dematched[i].resize(meta.cb_specific.full_length);
          unsigned inverse_rate = (meta.tb_common.base_graph == ldpc_base_graph_type::BG1) ? 3 : 5;
          unsigned msg_length = meta.cb_specific.full_length / inverse_rate;
          decoded.emplace_back(msg_length);
        }

        if (opt.decode_only) {
          for (size_t i = 0; i < nof_cb; ++i) {
            const auto& cb = prepared.codeblocks[i];
            span<log_likelihood_ratio> dematch_out(dematched[i]);
            local_dematcher->rate_dematch(dematch_out, cb.first, true, cb.second);
          }
        }

        uint64_t local_iters = 0;
        uint64_t local_blocks = 0;

        while (true) {
          unsigned task = next_task.fetch_add(1, std::memory_order_relaxed);
          if (task >= active_tasks) {
            break;
          }

          unsigned cell_idx = task / nof_cb;
          size_t cb_idx = static_cast<size_t>(task % nof_cb);

          const auto& cb = prepared.codeblocks[cb_idx];
          const codeblock_metadata& meta = cb.second;

          span<log_likelihood_ratio> dematch_out(dematched[cb_idx]);
          if (!opt.decode_only) {
            local_dematcher->rate_dematch(dematch_out, cb.first, true, meta);
          }

          ldpc_decoder::configuration cfg_dec = {
              .base_graph = meta.tb_common.base_graph,
              .lifting_size = meta.tb_common.lifting_size,
              .nof_filler_bits = meta.cb_specific.nof_filler_bits,
              .nof_crc_bits = meta.cb_specific.nof_crc_bits,
              .max_iterations = opt.max_iterations,
          };

          std::optional<unsigned> iters =
              local_decoder->decode(decoded[cb_idx], dematch_out, local_crc ? local_crc.get() : nullptr, cfg_dec);
          if (iters) {
            local_iters += *iters;
          }
          local_blocks += 1;
          do_not_optimize(decoded[cb_idx]);

          auto t1 = std::chrono::high_resolution_clock::now();
          uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - batch_t0).count();
          atomic_max(tb_max_elapsed_ns[cell_idx], elapsed_ns);
        }

        results[tid].total_iters += local_iters;
        results[tid].total_blocks += local_blocks;
      });
    }

    for (auto& t : batch_threads) {
      t.join();
    }

    for (unsigned c = 0; c < batch_cells; ++c) {
      uint64_t elapsed = tb_max_elapsed_ns[c].load(std::memory_order_relaxed);
      unsigned rep_index = completed_reps + c;
      lat_ns.push_back(elapsed);
      if (opt.sample_stride > 0 && (rep_index % opt.sample_stride) == 0) {
        sample_ns.push_back(elapsed);
      }
    }
    completed_reps += batch_cells;
  }

  auto wall_end = std::chrono::high_resolution_clock::now();
  rapl_snapshot energy_end = rapl.ok_any ? read_rapl(rapl) : rapl_snapshot{};

  uint64_t total_iters = 0;
  uint64_t total_blocks = 0;
  unsigned active_workers = 0;
  for (const auto& r : results) {
    total_iters += r.total_iters;
    total_blocks += r.total_blocks;
    if (r.total_blocks > 0) {
      active_workers++;
    }
  }
  std::sort(lat_ns.begin(), lat_ns.end());
  std::sort(sample_ns.begin(), sample_ns.end());

  double wall_s = std::chrono::duration_cast<std::chrono::duration<double>>(wall_end - wall_start).count();
  uint64_t total_bits = static_cast<uint64_t>(tbs_bits) * static_cast<uint64_t>(lat_ns.size());

  rapl_energy run_energy = rapl.ok_any ? compute_rapl_energy(rapl, energy_start, energy_end) : rapl_energy{};
  rapl_energy idle_energy = rapl.ok_any && opt.idle_seconds > 0 ? compute_rapl_energy(rapl, idle_start, idle_end)
                                                                : rapl_energy{};

  uint64_t run_pkg_uj = 0;
  uint64_t run_dram_uj = 0;
  uint64_t idle_pkg_uj = 0;
  uint64_t idle_dram_uj = 0;
  scope_energy(run_energy, scope, run_pkg_uj, run_dram_uj);
  scope_energy(idle_energy, scope, idle_pkg_uj, idle_dram_uj);

  double run_pkg_j = static_cast<double>(run_pkg_uj) * 1e-6;
  double run_pkg_dram_j = static_cast<double>(run_pkg_uj + run_dram_uj) * 1e-6;
  double idle_pkg_w = (opt.idle_seconds > 0) ? (static_cast<double>(idle_pkg_uj) * 1e-6 / opt.idle_seconds) : 0.0;
  double idle_pkg_dram_w = (opt.idle_seconds > 0)
                               ? (static_cast<double>(idle_pkg_uj + idle_dram_uj) * 1e-6 / opt.idle_seconds)
                               : 0.0;

  double net_pkg_j = std::max(0.0, run_pkg_j - idle_pkg_w * wall_s);
  double net_pkg_dram_j = std::max(0.0, run_pkg_dram_j - idle_pkg_dram_w * wall_s);

  double net_pkg_j_per_bit = (total_bits > 0 && net_pkg_j > 0.0) ? (net_pkg_j / static_cast<double>(total_bits)) : 0.0;
  double net_pkg_dram_j_per_bit =
      (total_bits > 0 && net_pkg_dram_j > 0.0) ? (net_pkg_dram_j / static_cast<double>(total_bits)) : 0.0;

  if (!opt.silent) {
    const codeblock_metadata& meta = prepared.codeblocks.front().second;
    unsigned k_prime = meta.cb_specific.full_length;
    unsigned zc = static_cast<unsigned>(meta.tb_common.lifting_size);
    unsigned segs = static_cast<unsigned>(prepared.codeblocks.size());

    std::vector<uint64_t> lat_cb_ns;
    lat_cb_ns.reserve(lat_ns.size());
    if (segs > 0) {
      for (uint64_t ns : lat_ns) {
        lat_cb_ns.push_back(ns / segs);
      }
    }

    fmt::print("LDPC decode profile:\n");
    fmt::print("  decoder={} dematcher={} threads={} cells={} rv={} max_it={} crc={} force={} early_syndrome={} decode_only={}\n",
               opt.decoder_type,
               opt.dematcher_type,
               opt.nof_threads,
           opt.nof_cells,
               opt.rv,
               opt.max_iterations,
               opt.use_crc ? "on" : "off",
               opt.force_decoding ? "on" : "off",
           opt.early_stop_syndrome ? "on" : "off",
           opt.decode_only ? "on" : "off");
    fmt::print("  K'={} BG={} Zc={} rate={:.3f} maxIt={} seg={} trials={} warmup={}\n",
               k_prime,
               fmt::underlying(meta.tb_common.base_graph),
               zc,
               prepared.effective_rate,
               opt.max_iterations,
               segs,
               opt.nof_repetitions,
               opt.warmup);
    fmt::print("  tbs_bits={} cw_len={} mod={} rb={} symbols={} layers={}\n",
               tbs_bits,
               cw_length,
               to_string(opt.mod),
               opt.nof_rb,
               opt.nof_symbols,
               opt.nof_layers);

    percentiles lat = compute_latency_us(lat_ns);
    percentiles thr = compute_throughput_mbps(lat_ns, k_prime);
    percentiles thr_info = compute_throughput_mbps(lat_ns, tbs_bits);

    double avg_latency_ns = (lat_ns.empty()) ? 0.0
                                             : (static_cast<double>(std::accumulate(lat_ns.begin(), lat_ns.end(), 0ULL)) /
                                                static_cast<double>(lat_ns.size()));
    double avg_latency_ms = avg_latency_ns * 1e-6;
    double avg_latency_seg_ms = (segs > 0) ? (avg_latency_ms / static_cast<double>(segs)) : 0.0;
    double avg_iters = (total_blocks > 0) ? (static_cast<double>(total_iters) / static_cast<double>(total_blocks)) : 0.0;
    double latency_thr_mbps = (avg_latency_ns > 0.0) ? (static_cast<double>(tbs_bits) * 1000.0 / avg_latency_ns) : 0.0;
    double wall_thr_mbps = (wall_s > 0.0) ? (static_cast<double>(total_bits) / 1e6 / wall_s) : 0.0;

    if (opt.pure_output) {
      percentiles lat_cb = lat_cb_ns.empty() ? percentiles{} : compute_latency_us(lat_cb_ns);
      fmt::print("LDPC decode profile:\n");
      fmt::print("  decoder={} dematcher={} threads={} cells={} rv={} max_it={} crc={} force={} early_syndrome={} decode_only={}\n",
                 opt.decoder_type,
                 opt.dematcher_type,
                 opt.nof_threads,
             opt.nof_cells,
                 opt.rv,
                 opt.max_iterations,
                 opt.use_crc ? "on" : "off",
                 opt.force_decoding ? "on" : "off",
                 opt.early_stop_syndrome ? "on" : "off",
                 opt.decode_only ? "on" : "off");
      fmt::print("  K'={} BG={} Zc={} rate={:.3f} seg={} trials={} warmup={}\n",
                 k_prime,
                 fmt::underlying(meta.tb_common.base_graph),
                 zc,
                 prepared.effective_rate,
                 segs,
                 opt.nof_repetitions,
                 opt.warmup);
      fmt::print("  tbs_bits={} mod={} rb={} symbols={} layers={}\n\n",
                 tbs_bits,
                 to_string(opt.mod),
                 opt.nof_rb,
                 opt.nof_symbols,
                 opt.nof_layers);

      print_table("LDPC TB latency", "microseconds", "packet", lat);
      fmt::print("\n");
      print_table("LDPC single-CB latency", "microseconds", "cb", lat_cb);
      fmt::print("\nAvg iters   : {:.3f}\n", avg_iters);
      fmt::print("Latency thr : {:.3f} Mb/s (TBS bits)\n", latency_thr_mbps);
      fmt::print("Throughput  : {:.3f} Mb/s (TBS bits)\n", wall_thr_mbps);
      fmt::print("Workers used: {}/{} (threads)\n", active_workers, opt.nof_threads);

      if (rapl.ok_any) {
        fmt::print("Work power  : {:.6f} W ({} pkg)\n",
                   (wall_s > 0.0) ? (run_pkg_j / wall_s) : 0.0,
                   opt.energy_scope);
        fmt::print("Work power+dram: {:.6f} W ({} pkg+dram)\n",
                   (wall_s > 0.0) ? (run_pkg_dram_j / wall_s) : 0.0,
                   opt.energy_scope);
        if (opt.idle_seconds > 0) {
          fmt::print("Idle power  : {:.6f} W ({} pkg)\n", idle_pkg_w, opt.energy_scope);
          fmt::print("Idle power+dram: {:.6f} W ({} pkg+dram)\n", idle_pkg_dram_w, opt.energy_scope);
        } else {
          fmt::print("Idle power  : n/a (set -x)\n");
          fmt::print("Idle power+dram: n/a (set -x)\n");
        }
        fmt::print("Efficiency  : {:.3e} J/bit (net pkg)\n", net_pkg_j_per_bit);
        fmt::print("Efficiency+dram: {:.3e} J/bit (net pkg+dram)\n", net_pkg_dram_j_per_bit);
      } else {
        fmt::print("Energy: RAPL powercap not available on this system.\n");
      }
    } else {
      print_table("LDPC decode latency", "microseconds", "packet", lat);
      print_table("LDPC decode throughput", "megabits per second", "K' bits", thr);
      print_table("LDPC info throughput", "megabits per second", "TBS bits", thr_info);

      fmt::print("\nAvg latency : {:.3f} ms (per segment)\n\n", avg_latency_seg_ms);
      fmt::print("Throughput  : {:.3f} Mb/s (TBS bits)\n\n",
                 wall_thr_mbps);
      fmt::print("Avg iters   : {:.3f}\n\n", avg_iters);
      fmt::print("Workers used: {}/{} (threads)\n\n", active_workers, opt.nof_threads);
    }

    if (!opt.pure_output && !sample_ns.empty()) {
      double sample_avg_ms = (static_cast<double>(std::accumulate(sample_ns.begin(), sample_ns.end(), 0ULL)) /
                              static_cast<double>(sample_ns.size())) * 1e-6;
      fmt::print("Sampled avg latency : {:.3f} ms (per sampled call)\n\n", sample_avg_ms);
    }

    if (!opt.pure_output && rapl.ok_any) {
      fmt::print("--- Energy (RAPL) ---\n\n");
      fmt::print("Scope           : {}\n\n", opt.energy_scope);
      if (rapl.pkg0.ok) {
        fmt::print("Socket0 pkg     : {:.6f} J ({:.6f} W)\n",
                   static_cast<double>(run_energy.pkg0) * 1e-6,
                   (wall_s > 0.0) ? (static_cast<double>(run_energy.pkg0) * 1e-6 / wall_s) : 0.0);
      }
      if (rapl.dram0.ok) {
        fmt::print("Socket0 pkg+dram: {:.6f} J ({:.6f} W)\n",
                   static_cast<double>(run_energy.pkg0 + run_energy.dram0) * 1e-6,
                   (wall_s > 0.0) ? (static_cast<double>(run_energy.pkg0 + run_energy.dram0) * 1e-6 / wall_s) : 0.0);
      }
      if (rapl.pkg1.ok) {
        fmt::print("Socket1 pkg     : {:.6f} J ({:.6f} W)\n",
                   static_cast<double>(run_energy.pkg1) * 1e-6,
                   (wall_s > 0.0) ? (static_cast<double>(run_energy.pkg1) * 1e-6 / wall_s) : 0.0);
      }
      if (rapl.dram1.ok) {
        fmt::print("Socket1 pkg+dram: {:.6f} J ({:.6f} W)\n",
                   static_cast<double>(run_energy.pkg1 + run_energy.dram1) * 1e-6,
                   (wall_s > 0.0) ? (static_cast<double>(run_energy.pkg1 + run_energy.dram1) * 1e-6 / wall_s) : 0.0);
      }

      fmt::print("\nSum pkg         : {:.6f} J ({:.6f} W)\n",
                 static_cast<double>(run_energy.pkg0 + run_energy.pkg1) * 1e-6,
                 (wall_s > 0.0)
                     ? (static_cast<double>(run_energy.pkg0 + run_energy.pkg1) * 1e-6 / wall_s)
                     : 0.0);
      fmt::print("Sum pkg+dram    : {:.6f} J ({:.6f} W)\n\n",
                 static_cast<double>(run_energy.pkg0 + run_energy.pkg1 + run_energy.dram0 + run_energy.dram1) * 1e-6,
                 (wall_s > 0.0)
                     ? (static_cast<double>(run_energy.pkg0 + run_energy.pkg1 + run_energy.dram0 + run_energy.dram1) *
                        1e-6 / wall_s)
                     : 0.0);

      fmt::print("Selected pkg    : {:.6f} J ({:.6f} W)\n",
                 run_pkg_j,
                 (wall_s > 0.0) ? (run_pkg_j / wall_s) : 0.0);
      fmt::print("Selected pkg+dram: {:.6f} J ({:.6f} W)\n\n",
                 run_pkg_dram_j,
                 (wall_s > 0.0) ? (run_pkg_dram_j / wall_s) : 0.0);

      if (opt.idle_seconds > 0) {
        fmt::print("Idle pkg ({}s): {:.6f} W   [scope]\n", opt.idle_seconds, idle_pkg_w);
        fmt::print("Idle pkg+dram   : {:.6f} W   [scope]\n\n", idle_pkg_dram_w);
      }

      fmt::print("Net pkg         : {:.6f} J  ({:.3e} J/bit)\n",
                 net_pkg_j,
                 net_pkg_j_per_bit);
      fmt::print("Net pkg+dram    : {:.6f} J  ({:.3e} J/bit)\n",
                 net_pkg_dram_j,
                 net_pkg_dram_j_per_bit);
    } else if (!opt.pure_output) {
      fmt::print("Energy: RAPL powercap not available on this system.\n");
    }
  }

  return 0;
}

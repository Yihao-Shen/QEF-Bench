// SPDX-FileCopyrightText: Copyright (C) 2021-2026 Software Radio Systems Limited
// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
// Portions of this file may implement 3GPP specifications, which may be subject to additional licensing requirements.

/// \file
/// \brief LDPC rate dematcher benchmark.

#include "ocudu/phy/upper/channel_coding/channel_coding_factories.h"
#include "ocudu/support/benchmark_utils.h"
#include "ocudu/support/ocudu_test.h"
#include <getopt.h>
#include <random>

using namespace ocudu;
using namespace ocudu::ldpc;

static std::mt19937 rgen(0);

static std::string dec_type        = "auto";
static unsigned    nof_repetitions = 1000;
static unsigned    selected_bg     = 1;
static unsigned    l_size          = 384;
static unsigned    rm_length       = 9216;
static unsigned    rv              = 0;
static std::string selected_mod    = "256QAM";
static bool        silent          = false;

static void usage(const char* prog)
{
  fmt::print("Usage: {} [-R repetitions] [-T dematcher type] [-B base graph] [-L lifting size] [-E rm length] "
             "[-Q modulation] [-v rv] [-s silent]\n",
             prog);
  fmt::print("\t-R Repetitions [Default {}]\n", nof_repetitions);
  fmt::print("\t-T Dematcher type auto, generic, avx2, avx512 or neon [Default {}]\n", dec_type);
  fmt::print("\t-B Base graph 1 or 2 [Default {}]\n", selected_bg);
  fmt::print("\t-L Lifting size [Default {}]\n", l_size);
  fmt::print("\t-E Rate-matched codeblock length in bits [Default {}]\n", rm_length);
  fmt::print("\t-Q Modulation BPSK, QPSK, 16QAM, 64QAM or 256QAM [Default {}]\n", selected_mod);
  fmt::print("\t-v Redundancy version 0, 1, 2 or 3 [Default {}]\n", rv);
  fmt::print("\t-s Toggle silent operation [Default {}]\n", silent);
  fmt::print("\t-h Show this message\n");
}

static modulation_scheme parse_modulation(const std::string& value)
{
  for (modulation_scheme modulation :
       {modulation_scheme::BPSK, modulation_scheme::QPSK, modulation_scheme::QAM16, modulation_scheme::QAM64, modulation_scheme::QAM256}) {
    if (value == to_string(modulation)) {
      return modulation;
    }
  }
  fmt::print("Invalid modulation {}.\n", value);
  std::exit(1);
}

static void parse_args(int argc, char** argv)
{
  int opt = 0;
  while ((opt = getopt(argc, argv, "R:T:B:L:E:Q:v:sh")) != -1) {
    switch (opt) {
      case 'R':
        nof_repetitions = std::strtol(optarg, nullptr, 10);
        break;
      case 'T':
        dec_type = std::string(optarg);
        break;
      case 'B':
        selected_bg = std::strtol(optarg, nullptr, 10);
        break;
      case 'L':
        l_size = std::strtol(optarg, nullptr, 10);
        break;
      case 'E':
        rm_length = std::strtol(optarg, nullptr, 10);
        break;
      case 'Q':
        selected_mod = std::string(optarg);
        break;
      case 'v':
        rv = std::strtol(optarg, nullptr, 10);
        break;
      case 's':
        silent = (!silent);
        break;
      case 'h':
      default:
        usage(argv[0]);
        std::exit(0);
    }
  }

  if ((selected_bg != 1) && (selected_bg != 2)) {
    fmt::print("Invalid base graph {}.\n", selected_bg);
    std::exit(1);
  }
  if (rv > 3) {
    fmt::print("Invalid redundancy version {}.\n", rv);
    std::exit(1);
  }
}

int main(int argc, char** argv)
{
  parse_args(argc, argv);

  modulation_scheme modulation = parse_modulation(selected_mod);
  unsigned          qm         = get_bits_per_symbol(modulation);
  TESTASSERT((rm_length % qm) == 0, "Rate-matched length must be a multiple of the modulation order.");

  ldpc_base_graph_type bg          = (selected_bg == 1) ? ldpc_base_graph_type::BG1 : ldpc_base_graph_type::BG2;
  unsigned             full_length = ((selected_bg == 1) ? 66 : 50) * l_size;

  std::shared_ptr<ldpc_rate_dematcher_factory> dematcher_factory = create_ldpc_rate_dematcher_factory_sw(dec_type);
  TESTASSERT(dematcher_factory);
  std::unique_ptr<ldpc_rate_dematcher> dematcher = dematcher_factory->create();
  TESTASSERT(dematcher);

  std::uniform_int_distribution<int> llr_dist(-10, 10);
  std::vector<log_likelihood_ratio>  input(rm_length);
  std::vector<log_likelihood_ratio>  output(full_length);
  std::generate(input.begin(), input.end(), [&]() { return static_cast<int8_t>(llr_dist(rgen)); });

  codeblock_metadata metadata = {};
  metadata.tb_common.base_graph   = bg;
  metadata.tb_common.lifting_size = static_cast<lifting_size_t>(l_size);
  metadata.tb_common.rv           = rv;
  metadata.tb_common.mod          = modulation;
  metadata.tb_common.Nref         = 0;
  metadata.tb_common.cw_length    = rm_length;
  metadata.cb_specific.full_length     = full_length;
  metadata.cb_specific.rm_length       = rm_length;
  metadata.cb_specific.nof_filler_bits = 0;
  metadata.cb_specific.cw_offset       = 0;
  metadata.cb_specific.nof_crc_bits    = 16;

  benchmarker perf_meas("LDPC rate dematcher", nof_repetitions);

  std::string descr = fmt::format(
      "{} BG={} LS={} E={} full={} rv={} Mod={}", dec_type, selected_bg, l_size, rm_length, full_length, rv, selected_mod);
  perf_meas.new_measure(descr, rm_length, [&]() {
    dematcher->rate_dematch(output, input, true, metadata);
    do_not_optimize(output);
  });

  if (!silent) {
    perf_meas.print_percentiles_throughput("bits");
  }
}

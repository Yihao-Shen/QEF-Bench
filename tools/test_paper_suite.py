import unittest
from run_paper_suite import parser, make_plan


class PaperSuiteTests(unittest.TestCase):
    def args(self, platform, suite):
        extra = ['--bench-script','bench.py','--latency-source','host_wall'] if platform == 'gpu' else ['--benchmark','bench','--cpus','0','--energy-scope','socket0']
        return parser().parse_args(['--platform',platform,'--suite',suite,'--scan-script','scan.py','--out','new-results']+extra)

    def test_gpu_urllc_exact_grid(self):
        p=make_plan(self.args('gpu','urllc'))
        self.assertEqual(p['nominal_grid_points'],480)
        self.assertEqual(p['workload_config']['rb'],[6,12,20,25,37,50])
        self.assertEqual({j['iterations'] for j in p['jobs']},{2,5})
        for j in p['jobs']:
            argv=j['argv']
            self.assertEqual(argv[argv.index('--latency-limit-us')+1],'125')
            self.assertEqual(argv[argv.index('--num-streams')+1],'1')
            self.assertEqual(argv[argv.index('--batches')+1],str(j['aggregation']))
            self.assertIn('--prefer-net-energy',argv)

    def test_cpu_embb_includes_mcs28_and_full_range(self):
        p=make_plan(self.args('cpu','embb'))
        self.assertEqual(p['nominal_grid_points'],21888)
        self.assertEqual(len(p['jobs']),64)
        self.assertEqual(p['workload_config']['mcs'][-1],28)
        self.assertEqual(p['workload_config']['qos_us'],1000)

    def test_reject_invalid_aggregation(self):
        a=self.args('gpu','urllc'); a.aggregation='1,8'
        with self.assertRaises(ValueError): make_plan(a)


if __name__ == '__main__': unittest.main()

import unittest
from replay_historical import replay, summarize


class HistoricalTests(unittest.TestCase):
    def test_release_integrity(self):
        datasets = replay()['datasets']
        self.assertEqual(sum(x['recorded_rows'] for x in datasets), 468)
        self.assertEqual(sum(x['failed_rows'] for x in datasets), 1)
        self.assertEqual(sum(x['missing_configurations'] for x in datasets), 0)

    def test_missing_and_failed_remain_in_denominator(self):
        spec = dict(id='fixture', backend='cpu', latency_column='lat',
                    energy_column='energy', energy_to_nj=1e9, deadline_us=125)
        cfg = dict(rb_list='6', layers_list='1', mcs_list='0-2', cells_list='1')
        row = dict(rb='6', layers='1', mcs='0', cells='1', success='True', lat='100', energy='1e-8')
        failed = {**row, 'mcs': '1', 'success': 'False'}
        result = summarize([row, failed], spec, cfg, [10])
        self.assertEqual(result['missing_configurations'], 1)
        self.assertEqual(result['joint_coverage'][0]['fraction'], 1 / 3)
        with self.assertRaises(ValueError):
            summarize([row, row], spec, cfg, [10])

    def test_no_fallback_or_zero_energy_credit(self):
        spec = dict(id='fixture', backend='gpu', latency_column='lat',
                    energy_column='net', energy_to_nj=1, deadline_us=125)
        cfg = dict(rb_list='6', layers_list='1', mcs_list='0', batches='1')
        row = dict(rb='6', layers='1', mcs='0', batch='1', success='True', lat='100', net='0', gross='10')
        result = summarize([row], spec, cfg, [100])
        self.assertEqual(result['invalid_success_rows'], 1)
        self.assertEqual(result['joint_coverage'][0]['fraction'], 0)

import csv
import json
from pathlib import Path
import tempfile
import unittest
from import_gpu_trace import import_run


class ImportTests(unittest.TestCase):
    def test_preserves_workloads_failures_and_energy_units(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            trace = root / 'trace.csv'
            fields = ['rb','layers','mcs','batch','success','p99_999_us','net_nj_per_bit']
            with trace.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerow(dict(zip(fields, [6,1,0,1,'True',100,10])))
                w.writerow(dict(zip(fields, [6,1,1,1,'False','',''])))
                w.writerow(dict(zip(fields, [6,1,2,1,'True',110,-2])))
            metadata = root / 'meta.json'
            metadata.write_text(json.dumps({'config': {'num_iterations': 2}, 'total_rows': 3, 'successful_rows': 2}))
            records, audit = import_run(trace, metadata)
            self.assertEqual(len(records), 3)
            self.assertEqual(records[0]['energy']['net_nj_per_bit'], 10)
            self.assertEqual(records[0]['energy']['net_nj_per_bit_per_configured_iteration'], 5)
            self.assertIsNone(records[1]['energy']['net_nj_per_bit'])
            self.assertFalse(records[1]['run_success'])
            self.assertEqual(records[2]['energy']['net_nj_per_bit'], -2)
            self.assertIn('negative_net_energy', [v['issue'] for v in audit['findings']])

    def test_rejects_summary_in_place_of_trace(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'trace.csv').write_text('rb,selected\n6,True\n')
            (root/'meta.json').write_text('{"config":{}}')
            with self.assertRaises(ValueError):
                import_run(root/'trace.csv',root/'meta.json')


if __name__ == '__main__':
    unittest.main()

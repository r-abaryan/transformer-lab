"""Synthetic artifact checks; never import frozen experiment or training code."""
import copy
import csv
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
import validate_copy_exposure_continuation as check


FIELDS = 'sequences targets exact_accuracy generated_content_exact token_accuracy content_accuracy loss early_eos on_time_eos late_eos missing_eos first_accuracy outranked_fraction near_tie_fraction'.split()


def fixture():
    counts = dict(train_3=(32, 32), train_4=(128, 128), train_5=(128, 128), train_6=(128, 128),
                  validation_3=(8, 8), validation_4=(32, 32), validation_5=(32, 32), validation_6=(32, 32),
                  test_3=(8, 8), test_4=(64, 64), test_5=(64, 64), test_6=(64, 64), test_8=(64, 64), test_10=(0, 128))
    plan = dict(start=6000, stages=[6500, 7000, 7500, 8000], arms=['cross_only', 'both'])
    report = dict(status='complete', runs=[], gates={}, length10_gain=.25, exposure_specific_benefit=False)
    rows = []
    for step in (6500, 7000, 7500, 8000):
        for arm in ('cross_only', 'both'):
            metrics = {}
            for pool, (distinct, repeated) in counts.items():
                n = int(pool.rsplit('_', 1)[1])
                metrics[pool] = {}
                for group, size in (('all', distinct + repeated), ('distinct', distinct), ('repeated', repeated)):
                    if size == 0:
                        metrics[pool][group] = None
                        continue
                    acc = (.5 if arm == 'cross_only' else .75) if n == 10 else 1.
                    values = dict(zip(FIELDS, (size, size * (n + 1), acc, acc, acc, acc, .1, 1 - acc, acc, 0., 0., 1., 0., 0.)))
                    metrics[pool][group] = values
                    rows.append(dict(arm=arm, step=str(step), pool=pool, group=group, **{k: str(v) for k, v in values.items()}))
            report['runs'].append(dict(arm=arm, step=step, evaluation=dict(metrics=metrics)))
            report['gates'][arm] = dict(short_preserved=True, long_recovered=False, repair=False)
    report['initial'] = dict(metrics=copy.deepcopy(report['runs'][0]['evaluation']['metrics']))
    return plan, report, rows


def csv_text(rows, fields=None):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=fields or ['arm', 'step', 'pool', 'group', *FIELDS])
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def write_fixture(directory, plan, report, rows):
    plan_raw = json.dumps(plan).encode()
    csv_raw = csv_text(rows).encode()
    report = copy.deepcopy(report)
    report.update(plan_sha256=hashlib.sha256(plan_raw).hexdigest(), metrics_sha256=hashlib.sha256(csv_raw).hexdigest())
    for name, raw in (('plan.json', plan_raw), ('metrics.csv', csv_raw), ('report.json', json.dumps(report).encode())):
        (directory / name).write_bytes(raw)


class ContinuationValidationTests(unittest.TestCase):
    def setUp(self):
        self.plan, self.report, self.rows = fixture()

    def validate(self):
        return check.validate(self.plan, self.report, csv_text(self.rows))

    def test_valid_complete_fixture_and_reordered_rows(self):
        self.rows.reverse()
        result = self.validate()
        self.assertEqual(result['runs'], 8)
        self.assertEqual(result['csv_rows'], 328)
        self.assertEqual(result['length10_gain'], {'all': .25, 'repeated': .25})
        self.assertFalse(result['exposure_specific_benefit'])

    def test_duplicate_replacement_passes_old_count_logic_but_is_rejected(self):
        self.rows[1] = dict(self.rows[0])
        by_run = {(r['arm'], r['step']): r for r in self.report['runs']}
        for row in self.rows:
            expected = by_run[(row['arm'], int(row['step']))]['evaluation']['metrics'][row['pool']][row['group']]
            self.assertTrue(all(abs(float(row[k]) - v) < 1e-10 for k, v in expected.items()))
        self.assertEqual(len(self.rows), 328)
        with self.assertRaisesRegex(ValueError, 'Duplicate CSV record'):
            self.validate()

    def test_missing_and_unexpected_records(self):
        for mode in ('missing', 'unexpected', 'null_group'):
            plan, report, rows = fixture()
            if mode == 'missing': rows.pop()
            else: rows[-1]['group'] = 'unknown' if mode == 'unexpected' else 'distinct'
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                check.validate(plan, report, csv_text(rows))

    def test_column_and_ragged_errors(self):
        text = csv_text(self.rows)
        corruptions = [text.replace('targets,', 'sequences,', 1),
                       text.replace('targets,', '', 1), text.replace('targets,', 'unexpected,', 1),
                       text.replace('cross_only,6500', 'extra,cross_only,6500', 1)]
        for bad in corruptions:
            with self.subTest(header=bad[:80]), self.assertRaises((ValueError, csv.Error)):
                check.validate(self.plan, self.report, bad)

    def test_missing_or_duplicate_runs_and_missing_pool(self):
        for mode in ('missing_run', 'duplicate_run', 'missing_pool', 'missing_group', 'missing_null'):
            plan, report, rows = fixture()
            if mode == 'missing_run': report['runs'].pop()
            elif mode == 'duplicate_run': report['runs'][-1] = copy.deepcopy(report['runs'][0])
            else:
                metrics = report['runs'][0]['evaluation']['metrics']
                if mode == 'missing_pool': metrics.pop('test_4')
                elif mode == 'missing_group': metrics['test_4'].pop('distinct')
                else: metrics['test_10'].pop('distinct')
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                check.validate(plan, report, csv_text(rows))

    def test_wrong_metric_schema_and_numbers(self):
        for field, value in (('loss', float('nan')), ('loss', float('inf')), ('loss', -.1),
                             ('exact_accuracy', 1.1), ('sequences', True), ('sequences', 64.0)):
            plan, report, rows = fixture()
            report['runs'][0]['evaluation']['metrics']['train_3']['all'][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                check.validate(plan, report, csv_text(rows))
        values = self.report['runs'][0]['evaluation']['metrics']['train_3']['all']
        values['unknown_metric'] = 0
        with self.assertRaisesRegex(ValueError, 'Wrong metric fields'):
            self.validate()

    def test_bad_csv_values(self):
        for field, value in (('loss', 'NaN'), ('loss', 'Infinity'), ('sequences', '64.0'),
                             ('exact_accuracy', ''), ('step', '06500'), ('loss', '.2')):
            plan, report, rows = fixture()
            rows[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                check.validate(plan, report, csv_text(rows))

    def test_gain_group_disagreement_and_wrong_decisions(self):
        for mode in ('group', 'gain', 'gate', 'benefit'):
            plan, report, rows = fixture()
            if mode == 'group': report['runs'][-1]['evaluation']['metrics']['test_10']['all']['exact_accuracy'] = .8
            elif mode == 'gain': report['length10_gain'] = .3
            elif mode == 'gate': report['gates']['both']['repair'] = True
            else: report['exposure_specific_benefit'] = True
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                check.validate(plan, report, csv_text(rows))

    def test_positive_repair_at_frozen_threshold(self):
        final = self.report['runs'][-1]['evaluation']['metrics']
        for pool, groups in final.items():
            if int(pool.rsplit('_', 1)[1]) <= 6:
                for values in groups.values():
                    values['exact_accuracy'] = .99
        for group in ('all', 'repeated'):
            final['test_10'][group].update(exact_accuracy=.95, on_time_eos=.95, early_eos=.05)
        self.report['length10_gain'] = .45
        self.report['gates']['both'] = dict(short_preserved=True, long_recovered=True, repair=True)
        self.report['exposure_specific_benefit'] = True
        for row in self.rows:
            if row['arm'] == 'both' and row['step'] == '8000':
                row.update({k: str(v) for k, v in final[row['pool']][row['group']].items()})
        self.assertTrue(self.validate()['exposure_specific_benefit'])

    def test_bad_json_root_types_fail_cli_cleanly(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            write_fixture(directory, self.plan, self.report, self.rows)
            (directory / 'report.json').write_text('[]', encoding='utf-8')
            run = subprocess.run([sys.executable, str(Path(check.__file__)), '--directory', str(directory)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 1)
            self.assertFalse(json.loads(run.stderr)['passed'])
            self.assertEqual(run.stdout, '')

    def test_json_duplicate_keys_and_nonfinite_constants(self):
        for raw in ('{"a": 1, "a": 2}', '{"a": NaN}', '{"a": Infinity}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                check.decode_json(raw)

    def test_cli_reads_only_and_does_not_import_torch(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            write_fixture(directory, self.plan, self.report, self.rows)
            before = {p.name: p.read_bytes() for p in directory.iterdir()}
            run = subprocess.run([sys.executable, str(Path(check.__file__)), '--directory', str(directory)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout)['csv_rows'], 328)
            self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir()})
            self.assertNotIn('torch', sys.modules)

    def test_input_hash_mismatch_and_missing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            write_fixture(directory, self.plan, self.report, self.rows)
            (directory / 'metrics.csv').write_text('changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Metrics hash mismatch'):
                check.validate_directory(directory)
            (directory / 'metrics.csv').unlink()
            with self.assertRaises(OSError):
                check.validate_directory(directory)


if __name__ == '__main__':
    unittest.main()

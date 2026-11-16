"""Benefit and serialization boundaries; these fixtures reuse the checked study schema."""
import copy
import csv
import io
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
import validate_copy_exposure_continuation as check


def fixture(control=.9, benefit=False):
    plan = dict(start=6000, stages=list(check.STAGES), arms=list(check.ARMS))
    report = dict(status='complete', runs=[], length10_gain=1. - control,
                  exposure_specific_benefit=benefit, gates={
                      'cross_only': dict(short_preserved=True, long_recovered=False, repair=False),
                      'both': dict(short_preserved=True, long_recovered=True, repair=True)})
    rows = []
    for step in check.STAGES:
        for arm in check.ARMS:
            metrics = {}
            for pool, sizes in check.COUNTS.items():
                length = int(pool.rsplit('_', 1)[1])
                metrics[pool] = {}
                for group, count in zip(check.GROUPS, (sum(sizes), *sizes)):
                    if not count:
                        metrics[pool][group] = None
                        continue
                    acc = control if pool == 'test_10' and arm == 'cross_only' else 1.
                    values = dict(zip(check.METRICS,
                                      (count, count * (length + 1), acc, acc, acc, acc,
                                       .1, 1 - acc, acc, 0., 0., 1., 0., 0.)))
                    metrics[pool][group] = values
                    rows.append(dict(arm=arm, step=str(step), pool=pool, group=group,
                                     **{k: str(v) for k, v in values.items()}))
            report['runs'].append(dict(arm=arm, step=step, evaluation=dict(metrics=metrics)))
    report['initial'] = dict(metrics=copy.deepcopy(report['runs'][0]['evaluation']['metrics']))
    return plan, report, rows


def score(plan, report, rows):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=[*check.KEYS, *check.METRICS])
    writer.writeheader()
    writer.writerows(rows)
    return check.validate(plan, report, out.getvalue())


class ContinuationBoundaryTests(unittest.TestCase):
    def test_literal_benefit_threshold_and_adjacent_floats(self):
        for gain, expected in ((math.nextafter(.10, -math.inf), False),
                               (.10, True), (math.nextafter(.10, math.inf), True)):
            with self.subTest(gain=repr(gain)):
                self.assertIs(check.exposure_benefit(True, gain), expected)

    def test_gain_cannot_bypass_failed_repair(self):
        for gain in (math.nextafter(.10, -math.inf), .10, math.nextafter(.10, math.inf), .5):
            with self.subTest(gain=repr(gain)):
                self.assertIs(check.exposure_benefit(False, gain), False)

    def test_computed_gain_preserves_subtraction_semantics(self):
        for control, expected in ((.9, False), (math.nextafter(.9, 0.), True)):
            plan, report, rows = fixture(control, expected)
            gain = 1. - control
            with self.subTest(control=repr(control), computed_gain=repr(gain)):
                self.assertNotEqual(gain, .10)
                self.assertEqual(gain >= .10, expected)
                result = score(plan, report, rows)
                self.assertEqual(result['length10_gain']['repeated'], gain)
                self.assertIs(result['exposure_specific_benefit'], expected)
                report['exposure_specific_benefit'] = not expected
                with self.assertRaisesRegex(ValueError, 'Stored benefit gate differs'):
                    score(plan, report, rows)

    def test_csv_differences_inside_and_outside_tolerance(self):
        for field in ('loss', 'exact_accuracy'):
            for sign in (-1, 1):
                for factor, passes in ((.5, True), (2., False)):
                    plan, report, rows = fixture()
                    row = next(r for r in rows if r['arm'] == 'cross_only' and r['step'] == '8000'
                               and r['pool'] == 'test_10' and r['group'] == 'all')
                    base = float(row[field])
                    changed = base + sign * factor * check.TOLERANCE
                    row[field] = repr(changed)
                    with self.subTest(field=field, sign=sign, factor=factor):
                        self.assertEqual(abs(changed - base) <= check.TOLERANCE, passes)
                        if passes:
                            self.assertTrue(score(plan, report, rows)['passed'])
                        else:
                            with self.assertRaisesRegex(ValueError, 'Numeric mismatch'):
                                score(plan, report, rows)

    def test_inclusive_exact_tolerance_and_adjacent_floats(self):
        for sign in (-1, 1):
            for distance in (math.nextafter(check.TOLERANCE, 0.), check.TOLERANCE):
                check.close(0., sign * distance, 'inclusive boundary')
            with self.assertRaisesRegex(ValueError, 'Numeric mismatch'):
                check.close(0., sign * math.nextafter(check.TOLERANCE, math.inf), 'outside boundary')

    def test_nearest_serialized_values_straddling_tolerance(self):
        for field in ('loss', 'exact_accuracy'):
            for sign in (-1, 1):
                plan, report, rows = fixture()
                row = next(r for r in rows if r['arm'] == 'cross_only' and r['step'] == '8000'
                           and r['pool'] == 'test_10' and r['group'] == 'all')
                base = float(row[field])
                inside = base + sign * check.TOLERANCE
                while abs(inside - base) > check.TOLERANCE:
                    inside = math.nextafter(inside, base)
                outside = math.nextafter(inside, math.inf if sign > 0 else -math.inf)
                self.assertLessEqual(abs(inside - base), check.TOLERANCE)
                self.assertGreater(abs(outside - base), check.TOLERANCE)
                with self.subTest(field=field, sign=sign):
                    row[field] = repr(inside)
                    self.assertTrue(score(plan, report, rows)['passed'])
                    row[field] = repr(outside)
                    with self.assertRaisesRegex(ValueError, 'Numeric mismatch'):
                        score(plan, report, rows)


if __name__ == '__main__':
    unittest.main()

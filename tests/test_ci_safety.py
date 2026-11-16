"""Synthetic checks for the CI summary gate; no platform coverage is simulated as real."""
import tempfile
import unittest
import json
import os
import sys
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import ci_safety as ci


def summary(windows=False):
    skipped = [] if windows else [dict(test=ci.NATIVE + ' (test_safe_runner.SafeRunnerTests.' + ci.NATIVE + ')',
                                      detail='Native Windows junction test; not simulated.')]
    hashes = {'tests/safe_runner.py': 'a' * 64, 'tests/test_safe_runner.py': 'b' * 64}
    return dict(schema_version=1, suite='runner-safety', passed=True, frozen_files_unchanged=True,
                exception=None, methods=22, failures=0, errors=0, training_updates=0,
                snapshot_errors={'before': [], 'after': []},
                integrity_changes={'changed': [], 'missing': [], 'unexpected': []},
                before=hashes, after=dict(hashes), skipped=len(skipped),
                test_details=dict(failures=[], errors=[], skipped=skipped))


class CISafetyTests(unittest.TestCase):
    def test_valid_platform_summaries(self):
        for windows in (False, True):
            with self.subTest(windows=windows):
                self.assertEqual(ci.verify_summary(summary(windows), windows=windows), summary(windows)['test_details']['skipped'])

    def test_windows_junction_skip_rejected(self):
        with self.assertRaises(ValueError):
            ci.verify_summary(summary(False), windows=True)

    def test_incomplete_or_failed_summaries_rejected(self):
        mutations = [('passed', False), ('methods', 0), ('failures', 1), ('errors', 1),
                     ('training_updates', 1), ('suite', 'archived-copy'), ('exception', {'type': 'Error'}),
                     ('frozen_files_unchanged', False), ('snapshot_errors', {}), ('integrity_changes', {}),
                     ('before', {}), ('after', {}), ('test_details', {}), ('skipped', 99)]
        for key, value in mutations:
            data = summary()
            data[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                ci.verify_summary(data, windows=False)

    def test_invalid_hash_rejected(self):
        data = summary()
        data['before']['tests/safe_runner.py'] = 'not-a-hash'
        data['after'] = dict(data['before'])
        with self.assertRaises(ValueError):
            ci.verify_summary(data, windows=False)

    def test_allowed_symlink_skip_remains_explicit(self):
        data = summary(True)
        data['test_details']['skipped'].append(dict(test='test_output_root_symlink_rejected (case)',
                                                   detail='Creating symlinks is not permitted on this platform.'))
        data['skipped'] = 1
        self.assertEqual(len(ci.verify_summary(data, windows=True)), 1)

    def test_unexpected_skip_rejected(self):
        data = summary(True)
        data['test_details']['skipped'].append(dict(test='test_unknown (case)', detail='ignored'))
        data['skipped'] = 1
        with self.assertRaises(ValueError):
            ci.verify_summary(data, windows=True)

    def test_missing_linux_native_skip_rejected(self):
        with self.assertRaises(ValueError):
            ci.verify_summary(summary(True), windows=False)

    def check_driver(self, failed_command=False, missing_summary=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            def fresh(base, name, **kwargs):
                target = base / name
                target.mkdir(parents=True)
                return target
            def logged(command, *, cwd, log):
                log.write_text('synthetic subprocess fixture', encoding='utf-8')
                if '--run-name' in command and not missing_summary:
                    path = root / 'results/test_runs' / command[-1]
                    path.mkdir()
                    (path / 'summary.json').write_text(json.dumps(summary(os.name == 'nt')), encoding='utf-8')
                return dict(command=command, cwd=str(cwd), returncode=7 if failed_command and log.stem == 'unit-safety' else 0, log=log.name)
            with patch.object(ci, 'ROOT', root), patch.object(ci, 'run_logged', side_effect=logged), \
                 patch.dict(sys.modules, {'safe_runner': SimpleNamespace(fresh_directory=fresh)}):
                code = ci.main()
            reports = list((root / 'results/test_runs').glob('ci_*/ci-report.json'))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text(encoding='utf-8'))
            self.assertEqual(len(report['commands']), 4)
            return code, report

    def test_driver_records_success(self):
        code, report = self.check_driver()
        self.assertEqual(code, 0)
        self.assertTrue(report['passed'])

    def test_driver_cannot_hide_nonzero_command(self):
        code, report = self.check_driver(failed_command=True)
        self.assertEqual(code, 1)
        self.assertFalse(report['passed'])
        self.assertEqual(report['commands'][0]['returncode'], 7)

    def test_driver_records_missing_summary(self):
        code, report = self.check_driver(missing_summary=True)
        self.assertEqual(code, 1)
        self.assertEqual(report['exception']['type'], 'FileNotFoundError')

    def test_command_failure_and_log_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / 'command.log'
            with patch.object(ci.subprocess, 'run', return_value=type('Result', (), {'returncode': 7})()) as run:
                result = ci.run_logged(['python', '-V'], cwd=Path(folder), log=log)
            self.assertEqual(result['returncode'], 7)
            self.assertTrue(log.exists())
            self.assertEqual(run.call_args.kwargs['timeout'], 180)
            self.assertEqual(run.call_args.kwargs['env']['PYTHONUTF8'], '1')


if __name__ == '__main__':
    unittest.main()

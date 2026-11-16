"""Synthetic runner regressions; no archived imports, checkpoints or training."""
import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import safe_runner as runner


class SafeRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        paths = ['tests/safe_runner.py', 'tests/test_safe_runner.py',
                 'tests/run_tests.py', 'tests/test_runner_safety.py']
        paths += ['tests/' + n + '.py' for n in runner.ARCHIVED]
        paths += ['docs/protocols/' + n.replace('_', '-') + '.md' for n in runner.FROZEN]
        paths += ['results/' + n + '/nested/checkpoint.bin' for n in runner.FROZEN]
        for relative in paths:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'protected fixture')
        self.target = self.root / 'results' / runner.FROZEN[0] / 'nested/checkpoint.bin'
        self.output = runner.fresh_directory(self.root / 'runs', 'fresh', root=self.root)

    def run_fixture(self, action=None, error=None, suite='archived-copy'):
        def load(name, directory):
            if action:
                action()
            if error:
                raise error
            return unittest.TestSuite([unittest.FunctionTestCase(lambda: None)])
        with patch.object(runner, 'load_suite', side_effect=load), contextlib.redirect_stderr(io.StringIO()):
            code = runner.execute(self.output, suite, self.root)
        return code, json.loads((self.output / 'summary.json').read_text())

    def test_success(self):
        code, report = self.run_fixture()
        self.assertEqual(code, 0)
        self.assertTrue(report['passed'])
        self.assertEqual(report['methods'], 1)
        self.assertIn(self.target.relative_to(self.root).as_posix(), report['before'])

    def test_import_failure_still_checks_integrity_and_records_summary(self):
        code, report = self.run_fixture(error=ImportError('deliberate import failure'))
        self.assertEqual(code, 1)
        self.assertEqual(report['exception']['type'], 'ImportError')
        self.assertTrue(report['frozen_files_unchanged'])
        self.assertEqual(report['methods'], 0)

    def test_fixture_setup_failure_records_mutation(self):
        code, report = self.run_fixture(lambda: self.target.write_bytes(b'changed'), RuntimeError('setup failed'))
        self.assertEqual(code, 1)
        self.assertEqual(report['exception']['type'], 'RuntimeError')
        self.assertEqual(report['integrity_changes']['changed'], [self.target.relative_to(self.root).as_posix()])

    def test_deleted_protected_file(self):
        code, report = self.run_fixture(self.target.unlink)
        self.assertEqual(code, 1)
        self.assertIn(self.target.relative_to(self.root).as_posix(), report['integrity_changes']['missing'])
        self.assertFalse(report['frozen_files_unchanged'])

    def test_unexpected_nested_file(self):
        added = self.target.parent / 'unexpected.bin'
        code, report = self.run_fixture(lambda: added.write_bytes(b'extra'))
        self.assertEqual(code, 1)
        self.assertIn(added.relative_to(self.root).as_posix(), report['integrity_changes']['unexpected'])

    def test_missing_baseline_prevents_loading(self):
        (self.root / 'tests/run_tests.py').unlink()
        with patch.object(runner, 'load_suite') as load:
            self.assertEqual(runner.execute(self.output, 'archived-copy', self.root), 1)
        load.assert_not_called()
        report = json.loads((self.output / 'summary.json').read_text())
        self.assertTrue(report['snapshot_errors']['before'])
        self.assertFalse(report['passed'])

    def test_after_snapshot_exception_is_recorded(self):
        with patch.object(runner, 'snapshot', side_effect=[({'file': 'hash'}, []), OSError('unreadable')]), \
             patch.object(runner, 'load_suite', return_value=unittest.TestSuite([unittest.FunctionTestCase(lambda: None)])), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(runner.execute(self.output, 'runner-safety', self.root), 1)
        report = json.loads((self.output / 'summary.json').read_text())
        self.assertEqual(report['snapshot_errors']['after'][0]['type'], 'OSError')

    def test_existing_directory_preserves_sentinel(self):
        sentinel = self.output / 'sentinel'
        sentinel.write_bytes(b'keep')
        with self.assertRaises(FileExistsError):
            runner.fresh_directory(self.output.parent, self.output.name, root=self.root)
        self.assertEqual(sentinel.read_bytes(), b'keep')

    def test_invalid_names_do_not_create_parent(self):
        base = self.root / 'not-created'
        for name in ('../escape', '..', 'a/b', 'a\\b', 'C:\\escape', '', 'a' * 81):
            with self.subTest(name=name), self.assertRaises(ValueError):
                runner.fresh_directory(base, name, root=self.root)
        self.assertFalse(base.exists())

    def test_runner_safety_does_not_require_archive(self):
        self.target.unlink()
        code, report = self.run_fixture(suite='runner-safety')
        self.assertEqual(code, 0)
        self.assertEqual(set(report['before']), {'tests/safe_runner.py', 'tests/test_safe_runner.py'})

    def test_unknown_suite_rejected(self):
        with self.assertRaises(ValueError):
            runner.snapshot(self.root, 'everything')
        with self.assertRaises(ValueError):
            runner.load_suite('everything', self.output)

    def test_unittest_setup_error_is_reported(self):
        class Broken(unittest.TestCase):
            def setUp(self):
                raise RuntimeError('deliberate fixture setup error')

            def test_body(self):
                pass
        with patch.object(runner, 'load_suite', return_value=unittest.defaultTestLoader.loadTestsFromTestCase(Broken)), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(runner.execute(self.output, 'runner-safety', self.root), 1)
        report = json.loads((self.output / 'summary.json').read_text())
        self.assertEqual(report['errors'], 1)
        self.assertIn('deliberate fixture setup error', report['test_details']['errors'][0]['detail'])
        self.assertTrue(report['frozen_files_unchanged'])

    def test_existing_summary_is_not_overwritten(self):
        summary = self.output / 'summary.json'
        summary.write_bytes(b'keep existing summary')
        with patch.object(runner, 'load_suite', return_value=unittest.TestSuite()), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(runner.execute(self.output, 'runner-safety', self.root), 1)
        self.assertEqual(summary.read_bytes(), b'keep existing summary')

    def test_symlink_is_rejected_without_following(self):
        link = self.target.parent / 'linked.bin'
        self.make_symlink(link, self.root / 'tests/safe_runner.py', directory=False)
        entries, errors = runner.snapshot(self.root, 'archived-copy')
        self.assertNotIn(link.relative_to(self.root).as_posix(), entries)
        self.assertTrue(any(e['path'] == link.relative_to(self.root).as_posix() for e in errors))

    def test_empty_suite_fails(self):
        with patch.object(runner, 'load_suite', return_value=unittest.TestSuite()), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(runner.execute(self.output, 'runner-safety', self.root), 1)

    def make_symlink(self, link, destination, *, directory=True):
        try:
            link.symlink_to(destination, target_is_directory=directory)
        except (OSError, NotImplementedError):
            self.skipTest('Creating symlinks is not permitted on this platform.')

    def output_layout(self):
        checkout = self.root / 'checkout'
        outside = self.root / 'outside'
        checkout.mkdir()
        outside.mkdir()
        (outside / 'sentinel').write_bytes(b'untouched')
        return checkout, outside

    def assert_outside_unchanged(self, outside):
        self.assertEqual([p.name for p in outside.iterdir()], ['sentinel'])
        self.assertEqual((outside / 'sentinel').read_bytes(), b'untouched')

    def test_output_root_symlink_rejected(self):
        checkout, outside = self.output_layout()
        (checkout / 'results').mkdir()
        base = checkout / 'results/test_runs'
        self.make_symlink(base, outside)
        with self.assertRaises(ValueError):
            runner.fresh_directory(base, 'attempt', root=checkout)
        self.assert_outside_unchanged(outside)

    def test_output_parent_symlink_rejected(self):
        checkout, outside = self.output_layout()
        self.make_symlink(checkout / 'results', outside)
        with self.assertRaises(ValueError):
            runner.fresh_directory(checkout / 'results/test_runs', 'attempt', root=checkout)
        self.assert_outside_unchanged(outside)

    def test_dangling_output_symlink_rejected(self):
        checkout, outside = self.output_layout()
        destination = outside / 'missing'
        self.make_symlink(checkout / 'results', destination)
        with self.assertRaises(ValueError):
            runner.fresh_directory(checkout / 'results/test_runs', 'attempt', root=checkout)
        self.assertFalse(destination.exists())
        self.assert_outside_unchanged(outside)

    def test_outside_and_parent_traversal_rejected(self):
        checkout, outside = self.output_layout()
        for base in (outside, checkout / '..' / 'outside', Path('relative')):
            with self.subTest(base=base), self.assertRaises(ValueError):
                runner.fresh_directory(base, 'attempt', root=checkout)
        self.assert_outside_unchanged(outside)

    def test_output_reparse_flag_rejected(self):
        checkout, outside = self.output_layout()
        parent = checkout / 'results'
        parent.mkdir()
        original = Path.lstat
        def flagged(path, *args, **kwargs):
            info = original(path, *args, **kwargs)
            if path == parent:
                return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
            return info
        with patch.object(Path, 'lstat', flagged), self.assertRaises(ValueError):
            runner.fresh_directory(parent / 'test_runs', 'attempt', root=checkout)
        self.assertEqual(list(parent.iterdir()), [])
        self.assert_outside_unchanged(outside)

    @unittest.skipUnless(os.name == 'nt', 'Native Windows junction test; not simulated.')
    def test_native_windows_output_junction_rejected(self):
        checkout, outside = self.output_layout()
        link = checkout / 'results'
        process = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(outside)],
                                 capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.addCleanup(link.rmdir)
        with self.assertRaises(ValueError):
            runner.fresh_directory(link / 'test_runs', 'attempt', root=checkout)
        self.assert_outside_unchanged(outside)

    def test_cli_rejects_linked_output_before_loading(self):
        checkout, outside = self.output_layout()
        self.make_symlink(checkout / 'results', outside)
        with patch.object(runner, 'ROOT', checkout), patch.object(runner, 'execute') as execute, \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            runner.main(['--suite', 'runner-safety', '--run-name', 'attempt'])
        self.assertEqual(error.exception.code, 2)
        execute.assert_not_called()
        self.assert_outside_unchanged(outside)


if __name__ == '__main__':
    unittest.main()

"""Explicit test suites with retained fixtures and fail-closed integrity checks."""
import argparse
import hashlib
import importlib
import json
import re
import sys
import traceback
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITES = ('archived-copy', 'runner-safety')
ARCHIVED = ('test_copy_rope', 'test_copy_marker', 'test_copy_marker_alignment')
FROZEN = ('copy_rope_training', 'copy_marker_training', 'copy_marker_alignment')


def fresh_directory(base, name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', name):
        raise ValueError('Run name must be one alphanumeric, underscore or hyphen component.')
    base = base.resolve()
    base.mkdir(parents=True, exist_ok=True)
    target = base / name
    target.mkdir(exist_ok=False)
    return target


def snapshot(root, suite):
    if suite not in SUITES:
        raise ValueError('Unknown suite.')
    paths = ['tests/safe_runner.py', 'tests/test_safe_runner.py']
    if suite == 'archived-copy':
        paths += ['tests/run_tests.py', 'tests/test_runner_safety.py']
        paths += ['tests/' + name + '.py' for name in ARCHIVED]
        paths += ['docs/protocols/' + name.replace('_', '-') + '.md' for name in FROZEN]
        paths += ['results/' + name for name in FROZEN]
    entries, errors = {}, []

    def visit(path):
        key = path.relative_to(root).as_posix()
        try:
            if any((root / Path(*path.relative_to(root).parts[:i])).is_symlink()
                   for i in range(1, len(path.relative_to(root).parts) + 1)):
                raise ValueError('Symlinks are not supported in protected paths.')
            if path.is_dir():
                entries[key] = 'directory'
                for child in sorted(path.iterdir()):
                    visit(child)
            elif path.is_file():
                h = hashlib.sha256()
                with path.open('rb') as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b''):
                        h.update(block)
                entries[key] = h.hexdigest()
            else:
                raise FileNotFoundError('Required protected path is missing or not a regular file/directory.')
        except Exception as error:
            errors.append(dict(path=key, type=type(error).__name__, message=str(error)))

    for relative in paths:
        visit(root / relative)
    return entries, errors


def differences(before, after):
    return dict(changed=sorted(k for k in before.keys() & after.keys() if before[k] != after[k]),
                missing=sorted(before.keys() - after.keys()),
                unexpected=sorted(after.keys() - before.keys()))


def load_suite(name, directory):
    if name == 'runner-safety':
        module = importlib.import_module('test_safe_runner')
        return unittest.defaultTestLoader.loadTestsFromModule(module)
    if name != 'archived-copy':
        raise ValueError('Unknown suite.')
    import run_tests
    modules = [run_tests.load_archived(n, directory / n) for n in ARCHIVED]
    safety = importlib.import_module('test_runner_safety')
    safety.SAFETY_ROOT = directory / 'runner_safety'
    return unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m)
                              for m in (*modules, safety))


def execute(directory, suite, root=ROOT):
    summary = dict(schema_version=1, suite=suite, passed=False, methods=0,
                   failures=0, errors=0, skipped=0, training_updates=0, test_details={},
                   fixture_policy='Fresh retained folders; historical main blocks are not invoked.',
                   integrity_scope='Declared sources/protocols and recursive frozen directories only.',
                   before={}, after={}, snapshot_errors={}, exception=None)
    result = None
    try:
        summary['before'], summary['snapshot_errors']['before'] = snapshot(root, suite)
        if summary['snapshot_errors']['before']:
            raise RuntimeError('Protected baseline is incomplete; tests were not loaded.')
        tests = load_suite(suite, directory)
        result = unittest.TextTestRunner(verbosity=2).run(tests)
        summary.update(methods=result.testsRun, failures=len(result.failures),
                       errors=len(result.errors), skipped=len(result.skipped),
                       test_details={key: [dict(test=str(test), detail=detail) for test, detail in items]
                                     for key, items in (('failures', result.failures),
                                                        ('errors', result.errors), ('skipped', result.skipped))})
    except (Exception, KeyboardInterrupt, SystemExit) as error:
        summary['exception'] = dict(type=type(error).__name__, message=str(error),
                                    traceback=traceback.format_exc())
    finally:
        try:
            summary['after'], summary['snapshot_errors']['after'] = snapshot(root, suite)
        except (Exception, KeyboardInterrupt, SystemExit) as error:
            summary['snapshot_errors']['after'] = [dict(type=type(error).__name__, message=str(error))]
        summary['integrity_changes'] = differences(summary['before'], summary['after'])
        summary['frozen_files_unchanged'] = (not any(summary['snapshot_errors'].values())
                                            and not any(summary['integrity_changes'].values()))
        summary['passed'] = bool(result is not None and result.testsRun > 0
                                 and result.wasSuccessful() and summary['exception'] is None
                                 and summary['frozen_files_unchanged'])
        try:
            with (directory / 'summary.json').open('x', encoding='utf-8') as handle:
                json.dump(summary, handle, indent=2)
                handle.write('\n')
        except OSError as error:
            print('Cannot write failure/success summary: ' + str(error), file=sys.stderr)
            return 1
    return 0 if summary['passed'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=SUITES, default='archived-copy')
    parser.add_argument('--run-name', default='safe_' + uuid.uuid4().hex)
    args = parser.parse_args(argv)
    try:
        directory = fresh_directory(ROOT / 'results/test_runs', args.run_name)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    sys.path.insert(0, str(ROOT / 'experiments'))
    sys.path.insert(0, str(ROOT / 'tests'))
    code = execute(directory, args.suite)
    print('Run directory: ' + str(directory))
    return code


if __name__ == '__main__':
    sys.exit(main())

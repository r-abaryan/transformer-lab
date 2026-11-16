"""Standard-library CI checks only; never discover or run archived experiments."""
import json
import os
import platform
import re
import subprocess
import sys
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE = 'test_native_windows_output_junction_rejected'
SYMLINK_TESTS = {
    'test_symlink_is_rejected_without_following', 'test_output_root_symlink_rejected',
    'test_output_parent_symlink_rejected', 'test_dangling_output_symlink_rejected',
    'test_cli_rejects_linked_output_before_loading',
}


def verify_summary(data, *, windows):
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(data.get('schema_version') == 1 and data.get('suite') == 'runner-safety', 'Wrong summary schema/suite.')
    require(data.get('passed') is True and data.get('frozen_files_unchanged') is True, 'Runner/integrity failed.')
    require(data.get('exception') is None, 'Runner raised an exception.')
    require(type(data.get('methods')) is int and data['methods'] > 0, 'No test methods recorded.')
    require(data.get('failures') == 0 and data.get('errors') == 0 and data.get('training_updates') == 0,
            'Failures, errors or training recorded.')
    require(data.get('snapshot_errors') == {'before': [], 'after': []}, 'Snapshot errors or missing checks.')
    require(data.get('integrity_changes') == {'changed': [], 'missing': [], 'unexpected': []}, 'Integrity changes.')
    before = data.get('before', {})
    require(set(before) == {'tests/safe_runner.py', 'tests/test_safe_runner.py'}, 'Unexpected source inventory.')
    require(all(isinstance(h, str) and re.fullmatch('[0-9a-f]{64}', h) for h in before.values()), 'Invalid source hashes.')
    require(before == data.get('after'), 'Source inventories differ.')
    details = data.get('test_details', {})
    require(details.get('failures') == [] and details.get('errors') == [], 'Missing or nonempty failure details.')
    skipped = details.get('skipped')
    require(isinstance(skipped, list) and type(data.get('skipped')) is int
            and len(skipped) == data['skipped'] < data['methods'], 'Invalid skip accounting.')
    native_skips = 0
    for entry in skipped:
        name, reason = entry['test'].split(' ', 1)[0], entry['detail']
        if name == NATIVE:
            require(not windows and reason == 'Native Windows junction test; not simulated.', 'Native Windows junction coverage skipped.')
            native_skips += 1
        else:
            require(name in SYMLINK_TESTS and reason == 'Creating symlinks is not permitted on this platform.', 'Unexpected skipped test.')
    require(native_skips == (0 if windows else 1), 'Unexpected native-junction skip accounting.')
    return skipped


def run_logged(command, *, cwd, log):
    with log.open('x', encoding='utf-8') as handle:
        process = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT,
                                 timeout=180, env={**os.environ, 'PYTHONUTF8': '1'})
    print(log.read_text(encoding='utf-8', errors='replace'), flush=True)
    return dict(command=command, cwd=str(cwd), returncode=process.returncode, log=log.name)


def main():
    from safe_runner import fresh_directory
    token = uuid.uuid4().hex
    directory = fresh_directory(ROOT / 'results/test_runs', 'ci_' + token, root=ROOT)
    cli_name = 'ci_cli_' + token
    report = dict(passed=False, python=sys.version, platform=platform.platform(), commands=[], exception=None)
    commands = [
        ('unit-safety', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_safe_runner.py', '-v'], ROOT),
        ('unit-ci', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_ci_safety.py', '-v'], ROOT),
        ('native-junction', [sys.executable, '-m', 'unittest', 'test_safe_runner.SafeRunnerTests.' + NATIVE, '-v'], ROOT / 'tests'),
        ('runner-cli', [sys.executable, 'tests/safe_runner.py', '--suite', 'runner-safety', '--run-name', cli_name], ROOT),
    ]
    try:
        for name, command, cwd in commands:
            result = run_logged(command, cwd=cwd, log=directory / (name + '.log'))
            report['commands'].append(dict(name=name, **result))
        source = ROOT / 'results/test_runs' / cli_name / 'summary.json'
        data = json.loads(source.read_text(encoding='utf-8'))
        with (directory / 'runner-summary.json').open('x', encoding='utf-8') as handle:
            json.dump(data, handle, indent=2)
        report['skipped_tests'] = verify_summary(data, windows=os.name == 'nt')
        print('Skipped tests: ' + json.dumps(report['skipped_tests']), flush=True)
        report['passed'] = all(r['returncode'] == 0 for r in report['commands'])
    except Exception as error:
        report['exception'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    finally:
        with (directory / 'ci-report.json').open('x', encoding='utf-8') as handle:
            json.dump(report, handle, indent=2)
            handle.write('\n')
    print('CI report: ' + str(directory / 'ci-report.json'), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())

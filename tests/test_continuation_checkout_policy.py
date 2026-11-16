"""Exercise Git checkout filters in isolated repositories; never rewrite real artifacts."""
import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = tuple('results/copy_exposure_continuation/' + name
                for name in ('plan.json', 'report.json', 'metrics.csv'))


@unittest.skipUnless(shutil.which('git'), 'Git is required for checkout-policy tests.')
class ContinuationCheckoutPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
        empty = self.root / 'empty-global-config'
        empty.write_bytes(b'')
        self.env.update(GIT_CONFIG_NOSYSTEM='1', GIT_ATTR_NOSYSTEM='1', GIT_CONFIG_GLOBAL=str(empty))
        self.git('init', '-q')
        self.git('config', 'core.autocrlf', 'false')
        self.git('config', 'core.attributesFile', str(empty))

    def git(self, *args):
        return subprocess.run(['git', *args], cwd=self.root, env=self.env,
                              check=True, capture_output=True).stdout

    def install_policy(self):
        (self.root / '.gitattributes').write_bytes((ROOT / '.gitattributes').read_bytes())
        self.git('add', '-f', '.gitattributes')

    def test_policy_is_limited_to_three_exact_paths(self):
        self.install_policy()
        others = ('results/copy_exposure_continuation/audit.json',
                  'results/copy_exposure_continuation/extra.json',
                  'results/other/plan.json', 'experiments/copy_exposure_continuation.py')
        for path in (*TARGETS, *others):
            lines = self.git('check-attr', 'text', 'eol', '--', path).decode().splitlines()
            values = {line.split(': ')[1]: line.split(': ')[2] for line in lines}
            expected = {'text': 'set', 'eol': 'crlf'} if path in TARGETS else {'text': 'unspecified', 'eol': 'unspecified'}
            self.assertEqual(values, expected, path)

    def test_fresh_materialization_restores_crlf_and_preserves_blobs(self):
        fixtures = {path: (b'column\nvalue\n' if path.endswith('.csv') else b'{\n  "fixture": true\n}\n')
                    for path in TARGETS}
        for path, raw in fixtures.items():
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            self.git('add', '-f', path)
        blobs = {path: self.git('rev-parse', ':' + path).strip() for path in TARGETS}
        self.install_policy()
        for setting in ('false', 'true'):
            self.git('config', 'core.autocrlf', setting)
            for path in TARGETS:
                (self.root / path).unlink()
            self.git('checkout-index', '--force', '--', *TARGETS)
            for path, raw in fixtures.items():
                with self.subTest(core_autocrlf=setting, path=path):
                    expected = raw.replace(b'\n', b'\r\n')
                    restored = (self.root / path).read_bytes()
                    self.assertEqual(restored, expected)
                    self.assertEqual(hashlib.sha256(restored).digest(), hashlib.sha256(expected).digest())
                    self.assertNotEqual(hashlib.sha256(restored).digest(), hashlib.sha256(raw).digest())
                    self.assertEqual(self.git('rev-parse', ':' + path).strip(), blobs[path])
                    self.assertEqual(self.git('cat-file', 'blob', ':' + path), raw)
            self.git('diff', '--exit-code', '--', *TARGETS)

    def test_missing_policy_does_not_satisfy_crlf_reference(self):
        path = TARGETS[0]
        target = self.root / path
        target.parent.mkdir(parents=True)
        raw = b'{\n  "fixture": true\n}\n'
        expected = hashlib.sha256(raw.replace(b'\n', b'\r\n')).digest()
        target.write_bytes(raw)
        self.git('add', '-f', path)
        target.unlink()
        self.git('checkout-index', '--force', '--', path)
        self.assertNotEqual(hashlib.sha256(target.read_bytes()).digest(), expected)
        self.install_policy()
        target.unlink()
        self.git('checkout-index', '--force', '--', path)
        self.assertEqual(hashlib.sha256(target.read_bytes()).digest(), expected)
        target.write_bytes(target.read_bytes().replace(b'true', b'false'))
        self.assertNotEqual(hashlib.sha256(target.read_bytes()).digest(), expected)


if __name__ == '__main__':
    unittest.main()

"""Script tests on fixtures; they neither build nor operate the real bot."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Scripts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='portable scripts ', dir=os.environ.get('TMPDIR'))
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'bot with spaces'
        self.root.mkdir()
        for name in ['launch.sh', 'clean.sh']:
            shutil.copy2(ROOT / name, self.root / name)
        for name in ['Cargo.toml', 'Cargo.lock', 'config/general.toml',
                     'src/tools/model/artifacts/btc5m/model.json',
                     'src/tools/model/artifacts/btc5m/manifest.json',
                     'src/api/static/dist/index.html']:
            self.put(name, 'fixture')
        self.bin = Path(self.tmp.name) / 'fake-bin'
        self.bin.mkdir()
        for name in ['cargo', 'rustc', 'cc', 'cmake', 'pkg-config']:
            self.command(name, 'exit 0')
        self.put('target/release/poly-bot', '#!/bin/bash\nprintf "%s" "$HRF_MODE" > launched\n')
        (self.root / 'target/release/poly-bot').chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.bin) + ':/usr/bin:/bin', HRF_MODE='live')

    def put(self, name, text):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def command(self, name, text):
        p = self.bin / name
        p.write_text('#!/bin/bash\n' + text + '\n')
        p.chmod(0o755)

    def run_script(self, name, *args, input=''):
        return subprocess.run(['bash', str(self.root / name), *args], cwd=self.tmp.name,
                              env=self.env, input=input, text=True, capture_output=True, timeout=20)

    def test_help_and_check_without_env(self):
        for name in ['launch.sh', 'clean.sh']:
            self.assertEqual(self.run_script(name, '--help').returncode, 0)
        self.assertEqual(self.run_script('launch.sh', '--check').returncode, 0)
        self.assertFalse((self.root / 'launched').exists())

    def test_default_paper_and_no_shell_env(self):
        self.put('.env', 'HRF_MODE=live\ntouch PWNED\n')
        result = self.run_script('launch.sh')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / 'launched').read_text(), 'paper')
        self.assertFalse((self.root / 'PWNED').exists())

    def test_cargo_failure_never_launches(self):
        self.command('cargo', 'exit 42')
        self.assertEqual(self.run_script('launch.sh').returncode, 42)
        self.assertFalse((self.root / 'launched').exists())

    def test_cargo_locked_single_binary_local_target(self):
        self.command('cargo', 'printf "%s\\n" "$*" "$CARGO_TARGET_DIR" > cargo-args')
        self.assertEqual(self.run_script('launch.sh').returncode, 0)
        self.assertEqual((self.root / 'cargo-args').read_text().splitlines(),
                         ['build --release --locked --bin poly-bot', str(self.root / 'target')])

    def test_missing_model_fails(self):
        (self.root / 'src/tools/model/artifacts/btc5m/model.json').unlink()
        self.assertNotEqual(self.run_script('launch.sh', '--check').returncode, 0)

    def test_frontend_install_build(self):
        (self.root / 'src/api/static/dist/index.html').unlink()
        self.command('node', 'exit 0')
        self.command('pnpm', 'printf "%s\\n" "$*" >> pnpm-calls\nmkdir -p src/api/static/dist\ntouch src/api/static/dist/index.html')
        self.assertEqual(self.run_script('launch.sh').returncode, 0)
        self.assertEqual((self.root / 'pnpm-calls').read_text().splitlines(),
                         ['--dir web install --frozen-lockfile', '--dir web build'])

    def test_unknown_args(self):
        for name in ['launch.sh', 'clean.sh']:
            self.assertNotEqual(self.run_script(name, '--nonsense').returncode, 0)

    def test_clean_declined_preserves_build(self):
        self.assertEqual(self.run_script('clean.sh', input='n\n').returncode, 0)
        self.assertTrue((self.root / 'target').exists())

    def test_clean_all_preserves_model_config_and_external(self):
        external = Path(self.tmp.name) / 'outside'
        external.mkdir()
        (external / 'saved.sqlite').write_text('keep')
        self.put('data/nested/db.sqlite', 'remove')
        (self.root / 'data/link').symlink_to(external, target_is_directory=True)
        self.put('local.sqlite-wal', 'remove')
        self.put('calibration_test/file', 'remove')
        self.assertEqual(self.run_script('clean.sh', '--all', '--yes').returncode, 0)
        for name in ['target', 'data/nested/db.sqlite', 'local.sqlite-wal', 'calibration_test']:
            self.assertFalse((self.root / name).exists(), name)
        self.assertTrue((external / 'saved.sqlite').exists())
        self.assertTrue((self.root / 'config/general.toml').exists())
        self.assertTrue((self.root / 'src/tools/model/artifacts/btc5m/model.json').exists())

    def test_clean_rejects_external_target(self):
        shutil.rmtree(self.root / 'target')
        (self.root / 'target').symlink_to(self.bin, target_is_directory=True)
        self.assertNotEqual(self.run_script('clean.sh', '--build', '--yes').returncode, 0)
        self.assertTrue((self.bin / 'cargo').exists())

    def test_clean_rejects_running_bot(self):
        sleeper = self.root / 'poly-bot'
        shutil.copy2('/usr/bin/sleep', sleeper)
        p = subprocess.Popen([str(sleeper), '30'], cwd=self.root)
        try:
            time.sleep(0.05)
            result = self.run_script('clean.sh', '--all', '--yes')
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertTrue((self.root / 'target').exists())
        finally:
            p.terminate()
            p.wait()

    def test_clean_data_only_and_sqlite_sidecars(self):
        names = [f'{directory}state.sqlite{suffix}'
                 for directory in ['', 'data/nested/', 'db/', 'runtime/']
                 for suffix in ['', '-wal', '-shm', '-journal']]
        for name in names:
            self.put(name, 'remove')
        self.put('calibration_keep/file', 'keep')
        result = self.run_script('clean.sh', '--data', '--yes')
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in names:
            self.assertFalse((self.root / name).exists(), name)
        self.assertTrue((self.root / 'target').exists())
        self.assertTrue((self.root / 'calibration_keep/file').exists())

    def test_clean_build_and_db_alias_scope(self):
        self.put('data/state.sqlite', 'remove only with db')
        self.put('calibration_keep/file', 'keep')
        result = self.run_script('clean.sh', '--build', '--yes')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.root / 'target').exists())
        self.assertTrue((self.root / 'data/state.sqlite').exists())
        self.put('target/cache', 'build')
        result = self.run_script('clean.sh', '--db', '--yes')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.root / 'target').exists())
        self.assertFalse((self.root / 'data/state.sqlite').exists())
        self.assertTrue((self.root / 'calibration_keep/file').exists())

    def test_clean_cal_only_and_external_links(self):
        external = Path(self.tmp.name) / 'outside-cal'
        external.mkdir()
        (external / 'keep').write_text('keep')
        self.put('calibration_local/file', 'remove')
        self.put('data/state.sqlite', 'keep')
        (self.root / 'calibration_link').symlink_to(external, target_is_directory=True)
        result = self.run_script('clean.sh', '--cal', '--yes')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.root / 'calibration_local').exists())
        self.assertTrue((external / 'keep').exists())
        self.assertTrue((self.root / 'calibration_link').is_symlink())
        self.assertTrue((self.root / 'data/state.sqlite').exists())
        self.assertTrue((self.root / 'target').exists())

    def test_clean_all_preserves_rl_and_research(self):
        preserved = ['src/tools/model/artifacts/rl_predictor/paper_bootstrap_v1/policy.json',
                     'config/strategies/rl_predictor.toml',
                     'data/rl_predictor_bootstrap_v1/active.json',
                     'data/rl_predictor_bootstrap_v1/checkpoints/v000001/policy.json',
                     'research/capture.sqlite', 'paper_rl/data/state_paper.sqlite',
                     'paper_rl/data/evidence.json', '.env', '.env.calib_backup']
        for name in preserved:
            self.put(name, 'synthetic preservation sentinel')
        result = self.run_script('clean.sh', '--all', '--yes')
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in preserved:
            self.assertEqual((self.root / name).read_text(),
                             'synthetic preservation sentinel', name)

    def test_clean_eof_declines_all_and_pack_has_no_effect(self):
        self.put('data/state.sqlite', 'keep')
        self.put('calibration_local/file', 'keep')
        self.assertEqual(self.run_script('clean.sh', '--all').returncode, 0)
        self.assertNotEqual(self.run_script('clean.sh', '--all', '--yes', '--pack').returncode, 0)
        for name in ['target', 'data/state.sqlite', 'calibration_local/file']:
            self.assertTrue((self.root / name).exists(), name)

    def test_clean_data_symlink_roots_and_files_untouched(self):
        external = Path(self.tmp.name) / 'external-data'
        external.mkdir()
        (external / 'state.sqlite').write_text('keep')
        for name in ['data', 'db', 'runtime']:
            (self.root / name).symlink_to(external, target_is_directory=True)
        (self.root / 'linked.sqlite').symlink_to(external / 'state.sqlite')
        result = self.run_script('clean.sh', '--data', '--yes')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((external / 'state.sqlite').read_text(), 'keep')
        self.assertTrue((self.root / 'linked.sqlite').is_symlink())

    def test_clean_all_is_idempotent(self):
        self.put('data/state.sqlite', 'remove')
        self.put('calibration_local/file', 'remove')
        for _ in range(2):
            result = self.run_script('clean.sh', '--all', '--yes')
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_clean_python_cache_default_all_and_explicit(self):
        for args in [('--cache',), ('--all',), ()]:
            with self.subTest(args=args):
                remove = ['research/pkg/__pycache__/x.pyc', 'paper_rl/__pycache__/smoke.pyc',
                          'tests/.pytest_cache/v/cache/nodeids', '.mypy_cache/cache',
                          '.ruff_cache/cache', 'tools/orphan.pyc']
                for name in remove:
                    self.put(name, 'cache')
                self.put('research/source.py', 'keep')
                self.put('.venv/lib/__pycache__/dependency.pyc', 'keep')
                result = self.run_script('clean.sh', *args, '--yes')
                self.assertEqual(result.returncode, 0, result.stderr)
                for name in remove:
                    self.assertFalse((self.root / name).exists(), name)
                self.assertTrue((self.root / 'research/source.py').exists())
                self.assertTrue((self.root / '.venv/lib/__pycache__/dependency.pyc').exists())

    def test_clean_logs_default_all_and_explicit_preserves_evidence(self):
        for args in [('--logs',), ('--all',), ()]:
            with self.subTest(args=args):
                for name in ['poly-bot.log', 'runtime/bot.log', 'bot.log.1', 'poly-bot.log.2.gz']:
                    self.put(name, 'operational log')
                self.put('paper_rl/data/smoke.log', 'evidence')
                self.put('research/report.log', 'evidence')
                result = self.run_script('clean.sh', *args, '--yes')
                self.assertEqual(result.returncode, 0, result.stderr)
                for name in ['poly-bot.log', 'runtime/bot.log', 'bot.log.1', 'poly-bot.log.2.gz']:
                    self.assertFalse((self.root / name).exists(), name)
                self.assertEqual((self.root / 'paper_rl/data/smoke.log').read_text(), 'evidence')
                self.assertEqual((self.root / 'research/report.log').read_text(), 'evidence')

    def test_clean_cache_logs_decline_links_and_active_guard(self):
        self.put('tools/__pycache__/x.pyc', 'keep')
        self.put('poly-bot.log', 'keep')
        external = Path(self.tmp.name) / 'external-cache'
        external.mkdir()
        (external / 'x.pyc').write_text('keep')
        (external / 'bot.log').write_text('keep')
        (self.root / '__pycache__').symlink_to(external, target_is_directory=True)
        (self.root / 'runtime').symlink_to(external, target_is_directory=True)
        result = self.run_script('clean.sh', '--cache', '--logs', input='n\nn\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / 'tools/__pycache__/x.pyc').exists())
        self.assertTrue((self.root / 'poly-bot.log').exists())
        sleeper = self.root / 'poly-bot'
        shutil.copy2('/usr/bin/sleep', sleeper)
        p = subprocess.Popen([str(sleeper), '30'], cwd=self.root)
        try:
            time.sleep(0.05)
            result = self.run_script('clean.sh', '--cache', '--logs', '--yes')
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue((self.root / 'poly-bot.log').exists())
        finally:
            p.terminate()
            p.wait()
        self.assertEqual(self.run_script('clean.sh', '--cache', '--logs', '--yes').returncode, 0)
        self.assertEqual((external / 'x.pyc').read_text(), 'keep')
        self.assertEqual((external / 'bot.log').read_text(), 'keep')
        self.assertTrue((self.root / 'target').exists())


if __name__ == '__main__':
    unittest.main()
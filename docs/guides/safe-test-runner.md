# Hardened test entry point

This additive launcher leaves `tests/run_tests.py` and historical test sources unchanged.
It uses an explicit suite allowlist, not blanket discovery.

From the repository root on native Windows:

```powershell
.venv-gpu/Scripts/python.exe tests/safe_runner.py --suite archived-copy
.venv-gpu/Scripts/python.exe tests/safe_runner.py --suite runner-safety
```

`archived-copy` loads the same three archived copy modules and original runner-safety
checks as the old launcher, using its existing fixture redirection. It requires the
corresponding frozen result directories and protocols to be present. No training is
performed. This is not the whole repository's test suite.

`runner-safety` runs only the new standard-library synthetic regression tests. It does
not import archived test modules, require checkpoints or train models. It protects its
two source files; its temporary synthetic files are disposable. Test it independently:

```powershell
python -m unittest discover -s tests -p test_safe_runner.py -v
```

Each launcher invocation creates a fresh retained folder under `results/test_runs/`.
`--run-name NAME` accepts one fresh alphanumeric/underscore/hyphen component (maximum
80 characters); existing names and path syntax are rejected. No historical main blocks
are invoked. Do not replace these commands with broad discovery of archived tests.

The archived-copy integrity inventory includes the old launcher, original safety tests,
three archived test sources, three matching protocols, the new runner sources, and all
files/directories recursively below the three declared frozen result directories.
Missing/unreadable prerequisites fail before test loading; protected symlinks are rejected.
The JSON summary includes before/after inventories, changed/missing/unexpected paths,
snapshot errors, test counts, and setup/import exception details. The post-run snapshot
and exclusive-create summary are attempted even when loading or execution raises.

Integrity hashing detects persistent changes after execution; it does not prevent
writes, roll back damage, detect a change subsequently restored, or sandbox test code.
Protection covers only declared paths, not every repository artifact. An inventory
of available files is not proof that an externally restored archive is complete. A killed process,
power loss, or unwritable/deleted output directory can prevent summary persistence;
such write errors return nonzero where the process remains able to respond.

Historical experiment sources, checkpoints, protocols, reported results and the old
launcher remain unchanged by this maintenance addition. Synthetic safety tests do not
constitute a rerun or independent reproduction of the archived copy experiments.

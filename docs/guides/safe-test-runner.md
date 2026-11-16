# Hardened test entry point

For automated Windows/Linux checks, artifacts and skip policy, see [synthetic safety CI](safety-ci.md).

This additive launcher leaves `tests/run_tests.py` and historical test sources unchanged.
It uses an explicit suite allowlist, not blanket discovery.

Run the archived suite in a disposable checkout with required artifacts restored.
Integrity checks are not a filesystem sandbox and cannot prevent arbitrary test writes.

From the repository root on native Windows:

```powershell
.venv-gpu/Scripts/python.exe tests/safe_runner.py --suite archived-copy
.venv-gpu/Scripts/python.exe tests/safe_runner.py --suite runner-safety
```

`archived-copy` loads the same three archived copy modules and original runner-safety
checks as the old launcher, using its existing fixture redirection. It requires the
corresponding frozen result directories and protocols to be present. No training is
performed. This is not the whole repository's test suite.

`runner-safety` runs only standard-library synthetic regression tests. It does not import
archived modules, require checkpoints or train models. Its temporary files are disposable;
the launcher inventories its two source files. Test it independently:

```powershell
python -m unittest discover -s tests -p test_safe_runner.py -v
```

Each launcher invocation creates a fresh retained folder under `results/test_runs/`.
`--run-name NAME` accepts one fresh alphanumeric/underscore/hyphen component (maximum
80 characters); existing names and path syntax are rejected. No historical main blocks
are invoked. Do not replace these commands with broad discovery of archived tests.

Only the trusted checkout root is resolved. Before creating output directories, the
launcher verifies lexical containment and checks existing output components with
`lstat`. Symlinks and Windows reparse points (including junctions) below that root are
rejected. Missing parents are created individually and checked again. The internal
`root` argument is an explicit trust boundary for synthetic fixtures, not a CLI option.
These checks do not protect against concurrent filesystem replacement races.

The archived-copy inventory includes the old launcher, original safety tests, three
archived test sources, three matching protocols, the new runner sources, and all
files/directories recursively below the three declared frozen result directories.
Missing/unreadable prerequisites fail before test loading; protected links/reparse points
are rejected. The JSON summary includes inventories, changed/missing/unexpected paths,
snapshot errors, test counts, and setup/import exception details. The post-run snapshot
and exclusive-create summary are attempted even when loading or execution raises.
Output-root rejection occurs before test execution and is reported as a CLI error;
no summary is written through a rejected output root.

Hashing detects persistent changes after execution; it does not prevent writes, roll
back damage, detect a change subsequently restored, or sandbox test code. Protection
covers declared paths only. An inventory does not prove restored-archive completeness.
A killed process or unwritable/deleted output directory can prevent summary persistence.

## Validation status after output-root correction

The synthetic suite and runner-safety CLI were exercised on Python 3.12/Linux using
only staged new PR files, not a restored historical checkout. The real Windows-junction
regression is skipped on non-Windows systems; a separate simulated reparse-attribute
check does not substitute for native testing. Symlink-creation skips are explicit.
Native Windows execution, the archived-copy suite, and historical reproduction remain
unvalidated. Before merging, run both synthetic commands above on native Windows,
inspect skips and the JSON summary, and test archived-copy in a disposable restored
checkout. Historical experiment sources and reported results remain unchanged.

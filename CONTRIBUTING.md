# Contributing

Contributions should preserve the benchmark's claim boundaries and never add
raw EEG recordings, participant-level predictions, credentials, checkpoints,
or generated caches to Git.

1. Open an issue describing the scientific or engineering change and the
   expected effect on frozen protocols.
2. Work in a focused branch and keep generated data under ignored directories.
3. Add or update synthetic/no-download checks for behavior changes.
4. Run `python scripts/check_public_smoke.py` and
   `python scripts/check_public_release.py` before requesting review.
5. Report dataset, split, model seed, attacker seed, utility, and all relevant
   leakage metrics for experimental changes.

Do not broaden a privacy claim from one attacker or dataset to another without
new evidence. Changes to frozen v1 configs should create a new versioned config
rather than silently altering the v1 protocol.

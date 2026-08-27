# v1.1 Publication QA

This note records release-boundary checks for the public `v1.1.0` candidate.
It supplements the frozen scientific gate records; it does not reinterpret or
replace them.

## Candidate boundary

- Review date: 2026-08-26
- Public base: `v1.0.0` (`bf7db7b`)
- Scientific source checkpoint: `c737c1a`
- Curated v1.1 evidence files: 12
- Raw EEG, posterior arrays, model checkpoints, and generated run outputs: not
  included

The v1.0.0 evidence remains unchanged. The v1.1 candidate adds the Cho2017
validation contract, aggregate gate records, claim audit, reproduction code,
and public documentation.

## Automated checks

The following checks passed from the staged public tree using the locked
environment:

- Python compilation and the no-download/no-training public smoke suite
- Dataset-cache plumbing: `7/7`
- Cho2017 validation contract: `28/28`
- Cho2017 split and attacker-partition plumbing: `40/40`
- Public inventory, path, size, encoding, link, credential-signature, commit
  identity, manifest, and checksum validation: `178` files passed
- Source distribution metadata and wheel build: package
  `eeg_privacy_benchmark-1.1.0-py3-none-any.whl` built successfully
- Git whitespace validation: no errors

The public smoke suite reported `training=0` and `downloads=0`. The release
checker rejected host-specific paths, private cache markers, credential-like
signatures, generated runtime paths, disallowed binary formats, files larger
than 1 MiB, and unapproved commit identities.

## Scientific claim boundary

The fixed compact bottleneck EEGNet (`dim=6`) passed the preregistered Cho2017
utility-noninferiority gate. The subsequent 600-job cache-only membership
evaluation recorded `NO_PROMOTION`: no membership-privacy improvement claim is
supported. Public summaries use the wording approved in
`docs/V1_1_CHO2017_CLAIM_AUDIT.md`.

## Remaining publication action

The candidate is intentionally local. Publishing requires an explicit review
of the staged commit followed by a separate authorization to push, tag
`v1.1.0`, and create the GitHub release.

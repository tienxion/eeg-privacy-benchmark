# v1.2 Publication QA

This note records release-boundary checks for the local public `v1.2.0`
candidate. It supplements the frozen scientific gate records; it does not
reinterpret or replace them.

## Candidate boundary

- Review date: 2026-08-27
- Public base: `v1.1.0` (`b9d003e`)
- Scientific source checkpoint: `0ed3c3c`
- Curated v1.2 evidence files: 9
- Raw EEG, feature or posterior arrays, model checkpoints, internal
  authorization records, and generated run outputs: not included

The candidate adds the Shin2017A historical contract, aggregate gate records,
claim audit, public loader and split plumbing, and public documentation. The
historical contract records the decisions used for the completed study; it is
not a transferable authorization to download data or execute new training.

## Automated checks

The following checks passed from the staged public tree using the locked
environment:

- Python compilation and the no-download/no-training public smoke suite
- Dataset-cache plumbing: `9/9`
- Shin2017A public aggregate evidence: `10/10`
- Shin2017A temporal split and attacker-partition plumbing: `14/14`
- Shin2017A synthetic loader and cache inventory: `6/6`
- Public inventory, path, size, encoding, link, credential-signature, commit
  identity, manifest, and checksum validation: `197` files passed
- Wheel build and isolated import: package
  `eeg_privacy_benchmark-1.2.0-py3-none-any.whl` built successfully and reported
  version `1.2.0`
- Git whitespace validation: no errors

The public smoke suite and all Shin2017A checks perform no training and no
download. The wheel was built locally without dependency installation or
network access.

## Scientific claim boundary

Plain compact EEGNet passed the named temporal utility gate, and the named
linear probe established subject-identity detectability for its fixed threat
model. The dim-6 bottleneck did not satisfy the strict utility-noninferiority
rule, so downstream bottleneck privacy evaluation was not run. Public summaries
do not promote a Shin2017A privacy-defense claim.

## Publication boundary

This candidate is intentionally local. Public push, tag `v1.2.0`, and GitHub
release creation require separate owner confirmation after the final staged
commit and QA results are reviewed.

# Evidence Provenance

- Release line: `1.0.0-rc1`
- Development checkpoint: `5dee43de85a2b4011fc97581efcf65f35fa5a4aa`
- History boundary: the public repository was created from a curated file set;
  development Git history was not copied
- Validated environment: Python 3.9.6, MOABB 1.2.0, MNE 1.8.0,
  scikit-learn 1.5.2, PyTorch 2.8.0
- Pre-release package QA: 18/18 checks, zero warnings
- Pre-release evidence-archive QA: 4/4 checks

Evidence files were selected from the audited development report layer,
assigned stable public paths, stripped of host-specific links, and checked
against the published claim boundaries. `scripts/generate_release_manifest.py`
records the public hashes without reading raw data or generated run outputs.

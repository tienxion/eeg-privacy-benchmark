# Evidence Provenance

- Public candidate version: `1.0.0-rc1`
- Private source checkpoint: `5dee43de85a2b4011fc97581efcf65f35fa5a4aa`
- Source branch role: private review source only; no private Git history copied
- Validated environment: Python 3.9.6, MOABB 1.2.0, MNE 1.8.0,
  scikit-learn 1.5.2, PyTorch 2.8.0
- Private package QA before curation: 18/18 checks, zero warnings
- Private lab archive QA before curation: 4/4 checks

Evidence files were copied from the audited private submission/report layer,
renamed to stable public paths, stripped of host-specific links, and checked
against the public claim boundaries. `scripts/generate_release_manifest.py`
records the final public hashes without reading private data or outputs.

# Data directory

Use this directory for:
- cached metadata
- generated dataset manifests
- frozen train and test splits
- processed artifacts that are small enough to keep under version control

Do not place large raw recordings here. Use `raw_data/` for that.

Version-control policy:
- Keep `splits/*.json` public so benchmark train/test assignments are reproducible.
- Do not commit generated `manifests/*.json`; rebuild them locally from the public source datasets.

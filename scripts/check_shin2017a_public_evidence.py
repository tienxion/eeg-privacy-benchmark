from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "shin2017a_temporal_validation_v1.yaml"
BASELINE_GATE = (
    ROOT / "configs" / "shin2017a_confirmatory_baseline_utility_gate_v1.json"
)
IDENTITY_GATE = (
    ROOT
    / "configs"
    / "shin2017a_confirmatory_subject_id_detectability_gate_v1.json"
)
BOTTLENECK_GATE = (
    ROOT / "configs" / "shin2017a_confirmatory_bottleneck_utility_gate_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_GATE.read_text(encoding="utf-8"))
    identity = json.loads(IDENTITY_GATE.read_text(encoding="utf-8"))
    bottleneck = json.loads(BOTTLENECK_GATE.read_text(encoding="utf-8"))
    contract_sha256 = _sha256(CONTRACT)
    cache_hashes = {
        baseline["provenance"]["cache_manifest_sha256"],
        identity["provenance"]["cache_manifest_sha256"],
        bottleneck["provenance"]["cache_manifest_sha256"],
    }
    checks = [
        (
            "contract_scope",
            contract["cohort"]["confirmatory_subjects"] == list(range(3, 30))
            and len(contract["outer_evaluation"]["folds"]) == 3
            and contract["seeds"]["task"] == [13, 17, 19, 23],
        ),
        (
            "contract_hashes_match_all_gates",
            baseline["provenance"]["contract_sha256"] == contract_sha256
            and identity["provenance"]["temporal_contract_sha256"] == contract_sha256
            and bottleneck["provenance"]["temporal_contract_sha256"] == contract_sha256,
        ),
        ("cache_manifest_hash_is_consistent", len(cache_hashes) == 1),
        (
            "baseline_gate_passed",
            baseline["status"] == "UTILITY_PASS_IDENTITY_PENDING"
            and baseline["provenance"]["validated_result_jobs"] == 24
            and baseline["plain_eegnet_utility_gate"]["utility_gate_pass"] is True,
        ),
        (
            "identity_detectability_passed",
            identity["status"]
            == "PASS_BOTTLENECK_ELIGIBLE_PENDING_AUTHORIZATION"
            and identity["provenance"]["validated_probe_jobs"] == 12
            and identity["detectability_gate"]["mean_threshold_pass"] is True,
        ),
        (
            "bottleneck_execution_validated",
            bottleneck["provenance"]["validated_bottleneck_jobs"] == 12
            and bottleneck["provenance"]["validated_bottleneck_feature_caches"] == 12,
        ),
        (
            "bottleneck_noninferiority_not_established",
            bottleneck["status"] == "STOP_NONINFERIORITY_NOT_ESTABLISHED"
            and bottleneck["bottleneck_utility_noninferiority_gate"]["mean_loss_pass"]
            is True
            and bottleneck["bottleneck_utility_noninferiority_gate"]["upper_bound_pass"]
            is False
            and bottleneck["bottleneck_utility_noninferiority_gate"]["gate_pass"]
            is False,
        ),
        (
            "privacy_stop_is_explicit",
            bottleneck["privacy_evaluation"]["status"]
            == "NOT_RUN_DUE_TO_PREDECLARED_UTILITY_STOP_RULE"
            and bottleneck["privacy_evaluation"]["membership_attacks_authorized"]
            is False,
        ),
        (
            "public_claim_audit_present",
            (ROOT / "docs" / "V1_2_SHIN2017A_CLAIM_AUDIT.md").is_file(),
        ),
        (
            "internal_authorization_records_excluded",
            not (ROOT / "configs" / "shin2017a_bottleneck_authorization_v1.yaml").exists()
            and not (
                ROOT / "configs" / "shin2017a_subject_id_probe_authorization_v1.yaml"
            ).exists(),
        ),
    ]
    for name, passed in checks:
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    passed_count = sum(bool(passed) for _, passed in checks)
    print(f"Shin2017A public evidence checks={passed_count}/{len(checks)}")
    if passed_count != len(checks):
        raise AssertionError("Shin2017A public evidence validation failed")


if __name__ == "__main__":
    main()

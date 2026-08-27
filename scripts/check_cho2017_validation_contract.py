from __future__ import annotations

import random
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "cho2017_confirmatory_validation_v1.yaml"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    checks: list[tuple[str, bool]] = []

    cohort = payload["cohort"]
    confirmatory = cohort["confirmatory_subjects"]
    blocks = cohort["shuffled_subject_blocks"]
    flattened = [subject for block in blocks for subject in block]
    expected = list(range(3, 53))

    reproduced = expected.copy()
    random.Random(cohort["partition_seed"]).shuffle(reproduced)
    checks.extend(
        [
            ("pilot_subjects_excluded", cohort["pilot_subjects_excluded_from_confirmation"] == [1, 2]),
            ("confirmatory_subjects_exact", confirmatory == expected),
            ("five_equal_blocks", len(blocks) == 5 and all(len(block) == 10 for block in blocks)),
            ("blocks_cover_cohort_once", sorted(flattened) == expected and len(set(flattened)) == 50),
            ("partition_seed_reproduces_blocks", flattened == reproduced),
        ]
    )

    fold_roles_valid = True
    role_counts = {subject: {"fit": 0, "validation": 0, "test": 0} for subject in expected}
    for fold_index in range(5):
        test = set(blocks[fold_index])
        validation = set(blocks[(fold_index + 1) % 5])
        fitting = set(expected) - test - validation
        fold_roles_valid &= (
            len(fitting) == 30
            and len(validation) == 10
            and len(test) == 10
            and not (fitting & validation or fitting & test or validation & test)
        )
        for subject in fitting:
            role_counts[subject]["fit"] += 1
        for subject in validation:
            role_counts[subject]["validation"] += 1
        for subject in test:
            role_counts[subject]["test"] += 1
    checks.extend(
        [
            ("fold_roles_disjoint", fold_roles_valid),
            (
                "each_subject_has_expected_roles",
                all(counts == {"fit": 3, "validation": 1, "test": 1} for counts in role_counts.values()),
            ),
        ]
    )

    membership = payload["membership_design"]
    checks.extend(
        [
            ("membership_reserve_fraction", membership["nonmember_fraction_per_fitting_subject"] == 0.20),
            ("membership_same_subject_pool", membership["member_and_nonmember_subject_pool_must_match"] is True),
            ("membership_nonmembers_strict", membership["reserved_nonmembers_never_used_for_fitting_or_early_stopping"] is True),
            ("attacker_subjects_disjoint", membership["attacker_training_and_evaluation_subjects_disjoint"] is True),
            ("outer_subjects_not_membership_controls", membership["outer_validation_and_test_subjects_excluded_from_membership_attack"] is True),
            ("attacker_subject_split", membership["attacker_subject_split_per_fold"] == [15, 15]),
            ("three_attacker_families", set(membership["attacker_families"]) == {"threshold", "logistic_regression", "mlp"}),
        ]
    )

    seeds = payload["seeds"]
    models = payload["models"]
    checks.extend(
        [
            ("task_seeds_frozen", seeds["task"] == [13, 17, 19, 23]),
            ("attacker_seeds_frozen", seeds["membership_attacker"] == [101, 103, 107]),
            ("task_and_attacker_seeds_disjoint", not set(seeds["task"]) & set(seeds["membership_attacker"])),
            (
                "models_frozen",
                models["candidates"]
                == ["csp_lda", "compact_eegnet", "compact_bottleneck_eegnet_dim6"],
            ),
        ]
    )

    promotion = payload["promotion_rules"]
    bad_trial_policy = payload["data_qa"]["bad_trial_policy"]
    checks.extend(
        [
            ("baseline_utility_gate", promotion["plain_eegnet_utility"]["minimum_mean_subject_balanced_accuracy"] == 0.60),
            ("utility_noninferiority_margin", promotion["bottleneck_utility_noninferiority"]["maximum_mean_balanced_accuracy_loss"] == 0.030),
            ("privacy_auc_threshold", promotion["membership_privacy"]["minimum_mean_auc_reduction"] == 0.010),
            ("privacy_requires_two_families", promotion["membership_privacy"]["minimum_improved_families"] == 2),
            (
                "primary_uses_moabb_all_trials",
                bad_trial_policy["primary_analysis"] == "moabb_all_trials",
            ),
            (
                "primary_does_not_infer_bad_trial_mapping",
                bad_trial_policy["exclude_bad_trial_indices_from_primary"] is False,
            ),
            (
                "artifact_sensitivity_requires_provenance",
                bad_trial_policy[
                    "artifact_filtered_sensitivity_requires_authoritative_mapping"
                ]
                is True,
            ),
            ("qa_stop_limit", payload["data_qa"]["maximum_subject_exclusions_before_stop"] == 2),
            ("subject_is_inferential_unit", cohort["inferential_unit"] == "subject"),
            ("precompute_status", payload["project"]["status"] == "preregistered_precompute"),
        ]
    )

    for name, passed in checks:
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    passed_count = sum(passed for _, passed in checks)
    print(f"passed={passed_count}/{len(checks)}")
    _require(passed_count == len(checks), "Cho2017 validation contract check failed")


if __name__ == "__main__":
    main()

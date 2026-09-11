#!/usr/bin/env python3
"""Synthetic-only checks for the portable output-stream identity runner."""
from __future__ import annotations

from collections import Counter
import copy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import output_stream_identity as run


RUNNER = Path(__file__).with_name("output_stream_identity.py")
IDENTITIES = [2, 11, 37]


def synthetic_source():
    scores = {2: [.625, .375], 11: [.875, .125], 37: [.125, .875]}
    return {"schema_version": 1, "model_id": "synthetic-model", "score_kind": "probabilities",
            "trials": [{"trial_id": f"source-{person}-{trial}", "identity": person, "scores": scores[person].copy()}
                       for person in (11, 37, 2) for trial in range(20)]}


def synthetic_evaluation():
    scores = {2: [.625, .375], 11: [.875, .125], 37: [.125, .875]}
    return {"schema_version": 1, "model_id": "synthetic-model", "score_kind": "probabilities",
            "bags": [{"identity": person,
                      "trials": [{"trial_id": f"evaluation-{person}-{bag}-{trial}", "scores": scores[person].copy()} for trial in range(2)]}
                     for person in (37, 2, 11) for bag in range(2)]}


def enrollment():
    source = synthetic_source()
    digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    return run.fit(source, 2, digest)


def independent_predict(counts, observations, surface):
    """Fraction likelihoods, grouped by multiplicity instead of integer products."""
    answers = []
    for bag in observations:
        scores = []
        for row in counts:
            score = Fraction(1)
            for symbol, multiplicity in Counter(bag).items():
                numerator = sum(row[8 * symbol:8 * symbol + 8]) + 8 if surface == "hard_label" else row[symbol] + 1
                score *= Fraction(numerator, sum(row) + 16) ** multiplicity
            scores.append(score)
        answers.append(scores.index(max(scores)))
    return answers


def independent_metric(y, predictions, identities):
    cm = {actual: Counter() for actual in identities}
    for actual, predicted in zip(y, predictions):
        cm[actual][predicted] += 1
    recalls, f1s = [], []
    for person in identities:
        actual_n, predicted_n = sum(cm[person].values()), sum(row[person] for row in cm.values())
        recalls.append(Fraction(cm[person][person], actual_n))
        f1s.append(Fraction(2 * cm[person][person], actual_n + predicted_n))
    return {"balanced_accuracy": float(sum(recalls) / len(identities)), "macro_f1": float(sum(f1s) / len(identities)),
            "correct": sum(a == b for a, b in zip(y, predictions)), "total": len(y)}


def cli(*args):
    return subprocess.run([sys.executable, str(RUNNER), *map(str, args)], capture_output=True, text=True, timeout=30)


class PortableRunnerTests(unittest.TestCase):
    def test_disclosure_extremes_ties_and_float32_probability_normalization(self):
        maximum = sys.float_info.max
        self.assertEqual(run.disclose([0, 0], "logits"), (0, 0))
        self.assertEqual(run.disclose([maximum, -maximum], "logits"), (0, 7))
        self.assertEqual(run.disclose([-maximum, maximum], "logits"), (1, 15))
        self.assertEqual(run.disclose([.5, .5], "probabilities"), (0, 0))
        self.assertEqual(run.disclose([0, 1], "probabilities"), (1, 15))
        self.assertEqual(run.disclose([1, 0], "probabilities"), (0, 7))
        q = [struct.unpack("f", struct.pack("f", x))[0] for x in (.3, .7)]
        expected_band = math.floor(16 * (max(q) / sum(q) - .5))
        self.assertEqual(run.disclose(q, "probabilities"), (1, 8 + expected_band))

    def test_all_bin_boundaries_and_exact_coarsening(self):
        for band in range(1, 8):
            boundary = .5 + band / 16
            margin = math.log((1 + band / 8) / (1 - band / 8))
            for label in (0, 1):
                for epsilon in (-1e-9, 0, 1e-9):
                    confidence = boundary + epsilon
                    q = [confidence, 1 - confidence] if label == 0 else [1 - confidence, confidence]
                    hard, fine = run.disclose(q, "probabilities")
                    self.assertEqual((hard, fine // 8), (label, label))
                    self.assertEqual(fine % 8, band - 1 if epsilon < 0 else band)
                for epsilon in (-1e-8, 1e-8):
                    z = [margin + epsilon, 0] if label == 0 else [0, margin + epsilon]
                    hard, fine = run.disclose(z, "logits")
                    self.assertEqual((hard, fine // 8), (label, label))
                    self.assertEqual(fine % 8, band - 1 if epsilon < 0 else band)

    def test_nonfinite_invalid_probabilities_booleans_and_json_rejected(self):
        for scores, kind in (([], "logits"), ([0, 1, 2], "logits"), ([True, 0], "logits"),
                             ([0, float("nan")], "logits"), ([float("inf"), 1], "logits"),
                             ([0, 0], "probabilities"), ([-.1, 1.1], "probabilities"),
                             ([.4, .4], "probabilities"), ([.5, .5], "unknown")):
            with self.subTest(scores=scores, kind=kind), self.assertRaises(ValueError):
                run.disclose(scores, kind)
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}', '{"x":1e999}'):
            with self.assertRaises(ValueError):
                run.decode(text)

    def test_balanced_enrollment_numeric_order_and_coherent_prior(self):
        table = enrollment()
        run.validate_enrollment(table)
        self.assertEqual(table["identities"], IDENTITIES)
        self.assertEqual(table["enrollment_per_identity"], 20)
        self.assertEqual(table["method"], {"fine_categories": 16, "fine_pseudocount": 1, "hard_pseudocount": 8,
                                           "denominator": 36, "tie_rule": "lowest_numeric_identity"})
        self.assertEqual(set(table["source_trial_ids"]), {r["trial_id"] for r in synthetic_source()["trials"]})
        for row, symbol in zip(table["counts16"], (2, 6, 14)):
            self.assertEqual(row, [20 if i == symbol else 0 for i in range(16)])
            for label in (0, 1):
                self.assertEqual(sum(v + 1 for v in row[8 * label:8 * label + 8]), sum(row[8 * label:8 * label + 8]) + 8)
        for budget in (1, True, 129):
            with self.assertRaises(ValueError):
                run.fit(synthetic_source(), budget, "a" * 64)

    def test_source_identity_trial_balance_and_schema_fail_closed(self):
        original = synthetic_source()
        changes = [lambda x: x["trials"].pop(),
                   lambda x: x["trials"][1].update(trial_id=x["trials"][0]["trial_id"]),
                   lambda x: x["trials"][0].update(identity=True),
                   lambda x: x["trials"][0].update(identity=-1),
                   lambda x: x.update(validation=[]),
                   lambda x: x["trials"][0].update(task_label=0)]
        for change in changes:
            source = copy.deepcopy(original)
            change(source)
            with self.assertRaises(ValueError):
                run.fit(source, 2, "a" * 64)
        for change in (lambda x: x["counts16"][0].__setitem__(2, True),
                       lambda x: x["identities"].reverse(),
                       lambda x: x["method"].update(hard_pseudocount=1),
                       lambda x: x.update(validation=[])):
            table = enrollment()
            change(table)
            with self.assertRaises(ValueError):
                run.validate_enrollment(table)

    def test_exact_products_long_streams_ties_and_invalid_inputs(self):
        counts = enrollment()["counts16"]
        for surface, bags in (("confidence16", [[2] * 2, [6] * 2, [14] * 2]),
                              ("hard_label", [[0] * 2, [1] * 2, [1, 0]])):
            self.assertEqual(run.predict(counts, bags, surface), independent_predict(counts, bags, surface))
        self.assertEqual(run.predict(counts, [[0]], "hard_label"), [0])
        large = [[100000] + [0] * 15, [0, 100000] + [0] * 14]
        self.assertGreater(100001 ** 128, 2 ** 64)
        self.assertEqual(run.predict(large, [[0] * 128, [1] * 128], "confidence16"), [0, 1])
        self.assertEqual(run.predict(large, [[0] * 64 + [1] * 64], "confidence16"), [0])
        for bags, surface in (([], "hard_label"), ([[0], [0, 1]], "hard_label"), ([[True]], "hard_label"),
                              ([[2]], "hard_label"), ([[16]], "confidence16"), ([[0] * 129], "hard_label")):
            with self.assertRaises(ValueError):
                run.predict(counts, bags, surface)

    def test_metrics_use_actual_identity_labels_and_fraction_macro_averages(self):
        y = [2, 11, 11, 37, 37, 37]
        predicted = [2, 2, 11, 11, 37, 37]
        self.assertEqual(run.metric(y, predicted, IDENTITIES), independent_metric(y, predicted, IDENTITIES))
        self.assertEqual(run.metric(y, [2] * len(y), IDENTITIES)["balanced_accuracy"], 1 / 3)
        for actual, guesses in (([2, 11], [2, 11]), (y, [2] * 5), (y, [True] * 6), (y, [99] * 6)):
            with self.assertRaises(ValueError):
                run.metric(actual, guesses, IDENTITIES)

    def test_evaluation_overlap_duplicates_mismatches_and_width_before_predict(self):
        table, original = enrollment(), synthetic_evaluation()
        changes = [lambda x: x.update(model_id="another-model"),
                   lambda x: x.update(score_kind="logits"),
                   lambda x: x.update(validation=[]),
                   lambda x: x["bags"][0]["trials"].pop(),
                   lambda x: x["bags"][0]["trials"][0].update(trial_id=table["source_trial_ids"][0]),
                   lambda x: x["bags"][1]["trials"][0].update(trial_id=x["bags"][0]["trials"][0]["trial_id"]),
                   lambda x: x["bags"].pop(),
                   lambda x: x["bags"][0].update(identity=99)]
        for change in changes:
            evaluation = copy.deepcopy(original)
            change(evaluation)
            with patch.object(run, "predict") as predict:
                with self.assertRaises(ValueError):
                    run.evaluate(table, evaluation)
                predict.assert_not_called()

    def test_evaluation_independent_recount_and_aggregate_only_output(self):
        table, evaluation = enrollment(), synthetic_evaluation()
        with patch.object(run, "predict", wraps=run.predict) as kernel:
            result = run.evaluate(table, evaluation)
        self.assertEqual(kernel.call_count, 4)
        self.assertTrue(all(len(call.args) == 3 and not call.kwargs for call in kernel.call_args_list))
        truth = [bag["identity"] for bag in evaluation["bags"]]
        symbols = {2: 2, 11: 6, 37: 14}
        fine = [[symbols[person]] * 2 for person in truth]
        expected_rows = []
        for budget in (1, 2):
            y = [person for person in truth for _ in range(2)] if budget == 1 else truth
            for surface in ("hard_label", "confidence16"):
                bags = [[symbol // 8 if surface == "hard_label" else symbol for symbol in bag] for bag in fine]
                observations = [[symbol] for bag in bags for symbol in bag] if budget == 1 else bags
                guessed_rows = independent_predict(table["counts16"], observations, surface)
                predicted = [IDENTITIES[i] for i in guessed_rows]
                expected_rows.append({"budget": budget, "surface": surface, **independent_metric(y, predicted, IDENTITIES)})
        self.assertEqual(result["rows"], expected_rows)
        self.assertEqual(result["primary_exact_fraction"], "1/3")
        self.assertEqual(result["primary_confidence_minus_hard_BA"], 1 / 3)
        self.assertEqual((result["identities_count"], result["evaluation_trials"], result["evaluation_stream_bags"]), (3, 12, 6))
        encoded = json.dumps(result)
        self.assertNotIn("source-", encoded)
        self.assertNotIn("evaluation-", encoded)
        self.assertNotIn('"predictions"', encoded)
        self.assertNotIn('"trial_ids"', encoded)

    def test_recorded_hash_and_table_validation_precede_evaluation_read(self):
        table_path, evaluation_path = Path("synthetic-enrollment.json"), Path("synthetic-evaluation.json")
        with patch.object(run, "load_document", return_value=(enrollment(), "a" * 64)) as read, patch.object(run, "evaluate") as score:
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                run.evaluate_files(table_path, "b" * 64, evaluation_path, Path("synthetic-summary.json"))
            read.assert_called_once_with(table_path)
            score.assert_not_called()
        invalid = enrollment()
        invalid["method"]["hard_pseudocount"] = 1
        with patch.object(run, "load_document", return_value=(invalid, "a" * 64)) as read, patch.object(run, "evaluate") as score:
            with self.assertRaises(ValueError):
                run.evaluate_files(table_path, "a" * 64, evaluation_path, Path("synthetic-summary.json"))
            read.assert_called_once_with(table_path)
            score.assert_not_called()

    def test_exclusive_owner_only_writes_input_cap_and_network_guard(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.json"
            digest = run.write_new(path, {"synthetic": True})
            original = path.read_bytes()
            self.assertEqual(hashlib.sha256(original).hexdigest(), digest)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(run.read_json(path), {"synthetic": True})
            with self.assertRaises(FileExistsError):
                run.write_new(path, {"synthetic": False})
            self.assertEqual(path.read_bytes(), original)
            oversized = Path(temporary) / "oversized.json"
            oversized.write_bytes(b" " * 33)
            with patch.object(run, "MAX_BYTES", 32), self.assertRaisesRegex(ValueError, "limit"):
                run.read_json(oversized)
        with self.assertRaises(RuntimeError):
            run.no_network("socket.connect", ())

    def test_black_box_two_stage_cli_and_synthetic_demo(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source, heldout = directory / "source.json", directory / "evaluation.json"
            table_path, summary_path = directory / "table.json", directory / "summary.json"
            run.write_new(source, synthetic_source())
            run.write_new(heldout, synthetic_evaluation())
            fitted = cli("fit", "--source", source, "--stream-budget", 2, "--output", table_path)
            self.assertEqual(fitted.returncode, 0, fitted.stderr)
            digest = hashlib.sha256(table_path.read_bytes()).hexdigest()
            self.assertIn(digest, fitted.stdout)
            rejected = cli("evaluate", "--enrollment", table_path, "--expected-sha256", "0" * 64,
                           "--evaluation", directory / "missing-evaluation.json", "--output", summary_path)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("evaluation input was not opened", rejected.stderr)
            self.assertFalse(summary_path.exists())
            missing_hash = cli("evaluate", "--enrollment", table_path, "--evaluation", heldout, "--output", summary_path)
            self.assertNotEqual(missing_hash.returncode, 0)
            self.assertFalse(summary_path.exists())
            completed = cli("evaluate", "--enrollment", table_path, "--expected-sha256", digest,
                            "--evaluation", heldout, "--output", summary_path)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(run.read_json(summary_path)["primary_exact_fraction"], "1/3")
            demo_directory = directory / "demo"
            demo = cli("demo", "--output-dir", demo_directory)
            self.assertEqual(demo.returncode, 0, demo.stderr)
            self.assertIn("SYNTHETIC", demo.stdout)
            files = {p.name: p for p in demo_directory.iterdir()}
            self.assertEqual(set(files), {"source.json", "evaluation.json", "enrollment.json", "summary.json"})
            before = {name: path.read_bytes() for name, path in files.items()}
            self.assertTrue(all(path.stat().st_mode & 0o777 == 0o600 for path in files.values()))
            result = run.read_json(files["summary.json"])
            for row in result["rows"]:
                self.assertEqual(row["balanced_accuracy"], 1 if row["surface"] == "confidence16" else 2 / 3)
            repeated = cli("demo", "--output-dir", demo_directory)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual({name: path.read_bytes() for name, path in files.items()}, before)


def main():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PortableRunnerTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()

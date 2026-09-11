#!/usr/bin/env python3
"""Fixed passive output-stream identity attack on caller-supplied local JSON.

Standard library only; no EEG training, inference, download, or upload. Enrollment
tables and inputs contain identifying information: keep them private. This is a
development runner, not a privacy guarantee or full historical-data reproduction.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

MAX_BYTES = 64 * 1024 * 1024
MAX_TRIALS = 100000
SURFACES = ("hard_label", "confidence16")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def fields(value, names, label):
    require(isinstance(value, dict) and set(value) == set(names), f"{label}: incorrect fields")


def integer(value, low, high, label):
    require(type(value) is int and low <= value <= high, f"{label}: integer out of range")


def identifier(value, label):
    require(isinstance(value, str) and 0 < len(value) <= 200 and value.strip() == value
        and all(ord(c) >= 32 and ord(c) != 127 for c in value), f"{label}: invalid identifier")


def digest_text(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Expected a lowercase SHA-256 digest")


def decode(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON field")
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    def finite_float(value):
        parsed = float(value)
        require(math.isfinite(parsed), "Nonfinite JSON number")
        return parsed
    return json.loads(text, object_pairs_hook=unique, parse_constant=reject_constant, parse_float=finite_float)


def load_document(path):
    with Path(path).open("rb") as handle:
        raw = handle.read(MAX_BYTES + 1)
    require(len(raw) <= MAX_BYTES, "JSON input exceeds the 64 MiB limit")
    return decode(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def read_json(path):
    return load_document(path)[0]


def write_new(path, payload):
    """Exclusive creation with owner-only permissions; never replace a prior result."""
    path = Path(path)
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(raw).hexdigest()


def disclose(scores, score_kind):
    require(score_kind in ("logits", "probabilities"), "Unknown score kind")
    require(isinstance(scores, (list, tuple)) and len(scores) == 2
        and all(type(v) in (int, float) for v in scores), "Two real nonboolean scores required")
    try:
        a, b = map(float, scores)
    except (ValueError, OverflowError) as exc:
        raise ValueError("Scores must be finite floating-point values") from exc
    require(math.isfinite(a) and math.isfinite(b), "Scores must be finite")
    label = int(b > a)
    if score_kind == "logits":
        # Divide before subtraction to keep finite extreme logits from overflowing.
        band = min(7, math.floor(8 * math.tanh(abs(b / 2 - a / 2))))
    else:
        require(0 <= a <= 1 and 0 <= b <= 1 and a + b > 0 and abs(a + b - 1) <= 1e-5,
            "Probabilities must lie in [0,1] and sum to one within 1e-5")
        band = min(7, math.floor(16 * (max(a, b) / (a + b) - .5)))
    return label, 8 * label + band


def validate_counts(counts):
    require(isinstance(counts, list) and 2 <= len(counts) <= 256, "Expected 2..256 enrollment rows")
    for row in counts:
        require(isinstance(row, list) and len(row) == 16, "Sixteen fine counts required")
        for value in row:
            integer(value, 0, MAX_TRIALS, "fine count")
    n = sum(counts[0])
    require(1 <= n <= MAX_TRIALS and all(sum(row) == n for row in counts), "Enrollment must be nonempty and balanced")
    return n


def method(n):
    return {"fine_categories": 16, "fine_pseudocount": 1, "hard_pseudocount": 8,
        "denominator": n + 16, "tie_rule": "lowest_numeric_identity"}


def fit(source, stream_budget, source_sha256):
    fields(source, ("schema_version", "model_id", "score_kind", "trials"), "source")
    integer(source["schema_version"], 1, 1, "source schema")
    identifier(source["model_id"], "model_id")
    require(source["score_kind"] in ("logits", "probabilities"), "Unknown source score kind")
    integer(stream_budget, 2, 128, "stream budget")
    digest_text(source_sha256)
    trials = source["trials"]
    require(isinstance(trials, list) and 2 <= len(trials) <= MAX_TRIALS, "Source trial count out of range")
    for row in trials:
        fields(row, ("trial_id", "identity", "scores"), "source trial")
        identifier(row["trial_id"], "source trial_id")
        integer(row["identity"], 0, 2 ** 31 - 1, "source identity")
    source_ids = [r["trial_id"] for r in trials]
    require(len(set(source_ids)) == len(source_ids), "Duplicate source trial identifier")
    identities = sorted({r["identity"] for r in trials})
    require(2 <= len(identities) <= 256, "Expected 2..256 enrolled identities")
    index = {p: i for i, p in enumerate(identities)}
    counts = [[0] * 16 for _ in identities]
    for row in trials:
        _, fine = disclose(row["scores"], source["score_kind"])
        counts[index[row["identity"]]][fine] += 1
    n = validate_counts(counts)
    return {"schema_version": 1, "artifact_type": "output_stream_enrollment",
        "model_id": source["model_id"], "score_kind": source["score_kind"], "stream_budget": stream_budget,
        "identities": identities, "enrollment_per_identity": n, "source_sha256": source_sha256,
        "source_trial_ids": source_ids, "counts16": counts, "method": method(n)}


def validate_enrollment(table):
    fields(table, ("schema_version", "artifact_type", "model_id", "score_kind", "stream_budget",
        "identities", "enrollment_per_identity", "source_sha256", "source_trial_ids", "counts16", "method"), "enrollment")
    integer(table["schema_version"], 1, 1, "enrollment schema")
    require(table["artifact_type"] == "output_stream_enrollment", "Wrong enrollment artifact type")
    identifier(table["model_id"], "model_id")
    require(table["score_kind"] in ("logits", "probabilities"), "Unknown enrollment score kind")
    integer(table["stream_budget"], 2, 128, "stream budget")
    digest_text(table["source_sha256"])
    ids = table["identities"]
    require(isinstance(ids, list) and 2 <= len(ids) <= 256, "Expected 2..256 identities")
    for person in ids:
        integer(person, 0, 2 ** 31 - 1, "enrolled identity")
    require(ids == sorted(set(ids)), "Identities must be unique and numerically sorted")
    n = validate_counts(table["counts16"])
    require(len(table["counts16"]) == len(ids), "Identity/count alignment mismatch")
    integer(table["enrollment_per_identity"], n, n, "enrollment size")
    source_ids = table["source_trial_ids"]
    require(isinstance(source_ids, list) and len(source_ids) == n * len(ids) <= MAX_TRIALS, "Source ID inventory mismatch")
    for trial in source_ids:
        identifier(trial, "source trial_id")
    require(len(set(source_ids)) == len(source_ids), "Duplicate enrolled trial")
    fields(table["method"], method(n), "method")
    for key, expected in method(n).items():
        if type(expected) is int:
            integer(table["method"][key], expected, expected, key)
        else:
            require(table["method"][key] == expected, "Changed fixed method")


def predict(counts, observations, surface):
    validate_counts(counts)
    require(surface in SURFACES, "Unknown disclosure surface")
    require(isinstance(observations, list) and 0 < len(observations) <= MAX_TRIALS, "Nonempty observations required")
    width = len(observations[0]) if isinstance(observations[0], list) else 0
    require(1 <= width <= 128, "Observation width out of range")
    for bag in observations:
        require(isinstance(bag, list) and len(bag) == width, "Inconsistent observation widths")
        for value in bag:
            integer(value, 0, 1 if surface == "hard_label" else 15, "disclosed symbol")
    smoothed = [[sum(row[:8]) + 8, sum(row[8:]) + 8] if surface == "hard_label"
        else [v + 1 for v in row] for row in counts]
    # The equal row denominators cancel; Python integers preserve exact ties.
    return [max(range(len(counts)), key=lambda person: math.prod(smoothed[person][symbol] for symbol in bag)) for bag in observations]


def metric(y, predictions, identities):
    require(isinstance(identities, list) and 2 <= len(identities) <= 256, "Identity inventory required")
    for person in identities:
        integer(person, 0, 2 ** 31 - 1, "metric identity")
    require(identities == sorted(set(identities)), "Unique sorted identities required")
    require(isinstance(y, list) and isinstance(predictions, list) and len(y) == len(predictions) > 0,
        "Aligned nonempty truth and predictions required")
    for person in y + predictions:
        integer(person, 0, 2 ** 31 - 1, "prediction or truth")
    require(set(y) == set(identities) and set(predictions) <= set(identities), "Missing or unknown identity")
    cells, actual, predicted = Counter(zip(y, predictions)), Counter(y), Counter(predictions)
    recall = sum(Fraction(cells[p, p], actual[p]) for p in identities) / len(identities)
    f1 = sum(Fraction(2 * cells[p, p], actual[p] + predicted[p]) for p in identities) / len(identities)
    return {"balanced_accuracy": float(recall), "macro_f1": float(f1),
        "correct": sum(a == b for a, b in zip(y, predictions)), "total": len(y)}


def evaluate(table, evaluation):
    validate_enrollment(table)
    fields(evaluation, ("schema_version", "model_id", "score_kind", "bags"), "evaluation")
    integer(evaluation["schema_version"], 1, 1, "evaluation schema")
    require(evaluation["model_id"] == table["model_id"] and evaluation["score_kind"] == table["score_kind"],
        "Evaluation model or score kind differs from enrollment")
    bags, k, ids = evaluation["bags"], table["stream_budget"], table["identities"]
    require(isinstance(bags, list) and 0 < len(bags) * k <= MAX_TRIALS, "Evaluation trial count out of range")
    seen, source_ids, truth = set(), set(table["source_trial_ids"]), []
    for bag in bags:
        fields(bag, ("identity", "trials"), "evaluation bag")
        integer(bag["identity"], 0, 2 ** 31 - 1, "evaluation identity")
        require(bag["identity"] in ids, "Unknown evaluation identity")
        require(isinstance(bag["trials"], list) and len(bag["trials"]) == k, "Bag differs from frozen stream budget")
        truth.append(bag["identity"])
        for row in bag["trials"]:
            fields(row, ("trial_id", "scores"), "evaluation trial")
            identifier(row["trial_id"], "evaluation trial_id")
            require(row["trial_id"] not in source_ids, "Enrollment/evaluation trial overlap")
            require(row["trial_id"] not in seen, "Duplicate evaluation trial identifier")
            seen.add(row["trial_id"])
    sizes = Counter(truth)
    require(set(sizes) == set(ids) and len(set(sizes.values())) == 1, "Evaluation bags must be balanced across enrolled identities")
    # Evaluator metadata is not passed to the prediction kernel.
    fine = [[disclose(row["scores"], table["score_kind"])[1] for row in bag["trials"]] for bag in bags]
    hard = [[symbol // 8 for symbol in bag] for bag in fine]
    rows = []
    for budget in (1, k):
        y = [p for p in truth for _ in range(k)] if budget == 1 else truth
        for surface, symbols in (("hard_label", hard), ("confidence16", fine)):
            observed = [[symbol] for bag in symbols for symbol in bag] if budget == 1 else symbols
            predicted = [ids[i] for i in predict(table["counts16"], observed, surface)]
            rows.append({"budget": budget, "surface": surface, **metric(y, predicted, ids)})
    delta = Fraction(rows[3]["correct"] - rows[2]["correct"], len(bags))
    return {"schema_version": 1, "status": "FIXED_OUTPUT_STREAM_EVALUATION_COMPLETE",
        "model_id": table["model_id"], "score_kind": table["score_kind"], "identities_count": len(ids),
        "enrollment_per_identity": table["enrollment_per_identity"], "stream_budget": k,
        "evaluation_trials": len(seen), "evaluation_stream_bags": len(bags), "rows": rows,
        "primary_confidence_minus_hard_BA": float(delta), "primary_exact_fraction": str(delta),
        "uniform_identity_chance": 1 / len(ids), "single_binary_alphabet_BA_ceiling": min(1, 2 / len(ids)),
        "method": table["method"],
        "limitations": "Caller-supplied IDs/roles and one-model provenance must be trustworthy; renaming duplicated trials defeats ID overlap checks. Known same-person grouping and labeled enrollment are strong auxiliary information. Achieved accuracy is not optimal leakage or a privacy guarantee. Hard decisions remove confidence-dependent utility. No EEG task utility is measured. Single-symbol bounds do not apply to streams. Choose the design and record the table hash before inspecting evaluation outcomes; this software cannot certify outcome-blind research."}


def evaluate_files(enrollment_path, expected_sha256, evaluation_path, output_path):
    digest_text(expected_sha256)
    table, actual = load_document(enrollment_path)
    require(actual == expected_sha256, "Enrollment SHA-256 mismatch; evaluation input was not opened")
    validate_enrollment(table)
    heldout, heldout_sha = load_document(evaluation_path)
    result = evaluate(table, heldout)
    result.update(enrollment_sha256=actual, evaluation_sha256=heldout_sha,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_new(output_path, result)
    return result


def demo(output_dir):
    directory = Path(output_dir)
    source_path, eval_path, table_path, result_path = [directory / name for name in
        ("source.json", "evaluation.json", "enrollment.json", "summary.json")]
    require(not any(p.exists() or p.is_symlink() for p in (source_path, eval_path, table_path, result_path)), "Demo outputs already exist; use a new output directory")
    prototypes = {2: [.95, .05], 11: [.05, .95], 37: [.65, .35]}
    common = {"schema_version": 1, "model_id": "synthetic-demo-not-research", "score_kind": "probabilities"}
    source = {**common, "trials": [{"trial_id": f"source-{p}-{i}", "identity": p, "scores": q}
        for p, q in prototypes.items() for i in range(20)]}
    heldout = {**common, "bags": [{"identity": p, "trials": [
        {"trial_id": f"evaluation-{p}-{2 * bag + i}", "scores": q} for i in range(2)]}
        for p, q in prototypes.items() for bag in range(2)]}
    source_sha = write_new(source_path, source)
    write_new(eval_path, heldout)
    table_sha = write_new(table_path, fit(source, 2, source_sha))
    result = evaluate_files(table_path, table_sha, eval_path, result_path)
    print("SYNTHETIC DEMO ONLY: fine BA=1; hard BA=2/3. No research outcomes or EEG data.")
    print(f"enrollment_sha256={table_sha}")
    return result


def no_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
        raise RuntimeError("This local-only runner forbids network access")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    source = stages.add_parser("fit", help="Fit from enrollment JSON only; output is sensitive")
    source.add_argument("--source", type=Path, required=True)
    source.add_argument("--stream-budget", type=int, required=True)
    source.add_argument("--output", type=Path, required=True)
    evaluation = stages.add_parser("evaluate", help="Require a recorded enrollment hash before held-out input is opened")
    evaluation.add_argument("--enrollment", type=Path, required=True)
    evaluation.add_argument("--expected-sha256", required=True)
    evaluation.add_argument("--evaluation", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    demonstration = stages.add_parser("demo", help="Create a synthetic example; never use it as research evidence")
    demonstration.add_argument("--output-dir", type=Path, default=Path("outputs/output-stream-demo"))
    args = parser.parse_args()
    sys.addaudithook(no_network)
    try:
        if args.stage == "fit":
            data, source_sha = load_document(args.source)
            table_sha = write_new(args.output, fit(data, args.stream_budget, source_sha))
            print(f"enrollment_sha256={table_sha}")
            print("Record this hash before evaluation. Enrollment inputs and tables must stay private.")
        elif args.stage == "evaluate":
            result = evaluate_files(args.enrollment, args.expected_sha256, args.evaluation, args.output)
            print(f"primary_confidence_minus_hard_BA={result['primary_confidence_minus_hard_BA']:.9f}")
        else:
            demo(args.output_dir)
    except (ValueError, OSError, UnicodeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

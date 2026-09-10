#!/usr/bin/env python3
"""Check aggregate arithmetic and its static figure, not raw attack reproduction.

Only the Python standard library is required. By default this command is
read-only. --render writes the deterministic SVG; --test runs synthetic tests.
Balanced identities/classes are declared properties of the published design,
not properties this aggregate-only checker can reconstruct from raw trials.
"""
from __future__ import annotations

import argparse
import copy
from fractions import Fraction
from html import escape
import json
import math
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/development/output-stream-identity.json"
FIGURE = ROOT / "results/development/output-stream-identity.svg"
SURFACES = ("hard_label", "confidence16")
CAPTION = "Different protocols; descriptive means, not independent-cohort inference"
DESIGNS = (
    {"id": "bnci-v19", "dataset": "BNCI2014-001", "identities": 9,
     "configurations": 4, "rotations": 1, "seeds": [101, 103, 107, 109],
     "enrollment_per_identity": 144, "stream_budget": 12,
     "evaluation_trials_per_configuration": 1296, "stream_bags_per_configuration": 108,
     "unique_evaluation_trials": 1296, "unique_stream_bags": 108,
     "probability_denominator": 160},
    {"id": "shin-v20", "dataset": "Shin2017A", "identities": 27,
     "configurations": 12, "rotations": 3, "seeds": [13, 17, 19, 23],
     "enrollment_per_identity": 20, "stream_budget": 10,
     "evaluation_trials_per_configuration": 540, "stream_bags_per_configuration": 54,
     "unique_evaluation_trials": 1620, "unique_stream_bags": 162,
     "probability_denominator": 36},
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keys(value, expected, name):
    require(isinstance(value, dict) and set(value) == set(expected), f"{name}: incorrect fields")


def integer(value, low, high, name):
    require(type(value) is int and low <= value <= high, f"{name}: integer out of range")


def prose(value, name):
    require(isinstance(value, str) and bool(value.strip()), f"{name}: nonempty text required")


def decode(text):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON field: {key}")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique_object)


def calculate(study):
    """Derive every reported endpoint using rational integer-count arithmetic."""
    rows, budget = study["rows"], str(study["stream_budget"])
    n = study["configurations"]
    denominators = {"1": study["evaluation_trials_per_configuration"], budget: study["stream_bags_per_configuration"]}
    means = {k: {surface: sum(Fraction(row["correct"][k][surface], denominator) for row in rows) / n
                 for surface in SURFACES} for k, denominator in denominators.items()}
    differences = [Fraction(row["correct"][budget]["confidence16"] - row["correct"][budget]["hard_label"], denominators[budget]) for row in rows]
    rotations = {str(rotation): sum(d for row, d in zip(rows, differences) if row["rotation"] == rotation) / len(study["seeds"])
                 for rotation in range(1, study["rotations"] + 1)}
    return {"mean_BA": means, "primary_mean_difference": sum(differences) / n,
            "positive_configurations": sum(d > 0 for d in differences),
            "negative_configurations": sum(d < 0 for d in differences),
            "tied_configurations": sum(d == 0 for d in differences),
            "rotation_mean_differences": rotations,
            "mean_task_BA": sum(Fraction(row["task_correct"], denominators["1"]) for row in rows) / n}


def compare_reported(actual, expected, location="reported"):
    if isinstance(expected, dict):
        keys(actual, expected, location)
        for key in expected:
            compare_reported(actual[key], expected[key], f"{location}.{key}")
    elif isinstance(expected, Fraction):
        require(type(actual) in (int, float) and math.isfinite(actual)
                and abs(actual - float(expected)) <= 1e-12, f"{location}: arithmetic mismatch")
    else:
        require(type(actual) is int and actual == expected, f"{location}: count mismatch")


def validate(bundle):
    keys(bundle, ("schema_version", "status", "date", "license", "scope", "threat_model", "method", "limitations", "studies"), "evidence")
    require(type(bundle["schema_version"]) is int and bundle["schema_version"] == 1, "Unknown schema")
    require(bundle["status"] == "descriptive_development_evidence", "Incorrect evidence status")
    require(bundle["date"] == "2026-09-10" and bundle["license"] == "CC-BY-4.0", "Incorrect date/license")
    for name in ("scope", "threat_model", "method"):
        prose(bundle[name], name)
    require(isinstance(bundle["limitations"], list) and bool(bundle["limitations"]), "Limitations required")
    for limitation in bundle["limitations"]:
        prose(limitation, "limitation")
    require(isinstance(bundle["studies"], list) and len(bundle["studies"]) == 2, "Exactly two ordered studies required")
    calculations = []
    for study, design in zip(bundle["studies"], DESIGNS):
        keys(study, set(design) | {"source_summary_sha256", "protocol", "rows", "reported"}, "study")
        for key, expected in design.items():
            if type(expected) is int:
                integer(study[key], expected, expected, key)
            elif isinstance(expected, list):
                require(type(study[key]) is list and study[key] == expected and all(type(v) is int for v in study[key]), f"Incorrect {key}")
            else:
                require(study[key] == expected, f"Incorrect {key}")
        require(isinstance(study["source_summary_sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", study["source_summary_sha256"]), "Invalid source-summary digest")
        prose(study["protocol"], "protocol")
        trials, bags, people = (study[k] for k in ("evaluation_trials_per_configuration", "stream_bags_per_configuration", "identities"))
        require(trials == bags * study["stream_budget"] and trials % people == bags % people == 0, "Unbalanced evaluation denominators")
        require(study["unique_evaluation_trials"] == trials * study["rotations"]
                and study["unique_stream_bags"] == bags * study["rotations"], "Seeds must not multiply unique trials/bags")
        require(study["probability_denominator"] == study["enrollment_per_identity"] + 16, "Incorrect coherent smoothing denominator")
        rows, budget = study["rows"], str(study["stream_budget"])
        expected_configs = [(r, s) for r in range(1, study["rotations"] + 1) for s in study["seeds"]]
        require(isinstance(rows, list) and len(rows) == study["configurations"] == len(expected_configs), "Incorrect configuration count")
        for row, (rotation, seed) in zip(rows, expected_configs):
            keys(row, ("rotation", "seed", "correct", "task_correct"), "configuration")
            integer(row["rotation"], rotation, rotation, "rotation")
            integer(row["seed"], seed, seed, "seed")
            integer(row["task_correct"], 0, trials, "task_correct")
            keys(row["correct"], ("1", budget), "correct")
            for k, denominator in (("1", trials), (budget, bags)):
                keys(row["correct"][k], SURFACES, "correct surface")
                for surface in SURFACES:
                    integer(row["correct"][k][surface], 0, denominator, "correct count")
        result = calculate(study)
        compare_reported(study["reported"], result)
        anchor = Fraction(13, 48) if study["id"] == "bnci-v19" else Fraction(-1, 324)
        require(result["primary_mean_difference"] == anchor, "Published primary anchor changed")
        if study["id"] == "shin-v20":
            require(result["rotation_mean_differences"]["1"] == 0, "Published Shin rotation-1 exact tie changed")
        calculations.append(result)
    return calculations


def render(bundle):
    values = validate(bundle)
    width, height = 1240, 690
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
           '<title id="title">Hard labels versus confidence disclosure: aggregate identity accuracy</title>',
           '<desc id="description">Two descriptive development studies. Each panel compares one output with a fixed same-person stream. Both panels use a zero-to-sixty-percent accuracy scale. No uncertainty intervals or independent-cohort inference.</desc>',
           '<rect width="1240" height="690" fill="#ffffff"/>',
           '<g font-family="Arial, Helvetica, sans-serif" fill="#17212b">']

    def text(x, y, label, size=15, anchor="start", color="#17212b", weight="normal", halo=False):
        outline = ' stroke="#ffffff" stroke-width="4" stroke-linejoin="round" paint-order="stroke fill"' if halo else ""
        out.append(f'<text x="{x:.3f}" y="{y:.3f}" font-size="{size}" text-anchor="{anchor}" fill="{color}" font-weight="{weight}"{outline}>{escape(str(label))}</text>')

    def line(x1, y1, x2, y2, color, stroke=1, dash=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" stroke="{color}" stroke-width="{stroke}"{extra}/>')

    text(620, 35, "Confidence disclosure and cross-session identity inference", 24, "middle", weight="bold")
    out.append('<text x="29" y="332" font-size="16" text-anchor="middle" transform="rotate(-90 29 332)">Identity balanced accuracy</text>')
    colors = {"hard_label": "#0072B2", "confidence16": "#D55E00"}
    top, bottom = 152, 480
    y_at = lambda value: bottom - float(value) * (bottom - top) / .6
    for index, (study, result) in enumerate(zip(bundle["studies"], values)):
        left, right = (100, 540) if index == 0 else (720, 1160)
        middle = (left + right) / 2
        points = (left + 60, right - 60)
        budget = str(study["stream_budget"])
        text(middle, 83, study["dataset"], 21, "middle", weight="bold")
        text(middle, 107, f'{study["identities"]} enrolled identities · {study["enrollment_per_identity"]} enrollment outputs/person', 14, "middle", "#48535f")
        chance = Fraction(1, study["identities"])
        line(right - 176, 129, right - 146, 129, "#65717e", 1.5, "6 4")
        text(right - 138, 134, f'Chance: {100 * float(chance):.2f}%', 13, color="#48535f")
        for step in range(7):
            value = Fraction(step, 10)
            y = y_at(value)
            line(left, y, right, y, "#dde3e9")
            text(left - 13, y + 5, f"{step * 10}%", 13, "end", "#48535f")
        line(left, top, left, bottom, "#7d8995")
        line(left, bottom, right, bottom, "#7d8995")
        line(left, y_at(chance), right, y_at(chance), "#65717e", 1.5, "6 4")
        for surface in SURFACES:
            means = [result["mean_BA"][k][surface] for k in ("1", budget)]
            require(all(0 <= v <= Fraction(3, 5) for v in means), "Figure value exceeds the fixed common axis")
            ys = [y_at(v) for v in means]
            line(points[0], ys[0], points[1], ys[1], colors[surface], 3)
            for x, y, value in zip(points, ys, means):
                if surface == "hard_label":
                    out.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="5.5" fill="{colors[surface]}" stroke="#ffffff" stroke-width="1.5"/>')
                else:
                    out.append(f'<rect x="{x - 5.5:.3f}" y="{y - 5.5:.3f}" width="11" height="11" fill="{colors[surface]}" stroke="#ffffff" stroke-width="1.5"/>')
                offset = 23 if surface == "hard_label" else -14
                text(x, y + offset, f'{100 * float(value):.2f}%', 15, "middle", colors[surface], "bold", halo=True)
        for x, k in zip(points, ("1", budget)):
            text(x, 509, f"K = {k}", 16, "middle")
        text(middle, 533, "Outputs in the known same-person bag", 14, "middle", "#48535f")
        delta = 100 * float(result["primary_mean_difference"])
        text(middle, 568, f"Stream contrast: {delta:+.2f} percentage points", 17, "middle", weight="bold")
        text(middle, 590, f'{study["configurations"]} configurations · {study["unique_stream_bags"]:,} unique stream bags', 13, "middle", "#48535f")
    line(386, 620, 420, 620, colors["hard_label"], 3)
    out.append(f'<circle cx="403" cy="620" r="5" fill="{colors["hard_label"]}"/>')
    text(430, 625, "Hard label (1 bit)", 15)
    line(663, 620, 697, 620, colors["confidence16"], 3)
    out.append(f'<rect x="675" y="615" width="10" height="10" fill="{colors["confidence16"]}"/>')
    text(707, 625, "Label + confidence band (4 bits)", 15)
    text(620, 657, CAPTION, 15, "middle")
    text(620, 679, "Aggregate arithmetic only; this figure does not reproduce the underlying attacks.", 12, "middle", "#48535f")
    out.extend(("</g>", "</svg>"))
    return "\n".join(out) + "\n"


def synthetic_fixture():
    def numbers(value):
        if isinstance(value, dict):
            return {k: numbers(v) for k, v in value.items()}
        return float(value) if isinstance(value, Fraction) else value
    studies = []
    for design in DESIGNS:
        study = copy.deepcopy(design)
        study.update(source_summary_sha256="a" * 64, protocol="Synthetic fixture, not research evidence.")
        differences = [29, 29, 29, 30] if study["id"] == "bnci-v19" else [-1, -3, 2, 2, -1, -1, 1, 2, -3, -2, 1, 1]
        study["rows"] = []
        for (rotation, seed), delta in zip([(r, s) for r in range(1, study["rotations"] + 1) for s in study["seeds"]], differences):
            hard = 20 if study["id"] == "bnci-v19" else 4
            study["rows"].append({"rotation": rotation, "seed": seed,
                "correct": {"1": {"hard_label": 60, "confidence16": 80}, str(study["stream_budget"]): {"hard_label": hard, "confidence16": hard + delta}},
                "task_correct": study["evaluation_trials_per_configuration"] // 2})
        study["reported"] = numbers(calculate(study))
        studies.append(study)
    return {"schema_version": 1, "status": "descriptive_development_evidence", "date": "2026-09-10", "license": "CC-BY-4.0",
            "scope": "Synthetic test fixture.", "threat_model": "Synthetic passive observer.", "method": "Synthetic fixed counts.",
            "limitations": ["Not observed research outcomes."], "studies": studies}


class EvidenceTests(unittest.TestCase):
    def test_exact_anchors_and_rotation_tie(self):
        values = validate(synthetic_fixture())
        self.assertEqual(values[0]["primary_mean_difference"], Fraction(13, 48))
        self.assertEqual(values[1]["primary_mean_difference"], Fraction(-1, 324))
        self.assertEqual(values[1]["rotation_mean_differences"], {"1": Fraction(0), "2": Fraction(1, 216), "3": Fraction(-1, 72)})
        self.assertEqual(values[1]["positive_configurations"], 6)
        self.assertEqual(values[1]["negative_configurations"], 6)

    def test_boolean_counts_and_bounds_rejected(self):
        for value in (True, -1, 109, 1.5):
            bundle = synthetic_fixture()
            bundle["studies"][0]["rows"][0]["correct"]["12"]["hard_label"] = value
            with self.assertRaises(ValueError):
                validate(bundle)

    def test_missing_duplicate_and_reordered_configurations_rejected(self):
        for operation in (lambda rows: rows.pop(), lambda rows: rows.__setitem__(-1, rows[0]), lambda rows: rows.reverse()):
            bundle = synthetic_fixture()
            operation(bundle["studies"][1]["rows"])
            with self.assertRaises(ValueError):
                validate(bundle)

    def test_reported_arithmetic_and_extra_fields_rejected(self):
        for value in (float("nan"), float("inf"), True, .25):
            bundle = synthetic_fixture()
            bundle["studies"][0]["reported"]["primary_mean_difference"] = value
            with self.assertRaises(ValueError):
                validate(bundle)
        bundle = synthetic_fixture()
        bundle["studies"][0]["rows"][0]["unexpected"] = []
        with self.assertRaises(ValueError):
            validate(bundle)
        with self.assertRaises(ValueError):
            decode('{"schema_version":1,"schema_version":1}')

    def test_unique_denominators_do_not_multiply_by_seeds(self):
        for key in ("unique_evaluation_trials", "unique_stream_bags", "probability_denominator"):
            bundle = synthetic_fixture()
            bundle["studies"][1][key] *= 4
            with self.assertRaises(ValueError):
                validate(bundle)

    def test_svg_is_deterministic_and_well_formed(self):
        bundle = synthetic_fixture()
        svg = render(bundle)
        self.assertEqual(svg, render(decode(json.dumps(bundle, sort_keys=True))))
        self.assertEqual(ET.fromstring(svg).tag, "{http://www.w3.org/2000/svg}svg")
        self.assertIn(CAPTION, svg)
        self.assertIn("Chance: 11.11%", svg)
        self.assertIn("Chance: 3.70%", svg)
        self.assertNotIn("<script", svg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--render", action="store_true", help="Validate evidence and write its deterministic static SVG")
    modes.add_argument("--test", action="store_true", help="Run synthetic tests only; do not read or change evidence")
    args = parser.parse_args()
    if args.test:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EvidenceTests))
        if not result.wasSuccessful():
            raise SystemExit(1)
        return
    bundle = decode(EVIDENCE.read_text(encoding="utf-8"))
    expected = render(bundle)
    if args.render:
        FIGURE.write_text(expected, encoding="utf-8")
        print("Rendered verified aggregate evidence figure; no underlying attacks were rerun.")
    else:
        require(FIGURE.read_text(encoding="utf-8") == expected, "SVG differs from deterministic rendering; run --render to regenerate")
        print("PASS: exact aggregate count arithmetic, frozen endpoints, and deterministic SVG; not raw attack reproduction.")


if __name__ == "__main__":
    main()

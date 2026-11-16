"""Contract tests use explicit logical fixtures, not experimental measurements."""
import unittest
from copy import deepcopy
from claim_metrics import evaluate, text_sha256, require_matched_comparison


def fixture():
    raw = [{"case_id": str(i), "model_id": "baseline", "seed": 11,
            "raw_response": text, "checkpoint_sha256": "test-checkpoint", "protocol_sha256": "test-protocol",
            "execution_status": "generated",
            "split": "test", "prompt": "same prompt", "source_case_sha256": str(i), "point_token_budget": 97}
           for i, text in enumerate(["a wrong category and location", "unknown object", "no object asserted"])]
    annotations = [{"case_id": str(i), "model_id": "baseline", "seed": 11,
                   "raw_response_sha256": text_sha256(r["raw_response"]), "annotation_source": "explicit unit fixture",
                   "error_types": ["wrong_category", "spatial"] if i == 0 else [],
                   "claims": [{"text": r["raw_response"], "verdict": "contradicted" if i == 0 else "unverifiable",
                               "evidence_refs": ["independent fixture GT"] if i == 0 else []}] if i < 2 else []}
                   for i, r in enumerate(raw)]
    return raw, annotations


class ClaimsContract(unittest.TestCase):
    def test_nonexclusive_sum_is_not_any_error_probability(self):
        raw, annotations = fixture()
        m = evaluate(raw, annotations)["metrics"][0]
        self.assertEqual(m["n_responses"], 3)
        self.assertAlmostEqual(m["nonexclusive_error_rate_sum"], 200 / 3)
        self.assertAlmostEqual(m["any_error_response_rate_pct"], 100 / 3)
        self.assertAlmostEqual(m["verifiable_response_coverage_pct"], 100 / 3)
        self.assertEqual(m["claim_verdict_counts"]["unverifiable"], 1)
        self.assertEqual(m["contradiction_rate_among_verifiable_claims_pct"], 100)

    def test_response_mutation_is_rejected(self):
        raw, annotations = fixture(); raw[0]["raw_response"] = "rewritten positive output"
        with self.assertRaises(ValueError): evaluate(raw, annotations)

    def test_missing_annotation_cannot_drop_difficult_responses(self):
        raw, annotations = fixture()
        with self.assertRaises(ValueError): evaluate(raw, annotations[:-1])

    def test_unsupported_is_not_contradicted(self):
        raw, annotations = fixture(); annotations[0]["claims"][0]["verdict"] = "unverifiable"
        m = evaluate(raw, annotations)["metrics"][0]
        self.assertIsNone(m["contradiction_rate_among_verifiable_claims_pct"])

    def test_mismatched_prompt_and_budget_rejected(self):
        raw, _ = fixture(); other = deepcopy(raw)
        for r in other: r["model_id"] = "rahum"
        self.assertEqual(require_matched_comparison(raw + other, "baseline", "rahum"), 3)
        other[0]["point_token_budget"] = 513
        with self.assertRaises(ValueError): require_matched_comparison(raw + other, "baseline", "rahum")

    def test_per_case_checkpoint_selection_rejected(self):
        raw, annotations = fixture(); raw[0]["checkpoint_sha256"] = "different checkpoint"
        with self.assertRaises(ValueError): evaluate(raw, annotations)

    def test_execution_failure_is_not_a_hallucination_free_response(self):
        raw, annotations = fixture(); raw[2]["execution_status"] = "input_or_generation_error"
        raw[2]["raw_response"] = ""; annotations[2]["raw_response_sha256"] = text_sha256("")
        with self.assertRaises(ValueError): evaluate(raw, annotations)

    def test_model_empty_output_is_retained_with_zero_verifiable_coverage(self):
        raw, annotations = fixture(); raw[2]["raw_response"] = ""
        annotations[2]["raw_response_sha256"] = text_sha256("")
        m = evaluate(raw, annotations)["metrics"][0]
        self.assertEqual(m["n_responses"], 3)
        self.assertEqual(m["n_empty_generated_responses"], 1)


if __name__ == "__main__": unittest.main()

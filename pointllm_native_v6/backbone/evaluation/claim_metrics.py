"""Evaluate independently annotated language claims, never point labels.

Raw generations are immutable JSONL records. The annotation file is a
separate artifact with evidence references and the exact raw response hash.
This module aggregates judgments; it does not infer hallucinations from
radar/reference disagreement or silently judge unknown claims as false.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ERROR_TYPES = ("phantom_existence", "wrong_category", "spatial", "temporal")
VERDICTS = {"supported", "contradicted", "unverifiable"}


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def key_of(record):
    return str(record["case_id"]), str(record["model_id"]), int(record["seed"])


def evaluate(raw_records, annotations):
    """Return both fixed-response rates and claim coverage.

    ``error_types`` is independently annotated per response because four
    errors can co-occur. Each type is counted at most once per response.
    Claim verdicts retain supported/contradicted/unverifiable distinctions.
    All raw responses, including abstentions and malformed model outputs,
    remain in the response-rate denominator.
    """
    raw = {}
    for record in raw_records:
        key = key_of(record)
        if key in raw:
            raise ValueError(f"Duplicate raw generation key: {key}")
        if not isinstance(record.get("raw_response"), str):
            raise ValueError(f"Missing unmodified raw response: {key}")
        if record.get("execution_status") != "generated":
            raise ValueError(
                f"Incomplete generation at {key}: retain this case and report execution failures; "
                "a failed or unverified execution cannot count as a hallucination-free response"
            )
        for required in ("checkpoint_sha256", "protocol_sha256", "split"):
            if not record.get(required):
                raise ValueError(f"Missing {required}: {key}")
        raw[key] = record
    if not raw:
        raise ValueError("No real generation records were supplied")
    annotated = {}
    for annotation in annotations:
        key = key_of(annotation)
        if key in annotated or key not in raw:
            raise ValueError(f"Duplicate or unmatched annotation: {key}")
        if annotation.get("raw_response_sha256") != text_sha256(raw[key]["raw_response"]):
            raise ValueError(f"Annotation does not match the immutable raw response: {key}")
        if not annotation.get("annotation_source"):
            raise ValueError(f"Missing annotation source: {key}")
        errors = annotation.get("error_types")
        if not isinstance(errors, list) or len(errors) != len(set(errors)) or set(errors) - set(ERROR_TYPES):
            raise ValueError(f"Invalid nonexclusive response error types: {key}")
        claims = annotation.get("claims")
        if not isinstance(claims, list):
            raise ValueError(f"Missing claim list: {key}")
        for claim in claims:
            if claim.get("verdict") not in VERDICTS or not claim.get("text"):
                raise ValueError(f"Malformed claim: {key}")
            if claim["verdict"] != "unverifiable" and not claim.get("evidence_refs"):
                raise ValueError(f"Verifiable claims require independent evidence references: {key}")
        annotated[key] = annotation
    if set(annotated) != set(raw):
        missing = sorted(set(raw) - set(annotated))
        raise ValueError(f"All generations must be annotated; missing {len(missing)} responses")
    groups = {}
    for key, record in raw.items():
        group_key = record["model_id"], int(record["seed"]), record["split"], record["protocol_sha256"]
        group = groups.setdefault(group_key, {"responses": 0, "any_error_responses": 0,
            "empty_responses": 0,
            "responses_with_verifiable_claim": 0, "claim_verdict_counts": {v: 0 for v in sorted(VERDICTS)},
            "response_error_counts": {e: 0 for e in ERROR_TYPES}, "checkpoints": set(), "cases": []})
        annotation = annotated[key]
        group["responses"] += 1
        group["empty_responses"] += not record["raw_response"].strip()
        group["any_error_responses"] += bool(annotation["error_types"])
        group["responses_with_verifiable_claim"] += any(c["verdict"] != "unverifiable" for c in annotation["claims"])
        for error in annotation["error_types"]:
            group["response_error_counts"][error] += 1
        for claim in annotation["claims"]:
            group["claim_verdict_counts"][claim["verdict"]] += 1
        group["checkpoints"].add(record["checkpoint_sha256"])
        group["cases"].append(record["case_id"])
    result = []
    for (model, seed, split, protocol), group in sorted(groups.items()):
        if len(group["checkpoints"]) != 1:
            raise ValueError(f"A model/seed/split combines different checkpoints: {(model, seed, split)}")
        n = group["responses"]
        verdicts = group["claim_verdict_counts"]
        verified = verdicts["supported"] + verdicts["contradicted"]
        result.append({"model_id": model, "seed": seed, "split": split, "protocol_sha256": protocol,
            "n_responses": n, "response_error_counts": group["response_error_counts"],
            "n_empty_generated_responses": group["empty_responses"],
            "response_error_rates_pct": {e: 100 * count / n for e, count in group["response_error_counts"].items()},
            "nonexclusive_error_rate_sum": 100 * sum(group["response_error_counts"].values()) / n,
            "any_error_response_rate_pct": 100 * group["any_error_responses"] / n,
            "verifiable_response_coverage_pct": 100 * group["responses_with_verifiable_claim"] / n,
            "claim_verdict_counts": verdicts,
            "contradiction_rate_among_verifiable_claims_pct": 100 * verdicts["contradicted"] / verified if verified else None,
            "checkpoint_sha256s": sorted(group["checkpoints"]), "case_ids": sorted(group["cases"])})
    return {"schema_version": 1, "judgment_scope": "independent annotations of raw generated language",
            "response_error_types_are_nonexclusive": True, "metrics": result}


def require_matched_comparison(raw_records, model_a, model_b):
    """Reject differing scenes, prompts, protocols or real-data splits."""
    paired = {}
    for record in raw_records:
        if record["model_id"] not in {model_a, model_b}:
            continue
        key = str(record["case_id"]), int(record["seed"])
        group = paired.setdefault(key, {})
        if record["model_id"] in group:
            raise ValueError(f"Repeated comparison record: {key}")
        group[record["model_id"]] = record
    if not paired:
        raise ValueError("No matched generations supplied")
    for key, group in paired.items():
        if set(group) != {model_a, model_b}:
            raise ValueError(f"Unpaired case/seed: {key}")
        a, b = group[model_a], group[model_b]
        for field in ("split", "protocol_sha256", "prompt", "source_case_sha256", "point_token_budget"):
            if field not in a or field not in b or a[field] != b[field]:
                raise ValueError(f"Comparison differs in {field}: {key}")
    return len(paired)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate(read_jsonl(args.raw), read_jsonl(args.annotations))
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

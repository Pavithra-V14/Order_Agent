"""
Tier 2 guardrails - architecture doc 8.6: fast classifier (sync,
blocking), 10-50ms. Structured-output schema validation + PII detection.

Structural validation: `guardrails-ai` is installed and verified working
in this environment (Guard.for_pydantic() correctly rejects malformed
decision objects) — but it registers an OTLP telemetry exporter on Guard
creation that unconditionally attempts a network call on interpreter
shutdown, adding several seconds of connection-retry delay every run even
with `settings.disable_tracing = True` set. That's a real problem for any
network-restricted or air-gapped deployment, not just this sandbox, so
Tier 2's default path validates directly against the same Pydantic schema
`Guard.for_pydantic()` would wrap anyway (Pydantic IS the underlying
validation mechanism) — same correctness, no telemetry dependency. See
`validate_with_guardrails_ai()` below for the guardrails-ai-backed
alternative, kept and tested for environments with normal network access.

PII detection: architecture doc 8.10/8.6 specifies LLM Guard (Presidio +
spaCy NER under the hood — a real transformer-based entity recognizer).
That pulls in spaCy language models (100MB+) and torch-adjacent
dependencies this sandbox's disk budget doesn't comfortably support
alongside everything else already installed. This ships a regex-based PII
scanner instead — genuinely functional for common structured patterns
(email, phone, SSN-shaped numbers, credit-card-shaped numbers), with the
real gap (unstructured/contextual PII a regex can't catch) documented
rather than silently absent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from app.guardrails.schema import ResolutionDecision


@dataclass
class Tier2Result:
    passed: bool
    validated_decision: ResolutionDecision = None
    errors: list = field(default_factory=list)
    pii_findings: list = field(default_factory=list)


def validate_structure(raw_decision: dict) -> Tier2Result:
    """Direct Pydantic validation against ResolutionDecision — see module
    docstring for why this is the default path over Guard.for_pydantic()."""
    try:
        decision = ResolutionDecision.model_validate(raw_decision)
        return Tier2Result(passed=True, validated_decision=decision)
    except ValidationError as e:
        errors = [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()]
        return Tier2Result(passed=False, errors=errors)


def validate_with_guardrails_ai(raw_decision: dict) -> Tier2Result:
    """Alternative path using the real guardrails-ai library — functionally
    equivalent to validate_structure(), kept for environments where the
    telemetry network call isn't a concern. Not used by default (see
    module docstring)."""
    from guardrails import Guard
    guard = Guard.for_pydantic(ResolutionDecision)
    import json
    result = guard.parse(json.dumps(raw_decision))
    if result.validation_passed:
        return Tier2Result(passed=True, validated_decision=ResolutionDecision.model_validate(result.validated_output))
    return Tier2Result(passed=False, errors=[str(result.validated_output)])


_PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn_shaped": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card_shaped": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
}


def scan_for_pii(text: str) -> list:
    """Regex-based PII scan — see module docstring for the LLM-Guard/
    Presidio swap point and the documented gap (contextual/unstructured
    PII a pattern match can't catch, e.g. a name mentioned in prose)."""
    findings = []
    for pii_type, pattern in _PII_PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append({"type": pii_type, "match": match.group(), "span": match.span()})
    return findings


def run_tier2(raw_decision: dict) -> Tier2Result:
    """Full Tier 2: structural validation, then a PII scan over the
    reasoning text (the one free-text field a model could leak PII into)."""
    result = validate_structure(raw_decision)
    if not result.passed:
        return result

    pii_findings = scan_for_pii(result.validated_decision.reasoning)
    if pii_findings:
        result.passed = False
        result.errors.append(f"PII detected in reasoning text: {[f['type'] for f in pii_findings]}")
        result.pii_findings = pii_findings

    return result

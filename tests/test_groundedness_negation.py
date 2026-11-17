"""Tests for negation handling in the Eq. (13) span check.

Manual inspection of 30 flagged spans from the 60-claim run found 29 of
one kind: a justification naming a figure in order to report its
*absence* -- "the snippet does not provide the 40% drop" -- which naive
containment testing flags as a fabricated quotation, inverting what the
justification said. Unfiltered, the measurement was 28.0%; filtered, it
is 0.9%. These tests keep the distinction pinned, because without it the
reported rate is almost entirely noise.
"""

from __future__ import annotations

import pytest

from fact_checker.groundedness import (
    aggregate_span_reports,
    audit_span_report,
    check_alignment,
    is_negated,
)
from fact_checker.schemas import Alignment, AuditReport, EvidenceItem, ScoutReport


def align(justification: str, source_id: str = "doc_1") -> Alignment:
    return Alignment(
        source_id=source_id, verdict="NOT_ENOUGH_INFO", justification=justification
    )


class TestIsNegated:
    @pytest.mark.parametrize(
        "justification, span",
        [
            ("The snippet does not provide the specific number of 98,000 deaths.", "98,000"),
            ("doc_2 does not mention a 60% drop.", "60"),
            ("This passage contains no reference to 2020.", "2020"),
            ("The snippet is silent on the 40% figure.", "40"),
            ("It never states 1971.", "1971"),
            ("The passage provides no information about March.", "March"),
        ],
    )
    def test_absence_claims_are_recognised(self, justification, span):
        assert is_negated(justification, span)

    @pytest.mark.parametrize(
        "justification, span",
        [
            ("The passage states the figure was 45,000.", "45,000"),
            ("This supports the claim of a 40% drop in oil prices.", "40"),
            ("Signed in March of that year.", "March"),
        ],
    )
    def test_positive_attributions_are_not(self, justification, span):
        assert not is_negated(justification, span)

    def test_negation_is_sought_within_the_span_s_sentence(self):
        """One clause may negate while another legitimately quotes."""
        text = "The passage does not mention 1969. It does state the figure 45,000."
        assert is_negated(text, "1969")
        assert not is_negated(text, "45,000")


class TestCheckAlignment:
    def test_a_negated_figure_is_not_a_violation(self):
        check = check_alignment(
            align("The snippet does not provide the specific number of 98,000 deaths."),
            "Recent releases from the National Center for Health Statistics.",
        )
        assert not check.violates
        assert "98,000" in check.negated

    def test_a_figure_attributed_to_the_passage_is(self):
        """The real failure: a figure the passage does not contain, asserted."""
        check = check_alignment(
            Alignment(
                source_id="doc_1",
                verdict="SUPPORTS",
                justification="This supports the claim of a 40% drop in oil prices.",
            ),
            "The audit attributed the decline to reduced oil price in the year.",
        )
        assert check.violates
        assert "40" in check.missing

    def test_a_negated_quotation_is_not_a_violation(self):
        """Quotes also name things in order to deny them."""
        check = check_alignment(
            align('The snippet does not discuss the “More Doctors” programme.'),
            "One found that 94% of users reported satisfaction with the programme.",
        )
        assert not check.violates
        assert "More Doctors" in check.negated

    def test_a_faithful_quotation_still_passes(self):
        check = check_alignment(
            align('the passage states "forty two people" directly'),
            "Officials said forty two people attended.",
        )
        assert not check.violates
        assert check.negated == []

    def test_negated_spans_are_kept_for_audit_not_discarded(self):
        check = check_alignment(
            align("It does not provide the 2019 figure."), "Unrelated text."
        )
        assert check.negated == ["2019"]


class TestAggregate:
    def test_negated_spans_are_counted_and_excluded_from_the_rate(self):
        scout = ScoutReport(
            queries=["q"],
            evidence=[EvidenceItem(source_id="doc_1", snippet="Unrelated passage text.")],
        )
        audit = AuditReport(
            alignments=[align("The snippet does not provide the 98,000 figure.")]
        )
        report = audit_span_report(audit, scout)
        assert report.n_violating == 0
        assert report.n_negated_spans == 1

        pooled = aggregate_span_reports([report.model_dump()])
        assert pooled["violation_rate"] == 0.0
        assert pooled["n_negated_spans"] == 1

    def test_a_genuine_violation_still_registers(self):
        scout = ScoutReport(
            queries=["q"],
            evidence=[EvidenceItem(source_id="doc_1", snippet="The decline was noted.")],
        )
        audit = AuditReport(
            alignments=[
                Alignment(
                    source_id="doc_1", verdict="SUPPORTS",
                    justification="This supports the claim of a 40% drop.",
                )
            ]
        )
        assert audit_span_report(audit, scout).n_violating == 1

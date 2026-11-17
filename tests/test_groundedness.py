"""Tests for the Equation (13) quoted-span diagnostic.

The check is deliberately conservative: it should catch a justification
that quotes words the passage does not contain, and should not manufacture
violations from paraphrase, possessives, or cosmetic differences in
punctuation and case.
"""

from __future__ import annotations

from fact_checker.groundedness import (
    aggregate_span_reports,
    audit_span_report,
    check_alignment,
    extract_figures,
    extract_quotes,
    normalise,
)
from fact_checker.schemas import Alignment, AuditReport, EvidenceItem, ScoutReport


def align(justification: str, source_id: str = "doc_1") -> Alignment:
    return Alignment(
        source_id=source_id, verdict="SUPPORTS", justification=justification
    )


def scout(*snippets: str) -> ScoutReport:
    return ScoutReport(
        queries=["q"],
        evidence=[
            EvidenceItem(source_id=f"doc_{i}", snippet=text)
            for i, text in enumerate(snippets, start=1)
        ],
    )


class TestNormalise:
    def test_case_and_whitespace_are_folded(self):
        assert normalise("  The   QUICK  fox ") == "the quick fox"

    def test_punctuation_is_stripped(self):
        assert normalise("don't stop, now!") == "don t stop now"

    def test_typographic_and_straight_quotes_agree(self):
        assert normalise("“hello”") == normalise('"hello"')

    def test_percent_survives(self):
        """A percentage sign carries meaning in a numeric claim."""
        assert "%" in normalise("62.4%")


class TestExtraction:
    def test_straight_double_quotes_are_found(self):
        assert extract_quotes('the text says "forty two people" clearly') == [
            "forty two people"
        ]

    def test_typographic_quotes_are_found(self):
        assert extract_quotes("it says “no such record” here") == [
            "no such record"
        ]

    def test_a_possessive_is_not_a_quotation(self):
        """Treating apostrophes as quotes would invent violations wholesale."""
        assert extract_quotes("the government's figure was higher") == []

    def test_unquoted_text_yields_nothing(self):
        assert extract_quotes("the passage broadly agrees") == []

    def test_figures_are_extracted_quoted_or_not(self):
        found = extract_figures("the 2019 report gives 45,000 and 62.4%")
        assert "2019" in found
        assert any("45,000" in f for f in found)

    def test_month_names_count_as_dates(self):
        assert "March" in extract_figures("signed in March that year")


class TestCheckAlignment:
    def test_a_faithful_quotation_passes(self):
        check = check_alignment(
            align('the passage states "forty two people" directly'),
            "Officials said forty two people attended.",
        )
        assert not check.violates
        assert check.n_quotes == 1
        assert check.n_quotes_found == 1

    def test_a_fabricated_quotation_is_caught(self):
        check = check_alignment(
            align('the passage states "ninety nine people"'),
            "Officials said forty two people attended.",
        )
        assert check.violates
        assert "ninety nine people" in check.missing

    def test_cosmetic_differences_do_not_count_as_fabrication(self):
        check = check_alignment(
            align('it says “Forty-Two People!”'),
            "officials said forty two people attended",
        )
        assert not check.violates

    def test_a_fabricated_figure_is_caught_even_unquoted(self):
        check = check_alignment(
            align("the passage gives the year as 1971"),
            "The treaty was signed in 1969.",
        )
        assert check.violates
        assert "1971" in check.missing

    def test_a_present_figure_passes(self):
        check = check_alignment(
            align("the passage gives the year as 1969"),
            "The treaty was signed in 1969.",
        )
        assert not check.violates

    def test_a_paraphrase_with_no_spans_is_uncheckable_not_violating(self):
        check = check_alignment(
            align("the passage broadly agrees with the claim"), "Some text."
        )
        assert not check.violates
        assert check.n_quotes == 0
        assert check.n_figures == 0
        assert not check.has_quote


class TestAuditSpanReport:
    def test_rates_are_computed_over_checkable_alignments(self):
        audit = AuditReport(
            alignments=[
                align('says "alpha"', "doc_1"),
                align('says "zulu"', "doc_2"),
                align("no spans at all", "doc_3"),
            ]
        )
        report = audit_span_report(audit, scout("alpha text", "bravo text", "charlie"))
        assert report.n_alignments == 3
        assert report.n_checkable == 2        # the third quotes nothing
        assert report.n_violating == 1        # "zulu" is absent from doc_2
        assert report.violation_rate == 0.5
        assert report.quote_rate == 2 / 3

    def test_alignments_naming_an_unknown_passage_are_skipped(self):
        """A missing identifier is reconciliation's problem, not Eq. 13's."""
        audit = AuditReport(alignments=[align('says "x"', "doc_9")])
        report = audit_span_report(audit, scout("some text"))
        assert report.n_alignments == 0
        assert report.violation_rate == 0.0

    def test_an_empty_audit_is_safe(self):
        report = audit_span_report(AuditReport(alignments=[]), scout())
        assert report.n_alignments == 0
        assert report.quote_rate == 0.0


class TestAggregate:
    def test_reports_pool_across_claims(self):
        reports = [
            audit_span_report(
                AuditReport(alignments=[align('says "alpha"', "doc_1")]),
                scout("alpha text"),
            ).model_dump(),
            audit_span_report(
                AuditReport(alignments=[align('says "zulu"', "doc_1")]),
                scout("bravo text"),
            ).model_dump(),
        ]
        pooled = aggregate_span_reports(reports)
        assert pooled["n_checkable"] == 2
        assert pooled["n_violating"] == 1
        assert pooled["violation_rate"] == 0.5
        assert "zulu" in pooled["example_missing_spans"]

    def test_empty_input_is_safe(self):
        pooled = aggregate_span_reports([])
        assert pooled["violation_rate"] == 0.0
        assert pooled["n_alignments"] == 0

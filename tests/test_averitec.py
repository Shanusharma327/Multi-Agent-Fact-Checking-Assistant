"""Tests for the AVeriTeC loader and the dataset registry.

Uses a local fixture -- no network access.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fact_checker.datasets import (
    DATASETS,
    SPLIT_CHOICES,
    UnknownDatasetError,
    load_benchmark,
)
from fact_checker.datasets.averitec import (
    AVeriTeCDownloadError,
    _normalise_label,
    load_averitec,
    local_path,
)
from fact_checker.datasets.base import BenchmarkClaim, label_distribution, sample_balanced

FIXTURE = [
    {
        "claim": "99% of people recover from COVID-19.",
        "label": "Refuted",
        "claim_date": "31-10-2020",
        "speaker": None,
        "reporting_source": "Facebook",
        "location_ISO_code": "US",
        "claim_types": ["Numerical Claim"],
        "justification": "The sources show otherwise.",
        "fact_checking_article": "https://example.org/a",
        "questions": [
            {"question": "q1", "answers": [{"answer": "a1"}, {"answer": "a2"}]},
            {"question": "q2", "answers": [{"answer": "a3"}]},
        ],
    },
    {
        "claim": "Ruth Bader Ginsburg lay in state at the U.S. Capitol.",
        "label": "Supported",
        "claim_date": "01-10-2020",
        "questions": [{"question": "q", "answers": [{"answer": "a"}]}],
    },
    {
        "claim": "There has been a 60% drop in government revenue.",
        "label": "Not Enough Evidence",
        "questions": [],
    },
    {
        "claim": "A claim with conflicting evidence.",
        "label": "Conflicting Evidence/Cherrypicking",
        "questions": [],
    },
    {"claim": "   ", "label": "Supported", "questions": []},   # empty -> unusable
    {"claim": "Has a bogus label.", "label": "Mostly True"},    # unmapped -> unusable
]


@pytest.fixture
def averitec_file(tmp_path: Path) -> Path:
    path = tmp_path / "dev.json"
    path.write_text(json.dumps(FIXTURE), encoding="utf-8")
    return path


class TestLabelNormalisation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Supported", "SUPPORTS"),
            ("Refuted", "REFUTES"),
            ("Not Enough Evidence", "NOT_ENOUGH_INFO"),
            ("  refuted  ", "REFUTES"),
            ("Conflicting Evidence/Cherrypicking", None),
            ("Mostly True", None),
            ("", None),
        ],
    )
    def test_maps_averitec_labels(self, raw, expected) -> None:
        assert _normalise_label(raw) == expected


class TestLoading:
    def test_keeps_only_mappable_claims(self, tmp_path, averitec_file) -> None:
        claims, report = load_averitec("dev", tmp_path, download=False)

        assert report["total"] == 6
        assert report["kept"] == 3
        assert report["conflicting"] == 1
        assert report["unusable"] == 2
        assert label_distribution(claims) == {
            "SUPPORTS": 1,
            "REFUTES": 1,
            "NOT_ENOUGH_INFO": 1,
        }

    def test_conflicting_is_excluded_not_silently_remapped(self, tmp_path, averitec_file) -> None:
        """The fourth label has no honest 3-way equivalent."""
        claims, report = load_averitec("dev", tmp_path, download=False)
        assert all("conflicting" not in c.claim.lower() for c in claims)
        assert report["conflicting"] == 1, "the exclusion must be reported"

    def test_conflicting_can_be_opted_into(self, tmp_path, averitec_file) -> None:
        claims, report = load_averitec(
            "dev", tmp_path, download=False, map_conflicting=True
        )
        assert report["kept"] == 4
        assert label_distribution(claims)["NOT_ENOUGH_INFO"] == 2

    def test_metadata_is_preserved(self, tmp_path, averitec_file) -> None:
        claims, _ = load_averitec("dev", tmp_path, download=False)
        covid = next(c for c in claims if "COVID" in c.claim)
        assert covid.label == "REFUTES"
        assert covid.claim_date == "31-10-2020"
        assert covid.reporting_source == "Facebook"
        assert covid.claim_types == ["Numerical Claim"]
        assert covid.country_code == "US"
        assert covid.n_questions == 2
        assert covid.n_answers == 3          # 2 + 1 across both questions
        assert covid.dataset == "averitec"

    def test_ids_are_synthesised_and_unique(self, tmp_path, averitec_file) -> None:
        """AVeriTeC carries no id field, so ids come from the record index."""
        claims, _ = load_averitec("dev", tmp_path, download=False)
        ids = [c.id for c in claims]
        assert len(set(ids)) == len(ids)
        assert all(i.startswith("averitec_dev_") for i in ids)

    def test_missing_file_without_download_raises(self, tmp_path) -> None:
        with pytest.raises(AVeriTeCDownloadError, match="not cached"):
            load_averitec("dev", tmp_path / "absent", download=False)

    def test_non_array_json_is_rejected(self, tmp_path) -> None:
        (tmp_path / "dev.json").write_text('{"not": "an array"}', encoding="utf-8")
        with pytest.raises(AVeriTeCDownloadError, match="Expected a JSON array"):
            load_averitec("dev", tmp_path, download=False)

    def test_unknown_split_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown split"):
            local_path("nonexistent")


class TestRegistry:
    def test_both_datasets_registered(self) -> None:
        assert set(DATASETS) == {"fever", "averitec"}
        assert "dev" in SPLIT_CHOICES["fever"]
        assert "dev" in SPLIT_CHOICES["averitec"]

    def test_unknown_dataset_rejected(self) -> None:
        with pytest.raises(UnknownDatasetError, match="Unknown dataset"):
            load_benchmark("squad")

    def test_registry_loads_averitec(self, tmp_path, averitec_file) -> None:
        claims, report = load_benchmark("averitec", "dev", str(tmp_path), download=False)
        assert report["kept"] == 3
        assert all(isinstance(c, BenchmarkClaim) for c in claims)

    def test_both_datasets_yield_the_same_interface(self, tmp_path, averitec_file) -> None:
        """The harness reads only id/claim/label/dataset, whatever the corpus."""
        claims, _ = load_benchmark("averitec", "dev", str(tmp_path), download=False)
        for claim in claims:
            assert isinstance(claim.id, str)
            assert isinstance(claim.claim, str)
            assert claim.label in {"SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"}
            assert claim.dataset == "averitec"


class TestSamplingOnImbalancedCorpus:
    def test_sample_caps_at_the_scarcest_label(self) -> None:
        """AVeriTeC dev has far fewer NEI claims than REFUTES."""
        population = (
            [BenchmarkClaim(id=f"s{i}", claim="c", label="SUPPORTS") for i in range(50)]
            + [BenchmarkClaim(id=f"r{i}", claim="c", label="REFUTES") for i in range(50)]
            + [BenchmarkClaim(id=f"n{i}", claim="c", label="NOT_ENOUGH_INFO") for i in range(4)]
        )
        sample = sample_balanced(population, 10, seed=1)
        dist = label_distribution(sample)
        assert dist == {"SUPPORTS": 10, "REFUTES": 10, "NOT_ENOUGH_INFO": 4}

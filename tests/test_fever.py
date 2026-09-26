"""Tests for the FEVER loader. Uses a local fixture -- no network access."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fact_checker.datasets.fever import (
    FeverClaim,
    FeverDownloadError,
    _extract_gold_pages,
    _normalise_label,
    iter_fever,
    label_distribution,
    load_fever,
    local_path,
    sample_balanced,
)

FIXTURE_ROWS = [
    {
        "id": 1,
        "verifiable": "VERIFIABLE",
        "label": "SUPPORTS",
        "claim": "Fox 2000 Pictures released the film Soul Food.",
        "evidence": [[[289914, 283015, "Soul_Food_-LRB-film-RRB-", 0]]],
    },
    {
        "id": 2,
        "verifiable": "NOT VERIFIABLE",
        "label": "NOT ENOUGH INFO",
        "claim": "Tilda Swinton is a vegan.",
        "evidence": [[[227768, None, None, None]]],
    },
    {
        "id": 3,
        "verifiable": "VERIFIABLE",
        "label": "REFUTES",
        "claim": "Meghan Markle was born in July.",
        "evidence": [[[1, 2, "Meghan_Markle", 0]]],
    },
    # Unlabelled row: must be skipped, not guessed at.
    {"id": 4, "verifiable": "NOT VERIFIABLE", "label": "", "claim": "No label here."},
    # Empty claim: also skipped.
    {"id": 5, "label": "SUPPORTS", "claim": "   "},
]


@pytest.fixture
def fever_file(tmp_path: Path) -> Path:
    path = tmp_path / "fever.jsonl"
    lines = [json.dumps(row) for row in FIXTURE_ROWS]
    lines.insert(2, "")          # blank line
    lines.append("{not json}")   # malformed line
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


class TestLabelNormalisation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("SUPPORTS", "SUPPORTS"),
            ("REFUTES", "REFUTES"),
            ("NOT ENOUGH INFO", "NOT_ENOUGH_INFO"),
            ("not enough info", "NOT_ENOUGH_INFO"),
            ("  SUPPORTS  ", "SUPPORTS"),
            ("NONSENSE", None),
            ("", None),
        ],
    )
    def test_maps_fever_labels_onto_our_verdicts(self, raw, expected) -> None:
        assert _normalise_label(raw) == expected


class TestGoldPages:
    def test_unescapes_wikipedia_titles(self) -> None:
        evidence = [[[1, 2, "Soul_Food_-LRB-film-RRB-", 0]]]
        assert _extract_gold_pages(evidence) == ["Soul Food (film)"]

    def test_null_pages_yield_nothing(self) -> None:
        assert _extract_gold_pages([[[1, None, None, None]]]) == []

    def test_deduplicates_across_annotators(self) -> None:
        evidence = [
            [[1, 2, "Soul_Food", 0]],
            [[3, 4, "Soul_Food", 0]],
        ]
        assert _extract_gold_pages(evidence) == ["Soul Food"]

    def test_malformed_evidence_is_tolerated(self) -> None:
        assert _extract_gold_pages(None) == []
        assert _extract_gold_pages("nope") == []
        assert _extract_gold_pages([["bad"]]) == []


class TestIterFever:
    def test_reads_only_usable_labelled_rows(self, fever_file: Path) -> None:
        claims = list(iter_fever(fever_file))
        assert [c.id for c in claims] == ["1", "2", "3"]   # ids are strings across datasets
        assert [c.label for c in claims] == ["SUPPORTS", "NOT_ENOUGH_INFO", "REFUTES"]

    def test_normalises_the_neutral_label(self, fever_file: Path) -> None:
        claims = {c.id: c for c in iter_fever(fever_file)}
        assert claims["2"].label == "NOT_ENOUGH_INFO"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FeverDownloadError, match="not found"):
            list(iter_fever(tmp_path / "absent.jsonl"))


class TestSampling:
    def _population(self) -> list[FeverClaim]:
        claims = []
        for index in range(30):
            label = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"][index % 3]
            claims.append(FeverClaim(id=str(index), claim=f"claim {index}", label=label))
        return claims

    def test_sample_is_balanced(self) -> None:
        sample = sample_balanced(self._population(), 3, seed=7)
        assert label_distribution(sample) == {
            "SUPPORTS": 3,
            "REFUTES": 3,
            "NOT_ENOUGH_INFO": 3,
        }

    def test_sampling_is_reproducible_for_a_seed(self) -> None:
        population = self._population()
        first = [c.id for c in sample_balanced(population, 4, seed=99)]
        second = [c.id for c in sample_balanced(population, 4, seed=99)]
        assert first == second

    def test_different_seeds_differ(self) -> None:
        population = self._population()
        first = [c.id for c in sample_balanced(population, 4, seed=1)]
        second = [c.id for c in sample_balanced(population, 4, seed=2)]
        assert first != second

    def test_requesting_more_than_available_takes_everything(self) -> None:
        sample = sample_balanced(self._population(), 100, seed=1)
        assert len(sample) == 30

    def test_labels_are_interleaved_not_grouped(self) -> None:
        """A partial run must still cover all three labels."""
        sample = sample_balanced(self._population(), 10, seed=5)
        assert len(set(c.label for c in sample[:6])) > 1


class TestPaths:
    def test_unknown_split_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown split"):
            local_path("nonexistent")

    def test_known_splits_map_to_files(self) -> None:
        assert local_path("dev", "cache").name == "shared_task_dev.jsonl"

    def test_load_without_download_raises_when_absent(self, tmp_path: Path) -> None:
        with pytest.raises(FeverDownloadError, match="not cached"):
            load_fever("dev", tmp_path, download=False)

"""Integration tests for the FEVER harness, with the LLM and search stubbed.

Drives the real code path in evaluate.py -- sampling, threaded execution,
incremental JSONL writes, resume, and scoring -- without a network call, an
API key, or a cent of spend.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import evaluate as harness
import fact_checker.agents.adjudicator as adjudicator_mod
import fact_checker.agents.auditor as auditor_mod
import fact_checker.agents.baseline as baseline_mod
import fact_checker.agents.scout as scout_mod
import fact_checker.search as search_module
from fact_checker.datasets.fever import FeverClaim
from fact_checker.schemas import Adjudication, Alignment, AuditReport, SearchQueries

GOLD = {1: "SUPPORTS", 2: "REFUTES", 3: "NOT_ENOUGH_INFO"}


@pytest.fixture
def stub_models(monkeypatch):
    """Grounded path always answers SUPPORTS; baseline is confidently wrong."""
    for module in (scout_mod, auditor_mod, adjudicator_mod, baseline_mod):
        monkeypatch.setattr(module, "build_llm", lambda *a, **k: object())

    monkeypatch.setattr(
        scout_mod,
        "structured_call",
        lambda *a, **k: SearchQueries(queries=["q1", "q2"]),
    )
    monkeypatch.setattr(
        search_module,
        "_raw_search",
        lambda query, max_results: [
            {"title": "T", "href": "https://x.example", "body": "evidence body"}
        ],
    )
    monkeypatch.setattr(
        auditor_mod,
        "structured_call",
        lambda *a, **k: AuditReport(
            alignments=[
                Alignment(source_id="doc_1", verdict="SUPPORTS", justification="j")
            ]
        ),
    )
    monkeypatch.setattr(
        adjudicator_mod,
        "structured_call",
        lambda *a, **k: Adjudication(
            claim="c",
            final_verdict="SUPPORTS",
            confidence=0.8,
            detailed_reasoning="grounded",
            cited_sources=["doc_1"],
        ),
    )
    monkeypatch.setattr(
        baseline_mod,
        "structured_call",
        lambda *a, **k: Adjudication(
            claim="c",
            final_verdict="REFUTES",
            confidence=0.95,
            detailed_reasoning="from memory",
            cited_sources=[],
        ),
    )


@pytest.fixture
def claims() -> list[FeverClaim]:
    return [
        FeverClaim(id=str(i), claim=f"claim {i}", label=label) for i, label in GOLD.items()
    ]


class TestCostEstimate:
    def test_scales_with_claim_count(self) -> None:
        one = harness.estimate_cost("claude-opus-5", 1)
        ten = harness.estimate_cost("claude-opus-5", 10)
        assert one is not None and ten == pytest.approx(one * 10)

    def test_cheaper_models_cost_less(self) -> None:
        opus = harness.estimate_cost("claude-opus-5", 10)
        haiku = harness.estimate_cost("claude-haiku-4-5", 10)
        assert haiku < opus

    def test_unknown_model_returns_none(self) -> None:
        assert harness.estimate_cost("some-other-model", 10) is None


class TestRunBenchmark:
    def test_writes_one_jsonl_record_per_claim(
        self, stub_models, claims, tmp_path: Path
    ) -> None:
        out = tmp_path / "run.jsonl"
        from fact_checker import Settings

        results = harness.run_benchmark(
            claims, Settings(api_key="k"), out, workers=2
        )

        assert len(results) == 3
        lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
        assert len(lines) == 3
        assert {rec["id"] for rec in lines} == {"1", "2", "3"}

    def test_records_both_systems(self, stub_models, claims, tmp_path: Path) -> None:
        from fact_checker import Settings

        out = tmp_path / "run.jsonl"
        results = harness.run_benchmark(claims, Settings(api_key="k"), out, workers=1)

        record = results[0]
        assert record["multi"]["verdict"] == "SUPPORTS"
        assert record["multi"]["cited_sources"] == ["doc_1"]
        assert record["baseline"]["verdict"] == "REFUTES"
        assert record["baseline"]["confidence"] == pytest.approx(0.95)
        assert record["errors"] == []

    def test_agent_failure_is_captured_not_raised(
        self, stub_models, claims, tmp_path: Path, monkeypatch
    ) -> None:
        from fact_checker import Settings
        from fact_checker.llm import AgentError

        def explode(*args, **kwargs):
            raise AgentError("provider exploded")

        monkeypatch.setattr(adjudicator_mod, "structured_call", explode)

        results = harness.run_benchmark(
            claims, Settings(api_key="k"), tmp_path / "run.jsonl", workers=1
        )

        assert all(r["multi"] is None for r in results)
        assert all("provider exploded" in r["errors"][0] for r in results)
        # The baseline still ran, so the run is not a total loss.
        assert all(r["baseline"] is not None for r in results)


class TestResume:
    def test_existing_results_are_loaded_by_id(self, tmp_path: Path) -> None:
        out = tmp_path / "run.jsonl"
        out.write_text(
            "\n".join(
                [
                    json.dumps({"id": 1, "gold": "SUPPORTS"}),
                    "",
                    "{bad json}",
                    json.dumps({"id": 2, "gold": "REFUTES"}),
                ]
            ),
            encoding="utf-8",
        )
        existing = harness.load_existing(out)
        assert set(existing) == {"1", "2"}

    def test_missing_file_is_empty(self, tmp_path: Path) -> None:
        assert harness.load_existing(tmp_path / "nope.jsonl") == {}


class TestScoring:
    def _records(self) -> list[dict]:
        return [
            {
                "id": 1,
                "gold": "SUPPORTS",
                "multi": {"verdict": "SUPPORTS", "confidence": 0.8, "seconds": 1.0},
                "baseline": {"verdict": "REFUTES", "confidence": 0.95, "seconds": 0.2},
                "errors": [],
            },
            {
                "id": 2,
                "gold": "REFUTES",
                "multi": {"verdict": "REFUTES", "confidence": 0.9, "seconds": 1.0},
                "baseline": {"verdict": "REFUTES", "confidence": 0.9, "seconds": 0.2},
                "errors": [],
            },
        ]

    def test_scores_each_system(self) -> None:
        multi, base, extra = harness.score_records(self._records())
        assert multi is not None and base is not None
        assert multi.accuracy == pytest.approx(1.0)
        assert base.accuracy == pytest.approx(0.5)
        assert extra["paired"] == 2
        assert extra["multi_only_correct"] == 1
        assert extra["baseline_only_correct"] == 0

    def test_baseline_overconfidence_is_visible(self) -> None:
        _, base, _ = harness.score_records(self._records())
        assert base.overconfidence > 0
        assert base.mean_confidence_when_wrong == pytest.approx(0.95)

    def test_failed_records_are_excluded_from_their_own_denominator(self) -> None:
        records = self._records()
        records.append(
            {"id": 3, "gold": "SUPPORTS", "multi": None, "baseline": None, "errors": ["x"]}
        )
        multi, base, extra = harness.score_records(records)
        assert multi.n == 2  # not 3
        assert base.n == 2
        assert extra["errors"] == 1

    def test_report_prints_without_error(self, capsys) -> None:
        harness.print_report(self._records())
        out = capsys.readouterr().out
        assert "FEVER BENCHMARK" in out
        assert "HEAD-TO-HEAD" in out
        assert "Caveat" in out


class TestCli:
    def test_score_only_reads_existing_results(self, tmp_path: Path, capsys) -> None:
        out = tmp_path / "run.jsonl"
        out.write_text(
            json.dumps(
                {
                    "id": 1,
                    "gold": "SUPPORTS",
                    "multi": {"verdict": "SUPPORTS", "confidence": 0.9},
                    "baseline": {"verdict": "SUPPORTS", "confidence": 0.9},
                    "errors": [],
                }
            ),
            encoding="utf-8",
        )
        code = harness.main(["--score-only", "--out", str(out)])
        assert code == 0
        assert "FEVER BENCHMARK" in capsys.readouterr().out

    def test_score_only_on_missing_file_fails_cleanly(self, tmp_path: Path) -> None:
        assert harness.main(["--score-only", "--out", str(tmp_path / "x.jsonl")]) == 1

    def test_dry_run_needs_no_api_key(self, tmp_path: Path, capsys, monkeypatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        fever = tmp_path / "shared_task_dev.jsonl"
        fever.write_text(
            "\n".join(
                json.dumps(
                    {"id": i, "label": label, "claim": f"claim {i}", "evidence": []}
                )
                for i, label in enumerate(
                    ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"] * 3
                )
            ),
            encoding="utf-8",
        )
        # With no key, the default backend is local, so no cost is quoted.
        code = harness.main(["--n", "1", "--dry-run", "--cache-dir", str(tmp_path)])
        assert code == 0
        out = capsys.readouterr().out
        assert "Sample preview" in out
        assert "no API cost" in out

        # Asking for the hosted backend explicitly does quote a cost. The
        # model is named rather than left to default, because FACTCHECK_MODEL
        # in the ambient environment legitimately overrides the default and
        # would otherwise decide what this test is measuring.
        code = harness.main(
            [
                "--n", "1", "--dry-run",
                "--cache-dir", str(tmp_path),
                "--provider", "anthropic",
                "--model", "claude-opus-5",
            ]
        )
        assert code == 0
        assert "Estimated cost" in capsys.readouterr().out

    def test_the_configured_model_is_honoured(self, tmp_path, monkeypatch, capsys):
        """FACTCHECK_MODEL must reach the run, not be silently replaced.

        The harness previously defaulted the model without consulting the
        environment, so a configured model was ignored while Settings
        honoured it -- the CLI and the library disagreed about which model a
        run had used.
        """
        monkeypatch.setenv("FACTCHECK_MODEL", "qwen2.5:3b")
        (tmp_path / "shared_task_dev.jsonl").write_text(
            "\n".join(
                json.dumps({"id": i, "claim": f"claim {i}", "label": "SUPPORTS"})
                for i in range(3)
            ),
            encoding="utf-8",
        )
        code = harness.main(
            ["--n", "1", "--dry-run", "--cache-dir", str(tmp_path), "--provider", "ollama"]
        )
        assert code == 0
        assert "qwen2.5:3b" in capsys.readouterr().out

    def test_an_explicit_model_beats_the_environment(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("FACTCHECK_MODEL", "qwen2.5:3b")
        (tmp_path / "shared_task_dev.jsonl").write_text(
            json.dumps({"id": 0, "claim": "claim 0", "label": "SUPPORTS"}),
            encoding="utf-8",
        )
        code = harness.main(
            [
                "--n", "1", "--dry-run", "--cache-dir", str(tmp_path),
                "--provider", "anthropic", "--model", "claude-opus-5",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "claude-opus-5" in out
        assert "qwen2.5:3b" not in out


class TestLabelGroupReporting:
    """NEI must be reported separately from the decidable claims.

    On FEVER, NEI is corpus-relative, so a single blended accuracy can hide an
    inverted result: a system that over-predicts NEI scores well overall while
    being worse on the claims that are actually decidable.
    """

    def _records(self) -> list[dict]:
        # multi-agent: perfect on NEI, poor on S/R.  baseline: the reverse.
        def rec(i, gold, multi, base):
            return {
                "id": str(i), "gold": gold,
                "multi": {"verdict": multi, "confidence": 0.8, "seconds": 1.0},
                "baseline": {"verdict": base, "confidence": 0.9, "seconds": 0.2},
                "errors": [],
            }

        return [
            rec(1, "SUPPORTS", "NOT_ENOUGH_INFO", "SUPPORTS"),
            rec(2, "REFUTES", "NOT_ENOUGH_INFO", "REFUTES"),
            rec(3, "NOT_ENOUGH_INFO", "NOT_ENOUGH_INFO", "SUPPORTS"),
            rec(4, "NOT_ENOUGH_INFO", "NOT_ENOUGH_INFO", "REFUTES"),
        ]

    def test_splits_decidable_from_nei(self, capsys) -> None:
        harness.print_report(self._records(), "fever")
        out = capsys.readouterr().out

        assert "REPORTED BY LABEL GROUP" in out
        assert "decidable (S/R)" in out
        assert "NEI only" in out

    def test_blended_accuracy_hides_the_inversion(self) -> None:
        records = self._records()
        multi, base, _ = harness.score_records(records)

        # Blended: both 50% -- indistinguishable.
        assert multi.accuracy == pytest.approx(0.5)
        assert base.accuracy == pytest.approx(0.5)

        from fact_checker.metrics import score_subset

        gold = [r["gold"] for r in records]
        m = [r["multi"]["verdict"] for r in records]
        b = [r["baseline"]["verdict"] for r in records]
        conf = [0.8] * len(records)

        m_dec = score_subset("m", gold, m, conf, ["SUPPORTS", "REFUTES"])
        b_dec = score_subset("b", gold, b, conf, ["SUPPORTS", "REFUTES"])
        # Split apart: the baseline is strictly better where it matters.
        assert m_dec.accuracy == pytest.approx(0.0)
        assert b_dec.accuracy == pytest.approx(1.0)

    def test_fever_caveat_only_shows_for_fever(self, capsys) -> None:
        harness.print_report(self._records(), "fever")
        assert "2017 Wikipedia dump" in capsys.readouterr().out

        harness.print_report(self._records(), "averitec")
        out = capsys.readouterr().out
        assert "2017 Wikipedia dump" not in out
        assert "real-world and open-web" in out

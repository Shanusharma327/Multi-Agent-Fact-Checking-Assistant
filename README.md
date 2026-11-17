# MAFCA — A Multi-Agent Fact-Checking Assistant

## 1. Problem statement

A single large language model asked both to **find evidence** and to
**reason over it** leaves no intermediate record. Three things follow from
that, and all three are structural rather than accidental:

1. A retrieval failure and a reasoning failure are **indistinguishable**
   from the model's output. If it answers wrongly, you cannot tell whether
   it found the wrong documents or misread the right ones.
2. Any citation it emits is **generated text**, not a reference to a
   document that was actually retrieved.
3. Constraints written into a prompt **cannot be verified** from the
   result. If the prompt says "use only the supplied passage" and the
   model does not, nothing in the pipeline notices.

MAFCA splits fact-checking across **three agents** that communicate
through typed, validated schemas, and enforces the boundaries in code
rather than in prose:

| | Agent | Does |
|---|---|---|
| 1 | **Evidence Scout** | Writes two neutral queries, retrieves passages, suppresses duplicates, filters out the claim's own fact-check article |
| 2 | **Alignment Auditor** | Labels *each passage separately* and quotes the words that decided the label |
| 3 | **Synthesis Adjudicator** | Writes the reasoning trace; the verdict itself is **computed** from the audit, not asked for |

The claim under test is about **auditability, not accuracy**. Two
properties hold by construction, so they do not depend on the model
complying and do not depend on the sample size either:

- **Deterministic post-conditions.** Audit completeness and citation
  validity are enforced in code after generation. Every emitted citation
  names a passage that was genuinely retrieved *and* that the
  verification stage labelled in the verdict's direction.
- **Computed aggregation.** The adjudication policy runs as arithmetic
  over the recorded audit — passage weights scaled by evidence strength,
  a weighted score per candidate label, and a Laplace-smoothed posterior
  for the confidence. The policy cannot return a verdict the audit does
  not support.

---

## 2. Dataset

**AVeriTeC** (Schlichtkrull et al.), real-world claims with annotated
evidence. Chosen because every annotated answer carries a **source URL**,
which makes retrieval quality measurable on its own terms rather than
inferred backwards from the final verdict — unlike benchmarks whose gold
evidence is a sentence index into a fixed Wikipedia snapshot.

The corpora are **not committed**. Fetch them with:

    python data/download_datasets.py

See [data/README.md](data/README.md) for sizes, sources and licensing.

**You do not need the corpus to reproduce the results.** Retrieval used a
live web service whose results change daily, so a fresh retrieval would
not reproduce these passages. What makes the experiment replayable is
committed instead:

- `cache/av250.jsonl` — every passage each condition saw
- `results/av250/`, `results/av250_oracle/` — per-claim verdicts

### Scope, stated plainly

The protocol was designed for **250 claims**. The search service degraded
partway through retrieval and the run was **halted at 90 claims** rather
than looped, because continuing against a failing endpoint fills the cache
with empty evidence sets indistinguishable from genuine "nothing found"
results. Everything reported here is that first **n = 90** draw. It is a
pilot, and Section 4 says what that does and does not permit.

---

## 3. Setup and run steps

### Requirements

- Python 3.12
- [Ollama](https://ollama.com) running locally — no API key, no cost

### Install

    git clone <this-repo>
    cd <this-repo>

    python -m venv .venv
    .venv\Scripts\activate          # Windows
    source .venv/bin/activate       # macOS / Linux

    pip install -r requirements.txt

    ollama pull qwen2.5:3b

    copy .env.example .env          # Windows
    cp .env.example .env            # macOS / Linux

### Reproduce every reported number — offline, no model calls

    python src/analysis/regenerate.py

This replays the committed cache and per-claim verdicts and rebuilds
**every** figure, table and quantity into `results/`. It runs no inference
and touches no network. Expect 13 figures, 8 tables and 3 macro files.
Console output from a real run is in [logs/regenerate.log](logs/regenerate.log).

### Rebuild the results report

    python src/analysis/build_results_doc.py

Writes [results/MAFCA_Results.pdf](results/MAFCA_Results.pdf) and
[results/MAFCA_Results.docx](results/MAFCA_Results.docx): 16 pages holding
every plot, every table, the real numeric values and the actual console
output, each with a caption naming the script that produced it. Both
documents are assembled from `results/macros/` by the same table
specification, so they cannot disagree with each other or with the run,
and a macro name that no longer exists raises rather than printing a
blank cell.

### Run the tests

    python -m pytest -q

434 tests, no network needed. Output from a real run is in
[logs/pytest.log](logs/pytest.log).

### Run the system yourself

    python src/app.py                 # check a single claim, interactively
    python src/evaluate.py            # MAFCA against the single-LLM baseline
    python src/run_conditions.py      # the full experiment (see --help)

`run_conditions.py` separates retrieval from everything downstream, which
is what makes the comparisons controlled: retrieval happens **once**, into
a cache, and every condition then replays that cache. Two conditions
differ in the thing under test and in nothing else — the run aborts if
that ever stops being true.

### Reproducibility

| | |
|---|---|
| Decoding | `temperature=0.0` — greedy, no sampling |
| RNG seed | `20240101` throughout (claim sampling, bootstrap, the random baseline) |
| Attribution bootstrap | seed `20261018`, 10,000 resamples |
| Model | `qwen2.5:3b`, digest `357c53fb659c`, `Q4_K_M` |

Nothing in `results/macros/` is written by hand. Every reported quantity
is a generated macro; editing one would make the write-up disagree with
the run, so the rule is to regenerate instead. A quantity that is cited
but no longer emitted becomes an undefined reference and fails the build —
which is how a stale number gets caught rather than printed.

---

## 4. Results summary

All figures below are generated into `results/macros/`, on the same
**n = 90** claims, every condition seeing identical cached passages so
comparisons are paired claim for claim.

### Conditions

| | |
|---|---|
| **B0** | Single LLM, no retrieval — the control |
| **B1** | Single-call RAG on identical evidence — isolates *decomposition* from *evidence access* |
| **B4** | Three agents, verdict **computed** from the audit — the primary configuration |
| **MAFCA** | Three agents, verdict **as reported by the agent** — ablation |

### Classification

| Condition | Accuracy (%) | Macro-F1 (%) | Decidable (%) | Abstention (%) |
|---|---|---|---|---|
| B0: single LLM | 42.2 | 39.8 | 24.6 | 73.3 |
| B1: single-call RAG | 44.4 | 42.0 | 27.9 | 66.7 |
| **B4: computed** | **48.9** | **47.4** | **34.4** | 61.1 |
| MAFCA: agent-reported | 47.8 | 45.2 | 29.5 | 67.8 |
| *Majority class* | *40.0* | *19.0* | ***59.0*** | *0.0* |

### No accuracy difference is demonstrated

B4 leads B0 by **+6.7** points. The 95% confidence interval is
**[-5.6, +18.9]** — it **contains zero**, and the exact McNemar p-value is
**0.377**.

At n = 90 the smallest paired difference this sample could resolve is
**17.6 points**. The observed gain is well inside that, so **a study this
size could not have settled the question either way**. That is a limit of
the experiment, not evidence of equivalence.

Two further cautions, both visible in the table above:

- **Abstention flatters accuracy.** B0 answers `NOT_ENOUGH_INFO` on 73.3%
  of claims. Declining is cheap, and accuracy rewards it whenever the gold
  label agrees — which is why the decidable subset is reported alongside.
- **On the decidable subset, a majority-class predictor reaches 59.0%
  against B4's 34.4%.** No result here demonstrates competent
  discrimination between `SUPPORTS` and `REFUTES`.

### What the decomposition does buy

The per-passage record makes every error **attributable to a stage**. A
single-call system emits no such record, so this analysis **cannot be run
on B1 at all** — which is the clearest thing the decomposition is worth.

| Stage | Live retrieval | Oracle evidence |
|---|---|---|
| Retrieval | **60.9%** [46.7, 74.5] | 5.7% [0.0, 14.6] |
| Verification | 39.1% [25.5, 53.3] | **94.3%** [85.4, 100.0] |
| Aggregation | **0.0%** [0.0, 0.0] | **0.0%** [0.0, 0.0] |

*95% bootstrap intervals over 10,000 resamples of the claims.*

Two readings. **Retrieval dominates** under live conditions; supplying the
corpus' own annotated evidence inverts the profile, leaving a verification
stage mishandling evidence it was handed. And **no error is attributable
to aggregation** in either condition — the computed policy cannot return a
verdict the audit does not support.

### Grounding, and its limits

- Validated citations accompany **38.9%** of B4's verdicts; B0 supplies
  **none**, having retrieved nothing to cite.
- Exact-URL Recall@k against the annotated evidence is **1.7%**. The
  system rarely recovers the sources the annotator used — it retrieves
  contemporaneous material instead, which the ungrounded baseline lacks
  entirely.
- Computing the confidence rather than eliciting it does **not** improve
  calibration past the baseline (ECE 0.280 against 0.192).

### Cost

Three agents. At replay time two model calls per claim are billed to the
verification and adjudication stages; the Evidence Scout's call is made
once during cache construction and is recorded there, so it does not
recur on replay.

### Conclusion

**The contribution is auditability, not accuracy.** Decomposition shows no
demonstrated accuracy advantage over either control at this scale. What it
provides is a record that makes failure diagnosable and citations
checkable — properties that hold by construction rather than by
measurement.

### Limitations

Pilot scale (n = 90 of a planned 250, cut short by a search outage); one
3B model; live retrieval that is not reproducible across dates; citation
support checked structurally, not semantically.

### The next test, stated so it can fail

Run `qwen2.5:7b` on the oracle condition over the same cached claims. If
the verification share of errors falls materially from **94.3%**, the
capacity hypothesis is supported; if it does not, that explanation is
wrong.

---

## 5. Repository layout

    README.md               this file
    requirements.txt        dependencies
    .env.example            the exact runtime config the reported run used

    src/
      fact_checker/         the implementation
        pipeline.py           the three-agent pipeline  <- start here
        agents/scout.py       Agent 1 - query generation and retrieval
        agents/auditor.py     Agent 2 - per-passage verification
        agents/adjudicator.py Agent 3 - verdict, confidence, citations
        agents/baseline.py    the single-LLM control (B0)
        agents/rag_baseline.py  the single-call RAG control (B1)
        aggregation.py        the verdict computed from the audit
        conditions.py         every experimental condition
        search.py             retrieval, leakage filter, circuit breaker
        cache.py              passage cache and the evidence fingerprint
        schemas.py            the typed contracts between agents
      analysis/             every table, figure and generated number
        regenerate.py         one entry point that rebuilds all of it
      app.py, evaluate.py, run_conditions.py, run_tier1.py

    tests/                  434 tests
    data/                   download script; corpora not committed
    cache/av250.jsonl       the passages every condition saw
    results/
      av250/                per-claim verdicts, per condition
      av250_oracle/         the same claims with annotated evidence
      figures/ tables/ macros/   regenerated artefacts
    checkpoints/            why there are no weights here
    logs/                   captured output from real runs
    notebooks/              MAFCA_reproduce.ipynb

**Start with [src/fact_checker/pipeline.py](src/fact_checker/pipeline.py)** —
it wires the three agents together. If you open one file to see the
system, open that one.

Note on paths: run every command from the repository root, not from
inside `src/`.

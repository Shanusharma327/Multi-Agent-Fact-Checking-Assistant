/**
 * Builds the 12-slide M.Tech project presentation.
 *
 *   node build_deck.js
 *
 * Every figure in this deck is a measured value from the project -- see
 * REPORT.md and analysis/dataset_stats.json.
 */

const pptxgen = require("pptxgenjs");

// Output name; override with: node build_deck.js MyDeck.pptx
const OUT = process.argv[2] || "Multi_Agent_Fact_Checker_PPE.pptx";

// --- palette -------------------------------------------------------------
const CRIMSON = "A51C30"; // Somaiya crimson
const NAVY = "1E2761";
const INK = "22262E";
const MUTED = "6B7280";
const WHITE = "FFFFFF";
const TINT = "F2F5FA";
const TINT_WARM = "FDF3F4";
const SUP = "2E7D32";
const REF = "C62828";
const NEI = "EF6C00";

const HEAD = "Cambria";
const BODY = "Calibri";

const W = 13.333;
const H = 7.5;
const M = 0.62; // page margin

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "K. J. Somaiya School of Engineering";
pres.title = "Multi-Agent Fact-Checking Assistant";

// --- helpers -------------------------------------------------------------

/** Standard content-slide header. */
function header(slide, title, kicker) {
  slide.addText(title, {
    x: M, y: 0.30, w: W - 2 * M, h: 0.62,
    fontFace: HEAD, fontSize: 30, bold: true, color: NAVY,
    align: "left", isTextBox: true, margin: 0,
  });
  if (kicker) {
    slide.addText(kicker, {
      x: M, y: 0.92, w: W - 2 * M, h: 0.3,
      fontFace: BODY, fontSize: 13, italic: true, color: CRIMSON,
      align: "left", isTextBox: true, margin: 0,
    });
  }
  slide.addText("K. J. Somaiya School of Engineering  |  Dept. of IT / AI-DS", {
    x: M, y: H - 0.62, w: 7.5, h: 0.28,
    fontFace: BODY, fontSize: 9.5, color: MUTED,
    align: "left", isTextBox: true, margin: 0,
  });
}

/** Rounded card with a subtle shadow (fresh options object each call). */
function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h,
    fill: { color: fill || TINT },
    line: { color: fill || TINT, width: 0 },
    rectRadius: 0.09,
    shadow: { type: "outer", angle: 90, blur: 7, offset: 0.05, color: "B9C0CC", opacity: 0.45 },
  });
}

/** Numbered circle used as the deck's repeating motif. */
function numberBadge(slide, x, y, d, label, fill) {
  slide.addShape(pres.ShapeType.ellipse, {
    x, y, w: d, h: d,
    fill: { color: fill || CRIMSON },
    line: { color: fill || CRIMSON, width: 0 },
  });
  slide.addText(label, {
    x, y, w: d, h: d,
    fontFace: HEAD, fontSize: 15, bold: true, color: WHITE,
    align: "center", valign: "middle", isTextBox: true, margin: 0,
  });
}

/** Big stat callout. */
function stat(slide, x, y, w, value, label, color) {
  slide.addText(value, {
    x, y, w, h: 0.72,
    fontFace: HEAD, fontSize: 34, bold: true, color: color || CRIMSON,
    align: "center", isTextBox: true, margin: 0,
  });
  slide.addText(label, {
    x, y: y + 0.70, w, h: 0.48,
    fontFace: BODY, fontSize: 11, color: MUTED,
    align: "center", isTextBox: true, margin: 0,
  });
}

// =========================================================================
// SLIDE 1 — TITLE
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };

  // Motif: oversized translucent verification circles.
  s.addShape(pres.ShapeType.ellipse, {
    x: 10.4, y: -1.25, w: 4.6, h: 4.6,
    fill: { color: CRIMSON, transparency: 72 }, line: { width: 0 },
  });
  s.addShape(pres.ShapeType.ellipse, {
    x: 11.7, y: 4.5, w: 3.0, h: 3.0,
    fill: { color: "FFFFFF", transparency: 90 }, line: { width: 0 },
  });

  s.addText("F.Y. M Tech  ·  Project Presentation", {
    x: M, y: 1.28, w: 9.6, h: 0.4,
    fontFace: BODY, fontSize: 15, color: "9FB0D8", charSpacing: 2,
    isTextBox: true, margin: 0,
  });

  s.addText("Multi-Agent Fact-Checking Assistant", {
    x: M, y: 1.85, w: 10.2, h: 1.5,
    fontFace: HEAD, fontSize: 44, bold: true, color: WHITE,
    isTextBox: true, margin: 0,
  });

  s.addText(
    "Retrieval-Grounded Claim Verification with a Calibration-Aware Benchmark",
    {
      x: M, y: 3.28, w: 10.2, h: 0.55,
      fontFace: BODY, fontSize: 17, color: "E4A0AC", italic: true,
      isTextBox: true, margin: 0,
    }
  );

  s.addShape(pres.ShapeType.rect, {
    x: M, y: 4.05, w: 2.1, h: 0.035,
    fill: { color: CRIMSON }, line: { width: 0 },
  });

  s.addText(
    [
      { text: "Presented by", options: { fontSize: 12, color: "9FB0D8", breakLine: true } },
      { text: "[Your Name]", options: { fontSize: 19, bold: true, color: WHITE, breakLine: true } },
      { text: "Roll No: [__________]", options: { fontSize: 13, color: "C9D3EA" } },
    ],
    { x: M, y: 4.42, w: 6.5, h: 1.3, fontFace: BODY, isTextBox: true, margin: 0 }
  );

  s.addText(
    "Department of IT / AI-DS,  K. J. Somaiya School of Engineering,  Vidyavihar, Mumbai",
    {
      x: M, y: H - 0.72, w: W - 2 * M, h: 0.35,
      fontFace: BODY, fontSize: 11.5, color: "8FA0C8",
      isTextBox: true, margin: 0,
    }
  );

  s.addNotes(
    "Good morning. I'm presenting a Multi-Agent Fact-Checking Assistant. The core " +
    "idea: a single LLM answers fact-checking questions from memory and cannot tell " +
    "you whether it actually knows something or merely recalls it. We decompose the " +
    "task across three agents grounded in live web evidence, and benchmark against a " +
    "single-LLM control."
  );
}

// =========================================================================
// SLIDE 2 — TABLE OF CONTENTS
// =========================================================================
{
  const s = pres.addSlide();
  header(s, "Table of Contents");

  const items = [
    "Introduction", "Problem Statement", "Objectives",
    "Dataset Description", "System Architecture", "Methodology — The 3 Agents",
    "Evaluation Metrics", "Results & Analysis", "Key Finding",
    "Limitations", "Future Scope", "Conclusion",
  ];
  const colors = [CRIMSON, NAVY, SUP, NEI, CRIMSON, NAVY, SUP, NEI, CRIMSON, NAVY, SUP, NEI];

  const cw = 5.75, ch = 0.72, gap = 0.22;
  items.forEach((label, i) => {
    const col = i < 6 ? 0 : 1;
    const row = i % 6;
    const x = M + col * (cw + 0.6);
    const y = 1.42 + row * (ch + gap);

    card(s, x, y, cw, ch, col === 0 ? TINT : TINT_WARM);
    numberBadge(s, x + 0.16, y + 0.13, 0.46, String(i + 1), colors[i]);
    s.addText(label, {
      x: x + 0.78, y, w: cw - 0.95, h: ch,
      fontFace: BODY, fontSize: 15, bold: true, color: INK,
      valign: "middle", isTextBox: true, margin: 0,
    });
  });

  s.addNotes("Twelve sections: motivation and problem, then how the system works, then how we measured it and what we found.");
}

// =========================================================================
// SLIDE 3 — INTRODUCTION
// =========================================================================
{
  const s = pres.addSlide();
  header(s, "Introduction", "Why a language model alone cannot be trusted to fact-check");

  s.addText(
    [
      { text: "Large Language Models answer from ", options: {} },
      { text: "parametric memory", options: { bold: true, color: CRIMSON } },
      { text: " — what they memorised during training, not from documents.", options: { breakLine: true } },
    ],
    { x: M, y: 1.45, w: 7.0, h: 0.75, fontFace: BODY, fontSize: 15, color: INK, isTextBox: true, margin: 0 }
  );

  const bullets = [
    { t: "They cannot distinguish \"I recall this\" from \"this is documented\".", b: true },
    { t: "They fail hardest on facts that changed after training, half-remembered details, and popular myths.", b: true },
    { t: "Critically, these failures arrive with no drop in confidence.", b: true },
    { t: "This project grounds every verdict in live web evidence and measures whether the confidence is honest.", b: true },
  ];
  s.addText(
    bullets.map((x, i) => ({
      text: x.t,
      options: { bullet: true, breakLine: i !== bullets.length - 1, paraSpaceAfter: 10 },
    })),
    { x: M, y: 2.30, w: 7.0, h: 2.6, fontFace: BODY, fontSize: 14.5, color: INK, isTextBox: true, margin: 0 }
  );

  // Right: contrast cards
  const rx = 8.0, rw = 4.7;
  card(s, rx, 1.45, rw, 1.95, TINT_WARM);
  s.addText("WITHOUT RETRIEVAL", {
    x: rx + 0.28, y: 1.62, w: rw - 0.56, h: 0.3,
    fontFace: BODY, fontSize: 10.5, bold: true, color: REF, charSpacing: 1.2, isTextBox: true, margin: 0,
  });
  s.addText("“Humans use 10% of their brains.”", {
    x: rx + 0.28, y: 1.94, w: rw - 0.56, h: 0.38,
    fontFace: BODY, fontSize: 12.5, italic: true, color: INK, isTextBox: true, margin: 0,
  });
  s.addText("REFUTES  ·  95% confident  ·  0 sources", {
    x: rx + 0.28, y: 2.36, w: rw - 0.56, h: 0.32,
    fontFace: BODY, fontSize: 12, bold: true, color: REF, isTextBox: true, margin: 0,
  });
  s.addText("Confidence reflects fluency, not evidence.", {
    x: rx + 0.28, y: 2.76, w: rw - 0.56, h: 0.42,
    fontFace: BODY, fontSize: 11, color: MUTED, isTextBox: true, margin: 0,
  });

  card(s, rx, 3.60, rw, 2.10, TINT);
  s.addText("WITH RETRIEVAL", {
    x: rx + 0.28, y: 3.78, w: rw - 0.56, h: 0.3,
    fontFace: BODY, fontSize: 10.5, bold: true, color: SUP, charSpacing: 1.2, isTextBox: true, margin: 0,
  });
  s.addText("Same claim, judged against 4 retrieved snippets", {
    x: rx + 0.28, y: 4.10, w: rw - 0.56, h: 0.38,
    fontFace: BODY, fontSize: 12.5, italic: true, color: INK, isTextBox: true, margin: 0,
  });
  s.addText("REFUTES  ·  90% confident  ·  cites doc_1, doc_2", {
    x: rx + 0.28, y: 4.52, w: rw - 0.56, h: 0.32,
    fontFace: BODY, fontSize: 12, bold: true, color: SUP, isTextBox: true, margin: 0,
  });
  s.addText("Every citation validated against real retrieved evidence.", {
    x: rx + 0.28, y: 4.92, w: rw - 0.56, h: 0.6,
    fontFace: BODY, fontSize: 11, color: MUTED, isTextBox: true, margin: 0,
  });

  s.addNotes("The point is not that the baseline is always wrong. It is that you cannot tell when it is wrong, because the confidence looks identical either way.");
}

// =========================================================================
// SLIDE 4 — PROBLEM STATEMENT
// =========================================================================
{
  const s = pres.addSlide();
  header(s, "Problem Statement");

  // Statement
  card(s, M, 1.32, W - 2 * M, 1.12, TINT_WARM);
  s.addText("STATEMENT", {
    x: M + 0.3, y: 1.45, w: 3.0, h: 0.3,
    fontFace: BODY, fontSize: 11, bold: true, color: CRIMSON, charSpacing: 1.4, isTextBox: true, margin: 0,
  });
  s.addText(
    "A single LLM estimates p(y | claim ; θ) from its parameters, not p(y | claim, Evidence) from " +
    "documents. The challenge is a verifier that returns a verdict, a calibrated confidence, and " +
    "citations grounded in retrieved evidence.",
    {
      x: M + 0.3, y: 1.74, w: W - 2 * M - 0.6, h: 0.62,
      fontFace: BODY, fontSize: 13.5, color: INK, isTextBox: true, margin: 0,
    }
  );

  // Challenges
  s.addText("KEY CHALLENGES", {
    x: M, y: 2.62, w: 5.0, h: 0.3,
    fontFace: BODY, fontSize: 11, bold: true, color: NAVY, charSpacing: 1.4, isTextBox: true, margin: 0,
  });

  const ch = [
    ["P1", "Unverifiable provenance", "With no retrieval, every citation is generated text. Hallucinated citations are the expected behaviour, not an anomaly.", REF],
    ["P2", "Temporal staleness", "Parameters freeze at the training cutoff. Stale answers are returned confidently, and the failure is silent.", NEI],
    ["P3", "Confidence decoupling", "Confidence tracks fluency and training-data frequency, not evidential support — so myths are asserted confidently.", CRIMSON],
    ["P4", "Confirmation bias", "An agent that both writes queries and knows the claim will search to confirm it, biasing evidence before inference.", NAVY],
  ];
  const cw = 2.96, gap = 0.16;
  ch.forEach((c, i) => {
    const x = M + i * (cw + gap);
    card(s, x, 2.96, cw, 2.10, TINT);
    numberBadge(s, x + 0.20, 3.14, 0.44, c[0], c[3]);
    s.addText(c[1], {
      x: x + 0.20, y: 3.68, w: cw - 0.4, h: 0.34,
      fontFace: BODY, fontSize: 12.5, bold: true, color: INK, isTextBox: true, margin: 0,
    });
    s.addText(c[2], {
      x: x + 0.20, y: 4.02, w: cw - 0.4, h: 0.95,
      fontFace: BODY, fontSize: 10.5, color: MUTED, isTextBox: true, margin: 0,
    });
  });

  // Solution
  card(s, M, 5.26, W - 2 * M, 1.02, "EAF3EC");
  s.addText("SOLUTION", {
    x: M + 0.3, y: 5.38, w: 3.0, h: 0.3,
    fontFace: BODY, fontSize: 11, bold: true, color: SUP, charSpacing: 1.4, isTextBox: true, margin: 0,
  });
  s.addText(
    "Decompose verification into three narrowly-scoped agents with schema contracts enforced in code — " +
    "so each failure mode becomes structurally impossible, not merely discouraged by prompt wording.",
    {
      x: M + 0.3, y: 5.66, w: W - 2 * M - 0.6, h: 0.5,
      fontFace: BODY, fontSize: 13, color: INK, isTextBox: true, margin: 0,
    }
  );

  s.addNotes("The key insight is on the solution line: a prompt saying 'use only the provided evidence' is unenforceable. Split the task across agents and each boundary becomes a place where violations are detectable in code.");
}

// =========================================================================
// SLIDE 5 — OBJECTIVES
// =========================================================================
{
  const s = pres.addSlide();
  header(s, "Objectives");

  const objs = [
    ["1", "Build a modular 3-agent pipeline", "Evidence retrieval → per-snippet inference → verdict synthesis", CRIMSON],
    ["2", "Implement a single-LLM baseline control", "Identical output schema, so the comparison is 1:1 on every field", NAVY],
    ["3", "Enforce deterministic post-conditions", "Validate citations and reconcile audits in code, not by prompt trust", SUP],
    ["4", "Evaluate on classification and calibration", "Accuracy, precision, recall, F1-score, plus ECE and Brier score", NEI],
    ["5", "Benchmark on two corpora", "FEVER for comparability, AVeriTeC for real-world validity", CRIMSON],
    ["6", "Deliver an interactive dashboard", "Streamlit UI running on free local inference — no API cost", NAVY],
  ];

  const rh = 0.78, gap = 0.16;
  objs.forEach((o, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = M + col * 6.15;
    const y = 1.45 + row * (rh + gap + 0.52);
    const w = 5.85;

    numberBadge(s, x, y + 0.10, 0.52, o[0], o[3]);
    s.addText(o[1], {
      x: x + 0.70, y: y + 0.02, w: w - 0.75, h: 0.36,
      fontFace: BODY, fontSize: 14.5, bold: true, color: INK, isTextBox: true, margin: 0,
    });
    s.addText(o[2], {
      x: x + 0.70, y: y + 0.38, w: w - 0.75, h: 0.62,
      fontFace: BODY, fontSize: 12, color: MUTED, isTextBox: true, margin: 0,
    });
  });

  s.addNotes("Objective 3 is what makes this more than a prompting exercise: the guarantees are in code, so they hold even when the model misbehaves.");
}

// =========================================================================
// SLIDE 6 — DATASET DESCRIPTION
// =========================================================================
{
  const s = pres.addSlide();
  header(s, "Dataset Description", "Two corpora, two purposes — they are expected to disagree");

  const rows = [
    [
      { text: "Property", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
      { text: "FEVER (2018)", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
      { text: "AVeriTeC (2023)", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
    ],
    ["Purpose", "Comparability — canonical benchmark", "Validity — real-world claims"],
    ["Claims used", "19,998  (dev split)", "462  of 500"],
    ["SUPPORTS", "6,666   (33.3%)", "122   (26.4%)"],
    ["REFUTES", "6,666   (33.3%)", "305   (66.0%)"],
    ["NOT ENOUGH INFO", "6,666   (33.3%)", "35   (7.6%)"],
    ["Mean claim length", "8.3 words", "17.1 words"],
    ["Evidence source", "Wikipedia, June 2017 dump", "Open web (QA pairs)"],
    ["Claim origin", "Mutated Wikipedia sentences", "Facebook, Twitter, Instagram"],
  ];

  s.addTable(rows, {
    x: M, y: 1.45, w: 7.35,
    colW: [2.15, 2.65, 2.55],
    rowH: 0.34,
    fontFace: BODY, fontSize: 11, color: INK,
    border: { type: "solid", color: "DCE1EA", pt: 0.5 },
    fill: { color: WHITE },
    valign: "middle",
    autoPage: false,
  });

  // Chart: label share comparison
  s.addChart(
    pres.ChartType.bar,
    [
      { name: "FEVER", labels: ["SUPPORTS", "REFUTES", "NEI"], values: [33.3, 33.3, 33.3] },
      { name: "AVeriTeC", labels: ["SUPPORTS", "REFUTES", "NEI"], values: [26.4, 66.0, 7.6] },
    ],
    {
      x: 8.15, y: 1.55, w: 4.55, h: 2.75,
      barDir: "col", barGrouping: "clustered",
      showTitle: true, title: "Label share (%)", titleFontSize: 12, titleColor: NAVY, titleFontFace: BODY,
      chartColors: [NAVY, CRIMSON],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 9, dataLabelColor: INK,
      showLegend: true, legendPos: "b", legendFontSize: 10,
      catAxisLabelColor: MUTED, catAxisLabelFontSize: 10,
      valAxisLabelColor: MUTED, valAxisLabelFontSize: 9, valAxisMaxVal: 80,
      valGridLine: { color: "E8EBF0", size: 0.5 },
      catGridLine: { style: "none" },
    }
  );

  card(s, 8.15, 4.52, 4.55, 1.72, TINT_WARM);
  s.addText("⚠  WHY TWO DATASETS?", {
    x: 8.40, y: 4.66, w: 4.05, h: 0.3,
    fontFace: BODY, fontSize: 11, bold: true, color: CRIMSON, charSpacing: 1, isTextBox: true, margin: 0,
  });
  s.addText(
    "FEVER is balanced and standard — but it was published in 2018, so it may sit inside the " +
    "models' training data, biasing the ungrounded baseline. AVeriTeC is real circulating " +
    "misinformation and post-dates it by five years.",
    {
      x: 8.40, y: 4.96, w: 4.05, h: 1.18,
      fontFace: BODY, fontSize: 10.5, color: INK, isTextBox: true, margin: 0,
    }
  );

  s.addNotes("FEVER is perfectly balanced at 6,666 per label, so accuracy is directly interpretable against a 33% chance baseline. AVeriTeC is two-thirds REFUTES, because fact-checkers mostly publish on claims they intend to debunk - so there we must lead with macro-F1.");
}

// =========================================================================
// SLIDE 7 — SYSTEM ARCHITECTURE
// =========================================================================
{
  const s = pres.addSlide();
  header(s, "System Architecture", "Three scoped agents, one shared output schema");

  // Heights chosen so the three stacked cards clear the callout band at y=5.55.
  const bx = 0.95, bw = 3.35, bh = 1.15, vgap = 0.22;
  const agents = [
    ["AGENT 1", "Evidence Scout", "2 neutral queries → DuckDuckGo\nSnippets as doc_1 … doc_n", CRIMSON],
    ["AGENT 2", "Alignment Auditor", "Closed-world NLI\nOne label per snippet", NAVY],
    ["AGENT 3", "Synthesis Adjudicator", "Weighs conflicts\nVerdict + confidence", SUP],
  ];

  agents.forEach((a, i) => {
    const y = 1.48 + i * (bh + vgap);
    card(s, bx, y, bw, bh, TINT);
    s.addShape(pres.ShapeType.ellipse, {
      x: bx + 0.22, y: y + 0.20, w: 0.48, h: 0.48,
      fill: { color: a[3] }, line: { width: 0 },
    });
    s.addText(String(i + 1), {
      x: bx + 0.22, y: y + 0.20, w: 0.48, h: 0.48,
      fontFace: HEAD, fontSize: 16, bold: true, color: WHITE,
      align: "center", valign: "middle", isTextBox: true, margin: 0,
    });
    s.addText(a[0], {
      x: bx + 0.84, y: y + 0.14, w: bw - 1.02, h: 0.22,
      fontFace: BODY, fontSize: 9, bold: true, color: a[3], charSpacing: 1.2, isTextBox: true, margin: 0,
    });
    s.addText(a[1], {
      x: bx + 0.84, y: y + 0.34, w: bw - 1.02, h: 0.30,
      fontFace: BODY, fontSize: 14, bold: true, color: INK, isTextBox: true, margin: 0,
    });
    s.addText(a[2], {
      x: bx + 0.84, y: y + 0.64, w: bw - 1.02, h: 0.44,
      fontFace: BODY, fontSize: 10, color: MUTED, isTextBox: true, margin: 0,
    });

    if (i < 2) {
      s.addShape(pres.ShapeType.downArrow, {
        x: bx + bw / 2 - 0.09, y: y + bh + 0.02, w: 0.18, h: 0.18,
        fill: { color: "C2C9D6" }, line: { width: 0 },
      });
    }
  });

  // Claim in
  s.addText("CLAIM", {
    x: bx, y: 1.16, w: bw, h: 0.26,
    fontFace: BODY, fontSize: 10, bold: true, color: MUTED, align: "center", charSpacing: 2, isTextBox: true, margin: 0,
  });

  // Right: baseline + verdict
  card(s, 5.10, 1.48, 3.05, 1.28, TINT_WARM);
  s.addText("CONTROL", {
    x: 5.32, y: 1.62, w: 2.6, h: 0.26,
    fontFace: BODY, fontSize: 9.5, bold: true, color: REF, charSpacing: 1.2, isTextBox: true, margin: 0,
  });
  s.addText("Single-LLM Baseline", {
    x: 5.32, y: 1.88, w: 2.6, h: 0.3,
    fontFace: BODY, fontSize: 14, bold: true, color: INK, isTextBox: true, margin: 0,
  });
  s.addText("Zero-shot, no search,\nidentical output schema", {
    x: 5.32, y: 2.18, w: 2.6, h: 0.5,
    fontFace: BODY, fontSize: 10.5, color: MUTED, isTextBox: true, margin: 0,
  });

  card(s, 5.10, 3.06, 3.05, 1.80, "EAF3EC");
  s.addText("OUTPUT SCHEMA", {
    x: 5.32, y: 3.20, w: 2.6, h: 0.26,
    fontFace: BODY, fontSize: 9.5, bold: true, color: SUP, charSpacing: 1.2, isTextBox: true, margin: 0,
  });
  s.addText(
    [
      { text: "final_verdict", options: { bold: true, breakLine: true } },
      { text: "confidence  (0.0 – 1.0)", options: { bold: true, breakLine: true } },
      { text: "detailed_reasoning", options: { bold: true, breakLine: true } },
      { text: "cited_sources[ ]", options: { bold: true } },
    ],
    { x: 5.32, y: 3.48, w: 2.6, h: 1.22, fontFace: "Courier New", fontSize: 11, color: INK, isTextBox: true, margin: 0 }
  );

  // Tech stack
  card(s, 8.55, 1.48, 4.15, 3.38, TINT);
  s.addText("TECHNOLOGY STACK", {
    x: 8.80, y: 1.64, w: 3.7, h: 0.28,
    fontFace: BODY, fontSize: 10, bold: true, color: NAVY, charSpacing: 1.2, isTextBox: true, margin: 0,
  });
  const stack = [
    ["Language", "Python 3.12"],
    ["Orchestration", "LangChain"],
    ["Schemas", "Pydantic v2"],
    ["Interface", "Streamlit"],
    ["Inference", "Ollama  (qwen2.5)"],
    ["Retrieval", "DuckDuckGo (ddgs)"],
    ["Testing", "pytest — 143 tests"],
  ];
  stack.forEach((row, i) => {
    const y = 2.00 + i * 0.39;
    s.addText(row[0], {
      x: 8.80, y, w: 1.55, h: 0.32,
      fontFace: BODY, fontSize: 11, color: MUTED, isTextBox: true, margin: 0,
    });
    s.addText(row[1], {
      x: 10.35, y, w: 2.15, h: 0.32,
      fontFace: BODY, fontSize: 11, bold: true, color: INK, isTextBox: true, margin: 0,
    });
  });

  card(s, M, 5.55, W - 2 * M, 0.92, TINT_WARM);
  s.addText(
    "The single controlled variable is access to retrieved evidence. Both paths emit the identical " +
    "schema, which is what makes the comparison exactly 1:1 rather than impressionistic.",
    {
      x: M + 0.30, y: 5.74, w: W - 2 * M - 0.6, h: 0.56,
      fontFace: BODY, fontSize: 13, italic: true, color: INK, isTextBox: true, margin: 0,
    }
  );

  s.addNotes("Note the baseline is not a straw man. It gets the same claim and produces the same fields. The only thing it lacks is retrieval.");
}

// =========================================================================
// SLIDE 8 — METHODOLOGY
// =========================================================================
{
  const s = pres.addSlide();
  header(s, "Methodology — The Three Agents", "Each agent's scope is narrowed so one failure mode becomes impossible");

  const cols = [
    ["Evidence Scout", "Writes 2 neutral queries, retrieves and de-duplicates snippets.",
      "Forbidden from judging the claim — blocks confirmation bias. Query 2 must take an independent angle.", CRIMSON],
    ["Alignment Auditor", "Labels each snippet independently against the claim.",
      "Closed-world rule — only that snippet's text may justify a label, and it must quote exact words or numbers.", NAVY],
    ["Synthesis Adjudicator", "Weighs the labels into one verdict plus a calibrated confidence.",
      "Specificity beats volume — one decisive snippet outranks four vague ones. Absence is never refutation.", SUP],
  ];

  const cw = 3.94, gap = 0.20;
  cols.forEach((c, i) => {
    const x = M + i * (cw + gap);
    card(s, x, 1.52, cw, 2.72, TINT);
    numberBadge(s, x + 0.24, 1.72, 0.50, String(i + 1), c[3]);
    s.addText(c[0], {
      x: x + 0.24, y: 2.32, w: cw - 0.48, h: 0.32,
      fontFace: BODY, fontSize: 15, bold: true, color: INK, isTextBox: true, margin: 0,
    });
    s.addText(c[1], {
      x: x + 0.24, y: 2.66, w: cw - 0.48, h: 0.52,
      fontFace: BODY, fontSize: 11.5, color: MUTED, isTextBox: true, margin: 0,
    });
    s.addText("CONSTRAINT", {
      x: x + 0.24, y: 3.16, w: cw - 0.48, h: 0.24,
      fontFace: BODY, fontSize: 9, bold: true, color: c[3], charSpacing: 1.2, isTextBox: true, margin: 0,
    });
    s.addText(c[2], {
      x: x + 0.24, y: 3.40, w: cw - 0.48, h: 0.76,
      fontFace: BODY, fontSize: 10.5, color: INK, isTextBox: true, margin: 0,
    });
  });

  s.addText("ENFORCED IN CODE, NOT BY PROMPT", {
    x: M, y: 4.44, w: 8.0, h: 0.30,
    fontFace: BODY, fontSize: 11, bold: true, color: NAVY, charSpacing: 1.4, isTextBox: true, margin: 0,
  });

  const guards = [
    ["Audit reconciliation", "Hallucinated doc_99 dropped; missing snippets backfilled as NOT_ENOUGH_INFO."],
    ["Citation validation", "Every cited ID checked against genuinely retrieved evidence."],
    ["Baseline citations discarded", "It has no sources, so anything it cites is fabricated by construction."],
  ];
  guards.forEach((g, i) => {
    const x = M + i * (cw + gap);
    card(s, x, 4.78, cw, 1.44, TINT_WARM);
    s.addText(g[0], {
      x: x + 0.24, y: 4.94, w: cw - 0.48, h: 0.32,
      fontFace: BODY, fontSize: 12, bold: true, color: CRIMSON, isTextBox: true, margin: 0,
    });
    s.addText(g[1], {
      x: x + 0.24, y: 5.26, w: cw - 0.48, h: 0.82,
      fontFace: BODY, fontSize: 10.5, color: INK, isTextBox: true, margin: 0,
    });
  });

  s.addNotes("The bottom row is the contribution. An LLM that violates a schema still returns syntactically valid output, so we repair it deterministically rather than trusting compliance.");
}

// =========================================================================
// SLIDE 9 — EVALUATION METRICS
// =========================================================================
{
  const s = pres.addSlide();
  header(s, "Evaluation Metrics", "Classification quality and calibration, reported together");

  const rows = [
    [
      { text: "Metric", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
      { text: "Formula", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
      { text: "What it tells us", options: { bold: true, color: WHITE, fill: { color: NAVY } } },
    ],
    ["Accuracy", "correct / total", "Overall correctness"],
    ["Precision", "TP / (TP + FP)", "“When it says REFUTES, is it right?”"],
    ["Recall", "TP / (TP + FN)", "“Does it catch all the false claims?”"],
    ["F1-Score", "2PR / (P + R)", "Harmonic mean — punishes imbalance"],
    ["Macro-F1", "mean of per-label F1", "Every class counts equally"],
    ["ECE", "Σ (|Bₘ|/N) · |acc − conf|", "Is the stated confidence honest?"],
    ["Brier score", "mean (κ − correct)²", "Rewards honest uncertainty"],
  ];

  s.addTable(rows, {
    x: M, y: 1.52, w: 7.6,
    colW: [1.70, 2.45, 3.45],
    rowH: 0.40,
    fontFace: BODY, fontSize: 11.5, color: INK,
    border: { type: "solid", color: "DCE1EA", pt: 0.5 },
    fill: { color: WHITE },
    valign: "middle",
    autoPage: false,
  });

  card(s, 8.45, 1.52, 4.25, 2.35, TINT_WARM);
  s.addText("WHY CALIBRATION MATTERS", {
    x: 8.70, y: 1.68, w: 3.8, h: 0.3,
    fontFace: BODY, fontSize: 11, bold: true, color: CRIMSON, charSpacing: 1.1, isTextBox: true, margin: 0,
  });
  s.addText(
    "Accuracy alone cannot express the point of this project. Two systems with equal accuracy are " +
    "not equivalent if one of them asserts 95% confidence while being wrong.",
    {
      x: 8.70, y: 2.00, w: 3.8, h: 1.0,
      fontFace: BODY, fontSize: 12, color: INK, isTextBox: true, margin: 0,
    }
  );
  s.addText(
    "A verifier that is 70% accurate and knows it is more useful than one that is 75% accurate and " +
    "claims 95% — only the first supports deferring to a human.",
    {
      x: 8.70, y: 2.96, w: 3.8, h: 0.8,
      fontFace: BODY, fontSize: 11, italic: true, color: MUTED, isTextBox: true, margin: 0,
    }
  );

  card(s, 8.45, 4.05, 4.25, 2.17, TINT);
  s.addText("REPORTING PROTOCOL", {
    x: 8.70, y: 4.20, w: 3.8, h: 0.3,
    fontFace: BODY, fontSize: 11, bold: true, color: NAVY, charSpacing: 1.1, isTextBox: true, margin: 0,
  });
  s.addText(
    [
      { text: "Decidable  ", options: { bold: true, color: SUP } },
      { text: "— gold is SUPPORTS or REFUTES", options: { breakLine: true } },
      { text: "NEI only  ", options: { bold: true, color: NEI } },
      { text: "— reported separately", options: { breakLine: true } },
    ],
    { x: 8.70, y: 4.52, w: 3.8, h: 0.72, fontFace: BODY, fontSize: 11.5, color: INK, isTextBox: true, margin: 0 }
  );
  s.addText(
    "On FEVER, NEI means “absent from a 2017 Wikipedia snapshot” — not “unverifiable in principle”. " +
    "Blending it into one accuracy conflates system quality with benchmark construction.",
    {
      x: 8.70, y: 5.22, w: 3.8, h: 0.9,
      fontFace: BODY, fontSize: 10.5, color: MUTED, isTextBox: true, margin: 0,
    }
  );

  s.addNotes("The last box on the right is what sets up the next slide, where the blended metric turns out to report the opposite of the label-group result.");
}

// =========================================================================
// SLIDE 10 — RESULTS & KEY FINDING
// =========================================================================
{
  const s = pres.addSlide();
  header(s, "Results & Key Finding", "Preliminary run — n = 20 claims, FEVER dev, qwen2.5:3b (local inference)");

  // Stat strip
  const sx = M, sw = 2.92, sgap = 0.14;
  const stats = [
    ["65.0%", "Multi-agent accuracy", CRIMSON],
    ["55.0%", "Baseline accuracy", NAVY],
    ["+27.2%", "Baseline overconfidence", REF],
    ["+0.0%", "Multi-agent overconfidence", SUP],
  ];
  stats.forEach((st, i) => {
    const x = sx + i * (sw + sgap);
    card(s, x, 1.50, sw, 1.30, TINT);
    stat(s, x, 1.62, sw, st[0], st[1], st[2]);
  });

  // Chart: the inversion
  s.addChart(
    pres.ChartType.bar,
    [
      { name: "Multi-agent", labels: ["Decidable (S/R)", "NEI only"], values: [46.2, 100.0] },
      { name: "Baseline", labels: ["Decidable (S/R)", "NEI only"], values: [69.2, 28.6] },
    ],
    {
      x: M, y: 3.00, w: 6.15, h: 3.05,
      barDir: "col", barGrouping: "clustered",
      showTitle: true, title: "Accuracy by label group (%)", titleFontSize: 12.5,
      titleColor: NAVY, titleFontFace: BODY,
      chartColors: [CRIMSON, NAVY],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelColor: INK,
      showLegend: true, legendPos: "b", legendFontSize: 10.5,
      catAxisLabelColor: MUTED, catAxisLabelFontSize: 10.5,
      valAxisLabelColor: MUTED, valAxisLabelFontSize: 9, valAxisMaxVal: 115,
      valGridLine: { color: "E8EBF0", size: 0.5 },
      catGridLine: { style: "none" },
    }
  );

  // The finding
  card(s, 7.05, 3.00, 5.65, 3.05, TINT_WARM);
  s.addText("🔍  THE HEADLINE INVERTS", {
    x: 7.30, y: 3.18, w: 5.15, h: 0.34,
    fontFace: HEAD, fontSize: 16, bold: true, color: CRIMSON, isTextBox: true, margin: 0,
  });
  s.addText(
    "The pipeline's +10-point advantage comes entirely from NOT_ENOUGH_INFO. On claims that are " +
    "actually decidable, the ungrounded baseline is better by 23 points.",
    {
      x: 7.30, y: 3.58, w: 5.15, h: 0.85,
      fontFace: BODY, fontSize: 12.5, color: INK, isTextBox: true, margin: 0,
    }
  );
  s.addText(
    "A conservative auditor that over-predicts NEI is rewarded by a corpus whose NEI class is defined " +
    "by absence from a 2017 snapshot — not by the system being better at verification.",
    {
      x: 7.30, y: 4.44, w: 5.15, h: 0.85,
      fontFace: BODY, fontSize: 11.5, color: MUTED, isTextBox: true, margin: 0,
    }
  );
  s.addText(
    "➡  Never report one blended accuracy on FEVER — it can say the opposite of the truth.",
    {
      x: 7.30, y: 5.28, w: 5.15, h: 0.42,
      fontFace: BODY, fontSize: 12, bold: true, color: CRIMSON, isTextBox: true, margin: 0,
    }
  );
  s.addText(
    "⚠  n = 20 → ±20 point confidence interval. Indicative, not conclusive.",
    {
      x: 7.30, y: 5.68, w: 5.15, h: 0.30,
      fontFace: BODY, fontSize: 10.5, italic: true, color: MUTED, isTextBox: true, margin: 0,
    }
  );

  s.addNotes("This is the most important slide. Most projects report a number. We report a number and then show why that number is misleading, which is why the reporting protocol on the previous slide exists. And we are explicit that n=20 is too small to conclude from.");
}

// =========================================================================
// SLIDE 11 — LIMITATIONS & FUTURE SCOPE
// =========================================================================
{
  const s = pres.addSlide();
  header(s, "Limitations & Future Scope");

  s.addText("LIMITATIONS", {
    x: M, y: 1.30, w: 6.0, h: 0.30,
    fontFace: BODY, fontSize: 11.5, bold: true, color: REF, charSpacing: 1.4, isTextBox: true, margin: 0,
  });

  const lims = [
    ["No large-scale evaluation yet", "n = 20; a defensible result needs n ≥ 300 with confidence intervals and multiple seeds."],
    ["Corpus / label mismatch", "FEVER's NEI means “absent from 2017 Wikipedia”. We search the live web, so the system can be right about the world and score as wrong."],
    ["Weak local model inflates the gap", "A 3B model is disproportionately worse at the baseline task, widening the apparent advantage."],
    ["No cross-snippet reasoning", "Snippets are judged independently, so evidence that jointly entails a claim is missed."],
    ["Verbalised confidence", "Not derived from logits — observed NOT_ENOUGH_INFO returned at 90% confidence."],
  ];
  lims.forEach((l, i) => {
    const y = 1.66 + i * 0.94;
    s.addShape(pres.ShapeType.ellipse, {
      x: M, y: y + 0.07, w: 0.17, h: 0.17,
      fill: { color: REF }, line: { width: 0 },
    });
    s.addText(l[0], {
      x: M + 0.32, y, w: 5.75, h: 0.30,
      fontFace: BODY, fontSize: 12.5, bold: true, color: INK, isTextBox: true, margin: 0,
    });
    s.addText(l[1], {
      x: M + 0.32, y: y + 0.29, w: 5.75, h: 0.60,
      fontFace: BODY, fontSize: 10.5, color: MUTED, isTextBox: true, margin: 0,
    });
  });

  s.addText("FUTURE SCOPE", {
    x: 7.25, y: 1.30, w: 5.45, h: 0.30,
    fontFace: BODY, fontSize: 11.5, bold: true, color: SUP, charSpacing: 1.4, isTextBox: true, margin: 0,
  });

  const fut = [
    ["Scale the evaluation", "Run n ≥ 300 across both corpora with bootstrap confidence intervals and paired significance testing.", SUP],
    ["Upgrade the model", "Move to qwen2.5:7b or a frontier model so the grounded-vs-ungrounded gap is measured fairly.", NAVY],
    ["Extend the architecture", "Iterative retrieval, source-credibility weighting in the aggregation, and cross-snippet entailment.", CRIMSON],
    ["Deployment", "Compound-claim decomposition, caching, and multilingual claim support.", NEI],
  ];
  fut.forEach((f, i) => {
    const y = 1.66 + i * 1.18;
    card(s, 7.25, y, 5.45, 1.02, TINT);
    numberBadge(s, 7.47, y + 0.24, 0.44, String(i + 1), f[2]);
    s.addText(f[0], {
      x: 8.05, y: y + 0.14, w: 4.45, h: 0.30,
      fontFace: BODY, fontSize: 12.5, bold: true, color: INK, isTextBox: true, margin: 0,
    });
    s.addText(f[1], {
      x: 8.05, y: y + 0.43, w: 4.45, h: 0.52,
      fontFace: BODY, fontSize: 10.5, color: MUTED, isTextBox: true, margin: 0,
    });
  });

  s.addNotes("Being explicit about limitations is deliberate. The honest position is that the contribution right now is the architecture and the evaluation protocol, not a performance claim.");
}

// =========================================================================
// SLIDE 12 — CONCLUSION
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: NAVY };

  s.addShape(pres.ShapeType.ellipse, {
    x: 11.0, y: 5.0, w: 3.6, h: 3.6,
    fill: { color: CRIMSON, transparency: 78 }, line: { width: 0 },
  });

  s.addText("Conclusion", {
    x: M, y: 0.42, w: 8.0, h: 0.72,
    fontFace: HEAD, fontSize: 34, bold: true, color: WHITE, isTextBox: true, margin: 0,
  });

  const points = [
    ["Built a working 3-agent retrieval-grounded pipeline", "with an ungrounded control emitting an identical output schema."],
    ["Enforced grounding structurally", "hallucinated citations and incomplete audits are repaired in code, not merely discouraged by prompting."],
    ["Showed calibration must be reported with accuracy", "the baseline was +27.2% overconfident while being 10 points less accurate."],
    ["Found that a blended FEVER accuracy can invert the true result", "motivating label-group-separated reporting."],
    ["Validated with 143 automated tests", "running end-to-end on free local inference — no API cost."],
  ];

  points.forEach((p, i) => {
    const y = 1.35 + i * 0.83;
    s.addShape(pres.ShapeType.ellipse, {
      x: M, y: y + 0.02, w: 0.46, h: 0.46,
      fill: { color: CRIMSON }, line: { width: 0 },
    });
    s.addText(String(i + 1), {
      x: M, y: y + 0.02, w: 0.46, h: 0.46,
      fontFace: HEAD, fontSize: 14, bold: true, color: WHITE,
      align: "center", valign: "middle", isTextBox: true, margin: 0,
    });
    s.addText(
      [
        { text: p[0] + "  ", options: { bold: true, color: WHITE } },
        { text: p[1], options: { color: "B9C6E4" } },
      ],
      { x: M + 0.64, y: y - 0.02, w: 7.05, h: 0.74, fontFace: BODY, fontSize: 12.5, isTextBox: true, margin: 0 }
    );
  });

  // Bigger picture panel
  s.addShape(pres.ShapeType.roundRect, {
    x: 8.25, y: 1.35, w: 4.45, h: 3.55,
    fill: { color: "FFFFFF", transparency: 92 },
    line: { color: "4A5B92", width: 1 },
    rectRadius: 0.10,
  });
  s.addText("THE BIGGER PICTURE", {
    x: 8.55, y: 1.58, w: 3.85, h: 0.3,
    fontFace: BODY, fontSize: 11, bold: true, color: "E4A0AC", charSpacing: 1.3, isTextBox: true, margin: 0,
  });
  s.addText(
    "Grounded verification with honest confidence enables selective deference — the system can say " +
    "“I am not sure, ask a human.”",
    {
      x: 8.55, y: 1.96, w: 3.85, h: 1.05,
      fontFace: BODY, fontSize: 14, color: WHITE, isTextBox: true, margin: 0,
    }
  );
  s.addText(
    "An overconfident model never can — and that, not raw accuracy, is what determines whether a " +
    "verdict can be safely acted upon.",
    {
      x: 8.55, y: 3.05, w: 3.85, h: 1.0,
      fontFace: BODY, fontSize: 12, italic: true, color: "B9C6E4", isTextBox: true, margin: 0,
    }
  );
  s.addText("143 tests  ·  2 corpora  ·  0 API cost", {
    x: 8.55, y: 4.24, w: 3.85, h: 0.4,
    fontFace: BODY, fontSize: 12.5, bold: true, color: WHITE, isTextBox: true, margin: 0,
  });

  s.addText("Thank You", {
    x: M, y: 5.62, w: 6.0, h: 0.72,
    fontFace: HEAD, fontSize: 30, bold: true, color: WHITE, isTextBox: true, margin: 0,
  });
  s.addText("Department of IT / AI-DS,  K. J. Somaiya School of Engineering,  Vidyavihar, Mumbai", {
    x: M, y: 6.42, w: 9.0, h: 0.35,
    fontFace: BODY, fontSize: 11, color: "8FA0C8", isTextBox: true, margin: 0,
  });

  s.addNotes("To close: the defensible contribution is the architecture, the enforcement mechanisms, and the evaluation protocol - including the finding that the obvious way to score this task is misleading. Happy to take questions.");
}

pres.writeFile({ fileName: OUT })
  .then((f) => console.log("Written:", f))
  .catch((e) => { console.error(e); process.exit(1); });

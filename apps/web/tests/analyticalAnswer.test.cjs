// Run from apps/web: node --test tests/analyticalAnswer.test.cjs
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const ts = require("typescript");
const React = require("react");
const { renderToStaticMarkup } = require("react-dom/server");

function load(relative, dependencies = {}) {
  const filename = path.join(__dirname, "..", relative);
  const { outputText } = ts.transpileModule(fs.readFileSync(filename, "utf8"), {
    compilerOptions: {module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.React, esModuleInterop: true, target: ts.ScriptTarget.ES2022},
  });
  const module_ = {exports: {}};
  new Function("require", "module", "exports", outputText)(id => id in dependencies ? dependencies[id] : require(id), module_, module_.exports);
  return module_.exports;
}
const presentation = load("src/lib/analyticalAnswer.ts");
const { presentAnalyticalAnswer: present } = presentation;
const Component = load("src/components/StructuredAnalysisAnswer.tsx", {
  "../lib/analyticalAnswer": presentation,
  "../lib/api": new Proxy({}, {get: () => () => {throw Error("A render test must not call the API");}}),
  "lucide-react": new Proxy({}, {get: () => () => null}),
}).default;

function fixture(kind = "DESCRIPTIVE", numbers = {}) {
  return {
    id: "synthetic-test", question: "What is the total revenue by region?", status: "COMPLETED", verdict: "OBSERVED",
    dataset_scope: ["sales"], provenance: {requested_dataset_ids: ["sales"]},
    analyst_result: {kind, numbers, descriptive: true, headline: "Recorded headline", details: [], caveats: []},
    evidence: [{id: "e1", validation_status: "VERIFIED"}], evidence_verifications: [{evidence_id: "e1", status: "PASSED"}],
    analysis_plan: {estimand: {target: "revenue"}, semantics: {grouping_columns: ["region"]}},
  };
}
const table = {West: {value: 1350, n: 6}, East: {value: 920, n: 6}, North: {value: 745, n: 6}, South: {value: 705, n: 6}};
const sales = () => fixture("DESCRIPTIVE", {operation: "SUM", target_column: "revenue", grouping_columns: ["region"], filters: [], population_row_count: 24, table});
const render = data => renderToStaticMarkup(React.createElement(Component, {analysis: data}));

test("totals answer explains complete breakdown, share and gap without changing values", () => {
  const data = sales(), before = JSON.stringify(data);
  const answer = present(data, "VERIFIED", null);
  assert.equal(answer.supported, true);
  assert.match(answer.answer, /Total revenue.*3,720/);
  assert.match(answer.answer, /West.*highest.*1,350.*South.*lowest.*705/);
  assert.match(answer.findings.join(" "), /36\.3%/);
  assert.match(answer.findings.join(" "), /645/);
  assert.ok(answer.calculation.includes("Calculation: SUM, grouped by region."));
  assert.equal(JSON.stringify(data), before);
});

test("ranking uses returned values, handles lowest-first direction and ties", () => {
  const data = fixture("RANKING", {aggregation: "sum", table, canonical_contract: {ranking_direction: "ASC"}});
  data.question = "Which region has the lowest total revenue?";
  const answer = present(data, "VERIFIED", null);
  assert.match(answer.answer, /South.*lowest.*705.*West.*highest.*1,350/);
  data.analyst_result.numbers.table = {A: {value: 5}, B: {value: 5}};
  assert.match(present(data, "VERIFIED", null).answer, /same.*5/);
});

test("means are never summed and invalid or missing values are never replaced with zero", () => {
  const data = sales();
  data.analyst_result.numbers.operation = "MEAN";
  const answer = present(data, "VERIFIED", null);
  assert.doesNotMatch(answer.answer, /3,720|Total revenue/);
  assert.doesNotMatch(answer.findings.join(" "), /36\.3%/);
  for (const value of [null, undefined, NaN, Infinity, "1350"]) {
    data.analyst_result.numbers.table = {West: {value}, South: {value: 705}};
    const invalid = present(data, "VERIFIED", null);
    assert.equal(invalid.supported, false);
    assert.match(invalid.answer, /missing or invalid/);
  }
});

test("comparison uses the named subject minus reference, not a highest-vs-rest comparison", () => {
  const data = fixture("GROUP_COMPARISON", {means: {West: 225, South: 117.5, East: 153.33}, subject: "South", reference: "West", diff: -107.5, ci_low: -150, ci_high: -65, p_value: 0.002});
  data.question = "Is average revenue lower in South than West?";
  const answer = present(data, "VERIFIED", null);
  assert.match(answer.answer, /117\.5 for South versus 225 for West.*107\.5 lower.*South minus West/);
  assert.deepEqual(answer.table.rows, [["South", "117.5"], ["West", "225"]]);
  assert.match(answer.findings.join(" "), /-150 to -65/);
  data.analyst_result.numbers.diff = 20;
  assert.equal(present(data, "VERIFIED", null).supported, false);
});

test("rate comparisons use percentage points; omnibus tests never imply a tested pair", () => {
  const data = fixture("RATE_COMPARISON", {rates: {A: 0.2, B: 0.1}, subject: "A", reference: "B", rate_diff: 0.1, p_value: 0.3});
  assert.match(present(data, "VERIFIED", null).answer, /10 percentage points higher/);
  const multi = fixture("GROUP_COMPARISON", {means: {A: 10, B: 20, C: 30}, anova_p: 0.01});
  const answer = present(multi, "VERIFIED", null);
  assert.match(answer.answer, /10 \(A\).*30 \(C\)/);
  assert.match(answer.limitations.join(" "), /does not identify which individual pairs differ/);
});

test("association presentation never repeats unsupported multi-predictor significance claims", () => {
  const data = fixture("ASSOCIATION", {correlations: {x: 0.7, z: -0.1}, pearson_p: 0.001, adequate_power: true, robust: true});
  data.question = "Is x or z related to revenue?";
  data.analyst_result.details = ["Both predictors are statistically significant."];
  const answer = present(data, "VERIFIED", null);
  assert.match(answer.answer, /x.*largest returned absolute correlation/);
  assert.match(answer.findings.join(" "), /z: r = -0\.1/);
  assert.doesNotMatch(answer.answer + answer.findings.join(" "), /statistically significant|adequate power|robust/i);
  assert.match(answer.limitations.join(" "), /Per-predictor significance cannot/);
});

test("missing and invalid p-values cannot become significant or a p-value of zero", () => {
  for (const p of [undefined, null, NaN, Infinity, -0.1, 1.1, "0.001"]) {
    const answer = present(fixture("ASSOCIATION", {pearson_r: 0.5, pearson_p: p}), "VERIFIED", null);
    assert.match(answer.limitations.join(" "), /valid p-value was not recorded/);
    assert.doesNotMatch(answer.findings.join(" "), /p </);
  }
});

test("trends and projections use recorded magnitudes, with missing uncertainty disclosed", () => {
  const data = fixture("TREND", {slope_per_period: -5, periods: 12, grain: "month", start_level: 100, end_level: 45, pct_change: -0.55, slope_p: 0.1});
  data.question = "What is the revenue trend?";
  assert.match(present(data, "VERIFIED", null).answer, /-5 per month.*12 periods/);
  assert.match(present(data, "VERIFIED", null).findings.join(" "), /-55\.0%/);
  data.question = "Forecast revenue next month";
  assert.equal(present(data, "VERIFIED", null).supported, false);
  data.analyst_result.numbers.run_rate_projection = {projected_value: 40, pi_low: 20, pi_high: 60};
  assert.match(present(data, "VERIFIED", null).findings.join(" "), /20 to 60/);
  delete data.analyst_result.numbers.run_rate_projection.pi_low;
  assert.match(present(data, "VERIFIED", null).limitations.join(" "), /uncertainty is unavailable/);
});

test("why questions do not turn a descriptive result or adjustment into a cause", () => {
  const data = sales();
  data.question = "Why is West revenue higher?";
  const answer = present(data, "VERIFIED", null);
  assert.match(answer.answer, /does not establish why/);
  assert.doesNotMatch(answer.answer, /West causes|because West/);
  const period = fixture("PERIOD_CHANGE", {previous: 900, current: 700, period: "February 2025", top_contributor: "West", top_contribution_share: 0.6});
  period.question = "Why did revenue drop in February 2025?";
  const change = present(period, "VERIFIED", null);
  assert.match(change.answer, /decreased from 900 to 700.*does not establish a cause/);
  assert.match(change.findings.join(" "), /West accounts for 60\.0%/);
  assert.match(change.findings.join(" "), /not a causal effect/);
});

test("rendered failures preserve badges, scope warning and evidence, and withhold confident answers", () => {
  const data = sales();
  assert.match(render(data), /Total revenue.*3,720/);
  assert.match(render(data), /Verification: VERIFIED/);
  data.provenance.requested_dataset_ids = ["different-source"];
  let html = render(data);
  assert.match(html, /Consistency: FAILED/);
  assert.match(html, /saved answer scope does not match/);
  assert.doesNotMatch(html, /Total revenue.*3,720/);
  data.provenance.requested_dataset_ids = ["sales"];
  data.evidence[0].validation_status = "FAILED";
  assert.match(render(data), /Verification: FAILED/);
  data.evidence[0].validation_status = "VERIFIED";
  data.evidence_verifications = [];
  html = render(data);
  assert.match(html, /Verification: UNVERIFIED/);
  assert.match(html, /A verified answer is not available/);
  data.status = "FAILED";
  data.direct_answer = "Could not complete: execution plan deviates from approved contract.";
  assert.match(render(data), /analysis failed/);
  assert.match(render(data), /execution plan deviates from approved contract/);
});

test("refused, qualified and insufficient results never acquire an answer from raw values", () => {
  const data = sales();
  for (const verification of ["REFUSED", "QUALIFIED", "UNVERIFIED", "FAILED"]) assert.equal(present(data, verification, null).supported, false);
  data.verdict = "INSUFFICIENT_DATA";
  assert.match(present(data, "VERIFIED", null).answer, /does not support a conclusive answer/);
});

test("a blocked why question discloses absent causal evidence and does not claim the plan ran", () => {
  const data = sales();
  data.question = "Why is revenue different by region?";
  data.analyst_result = null;
  data.verdict = "INCONCLUSIVE";
  data.direct_answer = "Investigation inconclusive: verified evidence is missing.";
  data.analysis_plan.estimand.requested_aggregation = "SUM";
  data.evidence_verifications = [];
  const answer = present(data, "QUALIFIED", null);
  assert.equal(answer.supported, false);
  assert.match(answer.answer, /does not establish a cause/);
  assert.deepEqual(answer.calculation, []);
  assert.match(answer.limitations.join(" "), /verified evidence is missing/);
  assert.doesNotMatch(render(data), /How the result was calculated/);
});

test("What MindEd did uses persisted execution, and mismatches block a confident answer", () => {
  const data = sales();
  data.execution_record = {
    execution_status: "SUCCESS", contract_status: "MATCHED", verification_status: "VERIFIED",
    dataset: {name: "Executed sales", dataset_id: "sales", dataset_version: 7, dataset_version_id: "v7"},
    aggregation: "SUM", target_column: "revenue", grouping_columns: ["region"], filters: [],
    population_row_count: 24, result_row_count: 4, sql: 'SELECT region, SUM("revenue") FROM actual_snapshot GROUP BY region',
    approved_contract: {target_column: "revenue", requested_aggregation: "SUM"},
  };
  let html = render(data);
  assert.match(html, /What MindEd did/);
  assert.match(html, /Executed sales/);
  assert.match(html, /v7/);
  assert.match(html, /actual_snapshot/);
  assert.match(html, /Approved plan \(separate from execution\)/);
  assert.doesNotMatch(html, /No reconciled execution record/);
  data.execution_record.contract_status = "MISMATCH";
  html = render(data);
  assert.match(html, /has not been verified against the approved plan/);
  assert.doesNotMatch(html, /Total revenue.*3,720/);
  assert.match(html, /actual_snapshot/);
  data.execution_record.contract_status = "MATCHED";
  data.execution_record.verification_status = "FAILED";
  html = render(data);
  assert.match(html, /Independent verification of the executed calculation did not pass/);
  assert.doesNotMatch(html, /Total revenue.*3,720/);
});

test("a missing execution record never presents planned SQL as executed", () => {
  const data = sales();
  data.analysis_plan.planned_sql = "SELECT secret_planned_only";
  const html = render(data);
  assert.match(html, /No reconciled execution record was saved/);
  assert.doesNotMatch(html, /secret_planned_only/);
});

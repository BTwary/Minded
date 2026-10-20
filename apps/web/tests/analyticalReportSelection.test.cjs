const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('/opt/nvm/versions/node/v22.16.0/lib/node_modules/typescript');

function loadTs(relativePath) {
  const filename = path.join(__dirname, '..', relativePath);
  const source = fs.readFileSync(filename, 'utf8');
  const compiled = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const module = { exports: {} };
  new Function('require', 'module', 'exports', compiled)((id) => {
    if (id === '../types') return {};
    throw new Error(`Unexpected runtime dependency: ${id}`);
  }, module, module.exports);
  return module.exports;
}

const { presentAnalyticalAnswer, selectAnalyticalCharts } = loadTs('src/lib/analyticalAnswer.ts');
function fixture(kind, numbers, question = 'Summarize revenue by region.') {
  return {
    id: 'report-fixture', question, status: 'COMPLETED', verdict: 'OBSERVED',
    analyst_result: { kind, headline: 'Recorded result', numbers, caveats: [], next_steps: [] },
    analysis_plan: { estimand: { target: numbers.target_column || 'revenue' }, semantics: { grouping_columns: numbers.grouping_columns || ['region'] } },
    evidence: [], findings: [],
  };
}
function present(data, verification = 'VERIFIED') {
  return presentAnalyticalAnswer(data, verification, null);
}

test('additive group totals get a sorted bar chart and a meaningful composition donut', () => {
  const data = fixture('RANKING', {
    operation: 'SUM', target_column: 'revenue', grouping_columns: ['region'],
    population_row_count: 30, result_row_count: 3,
    table: { A: { value: 120, n: 10 }, B: { value: 80, n: 10 }, C: { value: 40, n: 10 } },
    filters: [{ column: 'year', operator: '=', value: 2025 }], time_window: '2025',
  });
  const result = present(data);
  assert.equal(result.supported, true);
  assert.equal(result.charts.length, 2);
  assert.equal(result.charts[0].chart_type, 'bar');
  assert.deepEqual(result.charts[0].data.map((r) => r.region), ['A', 'B', 'C']);
  assert.equal(result.charts[1].chart_type, 'donut');
  assert.equal(result.charts[1].metadata.composition_total, 240);
  assert.match(result.interpreted.join(' '), /Target metric \/ outcome: revenue/);
  assert.match(result.interpreted.join(' '), /Grouping dimension\(s\): region/);
  assert.match(result.calculation.join(' '), /Aggregation logic: sum/);
  assert.match(result.calculation.join(' '), /sample sizes: A: n=10; B: n=10; C: n=10/);
});

test('large additive totals and ranking gaps retain their decimal precision', () => {
  const data = fixture('RANKING', {
    operation: 'SUM', target_column: 'realized_revenue', grouping_columns: ['region'],
    population_row_count: 9000, result_row_count: 4,
    table: {
      East: { value: 355091.84, n: 2288 },
      South: { value: 343625.00, n: 2342 },
      North: { value: 339704.02, n: 2172 },
      West: { value: 337192.80, n: 2198 },
    },
  }, 'Calculate total realized_revenue by region, ranked from highest to lowest.');
  const result = present(data);
  assert.equal(result.supported, true);
  assert.match(result.answer, /1,375,613\.66/);
  assert.match(result.answer, /355,091\.84/);
  assert.ok(result.findings.some((finding) => finding.includes('17,899.04')));
});

test('means and rates do not get a composition chart because they are not additive parts of a whole', () => {
  const means = fixture('GROUP_COMPARISON', { operation: 'MEAN', target_column: 'revenue', means: { A: 5, B: 10, C: 15 } });
  assert.equal(selectAnalyticalCharts(means).filter((c) => c.chart_type === 'donut').length, 0);
  const rates = fixture('RATE_COMPARISON', { operation: 'RATE', target_column: 'conversion', rates: { A: 0.1, B: 0.2 } });
  const charts = selectAnalyticalCharts(rates);
  assert.equal(charts[0].chart_type, 'bar');
  assert.equal(charts.some((c) => c.chart_type === 'donut'), false);
});

test('correlation outputs get an association bar chart without implying causality', () => {
  const result = fixture('ASSOCIATION', { target_column: 'revenue', correlations: { price: -0.7, visits: 0.5, discount: 0.2 } }, 'Which variables correlate with revenue?');
  const charts = selectAnalyticalCharts(result);
  assert.equal(charts.length, 1);
  assert.equal(charts[0].chart_type, 'bar');
  assert.deepEqual(charts[0].data.map((r) => r.variable), ['price', 'visits', 'discount']);
  assert.match(charts[0].explanation, /do not establish causation/);
});

test('box plot is selected only for a distribution-oriented question and uses saved percentiles', () => {
  const data = fixture('DESCRIPTIVE', { target_column: 'temperature', distribution_profile: { n: 100, p10: 4, p25: 8, p50: 11, p75: 16, p90: 20 } }, 'How has the temperature distribution and spread changed?');
  const charts = selectAnalyticalCharts(data);
  assert.equal(charts.length, 1);
  assert.equal(charts[0].chart_type, 'boxplot');
  assert.deepEqual(charts[0].data[0], { label: 'Observed distribution', p10: 4, p25: 8, p50: 11, p75: 16, p90: 20 });
});

test('period-change output visualizes only the saved waterfall bridge contributions', () => {
  const data = fixture('PERIOD_CHANGE', { target_column: 'revenue', waterfall_bridge: { steps: [
    { step: 'Prior Base', delta: 100, subtotal: 100 },
    { step: 'Region A', delta: -20, subtotal: 80 },
    { step: 'Net Change', delta: 0, subtotal: 80 },
  ] } }, 'What contributed to the revenue decline?');
  const charts = selectAnalyticalCharts(data);
  assert.equal(charts.length, 1);
  assert.equal(charts[0].chart_type, 'waterfall');
  assert.deepEqual(charts[0].data.map((row) => [row.start, row.end]), [[0, 100], [100, 80], [80, 80]]);
});

test('explicit trend chart payloads are preserved and deduplicated', () => {
  const chart = { chart_id: 'trend_revenue', chart_type: 'line', title: 'Revenue over time', x_axis: { field: 'time', type: 'temporal' }, y_axis: { field: 'value' }, data: [{ time: '2025-01', value: 10 }, { time: '2025-02', value: 12 }] };
  const result = fixture('TREND', { target_column: 'revenue', trend_chart: chart, trend_charts: [chart] }, 'How has revenue trended?');
  const charts = selectAnalyticalCharts(result);
  assert.equal(charts.length, 1);
  assert.equal(charts[0].chart_type, 'line');
  assert.equal(charts[0].data.length, 2);
});

test('unverified results never receive confident automatically selected charts', () => {
  const data = fixture('RANKING', { operation: 'SUM', target_column: 'revenue', grouping_columns: ['region'], table: { A: { value: 10 }, B: { value: 5 } } });
  const result = present(data, 'UNVERIFIED');
  assert.equal(result.supported, false);
  assert.equal(result.charts.length, 0);
});


test('shared API report contract controls interpretation, calculation, and chart selection', () => {
  const savedChart = { chart_id: 'saved_line', chart_type: 'line', title: 'Saved observed series', x_axis: { field: 'year', label: 'Year', type: 'quantitative' }, y_axis: { field: 'temperature', label: 'Temperature' }, data: [{ year: 2020, temperature: 1 }, { year: 2021, temperature: 2 }] };
  const data = fixture('RANKING', { operation: 'SUM', target_column: 'revenue', grouping_columns: ['region'], table: { A: { value: 10 }, B: { value: 5 } } });
  data.report_contract = { schema_version: '1.0', interpreted: ['Server-resolved outcome: temperature.'], calculation: ['Server recorded method: Sen slope.'], visualizations: [savedChart] };
  const result = present(data);
  assert.deepEqual(result.interpreted, ['Server-resolved outcome: temperature.']);
  assert.deepEqual(result.calculation, ['Server recorded method: Sen slope.']);
  assert.equal(result.charts.length, 1);
  assert.equal(result.charts[0].chart_id, 'saved_line');
  assert.equal(result.charts[0].title, 'Saved observed series');
});

test('an explicitly empty shared chart list prevents a divergent client-side auto-chart', () => {
  const data = fixture('RANKING', { operation: 'SUM', target_column: 'revenue', grouping_columns: ['region'], table: { A: { value: 10 }, B: { value: 5 } } });
  data.report_contract = { schema_version: '1.0', interpreted: [], calculation: [], visualizations: [] };
  const result = present(data);
  assert.deepEqual(result.charts, []);
});


test('saved paired observations produce a scatter plot instead of reconstructed synthetic points', () => {
  const analysis = fixture('RELATIONSHIP', {
    target_column: 'revenue', scatter_data: {
      x_field: 'marketing_spend', y_field: 'revenue',
      data: [{ marketing_spend: 10, revenue: 20 }, { marketing_spend: 15, revenue: 27 }, { marketing_spend: 20, revenue: 31 }],
    },
  }, 'How are marketing spend and revenue related?');
  const charts = selectAnalyticalCharts(analysis);
  assert.equal(charts[0].chart_type, 'scatter');
  assert.equal(charts[0].data.length, 3);
  assert.match(charts[0].explanation, /does not establish causation/i);
});

test('distribution question renders only persisted histogram bins', () => {
  const analysis = fixture('DESCRIPTIVE', {
    target_column: 'revenue', histogram_bins: [
      { bin_start: 0, bin_end: 10, count: 3 }, { bin_start: 10, bin_end: 20, count: 7 },
    ],
  }, 'Show the distribution of revenue');
  const charts = selectAnalyticalCharts(analysis);
  assert.equal(charts[0].chart_type, 'histogram');
  assert.deepEqual(charts[0].data, [{ bin_label: '0–10', count: 3 }, { bin_label: '10–20', count: 7 }]);
});


test('shared report contract exposes only evidence-supported narrative claims to the UI', () => {
  const data = fixture('RANKING', {
    operation: 'SUM', target_column: 'revenue', grouping_columns: ['region'],
    table: { A: { value: 10, n: 2 }, C: { value: 20, n: 2 } },
  });
  data.report_contract = {
    schema_version: '1.1', interpreted: [], calculation: [], visualizations: [],
    narrative: {
      family: 'comparison', title: 'What the comparison shows', status: 'PARTIAL',
      summary: 'Forged fallback summary 9999.',
      paragraphs: ['Forged paragraph 8888.'], supported_claim_count: 1,
      claims: [
        { text: 'C recorded the highest value (20).', validation_status: 'SUPPORTED', fact_ids: ['value:C'], evidence_refs: ['E-1'] },
        { text: 'C recorded 9999.', validation_status: 'WITHHELD', fact_ids: ['value:C'], evidence_refs: ['E-1'] },
      ],
    },
  };
  const result = present(data);
  assert.ok(result.narrative);
  assert.equal(result.narrative.summary, 'C recorded the highest value (20).');
  assert.deepEqual(result.narrative.paragraphs, []);
  assert.equal(result.narrative.claims.length, 1);
  assert.deepEqual(result.narrative.claims[0].evidenceRefs, ['E-1']);
});

test('why-did-it-change questions explain the recorded period movement and saved bridge instead of returning generic non-causal boilerplate', () => {
  const data = fixture('PERIOD_CHANGE', {
    target_column: 'revenue', aggregation: 'sum', previous: 100, current: 130, delta: 30,
    pct_change: 0.3, previous_start: 'Q2', period: 'Q3',
    top_contributor: 'West', top_dimension: 'region', top_contribution_share: 0.6,
    waterfall_bridge: { steps: [
      { step: 'Prior Base', delta: 100, subtotal: 100 },
      { step: 'region: West', delta: 18, subtotal: 118 },
      { step: 'region: East', delta: 12, subtotal: 130 },
    ] },
    pvm_decomposition: { volume_effect: 20, rate_effect: 8, cross_effect: 2 },
  }, 'Why did revenue change in Q3?');
  const result = present(data);
  assert.equal(result.supported, true);
  assert.match(result.answer, /increased from 100 in Q2 to 130 in Q3/);
  assert.match(result.answer, /net change \+30; \+30%/);
  assert.ok(result.findings.some((finding) => /West.*60\.0%/.test(finding)));
  assert.ok(result.findings.some((finding) => /region: West.*\+18/.test(finding)));
  assert.match(result.limitations.join(' '), /do not.*establish.*caused/i);
  assert.equal(result.charts[0]?.chart_type, 'waterfall');
});

test('period-change questions do not fabricate a driver when the saved prior/current bridge is incomplete', () => {
  const data = fixture('PERIOD_CHANGE', { target_column: 'revenue', current: 130, delta: 30 }, 'Why did revenue change in Q3?');
  const result = present(data);
  assert.equal(result.supported, false);
  assert.match(result.answer, /does not contain a consistent prior value/i);
});

test('root-cause analysis explains candidate-factor adjustment instead of treating it as a plain ranking', () => {
  const data = fixture('ROOT_CAUSE', {
    target_column: 'productivity', top_candidate: 'seniority', top_candidate_r: 0.82,
    unadjusted_diff: 18, adjusted_diff: 2, attenuation_pct: 88.9,
    confounding_classification: 'CANDIDATE_CONFOUNDER_SUBSTANTIAL_ATTENUATION',
    candidate_factors_scanned: ['seniority', 'region_size'],
  }, 'Why does Region A have higher productivity than Region B?');
  data.analyst_result.details = [
    'seniority is a candidate confounder and is associated with productivity and differs between regions.',
    'Observed data are consistent with confounding after adjustment.',
    'Observational evidence does not establish mechanistic causality without controlled intervention.',
  ];
  const result = present(data);
  assert.equal(result.supported, true);
  assert.match(result.answer, /seniority.*leading measured candidate confounder/i);
  assert.match(result.answer, /from 18 to 2/);
  assert.match(result.answer, /cannot establish that seniority caused/i);
  assert.ok(result.findings.some((finding) => /attenuation: 88\.9%/.test(finding)));
});

test('root-cause analysis reports when screening finds no candidate rather than inventing one', () => {
  const data = fixture('ROOT_CAUSE', { target_column: 'sales', candidate_factors_scanned: ['noise'] }, 'Why does North have higher sales?');
  data.analyst_result.details = [
    'Candidate factor screening: None of the scanned factors met the dual criteria for candidate confounders.',
  ];
  const result = present(data);
  assert.equal(result.supported, true);
  assert.match(result.answer, /did not identify a measured candidate factor/i);
  assert.doesNotMatch(result.answer, /noise is the cause/i);
});

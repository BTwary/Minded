const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('/opt/nvm/versions/node/v22.16.0/lib/node_modules/typescript');

function loadComponent(relativePath) {
  const filename = path.join(__dirname, '..', relativePath);
  const source = fs.readFileSync(filename, 'utf8');
  const compiled = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.React, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const ReactMock = {
    createElement: (type, props, ...children) => ({ type, props: props || {}, children: children.flat() }),
    useMemo: (fn) => fn(),
  };
  const reactModule = { __esModule: true, default: ReactMock, createElement: ReactMock.createElement, useMemo: ReactMock.useMemo };
  const lucide = new Proxy({}, { get: () => function Icon() { return null; } });
  const mod = { exports: {} };
  new Function('require', 'module', 'exports', compiled)((id) => {
    if (id === 'react') return reactModule;
    if (id === 'lucide-react') return lucide;
    return require(id);
  }, mod, mod.exports);
  return mod.exports.default;
}
function walk(node, result = []) {
  if (Array.isArray(node)) { for (const child of node) walk(child, result); return result; }
  if (!node || typeof node !== 'object') return result;
  result.push(node);
  for (const child of node.children || []) walk(child, result);
  return result;
}
function allText(node, result = []) {
  if (Array.isArray(node)) { for (const child of node) allText(child, result); return result; }
  if (typeof node === 'string' || typeof node === 'number') { result.push(String(node)); return result; }
  if (!node || typeof node !== 'object') return result;
  for (const child of node.children || []) allText(child, result);
  return result;
}
const PlotlyChart = loadComponent('src/components/PlotlyChart.tsx');
const render = (spec) => walk(PlotlyChart({ spec, height: 300 }));
const has = (nodes, type) => nodes.some((node) => node.type === type);

const base = {
  chart_id: 'renderer-test', title: 'Observed values', x_axis: { field: 'period', label: 'Period', type: 'temporal' },
  y_axis: { field: 'value', label: 'Value' }, data: [{ period: 2023, value: 10 }, { period: 2024, value: 15 }],
};

test('line chart renders an SVG path and observed points', () => {
  const nodes = render({ ...base, chart_type: 'line' });
  assert.ok(has(nodes, 'svg'));
  assert.ok(has(nodes, 'path'));
  assert.equal(nodes.filter((node) => node.type === 'circle').length, 2);
});

test('bar chart renders bars for every category', () => {
  const nodes = render({ ...base, chart_type: 'bar', x_axis: { field: 'region', type: 'nominal' }, data: [{ region: 'A', value: 5 }, { region: 'B', value: 8 }, { region: 'C', value: 3 }] });
  const rects = nodes.filter((node) => node.type === 'rect');
  assert.ok(rects.length >= 3);
});

test('donut chart renders segments and a usable legend', () => {
  const nodes = render({ ...base, chart_type: 'donut', x_axis: { field: 'region', type: 'nominal' }, data: [{ region: 'A', value: 60 }, { region: 'B', value: 40 }] });
  assert.ok(nodes.filter((node) => node.type === 'circle').length >= 3);
  assert.ok(allText(nodes).includes('A'));
});

test('invalid negative composition safely falls back instead of drawing a pie', () => {
  const nodes = render({ ...base, chart_type: 'donut', x_axis: { field: 'region', type: 'nominal' }, data: [{ region: 'A', value: 60 }, { region: 'B', value: -5 }] });
  assert.ok(!nodes.some((node) => node.type === 'svg' && node.props['aria-label']?.includes('Composition')));
  assert.ok(has(nodes, 'table'));
});

test('boxplot renders the percentile band and whiskers', () => {
  const nodes = render({ ...base, chart_type: 'boxplot', x_axis: { field: 'label', type: 'nominal' }, y_axis: { field: 'p50' }, data: [{ label: 'Observed', p10: 1, p25: 3, p50: 5, p75: 7, p90: 9 }] });
  assert.ok(has(nodes, 'svg'));
  assert.ok(has(nodes, 'rect'));
  assert.ok(nodes.some((node) => node.type === 'line'));
});

test('waterfall renders contribution bars from saved starts and ends', () => {
  const nodes = render({ ...base, chart_type: 'waterfall', x_axis: { field: 'step', type: 'nominal' }, y_axis: { field: 'delta' }, data: [{ step: 'Base', delta: 100, start: 0, end: 100 }, { step: 'Loss', delta: -20, start: 100, end: 80 }] });
  assert.ok(nodes.filter((node) => node.type === 'rect').length >= 2);
  assert.ok(nodes.some((node) => node.type === 'title' && String(node.children[0]).includes('subtotal')));
});

test('stacked bar segments share one category position instead of appearing side by side', () => {
  const nodes = render({ ...base, chart_type: 'stacked_bar', series_field: 'region',
    x_axis: { field: 'year', type: 'nominal' }, y_axis: { field: 'revenue' },
    data: [{ year: '2025', region: 'A', revenue: 10 }, { year: '2025', region: 'B', revenue: 15 }] });
  const bars = nodes.filter((node) => node.type === 'rect' && Number.isFinite(node.props.x) && node.props.width > 5);
  assert.equal(bars.length, 2);
  assert.equal(bars[0].props.x, bars[1].props.x, 'stack segments should share the same x position');
  assert.notEqual(bars[0].props.y, bars[1].props.y, 'the upper series should start above the lower segment');
});

test('line chart renders persisted confidence intervals even without a separate metadata flag', () => {
  const nodes = render({ ...base, chart_type: 'line',
    data: [{ period: 2023, value: 10, ci_low: 8, ci_high: 12 }, { period: 2024, value: 15, ci_low: 13, ci_high: 17 }] });
  const verticalIntervals = nodes.filter((node) => node.type === 'line'
    && node.props.x1 === node.props.x2 && node.props.y1 !== node.props.y2);
  assert.ok(verticalIntervals.length >= 2, 'each observed point should show its saved interval whisker');
  assert.equal(nodes.filter((node) => node.type === 'circle').length, 2);
});


test('histogram uses saved bin frequencies and dense bars', () => {
  const nodes = render({ ...base, chart_type: 'histogram',
    x_axis: { field: 'bin_label', type: 'nominal' }, y_axis: { field: 'count' },
    data: [{ bin_label: '0–10', count: 3 }, { bin_label: '10–20', count: 7 }, { bin_label: '20–30', count: 2 }] });
  const rects = nodes.filter((node) => node.type === 'rect' && Number.isFinite(node.props.x) && node.props.width > 5);
  assert.equal(rects.length, 3);
  assert.ok(rects[0].props.width > 100, 'histogram bars should use a nearly contiguous bin width');
});

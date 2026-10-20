# MindEd broad analytical reporting layer: implementation and verification report

Date: 2026-10-19

## Scope

This package combines the earlier analytical reliability fixes and professional-reporting changes within the real MindEd paths (analytics core, API response contract, Next.js answer renderer, SVG chart renderer, and PDF/DOCX exporter). The latest incremental patch is based on `MindEd_professional_reporting.zip`. It does not change the separate standalone `intent_adaptive_engine` prototype.

The intended user-facing requirement is: for a successfully completed and verified analysis, present the direct answer, suitable charts where supported and useful, what MindEd interpreted, how it calculated the result, evidence-backed insights, and limitations. The same saved result contract feeds the browser and downloadable report so they do not independently invent different interpretations or visualizations.

## Changes made

### 1. Shared report contract

- The investigation detail API now returns a `report_contract` containing `interpreted`, `calculation`, and `visualizations`, assembled from the saved question, analysis plan, execution record, analyst result, and evidence status.
- The browser prefers this server-generated contract. A saved empty visualization list means no charts are rendered; the browser will not independently invent another chart selection.
- The report hash includes the interpretation, execution record, and derived report content, so changes to the described calculation or chosen visuals alter the report digest.
- Charts are emitted only for completed results that meet the existing evidence verification/contract checks. Unsupported or unverified output does not receive a confident automatic chart.

### 2. Professional answer presentation

- The answer panel presents the direct answer, findings, “What MindEd interpreted”, calculation/execution details, limitations, and visualizations where available.
- Interpretation states target metric/outcome, aggregation/method, grouping dimensions, requested comparison values, time scope, filters, and dataset scope when those are present in the saved contract.
- Calculation details distinguish the interpreted plan from what the execution record says actually ran. They expose aggregation, grouping, executed filters and time window, eligible row count, returned rows/groups, exclusion counts and sample sizes when persisted.
- The presentation is gated on completion, verification and result consistency. An inconclusive or failed result gets an explanation rather than a made-up chart or definite finding.

### 3. General chart selection from saved outputs

The report selectors now support the following chart types when the matching saved data exists:

- **Time series and actual-vs-forecast:** `line` / `multi_line`, with saved confidence or prediction interval endpoints displayed when available.
- **Rankings and group comparisons:** sorted `bar`; `grouped_bar` and `stacked_bar` for saved multi-series data.
- **Composition:** `donut` only for small, non-negative additive totals that actually represent a meaningful returned whole; `stacked_bar` for saved composition-over-time aggregates.
- **Relationships:** `scatter` only from persisted paired observations. A correlation coefficient alone is not used to reconstruct fake scatter points.
- **Distribution:** `histogram` only from persisted bin boundaries/frequencies and `boxplot` only from stored percentile/quartile summaries.
- **Period/change analysis:** `waterfall` only when a saved change-contribution bridge exists.

All chart payloads are built from persisted output values; chart selection does not re-run an analysis, interpolate missing periods, invent raw points, or derive a time series from only a slope/endpoints. Long category lists are capped in the graphic with the full result table retained and the cap disclosed.

### 4. Browser chart renderer

- `PlotlyChart.tsx` accepts the chart-spec object used by analytics core as well as legacy Plotly and row-array payloads.
- SVG rendering covers time-series/multi-series lines, scatter plots, grouped and stacked bars, histograms, donut composition, percentile boxplots, and waterfall bridge charts.
- Persisted line/scatter intervals are rendered even if a redundant `shows_ci` flag is absent, and interval endpoints are included in the displayed y-axis range.
- Invalid composition values fall back to a table instead of being displayed as a misleading donut.

### 5. PDF/DOCX reporting

- PDF and DOCX now include “What MindEd interpreted”, “How the result was calculated”, “Visualizations”, findings, limitations/assumptions, and evidence/provenance.
- Saved chart specifications are rendered to embedded PNGs with Pillow, including trend/line intervals, stacked bars, histograms, scatter plots, donut charts, boxplots, and saved waterfall bridge values.
- Export remains subject to the review/acceptance gate introduced in the earlier integrated patch. The report hash covers key displayed inputs/results and derived report content.

### 6. Hybrid evidence-grounded narrative composer

- Added `packages/analytics_core/src/reporting/narrative_composer.py`, a shared composer using a small set of extensible report-family blueprints rather than a template per question. Families include comparison, trend, trend comparison, relationship, distribution, forecast, diagnostic, composition, and descriptive analysis. Combined questions may combine supported families where the saved result contains the necessary values.
- The current writer is deterministic and result-driven; it does **not** call an LLM or perform new analysis while writing. This is intentional for reproducibility. A future LLM paraphraser must operate only on the validated claims, preserve references, and pass the same claim validator before its prose is shown.
- Each generated claim includes a claim ID, family, fact IDs, source paths, evidence references, numeric literals, validation status, and issues. The validator rejects claims with missing facts/source paths, unverified evidence references, or numeric literals not derivable from that claim's linked facts. Unsupported claims remain in the audit object but are withheld from the reader.
- The shared report contract, browser answer, and PDF/DOCX all consume this narrative object. Evidence IDs are shown alongside displayed claims. The report integrity hash includes the narrative contract.
- The composer explains computed comparisons, gaps and relative differences, recorded trend slopes and intervals, distribution quartiles/IQR, correlations and their non-causal interpretation, saved forecast projections/ranges, and diagnostic change/contribution outputs. It does not infer values that the analysis did not persist.
- Added architecture decision record `docs/adr/002-evidence-grounded-report-composer.md` describing the LLM boundary and acceptance requirements.

### 7. Previously integrated analytical fixes retained

The prior integrated patch's question-role binding, categorical group comparison safety, time-window handling, trend estimation, explicit confounder handling, correction workflow, review endpoint repair, and export gate remain in this incremental package. These are described in the previous report history and are covered by the focused analytical tests named below; they are not all reimplemented in this reporting-only delta.

## Verification actually executed in this environment

| Command/check | Result | Scope and limits |
|---|---:|---|
| `PYTHONPATH=. python -m pytest tests/test_professional_report_contract.py tests/test_export_documents.py tests/independent_release/test_recently_discussed_fixes.py -q` | **35 passed, 0 failed** | Includes report-contract/narrative claim validation, fabricated-number withholding, combined report families, report hashes, PDF/DOCX sections and visuals, and previous interpretation/trend/confounder regression checks. |
| `node --test apps/web/tests/analyticalReportSelection.test.cjs apps/web/tests/visualizationRenderer.test.cjs` | **21 passed, 0 failed** | Tests chart selection, SVG renderer output, supported-claim-only report selection, stacked bar placement, and interval rendering. These use a renderer test harness, not a real browser. |
| Python `py_compile` on modified reporting/API Python modules | **Passed** | Syntax compilation only; does not import or start the complete app. |
| TypeScript transpile/parser check on changed `.ts`/`.tsx` source files | **7 files parsed without syntax errors** | Syntax only. Not a Next.js type check or production build. |
| `python -m pytest tests/independent_release/test_region_compare_end_to_end.py -q` | **6 passed, 7 setup errors** | The API-dependent tests could not import the real FastAPI app because `python-jose` (`jose`) is unavailable. These results do **not** prove the browser-to-API flow works. |
| Full Next.js build and live browser test | **Not run** | `apps/web/node_modules` is absent in this environment; React/Next runtime dependencies cannot be resolved. The environment has no network access for installing them. |

## Remaining gaps / release cautions

1. **The narrative writer is currently deterministic, not LLM-paraphrased.** It dynamically selects analysis-family sections and builds natural-language claims from structured values; a model-based paraphrasing layer remains a future, optional step and must not bypass claim validation.
2. **Full frontend integration is not yet verified.** Source syntax and renderer harness tests pass, but the Next.js app has not been built or exercised in a real browser here.
3. **Full API-to-worker-to-review-to-export is not yet verified in this environment.** The real end-to-end tests stop at application import due to the missing `python-jose` dependency; DuckDB/Polars are also unavailable in this runtime for parts of the application's analytical stack.
4. **Prebuilt static frontend bundle is stale.** `apps/api/src/static` has not been rebuilt/copied from the updated Next source because the production frontend build could not run. If the deployment serves this prebuilt offline bundle, it will not show the updated browser source until rebuilt.
5. Chart selection is constrained by persisted outputs. If a method has not saved the paired rows, histogram bins, period/group time points, or distribution statistics needed for a chart, the renderer deliberately omits that chart rather than fabricating data. Full automatic chart coverage requires the relevant analysis method to persist a supported result shape.
6. Live deployment acceptance is still required: validate that one actual UI analysis uses the same `report_contract` in the browser and in PDF/DOCX; verify corrections preserve unrelated plan choices; ensure export is unavailable before acceptance and available after acceptance; and compare displayed charts/numbers with independently calculated fixture values.

## Suggested verification on a fully provisioned checkout

From the repository root:

```bash
# Install backend dependencies as pinned by this repo.
python -m pip install -r requirements.txt

# Build the actual frontend.
cd apps/web
npm ci
npm run build
cd ../..

# Run focused tests and the actual API/controller regression test.
PYTHONPATH=. python -m pytest \
  tests/independent_release/test_recently_discussed_fixes.py \
  tests/independent_release/test_region_compare_end_to_end.py \
  tests/test_professional_report_contract.py \
  tests/test_export_documents.py -q

# The standalone trace creates a temporary database and deterministic fixture.
PYTHONPATH=. python scripts/trace_region_compare.py
```

Then run the app per `DEPLOYMENT.md` and verify in a browser: inspect the interpreted metric/group/time scope and plan; execute; inspect charts and calculation details; request an aggregation correction and confirm unrelated dimensions/filters/entities are preserved; accept the result; verify exports are gated until acceptance; and compare the browser report with PDF/DOCX. Do not label the full product “verified” until these real runtime steps pass.

# AA-OS Offline Conversational Analyst

## Purpose

MindEd AA-OS now treats conversational interaction as a local front-end to the canonical autonomous analyst rather than as a separate chatbot.

A user can ask questions such as:

- "Why did sales fall?"
- "Will sales likely increase next quarter?"
- "Which customers are driving the decline?"
- "Did the price increase cause sales to fall?"

The local conversational layer performs deterministic question routing and produces a structured interpretation. It does not calculate results, invent numbers, or override the canonical investigation controller.

## Execution contract

```text
Natural-language question
        ↓
LocalConversationalAnalyst
        ↓
Routing hint / problem class / horizon / metric hint
        ↓
Canonical InvestigationController
        ↓
Real deterministic computation
        ↓
Independent verification
        ↓
Evidence + calculation trace
        ↓
Local explanation
        ↓
Optional AI presentation augmentation
```

AI is therefore optional. The canonical evidence, verdict, uncertainty, and calculation trace do not depend on the conversational model.

## Offline behavior

The default chat mode is `DETERMINISTIC`. It requires no AI provider and no network connection. DuckDB, Polars, statistical libraries, local storage, and other execution components remain implementation details behind the analytical runtime.

Removing an optional AI provider does not disable chat. Removing an optional cloud provider does not disable chat. If an analytical execution backend is unavailable, AA-OS must fail honestly or route to another supported backend; it must never manufacture an answer.

## Forecast example

For:

> Will sales likely increase next quarter?

The local conversational layer identifies:

```text
problem_class = forecasting
user_intent = forecast_future_value_or_direction
target_horizon = quarter
requested_metric_hint = sales
```

The controller then owns the real forecasting investigation: data validation, time resolution, method selection, competing methods/experiments where supported, backtesting, uncertainty, verification, evidence, and traceability.

The user-facing reply may summarize the canonical result, but the underlying numbers must always be drillable to their calculation trace.

## Follow-up conversation

The conversation ID is retained so subsequent natural-language questions can remain within the same user/project conversation. Each analytical question still creates its own canonical investigation, preserving independent evidence and provenance.

## Safety boundary

The local conversational layer is an interpretation/presentation layer. It does not:

- calculate business metrics in free-form text;
- change a verdict;
- strengthen epistemic certainty;
- create a causal claim from correlation;
- replace independent verification;
- bypass calculation traceability.

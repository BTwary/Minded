# Release manifest changes

## 2026-10-14 — common descriptive intent expansion

Added required check `common_descriptive_intent_expansion` to `tests/independent_release/manifest.toml`. It covers newly supported mode, range, quartile, interquartile-range, and missing-value-count questions from natural language through canonical contract compilation and approved execution. This is a required regression check because these operations must not silently degrade to a different metric.

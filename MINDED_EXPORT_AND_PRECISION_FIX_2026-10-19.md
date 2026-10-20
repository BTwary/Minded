# MindEd AA-OS: Export Save Dialog and Numeric Precision Fix

Date: 2026-10-19

## Verified result for `northstar_commerce_analytics_test(1).csv`

The source CSV contains 9,000 rows and 51 columns. `region` and `realized_revenue` have no nulls. An independent group-by sum over the CSV returns:

| Rank | Region | SUM(realized_revenue) | Rows | Share |
|---:|---|---:|---:|---:|
| 1 | East | 355,091.84 | 2,288 | 25.8% |
| 2 | South | 343,625.00 | 2,342 | 25.0% |
| 3 | North | 339,704.02 | 2,172 | 24.7% |
| 4 | West | 337,192.80 | 2,198 | 24.5% |
| | **Total** | **1,375,613.66** | **9,000** | **100.0%** |

East-West difference: 17,899.04. Difference relative to West: 5.3083% (5.3% rounded to one decimal place).

The source-level grouping, rankings, group sums, counts and shares shown by MindEd match. The prior headline's `1,375,610` and key finding's `17,899` were presentation rounding defects, not the exact arithmetic. The frontend's six-significant-digit format rounded the total and gap too aggressively. The updated formatter uses fractional precision for normal values and retains significant-digit formatting only for very small coefficients.

## Export behavior change

Previously the UI fetched a PDF/DOCX blob, simulated an anchor download and immediately revoked the blob URL. It had no saved-path or success notice, which could make a desktop export appear to do nothing.

The updated desktop shell exposes a narrowly-scoped `save_export_file` bridge:

- Invokes the native OS **Save As** dialog and suggests `MindEd_Investigation_<first 8 id chars>.pdf` or `.docx`.
- Accepts only `.pdf` and `.docx`, checks format signatures, rejects empty/oversized/invalid payloads, and writes only to the path selected by the user.
- Returns the full saved path, which is displayed in the UI after the file is written.
- Canceling the dialog says that no file was written.

In an ordinary browser, the existing download behavior remains, but object-URL cleanup is delayed and the UI says to check the browser's Downloads folder/history instead of claiming a specific saved path.

## Validation performed in this environment

- Independent CSV recomputation: 9,000 rows, no null `region`/`realized_revenue`; exact regional sums match the displayed ranking table.
- Frontend report unit tests: 13 passed, including a new regression for the exact grand total and East-West gap.
- TypeScript/TSX syntax transpilation checks passed for the four changed frontend files.
- Python compile check passed for `packaging/windows/minded_entry.py`.
- Desktop save bridge smoke test passed using a mocked Save As selection and PDF bytes; invalid extension guard passed.

## Not yet verified

This change has not been rebuilt and exercised in the user's Windows desktop UI. Rebuild with the existing pinned Python 3.12 environment and run a real PDF and DOCX Save As test. A successful build is not itself proof that a generated report can be opened; open both exported files once saved.

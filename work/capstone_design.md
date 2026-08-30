# Refresh Opportunity Capstone Project Design

## Purpose

Build a reproducible Search Intelligence capstone that helps a central content editor decide which pages to review first each month. The project will use the predefined **Refresh / Content Opportunity Scoring** lane and will end with an executed notebook chain, a ranked action queue, and a public research paper deployed through GitHub Pages.

The research question is:

> Which pseudonymized content items should a content editor review first because their pre-decision search and content signals indicate elevated risk of a meaningful decline in the following month?

The output is decision-support. It does not automate edits and does not claim that refreshing a page causes recovery.

## Decision, action, and error costs

- **Decision owner:** a content editor managing a cross-client portfolio.
- **Decision:** which 50 eligible content items enter the next monthly review queue.
- **Unit of analysis:** one pseudonymized content item at one monthly decision point.
- **Output:** a descending priority score, a proposed review action, reason codes, and a confidence tier.
- **Allowed actions:** refresh, expand, protect, consolidate/prune, or monitor.
- **False-positive cost:** scarce editorial time is spent reviewing a healthy page, and a successful page may be changed unnecessarily.
- **False-negative cost:** a measurable decline remains unreviewed for another cycle.
- **Automation boundary:** a human makes the final action decision; no page is edited, merged, pruned, or published automatically.

## Data source and public-safety boundary

Use the gated FlyRank internship warehouse release `flyrank_pseudonymized_warehouse_release_v20260703`:

- `dim_clients`: 104 pseudonymized clients.
- `dim_content`: 519,606 pseudonymized content items.
- `fact_content_daily_performance`: 78,835,655 daily rows from 2025-01-27 through 2026-06-30.

The primary feature source is the daily performance fact table, joined to safe content metadata from `dim_content` and history-coverage fields from `dim_clients`. The query table is excluded from the first model because its fixed 90-day window overlaps the final months and creates unnecessary leakage risk.

The project will never publish client names, domains, URLs, raw queries, titles, credentials, access tokens, or data that can reverse pseudonyms. Pseudonymized IDs are used only for joins, grouping, validation, and the public-safe ranked queue; they are never model features. Credentials are read from `HF_TOKEN` or a notebook secret and never written into a notebook or committed file.

## Monthly analytical frame

Aggregate daily facts into page-month rows. Each modeling example uses information available by the end of feature month `t` and an outcome observed in month `t+1`.

### Development and evaluation windows

- **Training anchors:** September 2025 through March 2026; each anchor predicts its following month.
- **Validation anchor:** April 2026 predicting May 2026.
- **Sealed test anchor:** May 2026 predicting June 2026.
- **Final month rule:** June 2026 is never used for feature development, threshold selection, or model choice.

These exact windows may reduce the usable client set because the panel is unbalanced. That is expected and will be reported as a coverage result, not repaired by weakening the split.

### Eligibility

A page-month is eligible when all of the following hold:

1. the page existed by the end of the feature month;
2. its client has GSC history active during both the feature and outcome months;
3. the feature month contains at least 100 GSC impressions;
4. the client has observable GSC activity in the outcome month; and
5. the page and client identifiers are valid pseudonyms.

If an eligible page has no fact row in an otherwise active client outcome month, its outcome impressions are treated as zero. This choice will be counted and disclosed because it can mix true loss of visibility with page removal or measurement gaps.

### Target

The binary observed outcome is:

```text
future_decline = outcome_month_impressions < 0.80 × feature_month_impressions
```

This represents an observed month-over-month impression decline greater than 20%. It is not a refresh-success label and not a causal outcome. Sensitivity checks will report how the queue changes at 10% and 30% decline definitions without using those alternatives to select the final model after seeing the sealed test.

## Features and exclusions

All features are computed from information available at or before the monthly decision point.

### Candidate features

- log-transformed feature-month impressions and clicks;
- CTR as a percentage, with its correct 0–100 interpretation;
- impression-weighted average position, with unavailable position separated from valid values;
- number of active search days in the feature month;
- prior-month impressions and clicks;
- pre-decision momentum: feature-month impressions relative to the previous month;
- volatility across the prior daily window;
- content age and safe structured content type fields available by the decision date;
- explicit missingness and availability flags.

GA4 engagement fields will be admitted only for rows where `ga4_data_available IS TRUE` and only if coverage is sufficient for a declared analysis slice. They will not be silently zero-filled across unavailable history.

### Excluded fields

- target-month or later metrics;
- June 2026 values during development;
- product decisions or recreated product flags used as labels;
- raw or pseudonymized IDs as predictive features;
- raw client, domain, URL, title, keyword, or query fields;
- the fixed-window query-table totals in the first model;
- any feature whose timestamp or derivation cannot be shown to precede the decision point.

## Signal audit

Before modeling, test whether the proposed signals have stable, interpretable relationships with the future-decline outcome. The audit will include distributions, missingness by content type and client coverage, base rates by month, grouped comparisons, effect sizes, and at least three signal verdicts. Every verdict must be one of `keep`, `context only`, or `exclude`, with a written reason.

No signal will be described as a Google ranking factor. Associations are observational and may reflect seasonality, client mix, content maturity, measurement coverage, or other unobserved causes.

## Baseline and models

### Transparent baseline

Create a pre-decision rule score from only safe features:

1. meaningful exposure: higher log feature-month impressions;
2. existing negative momentum: lower feature-month versus prior-month impressions;
3. established content: sufficient age to avoid treating very new pages as refresh candidates;
4. position context: valid, nonzero search-position opportunity;
5. reason codes that expose which components contributed.

The exact weights are frozen in `w04_baseline_score.ipynb` before model training. The baseline writes a reproducible top-20 review and a full ranked CSV.

### Candidate models

Train two deliberately small candidates:

- logistic regression as the interpretable linear model;
- random forest as the nonlinear comparison.

Preprocessing is fitted on training data only. Numeric missingness and categorical missingness remain explicit; blind global `fillna(0)` is prohibited. The selected result is the simplest candidate that improves the validation queue over the baseline without a leakage or stability failure. If neither model earns its complexity, the baseline remains the recommended method.

## Validation and success criteria

The primary evaluation is time-forward: train on earlier anchors, select on April-to-May validation, and evaluate once on May-to-June sealed data. A client-grouped diagnostic on the development period checks whether apparent skill depends on familiar client patterns. No random row split is used as the headline result.

Metrics are computed for the baseline and every model on exactly the same eligible rows:

- primary: Precision@50;
- supporting: average precision, recall@50, base rate, and lift@50;
- stability: results by month, client coverage, and volume band;
- error review: high-scoring false positives and missed declines.

The model earns deployment as the ranking method only if it beats baseline Precision@50 on validation and does not reverse that advantage on the sealed month. A difference smaller than five percentage points is described as marginal. The paper reports the observed outcome even when the baseline wins.

Leakage checks must prove that feature windows end before outcome windows, excluded columns are absent, preprocessing fits only on training rows, June remains sealed until final evaluation, and IDs are not features.

## Ranked action policy

The final queue contains public-safe columns only: pseudonymized content ID, score, confidence tier, relevant observed feature summaries, reason codes, suggested action, and monitoring note.

Action mapping is conservative:

- **refresh/expand:** high decline risk with meaningful exposure and reviewable content context;
- **protect:** strong exposure and positive momentum where unnecessary editing could be harmful;
- **consolidate/prune review:** weak or persistently deteriorating opportunity, always requiring human confirmation;
- **monitor:** uncertain, new, sparse, unavailable, or conflicting evidence.

The no-go list prohibits automatic rewrites, deletion, publishing, causal claims, deanonymization, and recommendations based on unavailable measurement periods.

## Notebook and artifact flow

Complete and commit each existing skeleton in order:

1. `work/notebooks/w03_data_contract.ipynb` — release, grain, windows, fields, counts, and limits.
2. `work/notebooks/w03_feature_leakage_check.ipynb` — build the cached feature vector and prove leakage/privacy exclusions.
3. `work/notebooks/w04_signal_audit.ipynb` — distributions, missingness, signal tests, and verdicts.
4. `work/notebooks/w04_baseline_score.ipynb` — freeze the rule, reason codes, top-20 review, and baseline queue.
5. `work/notebooks/w05_model.ipynb` — train candidates and compare them with the baseline on the validation frame.
6. `work/notebooks/w06_validation_audit.ipynb` — grouped diagnostic, sealed-month evaluation, leakage audit, and claim rewrite.
7. `work/notebooks/w07_action_playbook.ipynb` — final ranked actions, monitoring policy, and paper exports.
8. `work/notebooks/capstone.ipynb` — mirror every final paper section and regenerate the reported figures and tables.

Derived warehouse aggregates and model-ready tables live under ignored `work/outputs/` or another repository-approved ignored cache path. Only small public-safe charts, tables, and ranked samples required to reproduce the paper are committed. The 79-million-row source is never downloaded into Git history.

## Research paper and deployment

Create a static paper at `docs/index.html` with supporting files under `docs/assets/`, then deploy the repository's `docs/` directory through GitHub Pages. The expected direct URL is:

```text
https://hasankhan05.github.io/flyrank-ml-internship/
```

The page contains all required sections:

1. Title and five-sentence abstract.
2. Introduction and problem statement.
3. Data release, tables, windows, exclusions, and public-safety scope.
4. Methodology, assumptions, features, target, baseline, validation, and leakage checks.
5. Results comparing model and baseline on the same rows, with accessible charts.
6. Limitations and careful observed/directional/decision-support framing.
7. Ranked recommendations and action playbook.
8. Reproducibility links to the notebooks and repository.
9. Acknowledgments and data credit containing the exact text `Built on the FlyRank ML Internship dataset` linked to `https://flyrank.ai`.

The page must remain readable on mobile, use semantic HTML, include alt text or equivalent descriptions for charts, and avoid external services that require a new account. After deployment, `submission/paper_url.txt` contains exactly the direct paper URL and one trailing newline.

## Failure handling

- Query one mid-panel month first and cache aggregates before running the full scan.
- On Hugging Face HTTP 429 responses, stop repeated scans, reuse cached results, and retry after backoff.
- If the token is absent, pause only the warehouse step and request that the user add `HF_TOKEN` securely; never ask for the token in chat or store it in the repo.
- If coverage is too thin under the declared eligibility rules, report the coverage and simplify the feature set before changing the target.
- If a model fails to beat the baseline, publish the baseline-led result and the negative modeling finding.
- If GitHub Pages is not yet enabled, enable Pages from `main/docs`, wait for deployment, and verify the public URL before writing it to the submission file.

## Verification and completion criteria

The capstone is complete only when all of the following are true:

- every assignment notebook and `capstone.ipynb` executes top to bottom with no error output;
- warehouse counts, date bounds, and grain probes match the release documentation;
- the feature/outcome window audit proves no overlap;
- baseline and model metrics use identical evaluation rows;
- the sealed-month result is run once after model selection;
- the final queue contains no prohibited data and has reason codes plus human-review actions;
- paper numbers match notebook outputs and charts;
- the deployed page contains all required sections and FlyRank data credit;
- every public notebook and chart renders from GitHub;
- `submission/paper_url.txt` contains exactly the verified direct URL;
- repository CI passes and the working tree is clean;
- each weekly artifact is preserved in a separate focused commit.

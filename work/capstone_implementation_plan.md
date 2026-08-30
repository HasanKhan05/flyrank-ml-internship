# Refresh Opportunity Capstone Implementation Plan

> Execute these tasks in order. Each task ends with fresh verification and a focused commit so earlier assignments remain independently reviewable.

**Goal:** Build, validate, explain, and deploy a public-safe monthly Refresh / Content Opportunity ranking system using the FlyRank warehouse.

**Architecture:** DuckDB aggregates the remote daily warehouse into an ignored local page-month Parquet cache. A small tested Python module owns label construction, ranking metrics, the frozen baseline, model comparison, and action mapping; the existing weekly notebooks call that module and preserve the reasoning and evidence. The final notebook feeds a static GitHub Pages research paper, and the deployed URL is recorded in `submission/paper_url.txt`.

**Tech stack:** Python 3.12, DuckDB, pandas, NumPy, scikit-learn, matplotlib, Jupyter, semantic HTML/CSS, GitHub Pages.

**Design:** `work/capstone_design.md`

## Global constraints

- Use warehouse release `flyrank_pseudonymized_warehouse_release_v20260703` only.
- Read authentication from `HF_TOKEN`; never print, paste into a cell, or commit the token.
- Never commit Parquet files, raw warehouse extracts, client names, domains, URLs, raw queries, titles, or credentials.
- IDs are for joins, grouping, splitting, and public-safe queue references only; never predictive features.
- Feature information ends before the outcome month begins.
- Training anchors are September 2025 through March 2026; validation is April predicting May; sealed test is May predicting June.
- June 2026 remains sealed until model selection and claim wording are frozen.
- Primary metric is Precision@50 on identical evaluation rows for baseline and model.
- Use observed, measured, directional, and decision-support language; make no causal refresh or Google-algorithm claims.
- Preserve `scripts/` unchanged; all new implementation lives under `work/`.
- Fix random seeds at `42`.
- Every notebook must execute top to bottom with zero error outputs before its commit.

---

### Task 1: Reusable capstone logic and tests

**Files:**

- Create: `work/scripts/refresh_capstone.py`
- Create: `work/tests/test_refresh_capstone.py`

**Interfaces:**

- `make_examples(monthly: pandas.DataFrame) -> pandas.DataFrame` consumes page-month metrics and creates prior-month features plus the next-month `future_decline` outcome.
- `precision_at_k(labels: pandas.Series, scores: pandas.Series, k: int = 50) -> float` evaluates a ranked queue.
- `baseline_score(frame: pandas.DataFrame) -> pandas.Series` returns a deterministic score in `[0, 100]` using pre-decision exposure, momentum, age, and position context.
- `assign_action(frame: pandas.DataFrame, scores: pandas.Series) -> pandas.DataFrame` returns `suggested_action`, `reason_codes`, and `confidence_tier`.
- `split_frames(frame: pandas.DataFrame) -> tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]` returns training, validation, and sealed-test rows using the declared anchor months.

- [ ] **Step 1: Write a failing unit test with a hand-checked page-month fixture**

```python
import unittest
import pandas as pd

from work.scripts.refresh_capstone import (
    assign_action,
    baseline_score,
    make_examples,
    precision_at_k,
    split_frames,
)


class RefreshCapstoneTests(unittest.TestCase):
    def test_future_label_uses_next_month_only(self):
        monthly = pd.DataFrame({
            "month": pd.to_datetime(["2026-04-01", "2026-05-01", "2026-06-01"]),
            "client_hash_id": ["client_a"] * 3,
            "content_hash_id": ["content_a"] * 3,
            "impressions": [200, 150, 90],
            "clicks": [20, 15, 9],
            "avg_position": [8.0, 9.0, 10.0],
            "active_days": [25, 24, 20],
            "content_age_days": [300, 330, 360],
        })
        result = make_examples(monthly)
        self.assertEqual(result["future_decline"].tolist(), [0, 1])
        self.assertEqual(result["outcome_impressions"].tolist(), [150, 90])

    def test_precision_at_k_uses_descending_scores(self):
        labels = pd.Series([0, 1, 1, 0])
        scores = pd.Series([0.2, 0.9, 0.8, 0.1])
        self.assertEqual(precision_at_k(labels, scores, k=2), 1.0)

    def test_splits_keep_june_sealed(self):
        frame = pd.DataFrame({
            "month": pd.to_datetime(["2026-03-01", "2026-04-01", "2026-05-01"])
        })
        train, validation, sealed = split_frames(frame)
        self.assertEqual(train["month"].dt.month.tolist(), [3])
        self.assertEqual(validation["month"].dt.month.tolist(), [4])
        self.assertEqual(sealed["month"].dt.month.tolist(), [5])
```

- [ ] **Step 2: Run the test and verify the module is missing**

Run:

```powershell
.venv\Scripts\python.exe -m unittest work.tests.test_refresh_capstone -v
```

Expected: failure importing `work.scripts.refresh_capstone`.

- [ ] **Step 3: Implement the minimal tested functions**

Implementation rules:

```python
TRAIN_MONTHS = pd.period_range("2025-09", "2026-03", freq="M")
VALIDATION_MONTH = pd.Period("2026-04", freq="M")
SEALED_MONTH = pd.Period("2026-05", freq="M")
DECLINE_RATIO = 0.80
RANDOM_SEED = 42
```

`make_examples` sorts by client, content, and month; uses groupwise `shift(1)` for prior metrics and `shift(-1)` for outcomes; requires consecutive calendar months; keeps feature impressions at least 100; and sets `future_decline = outcome_impressions < 0.8 * impressions`. `baseline_score` min-max scales four transparent components and records no target-month information. `precision_at_k` uses stable descending order and returns `NaN` when no rows exist. `assign_action` never emits an automatic-edit action.

- [ ] **Step 4: Run unit tests**

Run:

```powershell
.venv\Scripts\python.exe -m unittest work.tests.test_refresh_capstone -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit the reusable logic**

```powershell
git add -- work/scripts/refresh_capstone.py work/tests/test_refresh_capstone.py
git commit -m "Add tested refresh capstone logic"
```

---

### Task 2: Warehouse access, schema probe, and data contract

**Files:**

- Modify: `work/notebooks/w03_data_contract.ipynb`
- Create locally, ignored: `work/outputs/warehouse_schema.json`

**Interfaces:**

- Consumes `HF_TOKEN` and the three declared warehouse relations.
- Produces verified row counts, date bounds, grain results, availability coverage, and the exact column contract used by Task 3.

- [ ] **Step 1: Run an access preflight without exposing the token**

```powershell
.venv\Scripts\python.exe -c "import os; assert os.getenv('HF_TOKEN'), 'HF_TOKEN is not set'; print('HF_TOKEN is available without printing it')"
```

Expected: the availability message. If it fails, stop only this task and ask the user to add the token securely.

- [ ] **Step 2: Write the notebook completion check and watch it fail**

The check loads the notebook JSON and requires executed outputs containing the documented counts `104`, `519,606`, and `78,835,655`, the date bounds `2025-01-27` and `2026-06-30`, an empty duplicate-grain probe, and explicit feature/label/context/excluded field tables.

```powershell
.venv\Scripts\python.exe -c "import json,pathlib; n=json.loads(pathlib.Path(r'work/notebooks/w03_data_contract.ipynb').read_text(encoding='utf-8')); out=str(n); required=['78,835,655','2025-01-27','2026-06-30','feature','label','excluded']; missing=[x for x in required if x.lower() not in out.lower()]; assert not missing, missing"
```

Expected: failure listing missing executed evidence.

- [ ] **Step 3: Fill the data-contract notebook**

Use DuckDB `DESCRIBE` and these release relations:

```python
REL = "hf://datasets/FlyRank/internship-warehouse"
TABLES = {
    "dim_clients": f"read_parquet('{REL}/dim_clients.parquet')",
    "dim_content": f"read_parquet('{REL}/dim_content.parquet')",
    "fact_daily": f"read_parquet('{REL}/fact_content_daily_performance/**/*.parquet')",
}
```

The notebook must show `COUNT(*)`, `MIN/MAX(report_date)`, the daily grain probe, per-client GSC history coverage, `gsc_data_available IS TRUE` coverage, the monthly decision grain, the exact training/validation/sealed windows, the 100-impression eligibility floor, and the public-safety exclusions.

- [ ] **Step 4: Execute and rerun the completion check**

```powershell
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 work/notebooks/w03_data_contract.ipynb
```

Expected: exit code 0, all code cells have execution counts, no error outputs, completion check passes.

- [ ] **Step 5: Commit the executed data contract**

```powershell
git add -- work/notebooks/w03_data_contract.ipynb
git commit -m "Define and verify capstone data contract"
```

---

### Task 3: Monthly feature vector and leakage audit

**Files:**

- Create: `work/scripts/build_monthly_features.py`
- Modify: `work/notebooks/w03_feature_leakage_check.ipynb`
- Create locally, ignored: `work/outputs/page_month_features.parquet`
- Create locally, ignored: `work/outputs/model_examples.parquet`
- Create and commit: `work/outputs/feature_contract.json`

**Interfaces:**

- The script consumes remote warehouse relations and writes page-month aggregates.
- The notebook consumes the ignored Parquet cache, calls `make_examples`, and produces a public-safe feature contract and leakage audit.

- [ ] **Step 1: Add a failing synthetic aggregation test**

Extend `work/tests/test_refresh_capstone.py` with a fixture containing two pages, three consecutive months, a position value of zero/unavailable, and one missing GA4 availability flag. Assert that the resulting examples contain no target-month columns in the feature list, position availability is explicit, and unavailable GA4 is not interpreted as zero engagement.

- [ ] **Step 2: Verify the new test fails**

```powershell
.venv\Scripts\python.exe -m unittest work.tests.test_refresh_capstone -v
```

Expected: failure because feature-contract validation is not implemented.

- [ ] **Step 3: Implement the monthly builder and feature-contract validation**

The DuckDB query aggregates by `date_trunc('month', report_date)`, `client_hash_id`, and `content_hash_id` and computes impressions, clicks, impression-weighted valid position, active search days, and GSC/GA4 availability. Join `content_created_at` only after verifying its warehouse type. Do not scan the query table. First run `month=2026-03`; after the schema and grain checks pass, run the declared training-through-sealed partitions once and cache the result.

The committed JSON lists each field as `feature`, `label`, `context`, or `excluded`, plus source window and availability rule.

- [ ] **Step 4: Fill and execute the feature/leakage notebook**

The notebook shows row counts by anchor, client coverage, base rate, missingness, class balance, exact feature columns, forbidden-column assertions, window-order assertions, and privacy assertions.

```powershell
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 work/notebooks/w03_feature_leakage_check.ipynb
```

- [ ] **Step 5: Verify tests, notebook outputs, and ignored data**

```powershell
.venv\Scripts\python.exe -m unittest work.tests.test_refresh_capstone -v
git check-ignore work/outputs/page_month_features.parquet work/outputs/model_examples.parquet
git status --short
```

Expected: tests pass; both Parquet files are ignored; only code, notebook, test, and JSON receipt are tracked changes.

- [ ] **Step 6: Commit the feature-vector work**

```powershell
git add -- work/scripts/build_monthly_features.py work/scripts/refresh_capstone.py work/tests/test_refresh_capstone.py work/notebooks/w03_feature_leakage_check.ipynb work/outputs/feature_contract.json
git commit -m "Build leakage-safe monthly feature vector"
```

---

### Task 4: Signal audit with explicit verdicts

**Files:**

- Modify: `work/notebooks/w04_signal_audit.ipynb`
- Create and commit: `work/outputs/signal_audit.json`
- Create and commit: `work/figures/base_rate_by_month.png`
- Create and commit: `work/figures/signal_effects.png`

**Interfaces:**

- Consumes `model_examples.parquet` and `feature_contract.json`.
- Produces declared `keep`, `context only`, or `exclude` verdicts used by the baseline and model.

- [ ] **Step 1: Write a failing notebook-output check**

Require executed outputs for distributions, missingness by content/client coverage, monthly base rates, three named signal tests, effect sizes, and all three verdict labels.

- [ ] **Step 2: Verify the skeleton fails the check**

Expected: missing verdict and evidence outputs.

- [ ] **Step 3: Fill the signal-audit notebook**

Audit log impressions, prior momentum, valid average position, active days, content age, CTR, and availability flags. Each signal test must state the comparison, sample size, effect magnitude, verdict, and practical implication. Generate charts with titles, axis labels, notes explaining percentage scales, and no client-identifying labels.

- [ ] **Step 4: Execute and verify**

```powershell
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 work/notebooks/w04_signal_audit.ipynb
```

Confirm no error outputs, both figures exist and open, and the JSON receipt contains every verdict.

- [ ] **Step 5: Commit the audit**

```powershell
git add -- work/notebooks/w04_signal_audit.ipynb work/outputs/signal_audit.json work/figures/base_rate_by_month.png work/figures/signal_effects.png
git commit -m "Audit refresh opportunity signals"
```

---

### Task 5: Freeze and review the transparent baseline

**Files:**

- Modify: `work/scripts/refresh_capstone.py`
- Modify: `work/tests/test_refresh_capstone.py`
- Modify: `work/notebooks/w04_baseline_score.ipynb`
- Create locally, ignored: `work/outputs/baseline_queue.csv`
- Create and commit: `work/outputs/baseline_metrics.json`

**Interfaces:**

- Consumes only signals marked `keep` or `context only` before modeling.
- Produces the frozen baseline score, reason codes, top-20 review, and validation Precision@50 receipt.

- [ ] **Step 1: Add failing baseline behavior tests**

Use literal fixtures to assert that higher exposure plus worsening prior momentum ranks above low-exposure stable pages, score bounds are `[0, 100]`, ties are stable, and no target/outcome field changes the score.

- [ ] **Step 2: Verify the tests fail, then implement the frozen baseline**

Use the declared components only. Persist exact weights and thresholds in `baseline_metrics.json`; do not tune them on sealed May-to-June outcomes.

- [ ] **Step 3: Fill and execute the baseline notebook**

The notebook explains the rule, builds the ranked queue, shows reason-code counts, reviews the top 20, identifies weak picks, reports validation metrics, and repeats the leakage assertions.

```powershell
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 work/notebooks/w04_baseline_score.ipynb
```

- [ ] **Step 4: Verify and commit**

```powershell
.venv\Scripts\python.exe -m unittest work.tests.test_refresh_capstone -v
git check-ignore work/outputs/baseline_queue.csv
git add -- work/scripts/refresh_capstone.py work/tests/test_refresh_capstone.py work/notebooks/w04_baseline_score.ipynb work/outputs/baseline_metrics.json
git commit -m "Freeze transparent refresh baseline"
```

---

### Task 6: Train simple candidates and select on validation only

**Files:**

- Modify: `work/scripts/refresh_capstone.py`
- Modify: `work/tests/test_refresh_capstone.py`
- Modify: `work/notebooks/w05_model.ipynb`
- Create and commit: `work/outputs/model_selection.json`
- Create and commit: `work/figures/validation_precision_at_k.png`

**Interfaces:**

- Consumes frozen training/validation frames, selected features, and baseline scores.
- Produces logistic-regression and random-forest validation metrics, selected method, and validation error analysis. It does not read sealed labels.

- [ ] **Step 1: Add failing model-pipeline tests**

With a small synthetic frame, assert preprocessing fits on training rows, IDs and outcome fields are excluded, probability output stays in `[0, 1]`, and repeated runs with seed 42 are identical.

- [ ] **Step 2: Verify failure and implement candidate pipelines**

Use `ColumnTransformer`, median imputation plus missingness indicators for numeric features, explicit missing category for categorical features, standardized logistic regression, and `RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, min_samples_leaf=20)`. Select the simplest candidate that beats validation baseline Precision@50 without a stability failure.

- [ ] **Step 3: Fill and execute the model notebook**

Show method rationale, exact split counts, feature list, base rates, baseline and candidate metrics on identical rows, precision-at-K chart, and representative validation errors. Do not calculate sealed-month model metrics in this notebook.

```powershell
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 work/notebooks/w05_model.ipynb
```

- [ ] **Step 4: Verify and commit**

```powershell
.venv\Scripts\python.exe -m unittest work.tests.test_refresh_capstone -v
git add -- work/scripts/refresh_capstone.py work/tests/test_refresh_capstone.py work/notebooks/w05_model.ipynb work/outputs/model_selection.json work/figures/validation_precision_at_k.png
git commit -m "Compare refresh models on validation queue"
```

---

### Task 7: Honest validation and one-time sealed test

**Files:**

- Modify: `work/notebooks/w06_validation_audit.ipynb`
- Create and commit: `work/outputs/final_metrics.json`
- Create and commit: `work/figures/sealed_month_comparison.png`

**Interfaces:**

- Consumes the already selected method and untouched May-to-June sealed frame.
- Produces grouped diagnostics, one sealed result, leakage findings, error analysis, and final claim wording.

- [ ] **Step 1: Write a failing validation-receipt check**

Require the JSON receipt to name the selected method, baseline and model row counts, identical-row assertion, validation and sealed Precision@50, average precision, recall@50, lift@50, grouped diagnostic, and a `sealed_evaluated_once` flag.

- [ ] **Step 2: Verify the check fails before sealed evaluation**

Expected: `final_metrics.json` is absent.

- [ ] **Step 3: Fill the notebook and freeze claims before opening sealed labels**

First write the intended headline and failure-compatible wording. Then run the client-grouped development diagnostic, repeat all leakage checks, evaluate the selected method and baseline once on identical sealed rows, and record error cases and coverage limitations.

- [ ] **Step 4: Execute and verify**

```powershell
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 work/notebooks/w06_validation_audit.ipynb
```

Verify that paper claims match the measured result even if the baseline wins.

- [ ] **Step 5: Commit the validation audit**

```powershell
git add -- work/notebooks/w06_validation_audit.ipynb work/outputs/final_metrics.json work/figures/sealed_month_comparison.png
git commit -m "Validate refresh ranking on sealed month"
```

---

### Task 8: Ranked action playbook and public-safe exports

**Files:**

- Modify: `work/scripts/refresh_capstone.py`
- Modify: `work/tests/test_refresh_capstone.py`
- Modify: `work/notebooks/w07_action_playbook.ipynb`
- Create locally, ignored: `work/outputs/final_ranked_queue.csv`
- Create and commit: `work/outputs/action_summary.json`
- Create and commit: `work/figures/action_mix.png`

**Interfaces:**

- Consumes the selected ranking scores and safe context fields.
- Produces the final top-50 queue, conservative action mapping, reason codes, monitoring triggers, and aggregate public artifacts.

- [ ] **Step 1: Add failing action-policy tests**

Assert that every row receives one allowed action, high positive momentum can map to `protect`, uncertain/missing evidence maps to `monitor`, reason codes are nonempty, and no function emits auto-edit instructions or prohibited fields.

- [ ] **Step 2: Verify failure and implement action mapping**

Keep actions deterministic and human-review oriented. Confidence depends on rank band and evidence completeness, not model probability alone.

- [ ] **Step 3: Fill and execute the action notebook**

Show the top queue, reason-code and action distributions, intended use, no-go list, human review instructions, monitoring/retrain triggers, and public-safety column assertion. Export the ignored full CSV and committed aggregate JSON/chart.

- [ ] **Step 4: Verify and commit**

```powershell
.venv\Scripts\python.exe -m unittest work.tests.test_refresh_capstone -v
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 work/notebooks/w07_action_playbook.ipynb
git add -- work/scripts/refresh_capstone.py work/tests/test_refresh_capstone.py work/notebooks/w07_action_playbook.ipynb work/outputs/action_summary.json work/figures/action_mix.png
git commit -m "Build public-safe refresh action playbook"
```

---

### Task 9: Capstone notebook and research report source

**Files:**

- Modify: `work/notebooks/capstone.ipynb`
- Create: `work/capstone_report.md`

**Interfaces:**

- Consumes all committed metric receipts and figures.
- Produces a complete research narrative whose numbers are regenerated from receipts rather than typed independently.

- [ ] **Step 1: Write a failing completeness check**

Require title, five-sentence abstract, problem statement, data, methodology, results, limitations, ranked recommendations, reproducibility, data credit, demo outline, social-post cut, and employer-facing summary. Require every reported metric value to appear in `final_metrics.json` or another committed receipt.

- [ ] **Step 2: Verify the skeleton fails**

Expected: required sections and executed outputs are missing.

- [ ] **Step 3: Fill the capstone notebook and report**

Use `work/capstone_report_template.md` as the report structure. The notebook loads JSON receipts, displays the final comparison, embeds committed figures, shows a public-safe ranked sample, and states limitations before recommendations. Add the ML-12 closing material already required by `work/README.md`.

- [ ] **Step 4: Execute and cross-check every number**

```powershell
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 work/notebooks/capstone.ipynb
```

Run a script that extracts numeric strings from the results cells and confirms their source receipt; inspect all markdown for prohibited data and causal language.

- [ ] **Step 5: Commit the capstone source**

```powershell
git add -- work/notebooks/capstone.ipynb work/capstone_report.md
git commit -m "Complete refresh opportunity capstone report"
```

---

### Task 10: Static research paper and GitHub Pages deployment

**Files:**

- Create: `docs/index.html`
- Create: `docs/assets/paper.css`
- Copy committed paper figures into: `docs/assets/`
- Modify after live verification: `submission/paper_url.txt`

**Interfaces:**

- Consumes the capstone report, committed receipts, and charts.
- Produces the public paper URL `https://hasankhan05.github.io/flyrank-ml-internship/`.

- [ ] **Step 1: Write a failing static-paper check**

The checker parses `docs/index.html` and requires one `h1`, every required section heading, local chart assets with alt text, notebook/repository links, responsive viewport metadata, and an acknowledgment link whose text includes `Built on the FlyRank ML Internship dataset` and whose target is `https://flyrank.ai`.

- [ ] **Step 2: Verify the check fails because the page is absent**

- [ ] **Step 3: Build the semantic static paper**

Use no client-identifying examples and no external JavaScript. Include accessible tables, responsive CSS, a print stylesheet, concise chart interpretations, methodology details sufficient to rerun, and links to every executed notebook.

- [ ] **Step 4: Run local checks and inspect the page**

Serve `docs/` locally, inspect desktop and mobile widths, verify all local assets return HTTP 200, and run the static-paper checker.

- [ ] **Step 5: Commit and push the page before enabling Pages**

```powershell
git add -- docs/index.html docs/assets
git commit -m "Publish capstone research paper source"
git push origin publish:main
```

- [ ] **Step 6: Enable GitHub Pages from `main/docs` and wait for success**

Use the repository settings UI. Verify the exact public URL loads all sections and charts.

- [ ] **Step 7: Write and verify the mandatory submission file**

Write exactly:

```text
https://hasankhan05.github.io/flyrank-ml-internship/
```

Then verify one nonempty line, HTTP success, and required data credit before committing:

```powershell
git add -- submission/paper_url.txt
git commit -m "Record deployed capstone paper URL"
git push origin publish:main
```

---

### Task 11: End-to-end reproducibility and public verification

**Files:**

- Modify only files that fail verification.

**Interfaces:**

- Consumes the complete repository and public deployment.
- Produces the final evidence that the repo and paper meet the capstone definition of done.

- [ ] **Step 1: Run unit tests and validate every notebook**

```powershell
.venv\Scripts\python.exe -m unittest work.tests.test_refresh_capstone -v
```

For every required notebook, confirm valid notebook format, non-null execution counts for code cells, and zero error outputs.

- [ ] **Step 2: Run privacy and repository guards**

Search tracked files for credentials, raw domains/URLs/queries, local absolute paths, and AI-authorship claims. Confirm no Parquet, bulk CSV, archive, token, or `.env` file is tracked. Run `git diff --check` and the repository smoke-test workflow.

- [ ] **Step 3: Verify paper-to-receipt consistency**

Compare every headline number and chart caption against committed JSON receipts and executed notebook outputs. Confirm baseline and model rows are identical and the sealed-month flag is present exactly once.

- [ ] **Step 4: Verify public GitHub rendering**

Open every executed notebook on `github.com` and confirm outputs render. Open the deployed paper at desktop and mobile widths; verify all sections, figures, repository links, notebook links, and FlyRank credit.

- [ ] **Step 5: Verify final Git state**

```powershell
git status --short
git rev-parse HEAD
git rev-parse origin/main
```

Expected: clean working tree and identical local/remote hashes. All repository status checks must pass before submission.


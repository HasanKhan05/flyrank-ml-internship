# Refresh Opportunity Ranking Under Temporal Shift

- **Author:** Hasan Khan
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/HasanKhan05/flyrank-ml-internship
- **Date:** 2026-08-30

## 0. Abstract

This study asks which already-visible content items should receive limited human refresh review before the next month. It uses the FlyRank pseudonymized warehouse release dated 2026-07-03, aggregating daily search measurements into leakage-safe page-month examples. A fixed baseline, logistic regression, and random forest were compared with a time-aware training/validation/sealed design using Precision@50 as the primary metric. The transparent baseline won validation at 66% Precision@50 versus 54% for the strongest learned comparison, but the random forest reversed the order on the one-time sealed month at 98% versus 68%, revealing temporal instability rather than a universally superior method. The output is a ranked 50-item human-review queue with reason codes, confidence tiers, no-go rules, and monitoring triggers.

## 1. Problem framing

The decision is: **which content items deserve scarce editorial investigation first?** One unit is one pseudonymized content item at one monthly decision anchor. The output is a ranked review-priority score with a suggested human-review action, reasons, and confidence tier.

An editor can inspect the highest-ranked items for intent mismatch, stale or incomplete coverage, consolidation opportunity, or a reason to protect a currently growing page. A false positive wastes review time and may encourage an unnecessary change; a false negative misses a measured decline opportunity. Data helps because exposure, momentum, position, age, CTR, and measurement completeness interact and vary through time; no single threshold captured the observed patterns consistently.

## 2. Data safety

The work uses warehouse release flyrank_pseudonymized_warehouse_release_v20260703:

- dim_clients.parquet: 104 pseudonymized clients.
- dim_content.parquet: 519,606 structured content records.
- fact_content_daily_performance: 78,835,655 daily rows from 2025-01-27 through 2026-06-30.

Daily rows were aggregated by month, client_hash_id, and content_hash_id. Source months August 2025–June 2026 support prior-month features, feature anchors, and next-month labels. Training anchors are September 2025–March 2026; April predicts May for validation; May predicts June for the one-time sealed test.

Eligibility requires at least 100 feature-month impressions and at least 20 measured GSC days in both feature and outcome months. Missing GA4 or position measurement remains missing rather than becoming ordinary zero performance.

Client/content IDs are used only for joins, grouping, splitting, and a pseudonymous queue reference. Predictive inputs exclude IDs, outcome-month fields, raw names, domains, URLs, queries, titles, provider/model fields, and the fixed 90-day query table. No credential, client-identifying value, raw export, or private query is committed.

## 3. Baseline

The fixed 0–100 baseline was frozen before model training:

- 40% log-scaled current exposure.
- 30% negative prior-month momentum.
- 20% established-content age.
- 10% opportunity around a valid visible position.

Missing prior history contributes no momentum risk, and unavailable position contributes no position opportunity. On the identical 93,474-row April validation population, it achieved 66% Precision@50 against a 53.79% observed base rate: 1.23× Lift@50. Its average precision was 53.34%.

## 4. Model / analysis

The target is future_decline = 1 when next-month measured impressions are below 80% of feature-month impressions, provided both months meet the measurement-coverage rule.

Predictive features are impressions, clicks, CTR, prior impressions, prior clicks, impression momentum, average position, position availability, content age, content type, and search volume.

Numeric fields use training-fitted median imputation with missingness indicators; logistic inputs are standardized. Content type is imputed and one-hot encoded. Candidate methods are standardized logistic regression and a 300-tree random forest with seed 42 and min_samples_leaf=20.

The signal audit found measured directional separation for exposure, momentum, valid position, content age, and CTR. Active days stayed reviewer context; raw GSC/GA4 availability counts and sparse engaged sessions were excluded from prediction because they reflect access or coverage as much as content behavior.

## 5. Evaluation

The split is time-aware because the intended use is a future monthly queue. Selection uses September–March for training and April→May for validation. Only after the method and claim boundary were frozen was May→June evaluated once.

| Split and method | Rows | Base rate | Precision@50 | Average precision | Lift@50 |
|---|---:|---:|---:|---:|---:|
| Validation — fixed baseline | 93,474 | 53.79% | 66% | 53.34% | 1.23× |
| Validation — random forest | 93,474 | 53.79% | 54% | 60.42% | 1.00× |
| Sealed — fixed baseline | 93,048 | 64.04% | 68% | 67.02% | 1.06× |
| Sealed — random forest | 93,048 | 64.04% | 98% | 72.22% | 1.53× |

Logistic regression reached 40% validation Precision@50 and was not selected. The fixed baseline remained the validation-selected operational method. The random forest's sealed reversal is reported as temporal instability; it was not used to retune the selection rule.

False positives often combined high exposure, established age, and negative prior momentum without meeting the next-month decline threshold. False negatives illustrate that a top-50 cutoff has extremely low recall in a population with tens of thousands of positives. The system is therefore a capacity-constrained prioritizer, not a complete detector.

## 6. Interpretation

The strongest stable lesson is not that one algorithm always wins. March and the sealed period had markedly higher observed decline base rates, and Precision@K changed with review capacity. The forest ranked the full population better by average precision on validation, while the hand-built baseline was better at the exact top-50 cutoff; the order then reversed on the sealed month.

This supports three careful conclusions:

1. Exposure and recent momentum are useful review signals, but fixed weights can be brittle.
2. Nonlinear interactions can become valuable under a different month, but one strong sealed result is insufficient to replace the validation protocol.
3. Monitoring temporal base rate, missingness, and multiple K values is part of the system, not an optional afterthought.

## 7. Ranked recommendations

The May feature anchor produced 93,048 eligible items. The public-safe action layer exports the top 50 for human review. All 50 map to refresh_or_expand_review because every highest-ranked item had measured negative prior momentum; 34 also had meaningful exposure and all 50 had valid visible-position context. Confidence is high for 20 and medium for 30.

The action playbook is:

1. Review intent, accuracy, topic coverage, and competing content for the highest-ranked items.
2. Verify seasonality, tracking continuity, and business importance before recommending a change.
3. Protect positive-momentum items from unnecessary edits.
4. Consolidate or prune only after explicit cannibalization and value review.
5. Monitor incomplete-evidence cases instead of forcing an action.

No automatic edit, deletion, or causal refresh claim is authorized. Re-audit when the monthly base rate or key-feature missingness shifts by more than 10 percentage points, Precision@50 falls below the contemporaneous base rate, or the warehouse schema changes.

## 8. Reproducibility

The random seed is 42. From a fresh clone, create a virtual environment, install requirements, authenticate with hf auth login, run python -m work.scripts.build_monthly_features, then run python -m unittest work.tests.test_refresh_capstone -v.

Execute w03_feature_leakage_check.ipynb, w04_signal_audit.ipynb, w04_baseline_score.ipynb, w05_model.ipynb, w06_validation_audit.ipynb, w07_action_playbook.ipynb, and capstone.ipynb in order. Local Parquet/CSV caches are ignored. Committed JSON receipts contain the data contract, signal verdicts, baseline metrics, validation selection, sealed metrics, and action summary. The sealed-frame construction and sealed_evaluated_once receipt are both versioned.

## 9. Acknowledgments & data credit

[Built on the FlyRank ML Internship dataset](https://flyrank.ai).

## Closing material

### Five-minute demo outline

1. **0:00–0:40 — Decision:** explain the limited-review problem, unit of analysis, and cost of a wrong call.
2. **0:40–1:20 — Data safety:** show the monthly grain, time windows, eligibility rule, and exclusions.
3. **1:20–2:10 — Baseline and models:** explain the fixed score, logistic regression, random forest, and Precision@50.
4. **2:10–3:10 — Honest result:** show validation and sealed charts, emphasizing the temporal reversal and no retuning.
5. **3:10–4:20 — Action queue:** show reason codes, confidence, human review, and no-go rules.
6. **4:20–5:00 — Reproducibility:** point to notebooks, receipts, tests, and the deployed paper.

### Social-post cut

I built a leakage-safe refresh-opportunity ranking system on real pseudonymized search data. A transparent baseline beat two learned models at validation Precision@50, but a random forest reversed that result on the sealed month—an honest reminder that temporal stability matters as much as a headline metric. The final output is a reproducible 50-item human-review queue with reasons, confidence tiers, monitoring rules, and no causal claims.

### Employer-facing summary

I turned 78.8 million daily search-performance rows into a tested monthly decision system and public research paper. I designed time-aware validation, leakage and privacy checks, a transparent baseline, two learned comparisons, a one-time sealed audit, and a ranked human-review playbook. I reported the validation/sealed reversal without retuning, showing that I can balance modeling, operational action, and honest evidence.

import unittest

import pandas as pd

from work.scripts.build_monthly_features import (
    FEATURE_COLUMNS,
    add_derived_features,
    aggregate_monthly_frame,
    validate_feature_contract,
)
from work.scripts.refresh_capstone import (
    ALLOWED_ACTIONS,
    BASELINE_WEIGHTS,
    MODEL_FEATURES,
    PUBLIC_QUEUE_COLUMNS,
    build_candidate_pipelines,
    assign_action,
    baseline_score,
    make_examples,
    precision_at_k,
    split_frames,
)


class RefreshCapstoneTests(unittest.TestCase):
    def test_action_policy_covers_protect_monitor_and_public_safety(self):
        frame = pd.DataFrame(
            {
                "impressions": [2_000, 500],
                "prior_impressions": [1_000, 600],
                "content_age_days": [300, 300],
                "avg_position": [8.0, float("nan")],
                "position_available": [True, False],
            }
        )
        result = assign_action(frame, pd.Series([85.0, 85.0]))

        self.assertEqual(result.loc[0, "suggested_action"], "protect")
        self.assertEqual(result.loc[1, "suggested_action"], "monitor")
        self.assertTrue(set(result["suggested_action"]).issubset(ALLOWED_ACTIONS))
        self.assertTrue(result["reason_codes"].str.len().gt(0).all())
        prohibited = {"client_name", "domain", "url", "query", "title", "future_decline", "outcome_impressions"}
        self.assertTrue(prohibited.isdisjoint(PUBLIC_QUEUE_COLUMNS))

    def test_candidate_pipelines_are_safe_bounded_and_deterministic(self):
        frame = pd.DataFrame(
            {
                "impressions": [100, 200, 300, 400, 500, 600, 700, 800],
                "clicks": [1, 3, 2, 8, 5, 12, 7, 16],
                "ctr": [.01, .015, .007, .02, .01, .02, .01, .02],
                "prior_impressions": [120, 180, 350, 390, 550, 500, 800, 700],
                "prior_clicks": [1, 2, 3, 7, 6, 10, 8, 14],
                "impression_momentum": [.83, 1.11, .86, 1.03, .91, 1.2, .88, 1.14],
                "avg_position": [8, 12, 5, 20, 7, 15, 9, 18],
                "position_available": [True] * 8,
                "content_age_days": [100, 200, 300, 400, 120, 220, 320, 420],
                "content_type": ["article", "landing"] * 4,
                "search_volume": [10, 20, 30, 40, 15, 25, 35, 45],
                "client_hash_id": [f"client_{i}" for i in range(8)],
                "content_hash_id": [f"content_{i}" for i in range(8)],
                "future_decline": [0, 1, 0, 1, 0, 1, 0, 1],
                "outcome_impressions": [110, 100, 320, 200, 520, 250, 710, 300],
            }
        )
        labels = frame["future_decline"]
        first = build_candidate_pipelines(random_seed=42)["logistic_regression"]
        second = build_candidate_pipelines(random_seed=42)["logistic_regression"]
        first.fit(frame[list(MODEL_FEATURES)], labels)
        second.fit(frame[list(MODEL_FEATURES)], labels)
        first_probability = first.predict_proba(frame[list(MODEL_FEATURES)])[:, 1]
        second_probability = second.predict_proba(frame[list(MODEL_FEATURES)])[:, 1]

        self.assertTrue(((first_probability >= 0) & (first_probability <= 1)).all())
        self.assertEqual(first_probability.tolist(), second_probability.tolist())
        self.assertTrue({"client_hash_id", "content_hash_id", "future_decline", "outcome_impressions"}.isdisjoint(MODEL_FEATURES))

    def test_frozen_baseline_prioritizes_exposure_and_worsening_momentum(self):
        frame = pd.DataFrame(
            {
                "impressions": [8_000, 200, 8_000],
                "prior_impressions": [16_000, 200, 16_000],
                "content_age_days": [500, 500, 500],
                "avg_position": [8.0, 8.0, 8.0],
                "position_available": [True, True, True],
            }
        )
        scores = baseline_score(frame)

        self.assertEqual(sum(BASELINE_WEIGHTS.values()), 1.0)
        self.assertGreater(scores.iloc[0], scores.iloc[1])
        self.assertEqual(scores.iloc[0], scores.iloc[2])

    def test_monthly_aggregation_preserves_availability_and_safe_features(self):
        daily = pd.DataFrame(
            {
                "report_date": pd.to_datetime(
                    [
                        "2026-04-01", "2026-04-02", "2026-05-01",
                        "2026-05-02", "2026-06-01", "2026-06-02",
                        "2026-04-01", "2026-05-01", "2026-06-01",
                    ]
                ),
                "client_hash_id": ["client_a"] * 6 + ["client_b"] * 3,
                "content_hash_id": ["content_a"] * 6 + ["content_b"] * 3,
                "gsc_data_available": [True] * 9,
                "ga4_data_available": [False, False, True, False, True, True, False, False, False],
                "gsc_impressions": [100, 100, 90, 60, 50, 40, 120, 110, 100],
                "gsc_clicks": [10, 10, 9, 6, 5, 4, 12, 11, 10],
                "gsc_sum_position": [800, 0, 900, 600, 500, 400, 600, 660, 700],
                "ga4_engaged_sessions": [0, 0, 7, 0, 4, 3, 0, 0, 0],
                "content_created_date": pd.to_datetime(["2025-01-01"] * 9),
            }
        )

        monthly = aggregate_monthly_frame(daily)
        examples = add_derived_features(make_examples(monthly))

        april_a = monthly.loc[
            monthly["client_hash_id"].eq("client_a")
            & monthly["month"].eq(pd.Timestamp("2026-04-01"))
        ].iloc[0]
        april_b = monthly.loc[
            monthly["client_hash_id"].eq("client_b")
            & monthly["month"].eq(pd.Timestamp("2026-04-01"))
        ].iloc[0]
        self.assertTrue(april_a["position_available"])
        self.assertAlmostEqual(april_a["avg_position"], 8.0)
        self.assertEqual(april_a["ga4_available_days"], 0)
        self.assertTrue(pd.isna(april_a["engaged_sessions"]))
        self.assertTrue(pd.isna(april_b["engaged_sessions"]))
        self.assertFalse(any(name.startswith("outcome_") for name in FEATURE_COLUMNS))
        self.assertNotIn("future_decline", FEATURE_COLUMNS)
        validate_feature_contract(examples, FEATURE_COLUMNS)

    def test_future_label_uses_the_next_consecutive_month_only(self):
        monthly = pd.DataFrame(
            {
                "month": pd.to_datetime(
                    ["2026-04-01", "2026-05-01", "2026-06-01", "2026-04-01", "2026-06-01"]
                ),
                "client_hash_id": ["client_a"] * 3 + ["client_b"] * 2,
                "content_hash_id": ["content_a"] * 3 + ["content_b"] * 2,
                "impressions": [200, 170, 90, 300, 100],
                "clicks": [20, 17, 9, 30, 10],
                "avg_position": [8.0, 9.0, 10.0, 5.0, 12.0],
                "position_available": [True] * 5,
                "active_days": [25, 24, 20, 28, 20],
                "content_age_days": [300, 330, 360, 400, 460],
            }
        )

        result = make_examples(monthly)

        self.assertEqual(result["future_decline"].tolist(), [0, 1])
        self.assertEqual(result["outcome_impressions"].tolist(), [170, 90])
        self.assertEqual(result["content_hash_id"].unique().tolist(), ["content_a"])

    def test_precision_at_k_uses_descending_scores(self):
        labels = pd.Series([0, 1, 1, 0])
        scores = pd.Series([0.2, 0.9, 0.8, 0.1])

        self.assertEqual(precision_at_k(labels, scores, k=2), 1.0)

    def test_splits_keep_may_to_june_frame_sealed(self):
        frame = pd.DataFrame(
            {"month": pd.to_datetime(["2026-03-01", "2026-04-01", "2026-05-01"])}
        )

        train, validation, sealed = split_frames(frame)

        self.assertEqual(train["month"].dt.month.tolist(), [3])
        self.assertEqual(validation["month"].dt.month.tolist(), [4])
        self.assertEqual(sealed["month"].dt.month.tolist(), [5])

    def test_baseline_is_bounded_and_does_not_read_outcome_columns(self):
        frame = pd.DataFrame(
            {
                "impressions": [100, 10_000],
                "prior_impressions": [120, 20_000],
                "content_age_days": [100, 500],
                "avg_position": [40.0, 8.0],
                "position_available": [True, True],
                "future_decline": [0, 1],
                "outcome_impressions": [500, 1],
            }
        )

        first = baseline_score(frame)
        mutated = frame.assign(future_decline=[1, 0], outcome_impressions=[0, 999_999])
        second = baseline_score(mutated)

        self.assertTrue(first.between(0, 100).all())
        self.assertEqual(first.tolist(), second.tolist())
        self.assertGreater(first.iloc[1], first.iloc[0])

    def test_action_output_is_human_review_only(self):
        frame = pd.DataFrame(
            {
                "impressions": [5_000, 2_000, 150],
                "prior_impressions": [10_000, 1_000, 150],
                "content_age_days": [500, 400, 100],
                "avg_position": [8.0, 5.0, float("nan")],
                "position_available": [True, True, False],
            }
        )
        scores = pd.Series([95.0, 80.0, 20.0])

        actions = assign_action(frame, scores)

        allowed = {"refresh_or_expand_review", "protect", "consolidate_or_prune_review", "monitor"}
        self.assertTrue(set(actions["suggested_action"]).issubset(allowed))
        self.assertTrue(actions["reason_codes"].str.len().gt(0).all())
        self.assertTrue(actions["confidence_tier"].isin(["high", "medium", "low"]).all())


if __name__ == "__main__":
    unittest.main()

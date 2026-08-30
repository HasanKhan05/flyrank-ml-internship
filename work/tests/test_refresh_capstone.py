import unittest

import pandas as pd

from work.scripts.build_monthly_features import (
    FEATURE_COLUMNS,
    add_derived_features,
    aggregate_monthly_frame,
    validate_feature_contract,
)
from work.scripts.refresh_capstone import (
    BASELINE_WEIGHTS,
    assign_action,
    baseline_score,
    make_examples,
    precision_at_k,
    split_frames,
)


class RefreshCapstoneTests(unittest.TestCase):
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

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

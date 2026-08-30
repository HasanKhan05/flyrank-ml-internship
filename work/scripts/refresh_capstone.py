"""Reusable, public-safe logic for the Refresh Opportunity capstone."""

from __future__ import annotations

import numpy as np
import pandas as pd


TRAIN_MONTHS = pd.period_range("2025-09", "2026-03", freq="M")
VALIDATION_MONTH = pd.Period("2026-04", freq="M")
SEALED_MONTH = pd.Period("2026-05", freq="M")
DECLINE_RATIO = 0.80
MIN_GSC_COVERAGE_DAYS = 20
RANDOM_SEED = 42


def make_examples(monthly: pd.DataFrame) -> pd.DataFrame:
    """Create consecutive-month examples without exposing outcome data as features."""
    required = {
        "month",
        "client_hash_id",
        "content_hash_id",
        "impressions",
        "clicks",
        "avg_position",
        "active_days",
        "content_age_days",
    }
    missing = sorted(required.difference(monthly.columns))
    if missing:
        raise ValueError(f"Missing monthly columns: {missing}")

    frame = monthly.copy()
    frame["month"] = pd.to_datetime(frame["month"]).dt.to_period("M")
    group_columns = ["client_hash_id", "content_hash_id"]
    frame = frame.sort_values(group_columns + ["month"], kind="stable").reset_index(drop=True)
    grouped = frame.groupby(group_columns, sort=False, observed=True)

    frame["prior_month"] = grouped["month"].shift(1)
    frame["prior_impressions"] = grouped["impressions"].shift(1)
    frame["prior_clicks"] = grouped["clicks"].shift(1)
    frame["outcome_month"] = grouped["month"].shift(-1)
    frame["outcome_impressions"] = grouped["impressions"].shift(-1)
    if "gsc_available_days" in frame.columns:
        frame["outcome_gsc_available_days"] = grouped["gsc_available_days"].shift(-1)

    prior_is_consecutive = frame["prior_month"] == frame["month"].map(lambda value: value - 1)
    frame.loc[~prior_is_consecutive, ["prior_impressions", "prior_clicks"]] = np.nan
    outcome_is_consecutive = frame["outcome_month"] == frame["month"].map(lambda value: value + 1)

    eligible = outcome_is_consecutive & frame["impressions"].ge(100)
    if "gsc_available_days" in frame.columns:
        eligible &= frame["gsc_available_days"].ge(MIN_GSC_COVERAGE_DAYS)
        eligible &= frame["outcome_gsc_available_days"].ge(MIN_GSC_COVERAGE_DAYS)
    examples = frame.loc[eligible].copy()
    examples["future_decline"] = (
        examples["outcome_impressions"] < DECLINE_RATIO * examples["impressions"]
    ).astype("int8")
    examples["month"] = examples["month"].dt.to_timestamp()
    examples["outcome_month"] = examples["outcome_month"].dt.to_timestamp()
    examples["prior_month"] = examples["prior_month"].dt.to_timestamp()
    return examples.reset_index(drop=True)


def precision_at_k(labels: pd.Series, scores: pd.Series, k: int = 50) -> float:
    """Return positive-label share among the highest scores using stable tie handling."""
    label_values = np.asarray(labels)
    score_values = np.asarray(scores, dtype=float)
    if len(label_values) == 0 or k <= 0:
        return float("nan")
    if len(label_values) != len(score_values):
        raise ValueError("labels and scores must have the same length")
    order = np.argsort(-score_values, kind="stable")[: min(k, len(score_values))]
    return float(label_values[order].mean())


def _fixed_exposure_component(impressions: pd.Series) -> pd.Series:
    values = pd.to_numeric(impressions, errors="coerce").fillna(0).clip(lower=0)
    return (np.log1p(values) / np.log1p(100_000)).clip(0, 1)


def baseline_score(frame: pd.DataFrame) -> pd.Series:
    """Score pre-decision review priority with transparent fixed components."""
    required = {
        "impressions",
        "prior_impressions",
        "content_age_days",
        "avg_position",
        "position_available",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing baseline columns: {missing}")

    impressions = pd.to_numeric(frame["impressions"], errors="coerce").fillna(0).clip(lower=0)
    prior = pd.to_numeric(frame["prior_impressions"], errors="coerce")
    ratio = impressions.div(prior.where(prior.gt(0)))
    momentum_risk = (1 - ratio).clip(0, 1).fillna(0)
    age = pd.to_numeric(frame["content_age_days"], errors="coerce").fillna(0)
    established = (age / 365).clip(0, 1)
    position = pd.to_numeric(frame["avg_position"], errors="coerce")
    position_available = frame["position_available"].fillna(False).astype(bool)
    position_opportunity = (1 - (position - 10).abs() / 10).clip(0, 1)
    position_opportunity = position_opportunity.where(position_available & position.gt(0), 0).fillna(0)

    score = 100 * (
        0.40 * _fixed_exposure_component(impressions)
        + 0.30 * momentum_risk
        + 0.20 * established
        + 0.10 * position_opportunity
    )
    return score.clip(0, 100).rename("baseline_score")


def assign_action(frame: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
    """Map ranked evidence to conservative human-review actions and reasons."""
    if len(frame) != len(scores):
        raise ValueError("frame and scores must have the same length")

    result = pd.DataFrame(index=frame.index)
    impressions = pd.to_numeric(frame["impressions"], errors="coerce").fillna(0)
    prior = pd.to_numeric(frame["prior_impressions"], errors="coerce")
    momentum = impressions.div(prior.where(prior.gt(0)))
    position = pd.to_numeric(frame["avg_position"], errors="coerce")
    position_available = frame["position_available"].fillna(False).astype(bool)
    score_values = pd.Series(np.asarray(scores, dtype=float), index=frame.index)

    actions: list[str] = []
    reasons: list[str] = []
    confidence: list[str] = []
    for idx in frame.index:
        row_reasons: list[str] = []
        if impressions.loc[idx] >= 3_000:
            row_reasons.append("meaningful_exposure")
        elif impressions.loc[idx] < 500:
            row_reasons.append("limited_exposure")
        if pd.notna(momentum.loc[idx]) and momentum.loc[idx] < 0.80:
            row_reasons.append("negative_prior_momentum")
        if pd.notna(momentum.loc[idx]) and momentum.loc[idx] >= 1.20:
            row_reasons.append("positive_prior_momentum")
        if position_available.loc[idx] and 0 < position.loc[idx] <= 20:
            row_reasons.append("visible_position")
        if not position_available.loc[idx] or pd.isna(position.loc[idx]):
            row_reasons.append("position_unavailable")

        if impressions.loc[idx] < 100 or not position_available.loc[idx]:
            action = "monitor"
        elif pd.notna(momentum.loc[idx]) and momentum.loc[idx] >= 1.20:
            action = "protect"
        elif score_values.loc[idx] >= 70 and pd.notna(momentum.loc[idx]) and momentum.loc[idx] < 0.90:
            action = "refresh_or_expand_review"
        elif score_values.loc[idx] >= 50 and impressions.loc[idx] < 1_000:
            action = "consolidate_or_prune_review"
        else:
            action = "monitor"

        complete_evidence = position_available.loc[idx] and pd.notna(momentum.loc[idx])
        if score_values.loc[idx] >= 80 and complete_evidence:
            tier = "high"
        elif score_values.loc[idx] >= 50 and complete_evidence:
            tier = "medium"
        else:
            tier = "low"

        actions.append(action)
        reasons.append(";".join(row_reasons or ["insufficient_evidence"]))
        confidence.append(tier)

    result["suggested_action"] = actions
    result["reason_codes"] = reasons
    result["confidence_tier"] = confidence
    return result


def split_frames(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return development, validation, and sealed frames by feature month."""
    months = pd.to_datetime(frame["month"]).dt.to_period("M")
    train = frame.loc[months.isin(TRAIN_MONTHS)].copy()
    validation = frame.loc[months.eq(VALIDATION_MONTH)].copy()
    sealed = frame.loc[months.eq(SEALED_MONTH)].copy()
    return train, validation, sealed

"""Build leakage-safe page-month features for the refresh opportunity capstone."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import duckdb
import numpy as np
import pandas as pd
from huggingface_hub import get_token


HF_ROOT = "hf://datasets/FlyRank/internship-warehouse"
SOURCE_START = "2025-08-01"
SOURCE_END_EXCLUSIVE = "2026-07-01"

FEATURE_COLUMNS = (
    "impressions",
    "clicks",
    "ctr",
    "prior_impressions",
    "prior_clicks",
    "impression_momentum",
    "avg_position",
    "position_available",
    "active_days",
    "gsc_available_days",
    "ga4_available_days",
    "engaged_sessions",
    "content_age_days",
    "content_type",
    "search_volume",
)

FORBIDDEN_FEATURE_PARTS = (
    "outcome_",
    "future_",
    "target",
    "label",
    "client_hash_id",
    "content_hash_id",
    "url",
    "domain",
    "query",
    "title",
)


FEATURE_NOTES = {
    "impressions": ("feature month", "measured GSC exposure; eligible rows require at least 100"),
    "clicks": ("feature month", "measured GSC clicks; not filled across unavailable days"),
    "ctr": ("feature month", "clicks divided by impressions; missing when denominator is zero"),
    "prior_impressions": ("prior month", "previous measured exposure; missing without a consecutive month"),
    "prior_clicks": ("prior month", "previous measured clicks; missing without a consecutive month"),
    "impression_momentum": ("feature/prior month", "current divided by prior impressions; missing when prior is zero"),
    "avg_position": ("feature month", "impression-weighted valid GSC position only"),
    "position_available": ("feature month", "explicit flag for a valid position denominator"),
    "active_days": ("feature month", "days with measured positive GSC impressions"),
    "gsc_available_days": ("feature month", "measurement coverage; labels require 20 days in t and t+1"),
    "ga4_available_days": ("feature month", "GA4 measurement coverage, not engagement"),
    "engaged_sessions": ("feature month", "NULL when GA4 has no measured days; never zero-filled"),
    "content_age_days": ("feature month", "age at feature-month end from safe content metadata"),
    "content_type": ("known by feature month", "structured public-safe category; unknown is explicit"),
    "search_volume": ("known by feature month", "structured opportunity context; missing remains missing"),
}


def feature_contract_records() -> list[dict[str, object]]:
    """Return the committed, public-safe feature/label/context/exclusion receipt."""
    records = [
        {
            "field": field,
            "bucket": "feature",
            "source_window": FEATURE_NOTES[field][0],
            "availability_rule": FEATURE_NOTES[field][1],
        }
        for field in FEATURE_COLUMNS
    ]
    records.extend(
        [
            {"field": "future_decline", "bucket": "label", "source_window": "next month", "availability_rule": "1 only when next-month impressions are below 80% of feature month with at least 20 measured GSC days in both months"},
            {"field": "outcome_impressions", "bucket": "label", "source_window": "next month", "availability_rule": "label source only; never predictive"},
            {"field": "client_hash_id", "bucket": "context", "source_window": "stable pseudonym", "availability_rule": "grouping, joins, and validation only"},
            {"field": "content_hash_id", "bucket": "context", "source_window": "stable pseudonym", "availability_rule": "queue reference only; never predictive"},
            {"field": "month", "bucket": "context", "source_window": "feature month", "availability_rule": "time splitting only"},
            {"field": "raw names/domains/URLs/queries/titles", "bucket": "excluded", "source_window": "not used", "availability_rule": "private or identifying"},
            {"field": "all outcome-month fields", "bucket": "excluded", "source_window": "future", "availability_rule": "would leak the answer"},
            {"field": "IDs as model features", "bucket": "excluded", "source_window": "stable pseudonym", "availability_rule": "non-portable identity signal"},
            {"field": "provider_used/model_used", "bucket": "excluded", "source_window": "content metadata", "availability_rule": "not needed for the editorial decision"},
            {"field": "fact_content_query_90d", "bucket": "excluded", "source_window": "fixed rolling window", "availability_rule": "overlaps final outcomes and creates leakage risk"},
        ]
    )
    return records


def aggregate_monthly_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a small in-memory daily fixture with warehouse-equivalent rules."""
    required = {
        "report_date", "client_hash_id", "content_hash_id", "gsc_data_available",
        "ga4_data_available", "gsc_impressions", "gsc_clicks", "gsc_sum_position",
        "ga4_engaged_sessions", "content_created_date",
    }
    missing = sorted(required.difference(daily.columns))
    if missing:
        raise ValueError(f"Missing daily columns: {missing}")

    frame = daily.copy()
    frame["report_date"] = pd.to_datetime(frame["report_date"])
    frame["content_created_date"] = pd.to_datetime(frame["content_created_date"])
    frame["month"] = frame["report_date"].dt.to_period("M").dt.to_timestamp()
    frame["gsc_ok"] = frame["gsc_data_available"].fillna(False).astype(bool)
    frame["ga4_ok"] = frame["ga4_data_available"].fillna(False).astype(bool)
    frame["valid_position"] = (
        frame["gsc_ok"]
        & pd.to_numeric(frame["gsc_impressions"], errors="coerce").gt(0)
        & pd.to_numeric(frame["gsc_sum_position"], errors="coerce").gt(0)
    )
    keys = ["month", "client_hash_id", "content_hash_id"]

    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(keys, sort=True, observed=True):
        gsc = group.loc[group["gsc_ok"]]
        valid_position = group.loc[group["valid_position"]]
        ga4 = group.loc[group["ga4_ok"]]
        position_denominator = pd.to_numeric(
            valid_position["gsc_impressions"], errors="coerce"
        ).sum()
        position_numerator = pd.to_numeric(
            valid_position["gsc_sum_position"], errors="coerce"
        ).sum()
        impressions = pd.to_numeric(gsc["gsc_impressions"], errors="coerce").sum()
        clicks = pd.to_numeric(gsc["gsc_clicks"], errors="coerce").sum()
        month_end = pd.Timestamp(key[0]) + pd.offsets.MonthEnd(0)
        created = group["content_created_date"].min()
        rows.append(
            {
                "month": key[0],
                "client_hash_id": key[1],
                "content_hash_id": key[2],
                "impressions": impressions,
                "clicks": clicks,
                "ctr": clicks / impressions if impressions > 0 else np.nan,
                "avg_position": (
                    position_numerator / position_denominator
                    if position_denominator > 0 else np.nan
                ),
                "position_available": bool(position_denominator > 0),
                "active_days": int((pd.to_numeric(gsc["gsc_impressions"], errors="coerce") > 0).sum()),
                "gsc_available_days": int(group["gsc_ok"].sum()),
                "ga4_available_days": int(group["ga4_ok"].sum()),
                "engaged_sessions": (
                    pd.to_numeric(ga4["ga4_engaged_sessions"], errors="coerce").sum()
                    if len(ga4) else np.nan
                ),
                "content_age_days": max((month_end - created).days, 0),
                "content_type": (
                    group["content_type"].dropna().iloc[0]
                    if "content_type" in group and group["content_type"].notna().any()
                    else "unknown"
                ),
                "search_volume": (
                    pd.to_numeric(group["search_volume"], errors="coerce").max()
                    if "search_volume" in group else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(keys, kind="stable").reset_index(drop=True)


def validate_feature_contract(frame: pd.DataFrame, feature_columns: Sequence[str]) -> None:
    """Reject missing, identifying, or future-derived predictive columns."""
    columns = list(feature_columns)
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Feature columns absent from examples: {missing}")
    forbidden = [
        name for name in columns
        if any(part in name.lower() for part in FORBIDDEN_FEATURE_PARTS)
    ]
    if forbidden:
        raise ValueError(f"Forbidden predictive columns: {forbidden}")
    if len(columns) != len(set(columns)):
        raise ValueError("Feature columns must be unique")


def add_derived_features(examples: pd.DataFrame) -> pd.DataFrame:
    """Create pre-decision ratios without filling unavailable measurements with zero."""
    frame = examples.copy()
    impressions = pd.to_numeric(frame["impressions"], errors="coerce")
    clicks = pd.to_numeric(frame["clicks"], errors="coerce")
    prior_impressions = pd.to_numeric(frame["prior_impressions"], errors="coerce")
    frame["ctr"] = clicks.div(impressions.where(impressions.gt(0)))
    frame["impression_momentum"] = impressions.div(prior_impressions.where(prior_impressions.gt(0)))
    return frame


def _connection(extension_directory: Path) -> duckdb.DuckDBPyConnection:
    token = os.getenv("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError("Hugging Face authentication is required; use the device login first")
    connection = duckdb.connect()
    extension_directory.mkdir(parents=True, exist_ok=True)
    connection.execute("SET extension_directory=?", [str(extension_directory.resolve())])
    escaped_token = token.replace("'", "''")
    connection.execute(
        f"CREATE OR REPLACE SECRET hf_token (TYPE huggingface, TOKEN '{escaped_token}')"
    )
    return connection


def build_remote_cache(output_path: Path, extension_directory: Path) -> None:
    """Aggregate declared warehouse months and write only the ignored page-month cache."""
    connection = _connection(extension_directory)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fact = f"read_parquet('{HF_ROOT}/fact_content_daily_performance/**/*.parquet', hive_partitioning=true)"
    content = f"read_parquet('{HF_ROOT}/dim_content.parquet')"
    query = f"""
        WITH content_safe AS (
            SELECT
                client_hash_id,
                content_hash_id,
                min(content_created_date) AS content_created_date,
                any_value(content_type) AS content_type,
                max(search_volume) AS search_volume
            FROM {content}
            GROUP BY 1, 2
        ), monthly AS (
            SELECT
                date_trunc('month', f.report_date)::DATE AS month,
                f.client_hash_id,
                f.content_hash_id,
                sum(f.gsc_impressions) FILTER (WHERE f.gsc_data_available IS TRUE) AS impressions,
                sum(f.gsc_clicks) FILTER (WHERE f.gsc_data_available IS TRUE) AS clicks,
                sum(f.gsc_sum_position) FILTER (
                    WHERE f.gsc_data_available IS TRUE
                      AND f.gsc_impressions > 0 AND f.gsc_sum_position > 0
                ) / nullif(sum(f.gsc_impressions) FILTER (
                    WHERE f.gsc_data_available IS TRUE
                      AND f.gsc_impressions > 0 AND f.gsc_sum_position > 0
                ), 0) AS avg_position,
                count(*) FILTER (
                    WHERE f.gsc_data_available IS TRUE AND f.gsc_impressions > 0
                ) AS active_days,
                count(*) FILTER (WHERE f.gsc_data_available IS TRUE) AS gsc_available_days,
                count(*) FILTER (WHERE f.ga4_data_available IS TRUE) AS ga4_available_days,
                sum(f.ga4_engaged_sessions) FILTER (WHERE f.ga4_data_available IS TRUE) AS engaged_sessions
            FROM {fact} AS f
            WHERE f.report_date >= DATE '{SOURCE_START}'
              AND f.report_date < DATE '{SOURCE_END_EXCLUSIVE}'
            GROUP BY 1, 2, 3
        )
        SELECT
            m.month,
            m.client_hash_id,
            m.content_hash_id,
            coalesce(m.impressions, 0)::BIGINT AS impressions,
            coalesce(m.clicks, 0)::BIGINT AS clicks,
            m.clicks / nullif(m.impressions, 0) AS ctr,
            m.avg_position,
            (m.avg_position IS NOT NULL) AS position_available,
            m.active_days::INTEGER AS active_days,
            m.gsc_available_days::INTEGER AS gsc_available_days,
            m.ga4_available_days::INTEGER AS ga4_available_days,
            m.engaged_sessions,
            greatest(date_diff('day', c.content_created_date, last_day(m.month)), 0)::INTEGER AS content_age_days,
            coalesce(c.content_type, 'unknown') AS content_type,
            c.search_volume
        FROM monthly AS m
        LEFT JOIN content_safe AS c USING (client_hash_id, content_hash_id)
        ORDER BY m.client_hash_id, m.content_hash_id, m.month
    """
    escaped_output = str(output_path.resolve()).replace("'", "''")
    connection.execute(
        f"COPY ({query}) TO '{escaped_output}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("work/outputs/page_month_features.parquet")
    )
    parser.add_argument(
        "--extension-directory", type=Path,
        default=Path("work/outputs/.duckdb_extensions"),
    )
    args = parser.parse_args()
    build_remote_cache(args.output, args.extension_directory)
    print(f"Wrote monthly feature cache: {args.output}")


if __name__ == "__main__":
    main()

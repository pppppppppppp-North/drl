from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    date_column: str = "date"


@dataclass(frozen=True)
class WalkForwardConfig:
    train_window: int = 504
    validation_window: int = 126
    test_window: int = 126
    step_size: int = 126
    date_column: str = "date"
    expanding_train: bool = False


def _validate_split_config(config: SplitConfig) -> None:
    if not 0.0 < config.train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 < config.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if config.train_fraction + config.validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must leave a non-empty test split")


def split_config_from_mapping(mapping: dict | None) -> SplitConfig:
    mapping = mapping or {}
    return SplitConfig(
        train_fraction=float(mapping.get("train_fraction", SplitConfig.train_fraction)),
        validation_fraction=float(mapping.get("validation_fraction", SplitConfig.validation_fraction)),
        date_column=str(mapping.get("date_column", SplitConfig.date_column)),
    )


def walk_forward_config_from_mapping(mapping: dict | None) -> WalkForwardConfig:
    mapping = mapping or {}
    return WalkForwardConfig(
        train_window=int(mapping.get("train_window", WalkForwardConfig.train_window)),
        validation_window=int(mapping.get("validation_window", WalkForwardConfig.validation_window)),
        test_window=int(mapping.get("test_window", WalkForwardConfig.test_window)),
        step_size=int(mapping.get("step_size", WalkForwardConfig.step_size)),
        date_column=str(mapping.get("date_column", WalkForwardConfig.date_column)),
        expanding_train=bool(mapping.get("expanding_train", WalkForwardConfig.expanding_train)),
    )


def _validate_walk_forward_config(config: WalkForwardConfig) -> None:
    for field in ("train_window", "validation_window", "test_window", "step_size"):
        value = getattr(config, field)
        if value <= 0:
            raise ValueError(f"{field} must be positive")


def chronological_split(features: pd.DataFrame, config: SplitConfig | None = None) -> dict[str, pd.DataFrame]:
    config = config or SplitConfig()
    _validate_split_config(config)
    if config.date_column not in features.columns:
        raise ValueError(f"missing date column: {config.date_column}")

    df = features.copy()
    df[config.date_column] = pd.to_datetime(df[config.date_column])
    dates = pd.Index(df[config.date_column].drop_duplicates().sort_values())
    if len(dates) < 3:
        raise ValueError("at least three unique dates are required for train/validation/test splits")

    train_end = max(1, int(len(dates) * config.train_fraction))
    validation_end = max(train_end + 1, int(len(dates) * (config.train_fraction + config.validation_fraction)))
    if validation_end >= len(dates):
        validation_end = len(dates) - 1
    if train_end >= validation_end:
        train_end = validation_end - 1

    split_dates = {
        "train": dates[:train_end],
        "validation": dates[train_end:validation_end],
        "test": dates[validation_end:],
    }
    return {
        name: df[df[config.date_column].isin(values)].sort_values([config.date_column, "ticker"]).reset_index(drop=True)
        for name, values in split_dates.items()
    }


def _select_dates(df: pd.DataFrame, dates: pd.Index, date_column: str) -> pd.DataFrame:
    sort_columns = [date_column, "ticker"] if "ticker" in df.columns else [date_column]
    return df[df[date_column].isin(dates)].sort_values(sort_columns).reset_index(drop=True)


def generate_walk_forward_splits(
    features: pd.DataFrame,
    config: WalkForwardConfig | None = None,
) -> list[dict[str, pd.DataFrame]]:
    config = config or WalkForwardConfig()
    _validate_walk_forward_config(config)
    if config.date_column not in features.columns:
        raise ValueError(f"missing date column: {config.date_column}")

    df = features.copy()
    df[config.date_column] = pd.to_datetime(df[config.date_column])
    dates = pd.Index(df[config.date_column].drop_duplicates().sort_values())
    required_dates = config.train_window + config.validation_window + config.test_window
    if len(dates) < required_dates:
        raise ValueError(
            f"walk-forward config requires at least {required_dates} unique dates; found {len(dates)}"
        )

    windows: list[dict[str, pd.DataFrame]] = []
    start = 0
    while start + required_dates <= len(dates):
        train_start = 0 if config.expanding_train else start
        train_end = start + config.train_window
        validation_end = train_end + config.validation_window
        test_end = validation_end + config.test_window
        windows.append(
            {
                "train": _select_dates(df, dates[train_start:train_end], config.date_column),
                "validation": _select_dates(df, dates[train_end:validation_end], config.date_column),
                "test": _select_dates(df, dates[validation_end:test_end], config.date_column),
            }
        )
        start += config.step_size

    return windows


def select_chronological_split(
    features: pd.DataFrame,
    split: str = "all",
    config: SplitConfig | None = None,
) -> pd.DataFrame:
    if split == "all":
        return features.copy().reset_index(drop=True)
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {('all',) + SPLIT_NAMES}; got {split!r}")
    return chronological_split(features, config)[split]


def _split_summary_row(name: str, split: pd.DataFrame, date_column: str = "date") -> dict:
    dates = pd.to_datetime(split[date_column])
    return {
        "split": name,
        "start_date": dates.min().date().isoformat() if not split.empty else None,
        "end_date": dates.max().date().isoformat() if not split.empty else None,
        "rows": int(len(split)),
        "unique_dates": int(dates.nunique()),
        "tickers": int(split["ticker"].nunique()) if "ticker" in split else 0,
    }


def split_summary(splits: dict[str, pd.DataFrame], date_column: str = "date") -> pd.DataFrame:
    rows = [_split_summary_row(name, split, date_column) for name, split in splits.items()]
    return pd.DataFrame(rows)


def walk_forward_summary(windows: list[dict[str, pd.DataFrame]], date_column: str = "date") -> pd.DataFrame:
    rows = []
    for window_id, splits in enumerate(windows):
        for name, split in splits.items():
            row = {"window": window_id}
            row.update(_split_summary_row(name, split, date_column))
            rows.append(row)
    return pd.DataFrame(rows)

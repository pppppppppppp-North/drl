from __future__ import annotations

import pandas as pd
import pytest

from src.data.ingest_sentiment_csv import build_daily_sentiment, normalize_sentiment_csv


def test_normalize_sentiment_csv_combines_title_and_summary() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2023-01-03"],
            "published_at": ["2023-01-03 08:30:00+07:00"],
            "ticker": ["ptt.bk"],
            "title": ["PTT earnings improve"],
            "summary": ["Margins recovered"],
            "publisher": ["Example News"],
            "url": ["https://example.com/ptt"],
        }
    )

    normalized = normalize_sentiment_csv(
        raw,
        date_column="date",
        published_at_column="published_at",
        ticker_column="ticker",
        title_column="title",
        summary_column="summary",
        text_column="text",
        sentiment_column="sentiment_score",
        publisher_column="publisher",
        url_column="url",
        source_name="licensed_news_csv",
    )

    assert normalized["ticker"].tolist() == ["PTT.BK"]
    assert normalized["date"].tolist() == ["2023-01-03"]
    assert normalized["text"].tolist() == ["PTT earnings improve. Margins recovered"]
    assert normalized["source"].tolist() == ["licensed_news_csv"]


def test_build_daily_sentiment_uses_provided_scores() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2023-01-03", "2023-01-03", "2023-01-04"],
            "ticker": ["AOT.BK", "AOT.BK", "AOT.BK"],
            "title": ["a", "b", "c"],
            "sentiment_score": [0.5, -0.1, 0.2],
        }
    )
    normalized = normalize_sentiment_csv(
        raw,
        date_column="date",
        published_at_column="published_at",
        ticker_column="ticker",
        title_column="title",
        summary_column="summary",
        text_column="text",
        sentiment_column="sentiment_score",
        publisher_column="publisher",
        url_column="url",
        source_name="scored_news_csv",
    )

    daily = build_daily_sentiment(normalized, score_source="scored_news_csv")

    assert daily["news_count"].tolist() == [2, 1]
    assert daily["sentiment_score"].round(6).tolist() == [0.2, 0.2]
    assert set(daily["sentiment_source"]) == {"scored_news_csv"}


def test_normalize_sentiment_csv_rejects_rows_without_text_or_score() -> None:
    raw = pd.DataFrame({"date": ["2023-01-03"], "ticker": ["PTT.BK"], "title": [""]})

    with pytest.raises(ValueError, match="need text or a numeric sentiment score"):
        normalize_sentiment_csv(
            raw,
            date_column="date",
            published_at_column="published_at",
            ticker_column="ticker",
            title_column="title",
            summary_column="summary",
            text_column="text",
            sentiment_column="sentiment_score",
            publisher_column="publisher",
            url_column="url",
            source_name="empty_news_csv",
        )

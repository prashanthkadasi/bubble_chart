# processing.py

import pandas as pd
import numpy as np
from textblob import TextBlob
from datetime import datetime, timedelta

# --- Sentiment Analysis with TextBlob ---

def add_sentiment_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates sentiment scores for page titles using TextBlob.
    """
    if df.empty or 'PageTitle' not in df.columns:
        return df

    print("Running sentiment analysis with TextBlob...")

    features_to_analyze = ['topStories', 'videos', 'video']
    mask = df['SERPFeature'].isin(features_to_analyze)

    df['sentiment'] = np.nan

    if mask.sum() > 0:
        # Apply TextBlob sentiment analysis
        sentiment_scores = df.loc[mask, 'PageTitle'].apply(lambda title: TextBlob(str(title)).sentiment.polarity)
        df.loc[mask, 'sentiment'] = sentiment_scores

    print("Sentiment analysis complete.")
    return df


# --- Data Preparation for Charting ---

def prepare_chart_data(paa_df: pd.DataFrame, sv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicates the BigQuery logic to calculate growth and aggregate sentiment,
    then joins them to create the final data for the bubble chart.
    """
    if paa_df.empty or sv_df.empty:
        print("Warning: One or both input DataFrames are empty. Returning empty DataFrame.")
        return pd.DataFrame()

    # === 1. Replicate the `sentiment_aggregated` CTE ===
    print("Aggregating sentiment data...")

    paa_df['Date'] = pd.to_datetime(paa_df['Date'])
    date_filter = paa_df['Date'] >= (datetime.now() - timedelta(days=28))
    sentiment_recent = paa_df.loc[date_filter].copy()

    # Reverted to original TextBlob conditions
    conditions = [
        sentiment_recent['sentiment'] > 0,
        sentiment_recent['sentiment'] < 0
    ]
    choices = ['Positive', 'Negative']
    sentiment_recent['Label'] = np.select(conditions, choices, default='Neutral')

    sentiment_aggregated = sentiment_recent.groupby(['SearchTerm', 'Market']).agg(
        results_analysed=('Label', lambda x: (x.isin(['Positive', 'Negative'])).sum()),
        positive_count=('Label', lambda x: (x == 'Positive').sum()),
        negative_count=('Label', lambda x: (x == 'Negative').sum())
    ).reset_index()

    # === 2. Replicate the `Dubai_sv_growth` table calculation ===
    print("Calculating search volume growth...")

    sv_df['Month'] = pd.to_datetime(sv_df['Month'])
    sv_df = sv_df.sort_values(by=['SearchTerm', 'Market', 'Month'])
    sv_df['GrowthorDecline'] = sv_df.groupby(['SearchTerm', 'Market'])['SearchVolume'].pct_change() * 100
    sv_df['Last_12months_sv'] = sv_df.groupby(['SearchTerm', 'Market'])['SearchVolume'] \
                                     .transform(lambda x: x.rolling(12, min_periods=1).mean())
    sv_growth = sv_df.groupby(['SearchTerm', 'Market']).last().reset_index()

    # === 3. Replicate the final JOIN and SELECT ===
    print("Joining growth and sentiment data...")

    final_df = pd.merge(
        sv_growth,
        sentiment_aggregated,
        on=['SearchTerm', 'Market'],
        how='left'
    )

    cols_to_fill = ['results_analysed', 'positive_count', 'negative_count']
    final_df[cols_to_fill] = final_df[cols_to_fill].fillna(0)

    denominator = final_df['results_analysed'].replace(0, np.nan)
    final_df['Positive_Percentage'] = (final_df['positive_count'] / denominator * 100).fillna(0)
    final_df['Negative_Percentage'] = (final_df['negative_count'] / denominator * 100).fillna(0)
    final_df['Sentiment'] = ((final_df['positive_count'] - final_df['negative_count']) / denominator * 100).fillna(0)

    final_df = final_df[[
        'Market', 'SearchTerm', 'Category', 'Last_12months_sv', 'GrowthorDecline',
        'results_analysed', 'Positive_Percentage', 'Negative_Percentage', 'Sentiment'
    ]].copy()

    final_df.rename(columns={
        'Last_12months_sv': 'search_volume_12m_avg',
        'GrowthorDecline': 'growth_pct',
        'results_analysed': 'results_analysed',
        'Sentiment': 'sentiment_score'
    }, inplace=True)

    numeric_cols = ['search_volume_12m_avg', 'growth_pct', 'Positive_Percentage', 'Negative_Percentage', 'sentiment_score']
    final_df[numeric_cols] = final_df[numeric_cols].round(2)

    print("Data preparation complete.")
    return final_df

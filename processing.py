# processing.py

import pandas as pd
import numpy as np
from textblob import TextBlob
from datetime import datetime, timedelta

# The add_sentiment_scores function from before remains the same.
def add_sentiment_scores(df: pd.DataFrame) -> pd.DataFrame:
    # ... (code from previous step)
    if df.empty:
        return df
    features_to_analyze = ['topStories', 'videos', 'video']
    mask = df['SERPFeature'].isin(features_to_analyze)
    if mask.sum() == 0:
        df['sentiment'] = np.nan
        return df
    df['sentiment'] = np.nan
    sentiment_scores = df.loc[mask, 'PageTitle'].apply(lambda title: TextBlob(str(title)).sentiment.polarity)
    df.loc[mask, 'sentiment'] = sentiment_scores
    return df

# --- NEW FUNCTION TO REPLICATE THE BIGQUERY SCRIPT ---

def prepare_chart_data(paa_df: pd.DataFrame, sv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicates the BigQuery logic to calculate growth and aggregate sentiment,
    then joins them to create the final data for the bubble chart.

    Args:
        paa_df: DataFrame containing PAA data with a 'sentiment' column.
        sv_df: DataFrame containing raw monthly search volume data.

    Returns:
        A final DataFrame ready for plotting.
    """
    if paa_df.empty or sv_df.empty:
        print("Warning: One or both input DataFrames are empty. Returning empty DataFrame.")
        return pd.DataFrame()

    # === 1. Replicate the `sentiment_aggregated` CTE ===
    print("Aggregating sentiment data...")
    
    # Convert 'Date' column to datetime objects for filtering
    paa_df['Date'] = pd.to_datetime(paa_df['Date'])
    
    # Filter for the last 28 days
    date_filter = paa_df['Date'] >= (datetime.now() - timedelta(days=28))
    sentiment_recent = paa_df[date_filter].copy()
    
    # Create the 'Label' column from the numeric sentiment score
    conditions = [
        sentiment_recent['sentiment'] > 0,
        sentiment_recent['sentiment'] < 0
    ]
    choices = ['Positive', 'Negative']
    sentiment_recent['Label'] = np.select(conditions, choices, default='Neutral')
    
    # Group by SearchTerm and Market and perform the aggregations
    sentiment_aggregated = sentiment_recent.groupby(['SearchTerm', 'Market']).agg(
        results_analysed=('Label', lambda x: (x.isin(['Positive', 'Negative'])).sum()),
        positive_count=('Label', lambda x: (x == 'Positive').sum()),
        negative_count=('Label', lambda x: (x == 'Negative').sum())
    ).reset_index()
    
    # === 2. Replicate the `Dubai_sv_growth` table calculation ===
    print("Calculating search volume growth...")
    
    # Convert 'Month' to datetime and sort to ensure correct order for growth calculation
    sv_df['Month'] = pd.to_datetime(sv_df['Month'])
    sv_df = sv_df.sort_values(by=['SearchTerm', 'Market', 'Month'])
    
    # Calculate growth (percent change from previous month) within each group
    # This creates the 'percentchange' column
    sv_df['GrowthorDecline'] = sv_df.groupby(['SearchTerm', 'Market'])['SearchVolume'].pct_change() * 100
    
    # Calculate the average search volume over the last 12 months for each group
    sv_df['Last_12months_sv'] = sv_df.groupby(['SearchTerm', 'Market'])['SearchVolume'] \
                                     .transform(lambda x: x.rolling(12, min_periods=1).mean())

    # We only care about the MOST RECENT data for each search term
    sv_growth = sv_df.groupby(['SearchTerm', 'Market']).last().reset_index()

    # === 3. Replicate the final JOIN and SELECT ===
    print("Joining growth and sentiment data...")
    
    # Perform the LEFT JOIN from sv_growth to sentiment_aggregated
    final_df = pd.merge(
        sv_growth,
        sentiment_aggregated,
        on=['SearchTerm', 'Market'],
        how='left'
    )
    
    # Fill NaN values from the left join with 0, replicating COALESCE
    cols_to_fill = ['results_analysed', 'positive_count', 'negative_count']
    final_df[cols_to_fill] = final_df[cols_to_fill].fillna(0)

    # Calculate final percentage columns, replicating SAFE_DIVIDE
    # We replace 0 in the denominator with NaN to avoid division by zero, then fill resulting NaNs with 0
    denominator = final_df['results_analysed'].replace(0, np.nan)
    final_df['Positive_Percentage'] = (final_df['positive_count'] / denominator * 100).fillna(0)
    final_df['Negative_Percentage'] = (final_df['negative_count'] / denominator * 100).fillna(0)
    final_df['Sentiment'] = ((final_df['positive_count'] - final_df['negative_count']) / denominator * 100).fillna(0)

    # Select, rename, and round the final columns to match the SQL output
    final_df = final_df[[
        'Market',
        'SearchTerm',
        'Category',
        'Last_12months_sv',
        'GrowthorDecline',
        'results_analysed',
        'Positive_Percentage',
        'Negative_Percentage',
        'Sentiment'
    ]].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Rename columns to be more Python-friendly for plotting
    final_df.rename(columns={
        'Last_12months_sv': 'search_volume_12m_avg',
        'GrowthorDecline': 'growth_pct',
        'results_analysed': 'results_analysed',
        'Sentiment': 'sentiment_score'
    }, inplace=True)

    # Round numeric columns
    numeric_cols = ['search_volume_12m_avg', 'growth_pct', 'Positive_Percentage', 'Negative_Percentage', 'sentiment_score']
    final_df[numeric_cols] = final_df[numeric_cols].round(2)

    print("Data preparation complete.")
    return final_df


# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    # Create realistic sample data to test the entire pipeline
    paa_sample = pd.DataFrame({
        'Date': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30)],
        'Market': ['AE'] * 30,
        'SearchTerm': ['Dubai holidays'] * 15 + ['Dubai flights'] * 15,
        'PageTitle': ['Amazing Deals on Dubai Holidays'] * 5 + ['Dubai Holidays Canceled!'] * 5 + ['Neutral Title'] * 5 + ['Great Flight Prices'] * 15,
        'SERPFeature': ['topStories'] * 10 + ['organic'] * 5 + ['videos'] * 15,
        'sentiment': [0.8] * 5 + [-0.9] * 5 + [0.0] * 5 + [0.7] * 15
    })

    sv_sample = pd.DataFrame({
        'Month': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-01-01', '2023-02-01']),
        'Market': ['AE', 'AE', 'AE', 'AE'],
        'SearchTerm': ['Dubai holidays', 'Dubai holidays', 'Dubai flights', 'Dubai flights'],
        'SearchVolume': [1000, 1200, 5000, 4500],
        'Category': ['Travel', 'Travel', 'Flights', 'Flights']
    })

    # Run the new function
    final_chart_data = prepare_chart_data(paa_sample, sv_sample)

    print("\n--- Final Chart-Ready DataFrame ---")
    print(final_chart_data)
    
    # Expected output for 'Dubai holidays':
    # growth_pct should be 20.0 ((1200-1000)/1000 * 100)
    # results_analysed should be 10 (5 positive, 5 negative from topStories)
    # Positive_Percentage should be 50.0
    # Negative_Percentage should be 50.0
    # sentiment_score should be 0.0 ((5-5)/10 * 100)
    
    # Expected output for 'Dubai flights':
    # growth_pct should be -10.0 ((4500-5000)/5000 * 100)
    # results_analysed should be 15 (all 15 are positive from videos)
    # Positive_Percentage should be 100.0
    # Negative_Percentage should be 0.0
    # sentiment_score should be 100.0
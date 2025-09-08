import pandas as pd

PLOTLY_BUBBLE_SIZE_SCALE = 2500
### MODIFIED: Chart functions now take 'industry' to create unique filenames
def _generate_sentiment_bubble_chart(df: pd.DataFrame) -> str:
    """Generates a bubble chart and saves it with an industry-specific name."""
    logging.info(f"   -> Generating 'Sentiment vs. Growth' bubble chart for '{industry}'...")
    if df.empty:
        logging.warning("      - DataFrame is empty. Skipping bubble chart.")
        return None

    chart_df = df.copy()
    chart_df['Sentiment'] = pd.to_numeric(chart_df['Sentiment'], errors='coerce').fillna(0)
    chart_df['GrowthorDecline'] = pd.to_numeric(chart_df['GrowthorDecline'], errors='coerce').fillna(0)
    chart_df['Last12MonthsSV'] = pd.to_numeric(chart_df['Last12MonthsSV'], errors='coerce').fillna(0)
    chart_df['size_scaled'] = chart_df['Last12MonthsSV'] / PLOTLY_BUBBLE_SIZE_SCALE

    fig = px.scatter(
        chart_df,
        x='Sentiment',
        y='GrowthorDecline',
        size='size_scaled',
        color='Brand',
        hover_name='Brand',
        title=f"Brand Sentiment vs. Growth ({industry})",
        labels={'Sentiment': "Sentiment (%)", 'GrowthorDecline': "Growth or Decline (%)"}
    )
    fig.update_layout(height=450, showlegend=True)


# charting.py

import pandas as pd
import plotly.express as px


def create_bubble_chart(df: pd.DataFrame) -> str:
    """
    Generates an interactive bubble chart from the final processed data.

    Args:
        df: The final DataFrame from processing.py, containing growth, sentiment,
            and search volume data.

    Returns:
        An HTML string representing the interactive chart, or a message if no data exists.
    """
    if df.empty:
        return "<h3>No data available to generate a chart. Please check your inputs.</h3>"

    # --- Data Preparation and Column Mapping ---
    chart_df = df.copy()
    chart_df['sentiment_score'] = pd.to_numeric(
        chart_df['sentiment_score'], errors='coerce').fillna(0)
    chart_df['growth_pct'] = pd.to_numeric(
        chart_df['growth_pct'], errors='coerce').fillna(0)
    chart_df['search_volume_12m_avg'] = pd.to_numeric(
        chart_df['search_volume_12m_avg'], errors='coerce').fillna(0)

    # --- Create the Plotly Figure ---
    fig = px.scatter(
        chart_df,
        x='sentiment_score',
        y='growth_pct',
        size='search_volume_12m_avg',
        color='Category',
        hover_name='SearchTerm',
        size_max=60,
        title="Growth vs. Sentiment Analysis",
        labels={
            "sentiment_score": "Sentiment Score (%)",
            "growth_pct": "Month-over-Month Growth (%)",
            "search_volume_12m_avg": "Avg. Search Volume (12m)",
            "Category": "Category"
        },
        template="plotly_white"
    )

    # --- Enhance the Layout for Better Analysis ---
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="grey")
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")

    fig.update_layout(
        height=600,
        showlegend=True,
        xaxis_title="← Negative Sentiment | Positive Sentiment →",
        yaxis_title="← Decline | Growth →",
        title_x=0.5
    )

    # --- Convert to HTML ---
    chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    return chart_html

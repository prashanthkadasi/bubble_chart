# charting.py

import pandas as pd
import plotly.express as px
import json
from typing import Dict, Any

def create_bubble_chart(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates an interactive bubble chart and returns it along with the data.

    Args:
        df: The final DataFrame from processing.py.

    Returns:
        A dictionary containing the chart's HTML representation and the chart's data as a JSON string.
    """
    if df.empty:
        return {
            "html": "<h3>No data available to generate a chart. Please check your inputs.</h3>",
            "data_json": "{}"
        }

    # --- Data Preparation and Column Mapping ---
    chart_df = df.copy()

    # Ensure numeric columns are numeric, filling NaNs
    for col in ['sentiment_score', 'growth_pct', 'search_volume_12m_avg']:
        chart_df[col] = pd.to_numeric(chart_df[col], errors='coerce').fillna(0)

    # --- Create the Plotly Figure ---
    # Initial view: color by Category, size by search_volume_12m_avg
    fig = px.scatter(
        chart_df,
        x='sentiment_score',
        y='growth_pct',
        size='search_volume_12m_avg',
        color='Category',
        hover_name='SearchTerm',
        custom_data=['Market', 'Category', 'SearchTerm', 'search_volume_12m_avg'], # Add all potential data to custom_data
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
    fig.update_layout(
        height=700,
        showlegend=True,
        xaxis_title="← Negative Sentiment | Positive Sentiment →",
        yaxis_title="← Decline | Growth →",
        title_x=0.5,
        legend_title_text='Color By'
    )
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="grey")
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")

    # --- Convert to HTML and JSON ---
    chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn', div_id='bubble-chart')

    # Prepare data for frontend: Convert dataframe to a dictionary of lists
    chart_data_dict = chart_df.to_dict(orient='list')
    chart_data_json = json.dumps(chart_data_dict)

    return {
        "html": chart_html,
        "data_json": chart_data_json
    }

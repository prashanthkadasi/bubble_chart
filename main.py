import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd

# Import our custom modules
import data_provider
import processing
import charting
import utils

# --- App Initialization ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- Web Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def show_form(request: Request):
    """Displays the initial input form."""
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/generate-chart/", response_class=HTMLResponse)
async def generate_chart(request: Request, 
                         account_id: int = Form(...), 
                         market_configs_text: str = Form(...)):
    """
    Handles form submission and runs the entire data pipeline.
    """
    print("--- [START] Chart Generation Pipeline ---")

    # 1. Parse User Input
    print("Step 1: Parsing user input...")
    market_configs = utils.parse_market_input(market_configs_text)
    if not market_configs:
        return HTMLResponse("<h3>Error: Could not parse market configurations. Please check the format and go back.</h3>", status_code=400)

    # 2. Fetch Raw Data
    print(f"Step 2: Fetching data for Account ID: {account_id}...")
    final_paa_df, final_sv_df = data_provider.fetch_all_data(
        account_id=account_id,
        market_configs=market_configs
    )
    if final_paa_df.empty or final_sv_df.empty:
        return HTMLResponse("<h3>Error: No data could be fetched. Please check your inputs and API credentials (environment variables).</h3>", status_code=400)

    # 3. Add Sentiment Scores
    print("Step 3: Running sentiment analysis...")
    paa_with_sentiment_df = processing.add_sentiment_scores(final_paa_df)

    # 4. Prepare Final Chart Data (Growth, Aggregation, Join)
    print("Step 4: Calculating growth and preparing final data...")
    final_chart_df = processing.prepare_chart_data(paa_with_sentiment_df, final_sv_df)

    # 5. Generate the Interactive Chart
    print("Step 5: Generating the bubble chart...")
    chart_html = charting.create_bubble_chart(final_chart_df)

    print("--- [END] Pipeline Complete ---")

    # 6. Render the Results Page
    return templates.TemplateResponse("result.html", {
        "request": request,
        "chart_html": chart_html
    })

# --- How to Run This App ---
# 1. Make sure your PI_CLIENT_ID and PI_CLIENT_SECRET are set as environment variables.
# 2. In your terminal, run:
#    uvicorn main:app --reload
# 3. Open your browser to http://127.0.0.1:8000
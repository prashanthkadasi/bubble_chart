import os
import uuid
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import io

# Import our custom modules
import data_provider
import processing
import charting
import utils

# --- App Initialization ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")
TEMP_DATA_DIR = "temp_data"

# Ensure the temporary directory exists
os.makedirs(TEMP_DATA_DIR, exist_ok=True)


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
    Handles form submission, runs the data pipeline, saves the result,
    and renders the results page.
    """
    print("--- [START] Chart Generation Pipeline ---")

    # 1. Parse User Input
    print("Step 1: Parsing user input...")
    market_configs = utils.parse_market_input(market_configs_text)
    if not market_configs:
        raise HTTPException(
            status_code=400, detail="Could not parse market configurations. Please check the format.")

    # 2. Fetch Raw Data
    print(f"Step 2: Fetching data for Account ID: {account_id}...")
    try:
        final_paa_df, final_sv_df = await data_provider.fetch_all_data(
            account_id=account_id,
            market_configs=market_configs
        )
    except Exception as e:
        print(f"ERROR during data fetching: {e}")
        raise HTTPException(
            status_code=500, detail=f"An error occurred while fetching data: {e}")

    if final_paa_df.empty or final_sv_df.empty:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": "No data could be fetched. Please check your inputs and API credentials (environment variables)."
        }, status_code=400)

    # 3. Add Sentiment Scores & 4. Prepare Chart Data
    print("Step 3 & 4: Processing data...")
    paa_with_sentiment_df = processing.add_sentiment_scores(final_paa_df)
    final_chart_df = processing.prepare_chart_data(
        paa_with_sentiment_df, final_sv_df)

    if final_chart_df.empty:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": "The provided inputs resulted in no data to display after processing."
        }, status_code=400)

    # 5. Generate the Interactive Chart
    print("Step 5: Generating the bubble chart...")
    chart_html = charting.create_bubble_chart(final_chart_df)

    # 6. Save data for download
    file_id = str(uuid.uuid4())
    file_path = os.path.join(TEMP_DATA_DIR, f"{file_id}.csv")
    final_chart_df.to_csv(file_path, index=False)
    print(f"Result data saved to {file_path}")

    # 7. Calculate Summary Metrics
    summary = {
        "total_keywords": len(final_chart_df),
        "markets_analysed": final_chart_df['Market'].nunique(),
        "avg_sentiment": f"{final_chart_df['sentiment_score'].mean():.2f}%",
        "highest_growth_keyword": final_chart_df.loc[final_chart_df['growth_pct'].idxmax()]['SearchTerm'],
        "highest_growth_value": f"{final_chart_df['growth_pct'].max():.2f}%"
    }

    print("--- [END] Pipeline Complete ---")

    # 8. Render the Results Page
    return templates.TemplateResponse("result.html", {
        "request": request,
        "chart_html": chart_html,
        "file_id": file_id,
        "summary": summary
    })


@app.get("/download-csv/{file_id}")
async def download_csv(file_id: str):
    """Serves the generated data as a CSV file for download."""
    file_path = os.path.join(TEMP_DATA_DIR, f"{file_id}.csv")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")

    # Use an in-memory stream
    stream = io.StringIO()
    pd.read_csv(file_path).to_csv(stream, index=False)

    response = StreamingResponse(
        iter([stream.getvalue()]), media_type="text/csv")
    response.headers[
        "Content-Disposition"] = f"attachment; filename=sentiment_analysis_data_{file_id[:8]}.csv"

    return response

# --- Error Page Template (Good Practice) ---
# Create a templates/error.html to handle errors gracefully
# For this exercise, I'll assume it exists. If not, FastAPI's default JSON error is fine.

# --- How to Run This App ---
# 1. Make sure your PI_CLIENT_ID and PI_CLIENT_SECRET are set as environment variables.
# 2. In your terminal, run:
#    uvicorn main:app --reload
# 3. Open your browser to http://127.0.0.1:8000

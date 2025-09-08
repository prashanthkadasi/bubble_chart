# data_provider.py

import os
import requests
import pandas as pd
from datetime import date, timedelta
from requests.auth import HTTPBasicAuth
from typing import List, Dict, Tuple
from dotenv import load_dotenv  # <--- 1. IMPORT THE LIBRARY

load_dotenv()  # <--- 2. CALL THE FUNCTION TO LOAD THE .env FILE

# --- Configuration ---
CLIENT_ID = os.getenv("PI_CLIENT_ID")
CLIENT_SECRET = os.getenv("PI_CLIENT_SECRET")
# ACCOUNT_ID is no longer a constant here.

BASE_URL = "https://app.pi-datametrics.com/api"
AUTH_URL = f"{BASE_URL}/auth"

# --- Helper Functions ---

def get_access_token(client_id: str, client_secret: str) -> str:
    # This function remains the same
    print("Requesting new access token...")
    if not client_id or not client_secret:
        print("Error: Client ID or Secret is not set.")
        return None
    payload = {"grant_type": "client_credentials"}
    try:
        response = requests.post(AUTH_URL, auth=HTTPBasicAuth(client_id, client_secret), data=payload)
        response.raise_for_status()
        return response.json()["access_token"]
    except requests.exceptions.RequestException as e:
        print(f"Error getting access token: {e}")
        return None

def get_category_lookup(account_id: int, workspace_id: int, headers: dict) -> dict: # UPDATED: Added account_id
    """Fetches search term groups to create a category mapping."""
    print(f"Fetching category lookup for workspace {workspace_id}...")
    category_dict = {}
    # UPDATED: Use the account_id parameter in the URL
    stg_url = f"{BASE_URL}/accounts/{account_id}/workspaces/{workspace_id}/search-term-groups"
    try:
        stg_response = requests.get(stg_url, headers=headers, timeout=60)
        stg_response.raise_for_status()
        groups = stg_response.json().get('data', [])
        if not groups: return {}
        for group in groups:
            group_id, category_name = group['id'], group['name']
            terms_url = f"{stg_url}/{group_id}/search-terms"
            terms_response = requests.get(terms_url, headers=headers)
            terms_response.raise_for_status()
            terms = terms_response.json().get('data', [])
            for term in terms: category_dict[term] = category_name
        return category_dict
    except requests.exceptions.RequestException as e:
        print(f"Error fetching categories for workspace {workspace_id}: {e}")
        return {}

# --- Core Data Fetching Functions ---

def fetch_paa_data(account_id: int, workspace_id: int, workspace_name: str, search_engine_id: int, search_engine_name: str, headers: dict) -> pd.DataFrame: # UPDATED
    """Fetches the PAA/search results data for a single workspace."""
    # UPDATED: Pass account_id to the category lookup
    category_lookup = get_category_lookup(account_id, workspace_id, headers)
    yesterday_str = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    # UPDATED: Use account_id in the URL
    api_url = f"{BASE_URL}/accounts/{account_id}/workspaces/{workspace_id}/search-data/bulk-search-results?search-engine-id={search_engine_id}&period={yesterday_str}"
    # ... (rest of the function is the same)
    try:
        response = requests.get(api_url, headers=headers, timeout=60)
        response.raise_for_status()
        position_data = response.json().get('data', {})
    except requests.exceptions.RequestException as e:
        print(f"Error fetching PAA data for workspace {workspace_id}: {e}")
        return pd.DataFrame()
    rows = []
    for term_data in position_data:
        search_term = term_data['searchTerm']
        category = category_lookup.get(search_term, '')
        if term_data.get('results'):
            for result in term_data['results']:
                rows.append({'Date': yesterday_str, 'Market': workspace_name, 'SearchTerm': search_term, 'URL': result.get('url'), 'Position': result.get('position'), 'SERPFeature': result.get('feature'), 'PageTitle': result.get('title'), 'SearchEngine': search_engine_name, 'Attributes': str(result.get('attributes')), 'Category': category})
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).dropna(subset=['PageTitle'])


def fetch_sv_data(account_id: int, market: str, workspace_id: int, headers: dict) -> pd.DataFrame: # UPDATED
    """Fetches the Search Volume data for a single workspace."""
    # UPDATED: Pass account_id to the category lookup
    category_lookup = get_category_lookup(account_id, workspace_id, headers)
    # UPDATED: Use account_id in the URL
    api_url = f"{BASE_URL}/accounts/{account_id}/workspaces/{workspace_id}/volume-data/bulk-search-volume"
    # ... (rest of the function is the same)
    try:
        response = requests.get(api_url, headers=headers, timeout=60)
        response.raise_for_status()
        sv_data = response.json().get('data', {})
    except requests.exceptions.RequestException as e:
        print(f"Error fetching SV data for workspace {workspace_id}: {e}")
        return pd.DataFrame()
    rows = []
    for term_data in sv_data:
        if term_data.get('status') == 'available':
            search_term = term_data['search-term']
            category = category_lookup.get(search_term, '')
            cpc = term_data.get('cpc')
            for month, volume in term_data.get('monthly-volume', {}).items():
                rows.append({'Market': market, 'SearchTerm': search_term, 'Month': f"{month}-01", 'SearchVolume': volume, 'Category': category, 'CPC': cpc})
    return pd.DataFrame(rows)


# --- ORCHESTRATOR FUNCTION ---

def fetch_all_data(account_id: int, market_configs: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]: # UPDATED
    """Orchestrates fetching all data for a given account and list of markets."""
    access_token = get_access_token(CLIENT_ID, CLIENT_SECRET)
    if not access_token:
        return pd.DataFrame(), pd.DataFrame()

    headers = {'Authorization': f'Bearer {access_token}'}
    all_paa_dfs, all_sv_dfs = [], []

    for config in market_configs:
        market, w_id, se_id, se_name = config['market_code'], config['workspace_id'], config['search_engine_id'], config['search_engine_name']
        print(f"\n--- Fetching data for Market: {market} ({se_name}) ---")
        
        # UPDATED: Pass account_id to fetch functions
        paa_df = fetch_paa_data(account_id, w_id, market, se_id, se_name, headers)
        if not paa_df.empty:
            all_paa_dfs.append(paa_df)

        sv_df = fetch_sv_data(account_id, market, w_id, headers)
        if not sv_df.empty:
            all_sv_dfs.append(sv_df)

    final_paa_df = pd.concat(all_paa_dfs, ignore_index=True) if all_paa_dfs else pd.DataFrame()
    final_sv_df = pd.concat(all_sv_dfs, ignore_index=True) if all_sv_dfs else pd.DataFrame()
    
    print(f"\n--- Total Data Fetched for Account {account_id} ---")
    print(f"Total PAA rows: {len(final_paa_df)}")
    print(f"Total SV rows: {len(final_sv_df)}")
    
    return final_paa_df, final_sv_df
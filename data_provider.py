# data_provider.py

import os
import asyncio
import httpx
import pandas as pd
from datetime import date, timedelta
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from functools import wraps

load_dotenv()

# --- Configuration ---
CLIENT_ID = os.getenv("PI_CLIENT_ID")
CLIENT_SECRET = os.getenv("PI_CLIENT_SECRET")
BASE_URL = "https://app.pi-datametrics.com/api"
AUTH_URL = f"{BASE_URL}/auth"

# --- Caching ---

# Simple in-memory cache for the access token
_access_token_cache: Optional[str] = None
_token_expiry_time = None
# Tokens are valid for 1 hour (3600s), refresh a bit earlier
TOKEN_LIFESPAN_SECONDS = 3500


def cache_aware_async(func):
    """Decorator to cache results of an async function."""
    cache = {}

    @wraps(func)
    async def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key in cache:
            return cache[key]
        result = await func(*args, **kwargs)
        cache[key] = result
        return result
    return wrapper

# --- Helper Functions ---


async def get_access_token(client_id: str, client_secret: str, client: httpx.AsyncClient) -> Optional[str]:
    """Asynchronously gets a new access token."""
    global _access_token_cache, _token_expiry_time

    # Return cached token if it's still valid
    if _access_token_cache and _token_expiry_time and pd.Timestamp.now() < _token_expiry_time:
        print("Using cached access token.")
        return _access_token_cache

    print("Requesting new access token...")
    if not client_id or not client_secret:
        print("Error: Client ID or Secret is not set.")
        return None

    payload = {"grant_type": "client_credentials"}
    try:
        response = await client.post(AUTH_URL, auth=(client_id, client_secret), data=payload)
        response.raise_for_status()
        token_data = response.json()
        _access_token_cache = token_data["access_token"]
        _token_expiry_time = pd.Timestamp.now() + pd.Timedelta(seconds=TOKEN_LIFESPAN_SECONDS)
        print("New access token obtained and cached.")
        return _access_token_cache
    except httpx.RequestError as e:
        print(f"Error getting access token: {e}")
        return None


@cache_aware_async
async def get_category_lookup(account_id: int, workspace_id: int, headers: dict, client: httpx.AsyncClient) -> dict:
    """Asynchronously fetches search term groups and their terms in parallel to create a category mapping."""
    print(f"Fetching category lookup for workspace {workspace_id}...")
    stg_url = f"{BASE_URL}/accounts/{account_id}/workspaces/{workspace_id}/search-term-groups"

    try:
        stg_response = await client.get(stg_url, headers=headers, timeout=60)
        stg_response.raise_for_status()
        groups = stg_response.json().get('data', [])
        if not groups:
            return {}

        async def fetch_terms_for_group(group):
            group_id = group['id']
            category_name = group['name']
            terms_url = f"{stg_url}/{group_id}/search-terms"
            try:
                terms_response = await client.get(terms_url, headers=headers)
                terms_response.raise_for_status()
                terms = terms_response.json().get('data', [])
                return {term: category_name for term in terms}
            except httpx.RequestError as e:
                print(f"Error fetching terms for group {group_id}: {e}")
                return {}

        tasks = [fetch_terms_for_group(group) for group in groups]
        results = await asyncio.gather(*tasks)

        category_dict = {}
        for res in results:
            category_dict.update(res)

        return category_dict

    except httpx.RequestError as e:
        print(f"Error fetching categories for workspace {workspace_id}: {e}")
        return {}


# --- Core Data Fetching Functions ---

async def fetch_paa_data(account_id: int, workspace_id: int, workspace_name: str, search_engine_id: int, search_engine_name: str, headers: dict, client: httpx.AsyncClient) -> pd.DataFrame:
    """Asynchronously fetches PAA data for a single workspace."""
    category_lookup = await get_category_lookup(account_id, workspace_id, headers, client)
    yesterday_str = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    api_url = f"{BASE_URL}/accounts/{account_id}/workspaces/{workspace_id}/search-data/bulk-search-results?search-engine-id={search_engine_id}&period={yesterday_str}"

    try:
        # Increased timeout
        response = await client.get(api_url, headers=headers, timeout=120)
        response.raise_for_status()
        position_data = response.json().get('data', {})
    except httpx.RequestError as e:
        print(f"Error fetching PAA data for workspace {workspace_id}: {e}")
        return pd.DataFrame()

    rows = [
        {
            'Date': yesterday_str, 'Market': workspace_name, 'SearchTerm': term_data['searchTerm'],
            'URL': result.get('url'), 'Position': result.get('position'), 'SERPFeature': result.get('feature'),
            'PageTitle': result.get('title'), 'SearchEngine': search_engine_name,
            'Attributes': str(result.get('attributes')), 'Category': category_lookup.get(term_data['searchTerm'], '')
        }
        for term_data in position_data if term_data.get('results')
        for result in term_data['results']
    ]

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).dropna(subset=['PageTitle'])


async def fetch_sv_data(account_id: int, market: str, workspace_id: int, headers: dict, client: httpx.AsyncClient) -> pd.DataFrame:
    """Asynchronously fetches Search Volume data for a single workspace."""
    category_lookup = await get_category_lookup(account_id, workspace_id, headers, client)
    api_url = f"{BASE_URL}/accounts/{account_id}/workspaces/{workspace_id}/volume-data/bulk-search-volume"

    try:
        # Increased timeout
        response = await client.get(api_url, headers=headers, timeout=120)
        response.raise_for_status()
        sv_data = response.json().get('data', {})
    except httpx.RequestError as e:
        print(f"Error fetching SV data for workspace {workspace_id}: {e}")
        return pd.DataFrame()

    rows = [
        {
            'Market': market, 'SearchTerm': term_data['search-term'], 'Month': f"{month}-01",
            'SearchVolume': volume, 'Category': category_lookup.get(term_data['search-term'], ''),
            'CPC': term_data.get('cpc')
        }
        for term_data in sv_data if term_data.get('status') == 'available'
        for month, volume in term_data.get('monthly-volume', {}).items()
    ]

    return pd.DataFrame(rows)

# --- ORCHESTRATOR FUNCTION ---


async def fetch_all_data(account_id: int, market_configs: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Orchestrates fetching all data in parallel for a given account and list of markets."""
    async with httpx.AsyncClient() as client:
        access_token = await get_access_token(CLIENT_ID, CLIENT_SECRET, client)
        if not access_token:
            return pd.DataFrame(), pd.DataFrame()

        headers = {'Authorization': f'Bearer {access_token}'}

        paa_tasks, sv_tasks = [], []
        for config in market_configs:
            market, w_id, se_id, se_name = config['market_code'], config[
                'workspace_id'], config['search_engine_id'], config['search_engine_name']
            print(
                f"\n--- Queuing data fetch for Market: {market} ({se_name}) ---")

            paa_tasks.append(fetch_paa_data(account_id, w_id,
                             market, se_id, se_name, headers, client))
            sv_tasks.append(fetch_sv_data(
                account_id, market, w_id, headers, client))

        print("\n--- Fetching all data in parallel... ---")
        all_paa_dfs = await asyncio.gather(*paa_tasks)
        all_sv_dfs = await asyncio.gather(*sv_tasks)

    final_paa_df = pd.concat(
        all_paa_dfs, ignore_index=True) if all_paa_dfs else pd.DataFrame()
    final_sv_df = pd.concat(
        all_sv_dfs, ignore_index=True) if all_sv_dfs else pd.DataFrame()

    print(f"\n--- Total Data Fetched for Account {account_id} ---")
    print(f"Total PAA rows: {len(final_paa_df)}")
    print(f"Total SV rows: {len(final_sv_df)}")

    return final_paa_df, final_sv_df

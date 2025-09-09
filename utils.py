# utils.py

from typing import List, Dict


def parse_market_input(raw_text: str) -> List[Dict]:
    """
    Parses raw text input from a user into a list of market configuration dicts.
    Expected format per line: Market,WorkspaceID,SearchEngineID,DeviceName
    """
    configs = []
    lines = raw_text.strip().split('\n')

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue  # Skip empty lines

        parts = line.split(',')
        if len(parts) != 4:
            print(f"Warning: Skipping malformed line {i+1}: '{line}'")
            continue

        try:
            configs.append({
                'market_code': parts[0].strip(),
                'workspace_id': int(parts[1].strip()),
                'search_engine_id': int(parts[2].strip()),
                'search_engine_name': parts[3].strip()
            })
        except ValueError:
            print(
                f"Warning: Skipping line {i+1} with non-numeric ID: '{line}'")
            continue

    return configs

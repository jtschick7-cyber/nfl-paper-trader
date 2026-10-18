import os
import json
import base64
import gspread
import numpy as np
import pandas as pd
from google.oauth2.service_account import Credentials
from src.config import GSHEET_CREDENTIALS, SPREADSHEET_ID, MASTER_SHEET_NAME

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

def get_gspread_client():
    """Authenticates using Base64 decoded credentials from GitHub Secrets."""
    if not GSHEET_CREDENTIALS:
        raise ValueError("GSHEET_CREDENTIALS environment variable is missing.")

    creds_json = base64.b64decode(GSHEET_CREDENTIALS).decode('utf-8')
    creds_dict = json.loads(creds_json)
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    return gspread.authorize(creds)

def get_master_sheet():
    """Returns the master worksheet object."""
    client = get_gspread_client()
    sh = client.open_by_key(SPREADSHEET_ID)
    return sh.worksheet(MASTER_SHEET_NAME)

def get_sheet_as_df() -> pd.DataFrame:
    """Downloads the master sheet into a Pandas DataFrame.

    Safe when the sheet has only a header row (no bets yet),
    which previously caused KeyError: 'game_id'.
    """
    ws = get_master_sheet()
    all_values = ws.get_all_values()

    if not all_values:
        return pd.DataFrame()

    headers = all_values[0]
    data = all_values[1:]
    df = pd.DataFrame(data, columns=headers)

    # Treat blank cells as missing values
    df = df.replace('', np.nan)

    numeric_cols = ['week', 'pick_line', 'pick_odds', 'model_probability',
                    'implied_probability_no_vig', 'edge', 'opening_line',
                    'opening_odds', 'closing_line', 'closing_odds',
                    'clv_points', 'roi_units']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def update_cell(game_id: str, column_name: str, value):
    """Updates a specific cell for a given game_id. Idempotent."""
    ws = get_master_sheet()
    cell = ws.find(str(game_id))
    if cell:
        row_idx = cell.row
        headers = ws.row_values(1)
        if column_name in headers:
            col_idx = headers.index(column_name) + 1
            ws.update_cell(row_idx, col_idx, value)
        else:
            raise ValueError(f"Column {column_name} not found in sheet headers.")
    else:
        print(f"Warning: Game ID {game_id} not found in sheet.")
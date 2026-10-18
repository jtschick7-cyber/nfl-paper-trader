import base64
import json
import os
import time

import gspread
import numpy as np
import pandas as pd
from google.oauth2.service_account import Credentials
from gspread.exceptions import APIError

from src.config import GSHEET_CREDENTIALS, MASTER_SHEET_NAME, SPREADSHEET_ID

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def to_native(value):
    """
    Convert Pandas / NumPy scalar types into plain Python types.

    Google Sheets API cannot serialize numpy.int64, numpy.float64, etc.
    """
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def get_gspread_client():
    """Authenticates using Base64 decoded credentials from GitHub Secrets."""
    if not GSHEET_CREDENTIALS:
        raise ValueError("GSHEET_CREDENTIALS environment variable is missing.")

    creds_json = base64.b64decode(GSHEET_CREDENTIALS).decode("utf-8")
    creds_dict = json.loads(creds_json)
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    return gspread.authorize(creds)


def get_master_sheet():
    """Returns the master worksheet object."""
    client = get_gspread_client()
    sh = client.open_by_key(SPREADSHEET_ID)
    return sh.worksheet(MASTER_SHEET_NAME)


def get_sheet_as_df() -> pd.DataFrame:
    """
    Downloads the master sheet into a Pandas DataFrame.

    Safe when the sheet has only a header row and no data rows.
    """
    ws = get_master_sheet()
    all_values = ws.get_all_values()

    if not all_values:
        return pd.DataFrame()

    headers = all_values[0]
    data = all_values[1:]
    df = pd.DataFrame(data, columns=headers)

    df = df.replace("", np.nan)

    numeric_cols = [
        "week",
        "pick_line",
        "pick_odds",
        "model_probability",
        "implied_probability_no_vig",
        "edge",
        "opening_line",
        "opening_odds",
        "closing_line",
        "closing_odds",
        "clv_points",
        "roi_units",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def col_to_letter(col_index: int) -> str:
    """
    Convert 1-based column index to Excel-style column letter.

    Example:
        1 -> A
        26 -> Z
        27 -> AA
    """
    letters = ""
    while col_index > 0:
        col_index, remainder = divmod(col_index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def batch_update_cells(ws, updates, chunk_size=200):
    """
    Batch-update many cells using as few Google Sheets API requests as possible.

    updates should be a list of dicts:
        {"row": 2, "col": 5, "value": "DAL"}

    row and col are 1-based.
    """
    if not updates:
        return

    requests = []

    for update in updates:
        row = int(update["row"])
        col = int(update["col"])
        value = to_native(update["value"])
        cell_range = col_to_letter(col) + str(row)

        requests.append(
            {
                "range": cell_range,
                "values": [[value]],
            }
        )

    for i in range(0, len(requests), chunk_size):
        chunk = requests[i:i + chunk_size]

        max_attempts = 5
        delay_seconds = 10

        for attempt in range(1, max_attempts + 1):
            try:
                ws.batch_update(chunk, value_input_option="RAW")
                break
            except APIError as exc:
                error_text = str(exc)
                response = getattr(exc, "response", None)
                status_code = getattr(response, "status_code", None)

                retryable = (
                    status_code in {429, 500, 502, 503, 504}
                    or "429" in error_text
                    or "Quota exceeded" in error_text
                    or "rate limit" in error_text.lower()
                )

                if retryable and attempt < max_attempts:
                    sleep_for = delay_seconds * attempt
                    print(
                        "Google Sheets rate limit hit. "
                        "Waiting " + str(sleep_for) + " seconds before retry "
                        + str(attempt) + "/" + str(max_attempts - 1) + "..."
                    )
                    time.sleep(sleep_for)
                    continue

                raise
            except Exception as exc:
                error_text = str(exc)

                retryable = (
                    "429" in error_text
                    or "Quota exceeded" in error_text
                    or "rate limit" in error_text.lower()
                )

                if retryable and attempt < max_attempts:
                    sleep_for = delay_seconds * attempt
                    print(
                        "Google Sheets transient error. "
                        "Waiting " + str(sleep_for) + " seconds before retry "
                        + str(attempt) + "/" + str(max_attempts - 1) + "..."
                    )
                    time.sleep(sleep_for)
                    continue

                raise


def update_cell(game_id: str, column_name: str, value):
    """
    Convenience helper for updating one cell by game_id.

    Uses batch_update_cells internally so values are sanitized.
    """
    ws = get_master_sheet()
    cell = ws.find(str(game_id))

    if not cell:
        print("Warning: Game ID " + str(game_id) + " not found in sheet.")
        return

    row_idx = cell.row
    headers = ws.row_values(1)

    if column_name not in headers:
        raise ValueError("Column " + str(column_name) + " not found in sheet headers.")

    col_idx = headers.index(column_name) + 1

    batch_update_cells(
        ws,
        [
            {
                "row": row_idx,
                "col": col_idx,
                "value": value,
            }
        ],
    )

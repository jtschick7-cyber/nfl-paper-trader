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

_cached_client = None
_cached_spreadsheet = None
_worksheet_cache = {}


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
    """Authenticates using Base64 decoded credentials with singleton caching."""
    global _cached_client
    if _cached_client is not None:
        return _cached_client

    raw_creds = GSHEET_CREDENTIALS or os.getenv("GSHEET_CREDENTIALS")
    if not raw_creds:
        raise ValueError("GSHEET_CREDENTIALS environment variable is missing.")

    try:
        creds_json = base64.b64decode(raw_creds).decode("utf-8")
        creds_dict = json.loads(creds_json)
    except Exception as exc:
        raise ValueError(f"Failed to decode GSHEET_CREDENTIALS: {exc}")

    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    _cached_client = gspread.authorize(creds)
    return _cached_client


def get_spreadsheet():
    """Returns opened Spreadsheet instance with singleton caching."""
    global _cached_spreadsheet
    if _cached_spreadsheet is not None:
        return _cached_spreadsheet

    sheet_id = SPREADSHEET_ID or os.getenv("SPREADSHEET_ID")
    if not sheet_id:
        raise ValueError("SPREADSHEET_ID environment variable is missing.")

    client = get_gspread_client()
    _cached_spreadsheet = client.open_by_key(sheet_id)
    return _cached_spreadsheet


def get_or_create_worksheet(title: str, rows: str = "5000", cols: str = "50"):
    """
    Returns worksheet with given title, creating it if it doesn't exist.
    Features connection pooling and rate-limit backoff retry.
    """
    if title in _worksheet_cache:
        return _worksheet_cache[title]

    sh = get_spreadsheet()
    max_attempts = 5
    delay_seconds = 5

    for attempt in range(1, max_attempts + 1):
        try:
            try:
                ws = sh.worksheet(title)
            except gspread.exceptions.WorksheetNotFound:
                ws = sh.add_worksheet(title=title, rows=rows, cols=cols)
            _worksheet_cache[title] = ws
            return ws
        except Exception as exc:
            if attempt < max_attempts and ("429" in str(exc) or "Quota" in str(exc)):
                time.sleep(delay_seconds * attempt)
                continue
            raise


def retry_api_call(fn, *args, max_attempts=5, delay_seconds=5, **kwargs):
    """
    Executes a gspread function with automatic retries on 429/5xx and quota errors.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
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
                time.sleep(sleep_for)
                continue
            raise


def get_master_sheet():
    """Returns the master worksheet object."""
    return get_or_create_worksheet(MASTER_SHEET_NAME)


def get_sheet_as_df(ws=None) -> pd.DataFrame:
    """
    Downloads the sheet (master sheet if ws is None) into a Pandas DataFrame.

    Safe when the sheet has only a header row and no data rows.
    """
    if ws is None:
        ws = get_master_sheet()
    all_values = retry_api_call(ws.get_all_values)

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
        "opening_away_odds",
        "closing_line",
        "closing_odds",
        "closing_away_odds",
        "total_line",
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

import base64
import json
import os

import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import WorksheetNotFound

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
]


def get_gc():
    raw = os.getenv("GSHEET_CREDENTIALS")

    if not raw:
        raise SystemExit("Missing GSHEET_CREDENTIALS environment variable.")

    try:
        info = json.loads(base64.b64decode(raw).decode("utf-8"))
    except Exception as exc:
        raise SystemExit("Failed to decode GSHEET_CREDENTIALS: " + str(exc))

    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def get_or_create_worksheet(title):
    spreadsheet_id = os.getenv("SPREADSHEET_ID")

    if not spreadsheet_id:
        raise SystemExit("Missing SPREADSHEET_ID environment variable.")

    gc = get_gc()
    sh = gc.open_by_key(spreadsheet_id)

    try:
        ws = sh.worksheet(title)
    except WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows="5000", cols="50")

    return ws

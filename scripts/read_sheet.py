"""Read-only local access to the Google Sheet.

Loads SPREADSHEET_ID and GSHEET_KEY_FILE from the repo's .env and connects
with the read-only Sheets scope, so nothing here can write to the Sheet.

Usage:
    python scripts/read_sheet.py                 # list tabs
    python scripts/read_sheet.py Model_Scorecard  # print a tab as TSV
"""
import os
import sys
from pathlib import Path

import gspread
from google.oauth2.service_account import Credentials

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
REPO = Path(__file__).resolve().parent.parent


def load_env():
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def open_sheet():
    load_env()
    key_file = os.path.expanduser(os.environ["GSHEET_KEY_FILE"])
    creds = Credentials.from_service_account_file(key_file, scopes=SCOPES)
    return gspread.authorize(creds).open_by_key(os.environ["SPREADSHEET_ID"])


def main():
    sh = open_sheet()
    if len(sys.argv) < 2:
        for ws in sh.worksheets():
            print(f"{ws.title}\t{ws.row_count}x{ws.col_count}")
        return
    for row in sh.worksheet(sys.argv[1]).get_all_values():
        print("\t".join(row).rstrip())


if __name__ == "__main__":
    main()

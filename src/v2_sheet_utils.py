"""
Compatibility shim re-exporting unified Google Sheets utilities from src.sheet_utils.
"""

from src.sheet_utils import (
    get_gspread_client as get_gc,
    get_or_create_worksheet,
)

__all__ = ["get_gc", "get_or_create_worksheet"]

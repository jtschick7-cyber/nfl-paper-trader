"""Season schedules from nflverse's games.csv, in place of nfl_data_py.import_schedules.

nfl_data_py reads the file from a mirror at habitatring.com, which returned HTTP 404 from
2026-11-28 and broke Nightly Results. This reads the same file (same 46 columns) from the
nflverse/nfldata repository on GitHub and falls back to the old mirror. Same contract as
nfl_data_py.import_schedules: a list or range of seasons in, the matching rows out.
"""

import pandas as pd

URLS = (
    "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
    "http://www.habitatring.com/games.csv",
)


def import_schedules(years, urls=URLS):
    if not isinstance(years, (list, range)):
        raise ValueError("Input must be list or range.")
    if min(years) < 1999:
        raise ValueError("Data not available before 1999.")
    errors = []
    for url in urls:
        try:
            scheds = pd.read_csv(url, low_memory=False)
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            continue
        if "season" not in scheds.columns:
            errors.append(f"{url}: no 'season' column")
            continue
        return scheds[scheds["season"].isin(list(years))].copy()
    raise RuntimeError("No schedule source reachable: " + "; ".join(errors))

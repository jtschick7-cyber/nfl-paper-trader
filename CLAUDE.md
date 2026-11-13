# nfl-paper-trader

NFL spread paper-trading pipeline. v1 places paper bets in a Google Sheet; v2–v4 are logged in shadow tabs. GitHub Actions workflows are the only writers to the Sheet. `README.md` describes the models and workflows.

## Start of session

1. Read the "Current state" block of `PROGRESS.md`. It names the next task and its spec file.
2. Read `notes/implementation-plan/README.md` (rules, decisions taken, questions for the owner), then only the spec for the task in hand.
3. `FABLE_REVIEW.md` holds the evidence behind the plan. Open a section only when a spec points to it.

## Commands

```bash
.venv/bin/python -m pytest tests/ -q                 # full suite, a few seconds
.venv/bin/python scripts/read_sheet.py Master_Bets   # view-only read of a Sheet tab
```

## Rules

- One task per session, one `claude/` branch and PR per task. Update `PROGRESS.md` when it lands.
- Local sessions cannot write to the Sheet. Sheet-mutating one-off scripts default to dry-run.
- Do not retrain `models/*.pkl`, change `MIN_EDGE_THRESHOLD`, or change β by hand unless the task spec says so.
- Append Sheet columns; never reorder or rename them.

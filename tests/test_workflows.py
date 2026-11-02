import glob
import os

import yaml

WORKFLOWS = os.path.join(os.path.dirname(__file__), "..", ".github", "workflows")
SHEET_WRITERS = ["tuesday_opening.yml", "nightly_results.yml", "v2_shadow_logger.yml", "v3_shadow_logger.yml"]


def _load(name):
    with open(os.path.join(WORKFLOWS, name)) as f:
        return yaml.safe_load(f)


def test_short_sheet_writers_share_one_queue():
    for name in SHEET_WRITERS:
        conc = _load(name)["concurrency"]
        assert conc["group"] == "sheet-writer", name
        assert conc["cancel-in-progress"] is False, name


def test_closing_loop_keeps_own_group_but_scorecard_job_queues_with_writers():
    wf = _load("closing_line_monitor.yml")
    # The hours-long loop must not sit in sheet-writer, or the next closing cron
    # would cancel a Tuesday/nightly run left pending behind it.
    assert wf["concurrency"]["group"] == "closing-line-monitor"
    assert wf["jobs"]["scorecard"]["concurrency"]["group"] == "sheet-writer"
    assert wf["jobs"]["monitor"]["permissions"]["contents"] == "write"
    assert wf["permissions"]["contents"] == "read"


def test_all_workflows_use_python_312():
    for path in glob.glob(os.path.join(WORKFLOWS, "*.yml")):
        with open(path) as f:
            text = f.read()
        assert 'python-version: "3.12"' in text, path
        assert "3.10" not in text, path

// Cloudflare Worker that starts this repo's GitHub Actions workflows on time.
//
// GitHub's own `schedule:` triggers start 3-8h late and drop many firings, so
// each Cloudflare cron below calls the workflow_dispatch API instead. The
// GitHub crons stay in place as a backup; every job is safe to run twice
// (status-gated settling, de-duplicated shadow logs, row-checked closings).
//
// Keep SCHEDULE in sync with `crons` in wrangler.toml (a test checks this).

export const SCHEDULE = {
  // Nightly results: settle finished games.
  "17 5 * * *": ["nightly_results.yml"],
  // Tuesday opening capture, picks and shadow logs.
  "17 15 * * 2": ["tuesday_opening.yml"],
  // Closing-line loop on Mon/Thu/Fri/Sat. Each run sleeps until its kickoff
  // targets for up to 320 min and exits at once when there are none, so the
  // 13:30 and 17:30 starts only matter for afternoon games (Thanksgiving,
  // Christmas, Saturdays late in the season).
  "30 13,17,21 * * 1,4,5,6": ["closing_line_monitor.yml"],
  // Sunday closing loop: London games, 1pm/4pm ET slates, then SNF.
  "30 10,15,20 * * 0": ["closing_line_monitor.yml"],
};

const API = "https://api.github.com";

export async function dispatch(workflow, env, fetchImpl = fetch) {
  const url = `${API}/repos/${env.GITHUB_REPO}/actions/workflows/${workflow}/dispatches`;
  const init = {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.GITHUB_TOKEN}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "workflow-dispatch-trigger",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ref: env.GITHUB_REF || "main" }),
  };

  let last;
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const res = await fetchImpl(url, init);
      if (res.status === 204) return;
      last = new Error(`${workflow}: GitHub returned ${res.status} ${await res.text()}`);
      // 4xx other than rate limiting will not fix itself on retry.
      if (res.status < 500 && res.status !== 429) break;
    } catch (err) {
      last = err;
    }
  }
  throw last;
}

export async function runCron(cron, env, fetchImpl = fetch) {
  const workflows = SCHEDULE[cron];
  if (!workflows) throw new Error(`No workflows mapped to cron "${cron}"`);
  for (const wf of workflows) {
    await dispatch(wf, env, fetchImpl);
    console.log(`Dispatched ${wf} for cron "${cron}"`);
  }
}

export default {
  async scheduled(event, env, ctx) {
    // Throwing marks the invocation as failed in the Cloudflare dashboard.
    await runCron(event.cron, env);
  },
};

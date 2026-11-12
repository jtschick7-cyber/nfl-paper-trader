# External trigger (Cloudflare Worker)

GitHub's `schedule:` triggers in this repo start 3–8 hours late and skip many
firings (nightly: 4 of 9 ran on Nov 10–12, 2026; Tuesday opening started 6h late).
This Worker uses Cloudflare cron triggers to call GitHub's `workflow_dispatch`
API at the exact time. Dispatched runs start within seconds.

The GitHub crons stay as a backup. Duplicate starts are safe: settling is
status-gated, shadow logs de-duplicate, closings are row-checked, and the
concurrency groups queue rather than overlap.

| Cron (UTC) | Workflow | Why |
|---|---|---|
| `17 5 * * *` | `nightly_results.yml` | Settle finished games |
| `17 15 * * 2` | `tuesday_opening.yml` | Tuesday opening capture + picks + shadow logs |
| `30 13,17,21 * * 1,4,5,6` | `closing_line_monitor.yml` | Mon/Thu/Fri/Sat closing loop (exits at once if no games) |
| `30 10,15,20 * * 0` | `closing_line_monitor.yml` | Sunday: London, 1pm/4pm ET slates, SNF |

Edit `SCHEDULE` in `src/index.js` and `crons` in `wrangler.toml` together;
`npm test` fails if they drift. The free Cloudflare plan allows 5 cron triggers per Worker.

## One-time setup (you do these; they need your accounts)

1. **GitHub token.** GitHub → Settings → Developer settings → Fine-grained tokens →
   Generate. Repository access: *Only* `nfl-paper-trader`. Permissions:
   **Actions: Read and write** (Metadata: read is added automatically). Nothing else.
   Pick an expiry and put a reminder in your calendar.
2. **Cloudflare login** (free account):
   ```bash
   cd external_trigger && npx wrangler@4.151.2 login
   ```
3. **Store the token as a Worker secret** (paste it at the prompt; it is never written to disk):
   ```bash
   npx wrangler@4.151.2 secret put GITHUB_TOKEN
   ```
   On a first deploy Wrangler may ask to create the Worker; say yes.
4. **Deploy:**
   ```bash
   npm run deploy
   ```

## Checking it works

- Cloudflare dashboard → Workers → `nfl-paper-trader-trigger` → Settings → Triggers lists the crons;
  Logs / Metrics shows each invocation. A failed dispatch (bad token, 4xx) makes the invocation fail there.
- `gh run list --workflow nightly_results.yml` shows runs with event `workflow_dispatch` starting at the cron minute.

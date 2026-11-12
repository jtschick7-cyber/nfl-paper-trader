import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { SCHEDULE, dispatch, runCron } from "../src/index.js";

const here = dirname(fileURLToPath(import.meta.url));
const env = { GITHUB_REPO: "owner/repo", GITHUB_TOKEN: "t", GITHUB_REF: "main" };

function fakeFetch(statuses) {
  const calls = [];
  const fn = async (url, init) => {
    calls.push({ url, init });
    const status = statuses.shift();
    if (status instanceof Error) throw status;
    return { status, text: async () => "body" };
  };
  fn.calls = calls;
  return fn;
}

test("wrangler.toml crons match SCHEDULE keys", () => {
  const toml = readFileSync(join(here, "..", "wrangler.toml"), "utf8");
  const block = toml.match(/crons\s*=\s*\[([\s\S]*?)\]/)[1];
  const crons = [...block.matchAll(/"([^"]+)"/g)].map((m) => m[1]);
  assert.deepEqual([...crons].sort(), Object.keys(SCHEDULE).sort());
});

test("every scheduled workflow file exists and accepts workflow_dispatch", () => {
  for (const wfs of Object.values(SCHEDULE)) {
    for (const wf of wfs) {
      const path = join(here, "..", "..", ".github", "workflows", wf);
      assert.ok(existsSync(path), `${wf} missing`);
      assert.match(readFileSync(path, "utf8"), /^\s*workflow_dispatch:/m, `${wf} lacks workflow_dispatch`);
    }
  }
});

test("dispatch posts to the workflow's dispatches endpoint with the ref", async () => {
  const f = fakeFetch([204]);
  await dispatch("nightly_results.yml", env, f);
  assert.equal(f.calls.length, 1);
  assert.equal(
    f.calls[0].url,
    "https://api.github.com/repos/owner/repo/actions/workflows/nightly_results.yml/dispatches",
  );
  assert.equal(f.calls[0].init.method, "POST");
  assert.equal(f.calls[0].init.headers.Authorization, "Bearer t");
  assert.deepEqual(JSON.parse(f.calls[0].init.body), { ref: "main" });
});

test("dispatch retries 5xx and network errors, then succeeds", async () => {
  const f = fakeFetch([502, new Error("reset"), 204]);
  await dispatch("x.yml", env, f);
  assert.equal(f.calls.length, 3);
});

test("dispatch gives up after three failures", async () => {
  const f = fakeFetch([500, 500, 500]);
  await assert.rejects(dispatch("x.yml", env, f), /500/);
  assert.equal(f.calls.length, 3);
});

test("dispatch does not retry auth errors", async () => {
  const f = fakeFetch([401]);
  await assert.rejects(dispatch("x.yml", env, f), /401/);
  assert.equal(f.calls.length, 1);
});

test("runCron dispatches the mapped workflow and rejects unknown crons", async () => {
  const f = fakeFetch([204]);
  await runCron("17 15 * * 2", env, f);
  assert.match(f.calls[0].url, /tuesday_opening\.yml/);
  await assert.rejects(runCron("0 0 * * *", env, f), /No workflows/);
});

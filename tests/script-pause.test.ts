/**
 * Script-requested, resumable pause: pause() must leave the run paused with its
 * journal on disk (never completed, never failed), and a later resume must replay
 * the completed prefix, step over the pause, and run only what follows live.
 */

import { afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentRunOptions } from "../src/agent.js";
import { WorkflowManager, type WorkflowManagerOptions } from "../src/workflow-manager.js";
import { workflowProjectPaths } from "../src/workflow-paths.js";

const cwds: string[] = [];

function tempProject(): string {
  const cwd = mkdtempSync(join(tmpdir(), "omp-script-pause-"));
  cwds.push(cwd);
  return cwd;
}

afterEach(() => {
  for (const cwd of cwds.splice(0)) {
    rmSync(workflowProjectPaths(cwd).rootDir, { recursive: true, force: true });
    rmSync(cwd, { recursive: true, force: true });
  }
});

/** A manager whose agents record their prompts and resolve deterministically. */
function scriptedManager(cwd: string, calls: string[]): WorkflowManager {
  return new WorkflowManager({
    cwd,
    agent: {
      async run(prompt: string, _options?: AgentRunOptions<never>) {
        calls.push(prompt);
        return `ran:${prompt}`;
      },
    } as unknown as WorkflowManagerOptions["agent"],
  });
}

function scriptWithPauses(bodies: string[]): string {
  return (
    'export const meta = { name: "pausable", description: "probe", phases: [{ title: "work" }] };\n' +
    bodies.join("\n")
  );
}

/** Resolve on the next terminal/checkpoint event for one run (registered before the action). */
function nextEvent(
  manager: WorkflowManager,
  event: "complete" | "paused" | "error",
  runId: string,
): Promise<Record<string, unknown>> {
  const { promise, resolve } = Promise.withResolvers<Record<string, unknown>>();
  const listener = (payload: { runId: string }) => {
    if (payload.runId !== runId) return;
    manager.off(event, listener);
    resolve(payload as Record<string, unknown>);
  };
  manager.on(event, listener);
  return promise;
}

describe("script-requested pause", () => {
  test("settles the run as paused, keeps the journal, and emits no error", async () => {
    const cwd = tempProject();
    const calls: string[] = [];
    const manager = scriptedManager(cwd, calls);
    const errors: unknown[] = [];
    const paused: Record<string, unknown>[] = [];
    manager.on("error", (payload) => errors.push(payload));
    manager.on("paused", (payload) => paused.push(payload));

    const { runId, promise } = manager.startInBackground(
      scriptWithPauses(['await agent("one");', 'await pause("approve scope");', 'await agent("two");']),
    );
    await promise.catch(() => {});

    expect(manager.getRun(runId)?.status).toBe("paused");
    expect(paused.map((p) => p.reason)).toEqual(["script"]);
    expect(paused[0]!.pauseMessage).toContain("approve scope");
    // A pause is not a failure: no "error" event, and nothing after the pause ran.
    expect(errors).toEqual([]);
    expect(calls).toEqual(["one"]);

    const persisted = manager.getPersistence().load(runId)!;
    expect(persisted.status).toBe("paused");
    expect(persisted.pauseReason).toBe("script");
    // The journal — including the pause marker itself — survives on disk.
    expect(persisted.journal?.map((entry) => entry.index)).toEqual([0, 1]);
    expect(persisted.journal?.find((entry) => entry.index === 1)?.result).toBeNull();
  });

  test("resume replays the completed prefix, steps over the pause, and runs the rest live", async () => {
    const cwd = tempProject();
    const calls: string[] = [];
    const manager = scriptedManager(cwd, calls);
    manager.on("error", () => {});

    const { runId, promise } = manager.startInBackground(
      scriptWithPauses(['await agent("one");', 'await pause("approve scope");', 'await agent("two");']),
    );
    await promise.catch(() => {});
    calls.length = 0;

    const completed = nextEvent(manager, "complete", runId);
    expect(await manager.resume(runId)).toBe(true);
    await completed;

    // Only the call after the pause ran: "one" replayed from the journal and the
    // pause was stepped over instead of pausing again.
    expect(calls).toEqual(["two"]);
    expect(manager.getRun(runId)?.status).toBe("completed");
  });

  test("a second pause is a new control point: resume pauses again, then completes", async () => {
    const cwd = tempProject();
    const calls: string[] = [];
    const manager = scriptedManager(cwd, calls);
    manager.on("error", () => {});

    const script = scriptWithPauses([
      'await agent("a");',
      'await pause("gate 1");',
      'await agent("b");',
      'await pause("gate 2");',
      'await agent("c");',
    ]);
    const { runId, promise } = manager.startInBackground(script);
    await promise.catch(() => {});
    expect(calls).toEqual(["a"]);

    const pausedAgain = nextEvent(manager, "paused", runId);
    expect(await manager.resume(runId)).toBe(true);
    await pausedAgain;
    expect(manager.getRun(runId)?.status).toBe("paused");
    expect(calls).toEqual(["a", "b"]);

    const completed = nextEvent(manager, "complete", runId);
    expect(await manager.resume(runId)).toBe(true);
    await completed;
    expect(calls).toEqual(["a", "b", "c"]);
    expect(manager.getRun(runId)?.status).toBe("completed");
  });

  test("an edited pause reason re-pauses instead of resuming past the pause", async () => {
    const cwd = tempProject();
    const calls: string[] = [];
    const manager = scriptedManager(cwd, calls);
    manager.on("error", () => {});

    const { runId, promise } = manager.startInBackground(
      scriptWithPauses(['await agent("one");', 'await pause("approve scope");', 'await agent("two");']),
    );
    await promise.catch(() => {});
    calls.length = 0;

    const pausedAgain = nextEvent(manager, "paused", runId);
    await manager.resume(runId, {
      script: scriptWithPauses(['await agent("one");', 'await pause("approve a different scope");', 'await agent("two");']),
    });
    await pausedAgain;

    expect(manager.getRun(runId)?.status).toBe("paused");
    // "one" replayed (hash unchanged), the edited pause re-paused, so "two" never ran.
    expect(calls).toEqual([]);
  });

  test("a script that catches the pause throw still settles as paused", async () => {
    const cwd = tempProject();
    const calls: string[] = [];
    const manager = scriptedManager(cwd, calls);
    manager.on("error", () => {});

    const { runId, promise } = manager.startInBackground(
      scriptWithPauses([
        'await agent("one");',
        'try { await pause("approve scope"); } catch (error) { log("swallowed"); }',
        'await agent("two");',
      ]),
    );
    await promise.catch(() => {});

    expect(manager.getRun(runId)?.status).toBe("paused");
    expect(calls).toEqual(["one"]);
    expect(manager.getPersistence().load(runId)?.status).toBe("paused");
  });

  test("a run that never paused still refuses to resume", async () => {
    const cwd = tempProject();
    const calls: string[] = [];
    const manager = scriptedManager(cwd, calls);
    manager.on("error", () => {});

    const { runId, promise } = manager.startInBackground(scriptWithPauses(['return await agent("only");']));
    await promise;

    expect(manager.getRun(runId)?.status).toBe("completed");
    expect(await manager.resume(runId)).toBe(false);
  });
});

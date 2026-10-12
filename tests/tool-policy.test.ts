/**
 * Reviewer capability enforcement.
 *
 * 1. An agent-type tool policy must reach EVERY tool class. Native coding tools
 *    and extension tools already passed through `applyToolPolicy`; injected system
 *    tools (the shared-store `store_put`) used to bypass it, so a "read-only"
 *    reviewer could still mutate run shared state.
 * 2. `structured_output` is built from `opts.schema` and added outside the filter,
 *    so a schema keeps working under a restrictive allowlist.
 * 3. An explicitly declared `agentType` with no registered definition must fail
 *    closed rather than silently degrade to the default, unrestricted tool set.
 */
import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentRegistry } from "../src/agent-registry.js";
import { applyToolPolicy, loadAgentRegistry } from "../src/agent-registry.js";
import { WorkflowError, WorkflowErrorCode } from "../src/errors.js";
import { OMP_CODING_TOOL_NAMES } from "../src/omp-api.js";
import { createAgentStoreTools, SharedStore } from "../src/shared-store.js";
import { createStructuredOutputTool } from "../src/structured-output.js";
import { runWorkflow } from "../src/workflow.js";

const META = `export const meta = { name: "tool-policy-probe", description: "probe", phases: [{ title: "work" }] };\n`;
const REVIEWER_ALLOW = ["read", "grep", "glob", "store_get"];
const REVIEWER_DEFINITION =
  "---\nname: reviewer-readonly\ndescription: probe\ntools: read, grep, glob, store_get\n---\n\nRead-only reviewer.\n";
const nativeTools = [...OMP_CODING_TOOL_NAMES].map((name) => ({ name }));
const storeTools = createAgentStoreTools(new SharedStore(), "probe:0");

/** Run `body` against a real project registry built from a throwaway directory. */
async function withProjectRegistry<T>(
  agentDefinition: string | undefined,
  body: (registry: AgentRegistry, dir: string) => Promise<T> | T,
): Promise<T> {
  const dir = mkdtempSync(join(tmpdir(), "omp-agent-def-"));
  try {
    if (agentDefinition !== undefined) {
      mkdirSync(join(dir, ".omp", "agents"), { recursive: true });
      writeFileSync(join(dir, ".omp", "agents", "reviewer-readonly.md"), agentDefinition);
    }
    return await body(loadAgentRegistry(dir, { userDir: join(dir, "absent-user-agents") }), dir);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

describe("effective tool policy", () => {
  test("CASE A: no policy leaves native and system tools untouched", () => {
    expect(applyToolPolicy(nativeTools).map((tool) => tool.name).sort()).toEqual([...OMP_CODING_TOOL_NAMES].sort());
    expect(applyToolPolicy(storeTools).map((tool) => tool.name).sort()).toEqual(
      storeTools.map((tool) => tool.name).sort(),
    );
    // The shared store really does carry a write-capable tool.
    expect(storeTools.map((tool) => tool.name)).toContain("store_put");
  });

  test("CASE B: the reviewer allowlist drops every write-capable tool", () => {
    const keptNative = applyToolPolicy(nativeTools, REVIEWER_ALLOW).map((tool) => tool.name).sort();
    expect(keptNative).toEqual(["glob", "grep", "read"]);
    for (const banned of ["bash", "edit", "write"]) expect(keptNative).not.toContain(banned);

    const keptSystem = applyToolPolicy(storeTools, REVIEWER_ALLOW).map((tool) => tool.name).sort();
    expect(keptSystem).toEqual(["store_get"]);
    expect(keptSystem).not.toContain("store_put");
  });

  test("CASE B: a denylist alone also reaches the system tools", () => {
    expect(applyToolPolicy(storeTools, undefined, ["store_put"]).map((tool) => tool.name).sort()).toEqual(["store_get"]);
  });

  test("CASE C: structured_output is not a system tool, so the policy cannot remove it", () => {
    const capture = { called: false, value: undefined };
    const structured = createStructuredOutputTool({ schema: { type: "object", properties: {} }, capture });
    expect(structured.name).toBe("structured_output");
    // It is added by run() from opts.schema, never from the store tools...
    expect(storeTools.map((tool) => tool.name)).not.toContain("structured_output");
    // ...and routing it through a restrictive policy WOULD drop it, which is
    // exactly why run() must keep adding it outside the filter.
    expect(applyToolPolicy([structured], REVIEWER_ALLOW).map((tool) => tool.name)).toEqual([]);
  });
});

describe("agentType resolution", () => {
  test("parses a project reviewer definition with a read-only allowlist", async () => {
    await withProjectRegistry(REVIEWER_DEFINITION, (registry) => {
      expect(registry.get("reviewer-readonly")?.tools).toEqual(REVIEWER_ALLOW);
      expect(registry.get("reviewer-readonly")?.disallowedTools).toBeUndefined();
    });
  });

  test("CASE E: an agentType policy reaches the runner; an untagged agent keeps the default", async () => {
    await withProjectRegistry(REVIEWER_DEFINITION, async (registry, dir) => {
      const seen: Array<{ toolNames?: string[]; disallowedToolNames?: string[] }> = [];
      await runWorkflow(
        `${META}const tagged = await agent("x", { agentType: "reviewer-readonly" }); const plain = await agent("y"); return [tagged, plain];`,
        {
          cwd: dir,
          agentRegistry: registry,
          persistLogs: false,
          agent: {
            async run(_prompt, options) {
              seen.push({ toolNames: options?.toolNames, disallowedToolNames: options?.disallowedToolNames });
              return "ok";
            },
          },
        },
      );
      expect(seen[0]?.toolNames).toEqual(REVIEWER_ALLOW);
      expect(seen[1]?.toolNames).toBeUndefined();
      expect(seen[1]?.disallowedToolNames).toBeUndefined();
    });
  });

  test("CASE D: an explicit agentType with no definition fails closed", async () => {
    await withProjectRegistry(undefined, async (registry, dir) => {
      let runnerCalled = false;
      const failure = await runWorkflow(`${META}return await agent("x", { agentType: "definitely-does-not-exist" });`, {
        cwd: dir,
        agentRegistry: registry,
        persistLogs: false,
        agent: {
          async run() {
            runnerCalled = true;
            return "ok";
          },
        },
      }).then(
        () => undefined,
        (error: unknown) => error,
      );

      expect(failure).toBeInstanceOf(WorkflowError);
      expect((failure as WorkflowError).code).toBe(WorkflowErrorCode.AGENT_TYPE_NOT_FOUND);
      // The child was never started, so it never received the default tool set.
      expect(runnerCalled).toBe(false);
    });
  });
});

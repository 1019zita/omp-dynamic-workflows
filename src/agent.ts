import { randomUUID } from "node:crypto";
import { unlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { AssistantMessage, Model, TextContent } from "@oh-my-pi/pi-ai";
import {
  type CreateAgentSessionOptions,
  type ModelRegistry,
  OMP_CODING_TOOL_NAMES,
  type SessionManager,
  type Settings,
  type ToolDefinition,
} from "./omp-api.js";
import { host } from "./omp-host.js";
import { parseConfiguredThinkingLevel, warmSchemaValidator } from "./omp-lazy.js";
import type { Static, TSchema } from "./omp-typebox.js";
import { Check, Convert } from "./omp-typebox.js";
import { type AgentHistoryEntry, compactAgentHistory } from "./agent-history.js";
import { applyToolPolicy } from "./agent-registry.js";
import { classifyProviderLimit, WorkflowError, WorkflowErrorCode } from "./errors.js";
import { canonicalModelSpec, resolveModelSpecWithThinking } from "./model-spec.js";
import {
  formatTierFallbackNotice,
  loadModelTierConfig,
  type ModelTierConfig,
  type RankableModel,
  resolveTierModel,
} from "./model-tier-config.js";
import { createStructuredOutputTool, type StructuredOutputCapture } from "./structured-output.js";

/**
 * Find a JSON object/array in free-form text: a fenced ```json block if present,
 * else the first balanced {...} or [...]. Best-effort (the schema check is the
 * real gate). Returns the raw JSON string, or undefined when none is found.
 */
function findJsonBlock(text: string): string | undefined {
  const fence = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fence?.[1]) return fence[1].trim();
  const start = text.search(/[{[]/);
  if (start === -1) return undefined;
  const open = text[start];
  const close = open === "{" ? "}" : "]";
  let depth = 0;
  for (let i = start; i < text.length; i++) {
    if (text[i] === open) depth++;
    else if (text[i] === close && --depth === 0) return text.slice(start, i + 1);
  }
  return undefined;
}

/**
 * Last-resort structured-output recovery: extract a JSON block from prose, coerce
 * it toward the schema, and accept it only if it then validates. Never fabricates
 * — returns undefined unless the parsed value genuinely satisfies the schema.
 */
export function extractValidated<T>(text: string, schema: TSchema): T | undefined {
  const json = findJsonBlock(text);
  if (json === undefined) return undefined;
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch {
    return undefined;
  }
  try {
    const converted = Convert(schema, parsed);
    if (Check(schema, converted)) return converted as T;
  } catch {
    // typebox can throw on exotic schemas; treat as no match.
  }
  return undefined;
}

/**
 * The last assistant message's terminal metadata (stopReason/errorMessage). The pi
 * SDK does NOT throw provider usage/quota limits — it records them as an assistant
 * message with stopReason "error" and an errorMessage. This is the only place that
 * metadata is observable to the workflow layer.
 */
export function lastAssistantError(messages: unknown[]): { stopReason?: string; errorMessage?: string } | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i] as Partial<AssistantMessage> | undefined;
    if (message?.role !== "assistant") continue;
    return { stopReason: message.stopReason, errorMessage: message.errorMessage };
  }
  return undefined;
}

/**
 * If the subagent's turn ended in a provider usage/quota/rate-limit error, throw a
 * PROVIDER_USAGE_LIMIT WorkflowError carrying the real provider message + reset hint.
 * Gated on stopReason === "error" so a successful turn whose text merely mentions
 * "rate limit" is never misclassified. recoverable:false so the run checkpoints
 * (paused) rather than being retried into the same wall or collapsed to a silent null.
 */
export function throwIfProviderLimit(messages: unknown[], label?: string): void {
  const err = lastAssistantError(messages);
  if (err?.stopReason !== "error") return;
  const { matched, resetHint } = classifyProviderLimit(err.errorMessage);
  if (!matched) return;
  throw new WorkflowError(
    err.errorMessage ?? "Provider usage/quota limit reached",
    WorkflowErrorCode.PROVIDER_USAGE_LIMIT,
    { recoverable: false, agentLabel: label, resetHint },
  );
}

/** Minimal session surface resolveStructuredOutput needs (real session or a test double). */
export interface StructuredSession {
  prompt(text: string): Promise<unknown>;
  setActiveToolsByName?(names: string[]): void | Promise<void>;
  messages: unknown[];
}

/**
 * Resolve a schema agent's result. If the tool was called, return the captured
 * value. Otherwise re-prompt up to maxSchemaRetries (tools restricted to
 * structured_output), then try strict schema-validated prose extraction, else
 * throw SCHEMA_NONCOMPLIANCE (non-recoverable — surfaced, never a silent null).
 * Module-level with an injected `lastText` so it is unit-testable.
 */
export async function resolveStructuredOutput<T>(
  session: StructuredSession,
  capture: StructuredOutputCapture<T>,
  schema: TSchema,
  options: { maxSchemaRetries?: number; signal?: AbortSignal; label?: string },
  lastText: (messages: unknown[]) => string,
): Promise<T> {
  if (capture.called) return capture.value as T;

  const maxRetries = Math.max(0, options.maxSchemaRetries ?? 2);
  // Restrict to the schema tool so the only useful next action is calling it
  // (takes effect on the next prompt turn). Best-effort.
  try {
    await session.setActiveToolsByName?.(["structured_output"]);
  } catch {
    // ignore — the re-prompt alone still drives most models to comply
  }
  for (let attempt = 0; attempt < maxRetries && !capture.called; attempt++) {
    if (options.signal?.aborted) throw new Error("Subagent was aborted");
    await session.prompt(
      "You did not call the structured_output tool. Call structured_output now as your only action, with the required fields filled in. Do not write a prose answer.",
    );
  }
  if (capture.called) return capture.value as T;

  // extractValidated()'s Check() falls back to the host JSON-Schema validator for
  // schemas without safeParse; load it here, on the only path that can reach it.
  await warmSchemaValidator();
  const extracted = extractValidated<T>(lastText(session.messages), schema);
  if (extracted !== undefined) {
    console.warn(
      "[workflow] structured_output recovered from prose extraction (the model never called the tool); prefer a tool-reliable model",
    );
    return extracted;
  }

  // A repair re-prompt can itself hit the provider limit. Surface that as the real
  // (recoverable) cause instead of the misleading non-recoverable SCHEMA_NONCOMPLIANCE.
  throwIfProviderLimit(session.messages, options.label);

  throw new WorkflowError(
    "Subagent did not produce valid structured_output after repair attempts",
    WorkflowErrorCode.SCHEMA_NONCOMPLIANCE,
    { recoverable: false, agentLabel: options.label },
  );
}

/**
 * Resolve which concrete model spec a subagent should use. Precedence, most
 * specific first:
 *   1. options.model — an explicit per-agent model (also carries agentType /
 *      phase model, which the workflow layer folds into options.model).
 *   2. options.tier  — resolved via the model-tiers config, falling back to the
 *      session's main model when the tier has no configured entry.
 *   3. DEFAULT TIER — when neither is set but the user has a model-tiers config,
 *      untagged agents default to the "medium" tier so a configured tier set
 *      actually affects the whole workflow (not just agents the script tagged).
 *      Fresh-install medium == the session model, so this is a no-op until the
 *      user customizes tiers via /workflows-models.
 * Returns undefined when nothing applies, so the session default is used.
 *
 * `loadConfig` is injectable for testing; it defaults to reading from disk.
 */
export function resolveAgentModelSpec(
  options: { model?: string; tier?: string },
  mainModel: string | undefined,
  loadConfig: () => ModelTierConfig | null = loadModelTierConfig,
  onTierWithoutConfig?: (tier: string) => void,
): string | undefined {
  if (options.model) return options.model;
  const config = loadConfig();
  if (options.tier) {
    // Tier requested but unconfigured → it silently falls back to mainModel.
    // Let the caller surface that (once) so the no-op is discoverable.
    if (!config) onTierWithoutConfig?.(options.tier);
    return (config ? resolveTierModel(options.tier, config) : undefined) ?? mainModel;
  }
  // Untagged agent: default to the configured medium tier when one exists.
  if (config) {
    const medium = resolveTierModel("medium", config);
    if (medium) return medium;
  }
  return undefined;
}

export interface WorkflowAgentOptions {
  cwd?: string;
  /** Extra tools available to the subagent in addition to the structured output tool. */
  tools?: ToolDefinition[];
  /**
   * Extra tool NAMES to deny in the subagent session, on top of the always-on
   * defaults ({@link DEFAULT_EXCLUDED_SUBAGENT_TOOLS}). Lets the host exclude
   * other recursive-orchestration tools it registers (e.g. a pi-subagents tool)
   * so a workflow subagent can't fan out through them either (#107).
   */
  excludeTools?: string[];
  /** Override any createAgentSession option (model, modelRuntime, resourceLoader, etc.). */
  session?: Partial<CreateAgentSessionOptions>;
  /** Extra system guidance prepended to every subagent task. */
  instructions?: string;
  /**
   * The session's main model (`provider/modelId`). Used as a fallback when
   * resolving opts.tier and no model-tiers.json config exists. Without this,
   * a workflow using `{ tier: "small" }` would log a warning and fall through
   * to the session default when no config is saved yet.
   */
  mainModel?: string;
  /**
   * Shared model registry from the host Pi session. When provided, subagents
   * resolve tier/model specs against the same registry the main session uses,
   * including dynamically-registered providers such as ollama-cloud. Without
   * this, the agent builds an isolated registry from disk and may miss models
   * that are only available via extension registration.
   */
  modelRegistry?: ModelRegistry;
  /**
   * Persist each subagent transcript as a real pi session file under the
   * standard sessions directory (keyed by the runner's project cwd), instead
   * of the default in-memory session that is discarded when the run ends.
   * Default: false (current behavior).
   */
  persistAgentSessions?: boolean;
}

// OMP accepts a ModelRegistry directly. Build one lazily from the same
// agent-dir credential store used by createAgentSession().
let fallbackRegistryPromise: Promise<ModelRegistry> | undefined;
let fallbackRegistry: ModelRegistry | undefined;

function ensureFallbackRegistry(): Promise<ModelRegistry> {
  if (!fallbackRegistryPromise) {
    const { discoverAuthStorage, getAgentDir, ModelRegistry } = host();
    fallbackRegistryPromise = discoverAuthStorage(getAgentDir())
      .then((authStorage) => {
        const registry = new ModelRegistry(authStorage);
        fallbackRegistry = registry;
        return registry;
      })
      .catch((error) => {
        fallbackRegistryPromise = undefined;
        throw error;
      });
  }
  return fallbackRegistryPromise;
}

/**
 * List the user's currently available models (those with auth configured) with
 * the minimal fields tier ranking needs: canonical spec, output price, and
 * context window. This is the single place the SDK `Model` is projected into
 * the SDK-agnostic `RankableModel`. Best-effort: returns [] if the registry
 * can't be built (or while the disk-backed fallback is still initializing).
 */
export function listAvailableModels(registry?: ModelRegistry): RankableModel[] {
  try {
    const modelRegistry = registry ?? fallbackRegistry;
    if (!modelRegistry) {
      // Kick off the async fallback build; this call reports [] and later
      // calls (e.g. the tool's lazy promptGuidelines re-reads) see real specs.
      void ensureFallbackRegistry().catch(() => {});
      return [];
    }
    return modelRegistry.getAvailable().map((model) => ({
      spec: canonicalModelSpec(model),
      costOutput: model.cost?.output,
      contextWindow: model.contextWindow ?? undefined,
    }));
  } catch {
    return [];
  }
}

/**
 * List the user's currently available models as `provider/modelId` specs. Used
 * to tell the workflow author which models it may route agents to. Best-effort:
 * returns [] if the registry can't be built.
 */
export function listAvailableModelSpecs(registry?: ModelRegistry): string[] {
  return listAvailableModels(registry).map((model) => model.spec);
}

/**
 * Emitted at most once per process: when an agent asks for a tier but no
 * model-tiers.json exists, the tier silently falls back to the session model.
 * Surface that once (with the mapping the user would get by configuring) so the
 * no-op is discoverable. Diagnostics only — never lets a failure break a run.
 */
let warnedTierUnconfigured = false;
function warnTierUnconfiguredOnce(mainModel: string | undefined, registry: ModelRegistry): void {
  if (warnedTierUnconfigured) return;
  warnedTierUnconfigured = true;
  try {
    console.warn(formatTierFallbackNotice(mainModel, listAvailableModels(registry)));
  } catch {
    // best-effort diagnostic
  }
}

/**
 * Emitted at most once per process when persistAgentSessions is enabled and a
 * session is actually persisted: full subagent transcripts (which may include
 * secrets or other sensitive context) are being written to disk. Surface the
 * privacy trade-off at run time, not only in the docs.
 */
let warnedPersistSecrets = false;
function warnPersistSecretsOnce(sessionDir: string): void {
  if (warnedPersistSecrets) return;
  warnedPersistSecrets = true;
  console.warn(
    `[workflow] persistAgentSessions is ON: full subagent transcripts (which may include secrets or other sensitive context) are being written to disk under ${sessionDir}. Disable persistAgentSessions if that isn't intended.`,
  );
}

/** Real token/cost usage for a single subagent run, read from the SDK session. */
export interface AgentUsage {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
  total: number;
  cost: number;
}

/**
 * Map session stats to an AgentUsage, or undefined when the provider reported
 * no usage at all (all-zero stats). Returning undefined — instead of a zero
 * breakdown — lets displays fall back to their scalar token count, so setups
 * on non-reporting providers render the same as before the split existed.
 */
export function usageFromStats(stats: {
  tokens: { input: number; output: number; cacheRead: number; cacheWrite: number; total: number };
  cost: number;
}): AgentUsage | undefined {
  const { tokens, cost } = stats;
  if (tokens.total <= 0 && cost <= 0) return undefined;
  return {
    input: tokens.input,
    output: tokens.output,
    cacheRead: tokens.cacheRead,
    cacheWrite: tokens.cacheWrite,
    total: tokens.total,
    cost,
  };
}

/**
 * Cumulative usage across a session's assistant messages — the same accumulation
 * {@link usageFromStats} gets from the SDK, minus the context-usage estimate
 * `getSessionStats()` also recomputes (a full re-tokenization of the transcript).
 * Cheap enough to sample on every session update, so live progress can be polled
 * mid-run; the final figure still comes from `getSessionStats()`.
 */
export function usageFromMessages(messages: readonly unknown[]): AgentUsage | undefined {
  let input = 0;
  let output = 0;
  let cacheRead = 0;
  let cacheWrite = 0;
  let total = 0;
  let cost = 0;
  for (const message of messages) {
    const assistant = message as Partial<AssistantMessage> | undefined;
    if (assistant?.role !== "assistant" || !assistant.usage) continue;
    const usage = assistant.usage;
    input += usage.input ?? 0;
    output += usage.output ?? 0;
    cacheRead += usage.cacheRead ?? 0;
    cacheWrite += usage.cacheWrite ?? 0;
    total += usage.totalTokens ?? 0;
    cost += usage.cost?.total ?? 0;
  }
  if (total <= 0 && cost <= 0) return undefined;
  return { input, output, cacheRead, cacheWrite, total, cost };
}

export interface AgentRunOptions<TSchemaDef extends TSchema | undefined = undefined> {
  label?: string;
  /**
   * Display name recorded on the persisted session (session_info entry) when
   * `persistAgentSessions` is enabled, so transcripts are identifiable in
   * session pickers (e.g. `workflow:<runId> <label>`). Ignored for in-memory
   * sessions or when an explicit session.sessionManager override is injected.
   */
  sessionName?: string;
  schema?: TSchemaDef;
  tools?: ToolDefinition[];
  instructions?: string;
  signal?: AbortSignal;
  /**
   * Called once with this subagent's real usage, read from the session right
   * before disposal. Fires on both the success and error paths so partial
   * usage is never lost — but NOT when the provider reported no usage at all
   * (all-zero stats), so consumers keep their scalar fallback.
   */
  onUsage?: (usage: AgentUsage) => void;
  /**
   * Called repeatedly while the subagent runs, with its cumulative usage so far
   * (throttled, and only when the figures actually changed). Provider usage lands
   * per assistant message, so this ticks once per model round-trip rather than
   * continuously — enough for live displays to show tokens accruing instead of
   * jumping from nothing to the final number. The terminal `onUsage` call remains
   * authoritative; consumers MUST treat these as cumulative snapshots to overwrite,
   * never as increments to add up.
   */
  onUsageProgress?: (usage: AgentUsage) => void;
  /**
   * Model spec for this subagent: either `provider/modelId` (unambiguous) or a
   * bare `modelId`, parsed with the same grammar as Pi CLI's `--model`. When it
   * can't be resolved to a known model, `run()` throws MODEL_NOT_FOUND rather
   * than silently substituting the session default — a wrong-model run would
   * otherwise look successful while quietly answering with different (or
   * unauthenticated) weights. When omitted, the session default applies.
   */
  model?: string;
  /**
   * Model tier name (e.g. "small", "medium", "big"). When set (and no explicit
   * `model` is given), the model is resolved from the user's model-tiers.json
   * config before `run()` starts, falling back to the session's main model when
   * the tier has no configured entry. An explicit `model` always takes priority,
   * so workflow scripts can use `{ tier: "small" }` for coarse routing without
   * caring which concrete model backs that tier.
   *
   * A script-requested tier that resolves to an unavailable model spec is just
   * as loud as an explicit `model` pin — `run()` throws MODEL_NOT_FOUND naming
   * the tier and the spec it resolved to, e.g. `tier "big" from
   * model-tiers.json resolves to "deadprov/x", which is not available`.
   *
   * That's deliberately asymmetric with the IMPLICIT default tier an untagged
   * agent (neither `model` nor `tier` set) gets routed through: since the
   * script never asked for that tier, a broken default degrades to the
   * session default instead of failing every untagged agent in the run — see
   * onModelFallback below for how that degrade stays visible.
   */
  tier?: string;
  /** Called with the resolved model id once known (for display/telemetry). */
  onModelResolved?: (modelId: string) => void;
  /**
   * Called (at most once per WorkflowAgent instance) when an UNTAGGED agent's
   * implicit default "medium" tier resolves to a model spec that isn't
   * available. This is the one case that degrades to the session default
   * instead of throwing MODEL_NOT_FOUND (see `tier` above) — but the degrade
   * must still land in the run's own log/event stream, not just a
   * console.warn, or a broken default tier silently drifts every untagged
   * agent's model with zero trace in the run itself.
   */
  onModelFallback?: (info: { tier: string; requestedSpec: string }) => void;
  /** Called with a compact snapshot of this subagent's message/tool history. */
  onHistory?: (history: AgentHistoryEntry[]) => void;
  /** Run this agent in a different working directory (e.g. an isolated worktree). */
  cwd?: string;
  /**
   * Restrict the subagent's coding tools to these names (an agentType
   * definition's `tools` allowlist). Undefined = all coding tools. The
   * structured_output tool is always added after this filter, so a schema
   * still works under a restrictive allowlist.
   */
  toolNames?: string[];
  /** Remove these coding-tool names after the allowlist (an agentType `disallowedTools` denylist). */
  disallowedToolNames?: string[];
  /**
   * With `schema`: how many extra repair turns to allow if the model finishes
   * without calling structured_output. Each retry re-prompts (tools restricted to
   * structured_output) before falling back to strict prose extraction. Default 2.
   */
  maxSchemaRetries?: number;
  /**
   * Tools that are always injected AFTER the tool-policy filter (`toolNames` /
   * `disallowedToolNames`), so they are available even under a restrictive
   * allowlist. Used by the workflow runtime to inject shared-store tools into
   * every agent regardless of its agentType definition.
   */
  systemTools?: ToolDefinition[];
  /**
   * Per-run model registry override. Takes precedence over the constructor's
   * `modelRegistry` (WorkflowAgentOptions.modelRegistry) for both model
   * resolution and the `createAgentSession` call this run makes. Falls back to
   * the constructor's shared registry, then a lazily-built disk registry, when
   * omitted.
   */
  modelRegistry?: ModelRegistry;
}

export type AgentRunResult<TSchemaDef extends TSchema | undefined> = TSchemaDef extends TSchema
  ? Static<TSchemaDef>
  : string;

/**
 * Orchestration tools ALWAYS denied to workflow subagents. The `workflow` and
 * `workflow_control` tools are registered globally by the extension, so — unless
 * excluded — a subagent's session sees them and can start its own independent
 * background workflows. Those nested runs recursively fan out and are NOT bounded
 * by the parent run's maxAgents / concurrency / progress / accounting, and can
 * drain a shared provider quota and pile up paused runs (#107). Callers may deny
 * additional tool names via WorkflowAgentOptions.excludeTools.
 */
export const DEFAULT_EXCLUDED_SUBAGENT_TOOLS = ["workflow", "workflow_control"];

/**
 * The full subagent tool denylist: the always-on defaults plus any names the
 * caller added (via WorkflowAgentOptions.excludeTools) or set on the injected
 * session options. Extracted so the merge — and its order — is unit-testable;
 * a spread-order regression that dropped the defaults would slip past a test
 * that only asserts the constant. The SDK dedupes, so overlap is harmless.
 */
export function subagentExcludedTools(extra?: string[], sessionExclude?: string[]): string[] {
  return [...DEFAULT_EXCLUDED_SUBAGENT_TOOLS, ...(sessionExclude ?? []), ...(extra ?? [])];
}

/**
 * Monotonic per-process counter behind {@link workflowAgentIdentity}.
 */
let agentIdentitySeq = 0;

/**
 * A unique agent-registry identity for one workflow subagent.
 *
 * `createAgentSession` derives its registry id as `agentId ?? parentTaskPrefix
 * ?? "Main"`, and registering that id into the global AgentRegistry EVICTS the
 * previous holder. Passing none of the three made every workflow subagent
 * claim the host session's "Main" slot: two subagents initializing
 * concurrently raced, and the loser's own `attachSession()` found its ref
 * already replaced and threw `Agent "Main" was replaced during session
 * initialization.` — a red, errored agent seconds into a run, with no
 * connection to anything the script did. A distinct id per subagent removes
 * the contention (and makes each subagent addressable in the registry).
 */
export function workflowAgentIdentity(label?: string): string {
  const slug =
    (label ?? "")
      .replace(/[^\w.-]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 40) || "agent";
  return `wf-${slug}-${(++agentIdentitySeq).toString(36)}`;
}

export class WorkflowAgent {
  private readonly cwd: string;
  private readonly baseTools: ToolDefinition[];
  /** Extra subagent tool-name denylist, merged with the always-on defaults. */
  private readonly excludeTools: string[];
  private readonly sessionOptions: Partial<CreateAgentSessionOptions>;
  private readonly persistAgentSessions: boolean;
  private readonly instructions?: string;
  private readonly mainModel?: string;
  /** Shared registry from the host session, when provided. */
  private readonly sharedRegistry?: ModelRegistry;
  /** Lazily built once; shares the SDK's agentDir/auth so resolved models are authed. */
  private registry?: ModelRegistry;
  /**
   * Memoized model-tiers.json snapshot, boxed so a legitimately-null config
   * (file absent/invalid) is distinguishable from "not loaded yet". See
   * loadTierConfig() below for why this is scoped per-instance.
   */
  private tierConfigBox?: { value: ModelTierConfig | null };
  /** Shared OMP settings snapshot for every subagent in this workflow run. */
  private sharedSettingsPromise?: Promise<Settings>;
  /**
   * Emitted at most once per instance (~= once per run, see the class-level
   * lifetime note above): the untagged/default "medium" tier resolved to a
   * model spec that isn't available. Deliberately per-instance rather than a
   * MODEL_NOT_FOUND throw — an untagged agent never asked for that specific
   * model, so a broken default tier shouldn't fail every untagged agent in the
   * run. See onModelFallback below for the (still-loud) degrade path.
   */
  private warnedDefaultTierUnavailable = false;

  constructor(options: WorkflowAgentOptions = {}) {
    this.cwd = options.cwd ?? process.cwd();
    this.baseTools = options.tools ?? [];
    this.excludeTools = options.excludeTools ?? [];
    this.sessionOptions = options.session ?? {};
    this.persistAgentSessions = options.persistAgentSessions ?? false;
    this.instructions = options.instructions;
    this.mainModel = options.mainModel;
    this.sharedRegistry = options.modelRegistry;
  }

  /**
   * Load OMP settings once per workflow run. Extension discovery is disabled
   * separately on every child session, so no parent extension instance or
   * recursive workflow tool can leak into subagents.
   */
  private getSharedSettings(agentDir: string): Promise<Settings> {
    this.sharedSettingsPromise ??= host().Settings.loadIsolated({
      cwd: this.cwd,
      agentDir,
    }).catch((error) => {
      this.sharedSettingsPromise = undefined;
      throw error;
    });
    return this.sharedSettingsPromise;
  }

  /**
   * Resolve the registry for a run: an explicit per-run registry wins, then the
   * constructor's shared registry, then a lazily-built disk registry (shared
   * across calls once built). Async because pi >= 0.80.8 builds registries from
   * an async-created ModelRuntime.
   */
  private async getRegistry(perRunRegistry?: ModelRegistry): Promise<ModelRegistry> {
    if (perRunRegistry) {
      return perRunRegistry;
    }
    if (this.sharedRegistry) {
      return this.sharedRegistry;
    }
    if (!this.registry) {
      this.registry = await ensureFallbackRegistry();
    }
    return this.registry;
  }

  /**
   * Read+parse ~/.omp/workflows/model-tiers.json at most once for this
   * instance's lifetime, instead of on every run() call. `resolveAgentModelSpec`
   * previously received `loadModelTierConfig` directly (sync existsSync +
   * readFileSync + JSON.parse from disk), which it calls unconditionally for
   * any agent without an explicit options.model — so a large fan-out did N
   * redundant synchronous disk reads that blocked the event loop and stalled
   * concurrent agents' I/O.
   *
   * `runWorkflow()` constructs a fresh `WorkflowAgent` per run (see
   * `new WorkflowAgent(options)` in workflow.ts, unless a caller injects its
   * own `options.agent` runner — a test-only escape hatch per
   * WorkflowManagerOptions.agent's doc comment), so a WorkflowAgent instance's
   * lifetime is one run in production. Memoizing on `this` therefore has the
   * same scope and lifetime as the agentRegistry snapshot workflow.ts already
   * takes once per run "for determinism" — the config file isn't expected to
   * change mid-run, and two different runs (= two different WorkflowAgent
   * instances) each get their own fresh read of whatever is on disk at the
   * time, so this does not leak stale config across runs or break tests that
   * construct fresh agents with different configs.
   *
   * `loader` is injectable for tests (defaults to the real disk read); it is
   * only ever consulted once, on the first call, regardless of what is passed
   * on later calls.
   */
  private loadTierConfig(loader: () => ModelTierConfig | null = loadModelTierConfig): ModelTierConfig | null {
    if (!this.tierConfigBox) {
      this.tierConfigBox = { value: loader() };
    }
    return this.tierConfigBox.value;
  }

  /**
   * Session manager for one subagent run. File-backed (persisted under the
   * standard sessions dir, keyed by the runner's project cwd — never a
   * per-call worktree cwd) when persistAgentSessions is on; in-memory otherwise.
   *
   * SessionManager.create() only creates the session directory — the SDK writes
   * the session file lazily (synchronous fs calls, uncaught) on the first
   * assistant message, deep inside session.prompt(). A failure there would
   * otherwise throw mid-run and abort this subagent. Probe writability up front
   * so any create/write failure (permissions, disk full) degrades this single
   * agent to an in-memory session instead — the run continues, just without a
   * persisted transcript.
   */
  private createSessionManager(): SessionManager {
    const { SessionManager } = host();
    if (!this.persistAgentSessions) return SessionManager.inMemory();
    try {
      const manager = SessionManager.create(this.cwd);
      this.assertSessionDirWritable(manager.getSessionDir());
      warnPersistSecretsOnce(manager.getSessionDir());
      return manager;
    } catch (error) {
      console.warn(
        `[workflow] persistAgentSessions: could not persist this agent's session (${
          error instanceof Error ? error.message : String(error)
        }); continuing with an in-memory session`,
      );
      return SessionManager.inMemory();
    }
  }

  /** Best-effort write probe: throws if the session directory isn't actually writable. */
  private assertSessionDirWritable(dir: string): void {
    const probePath = join(dir, `.write-probe-${randomUUID()}`);
    writeFileSync(probePath, "");
    unlinkSync(probePath);
  }

  async run<TSchemaDef extends TSchema | undefined = undefined>(
    prompt: string,
    options: AgentRunOptions<TSchemaDef> = {},
  ): Promise<AgentRunResult<TSchemaDef>> {
    const capture: StructuredOutputCapture<any> = { called: false, value: undefined };
    const runCwd = options.cwd ?? this.cwd;
    const nativeToolNames = applyToolPolicy(
      (this.sessionOptions.toolNames ?? [...OMP_CODING_TOOL_NAMES]).map((name) => ({ name })),
      options.toolNames,
      options.disallowedToolNames,
    ).map(({ name }) => name);
    // Agent-type policy also applies to extension-supplied toolsets.
    const customTools: ToolDefinition[] = applyToolPolicy(
      [...this.baseTools, ...(options.tools ?? [])],
      options.toolNames,
      options.disallowedToolNames,
    );

    // System tools obey the same effective tool policy, so a write-capable system
    // tool (e.g. the shared-store `store_put`) cannot bypass an agent-type
    // allowlist/denylist. `structured_output` is NOT a system tool — it is added
    // separately below — so a schema still works under a restrictive allowlist.
    if (options.systemTools?.length) {
      customTools.push(...applyToolPolicy(options.systemTools, options.toolNames, options.disallowedToolNames));
    }

    if (options.schema) {
      // Strict OpenAI-compatible providers (e.g. DeepSeek) reject a tool whose top-level
      // parameters schema isn't a JSON object with a transport-level 400, before any of
      // this file's SCHEMA_NONCOMPLIANCE/empty-output classification ever runs. Fail fast
      // here instead, so a script's non-object opts.schema surfaces a clear workflow error.
      const schemaType = (options.schema as { type?: unknown }).type;
      if (schemaType !== "object") {
        throw new WorkflowError(
          `agent() opts.schema must be a top-level JSON object schema (type: "object") — got type: ${schemaType ?? "undefined"}; wrap array/primitive results in an object, e.g. { type: "object", properties: { items: <your schema> } }`,
          WorkflowErrorCode.SCRIPT_VALIDATION_ERROR,
          { recoverable: false },
        );
      }
      customTools.push(createStructuredOutputTool({ schema: options.schema, capture }) as unknown as ToolDefinition);
    }

    // Per-run modelRegistry wins over the constructor's shared registry, then
    // the lazily-built disk fallback. Used for tier diagnostics, model
    // resolution, and the subagent session's runtime below.
    const modelRegistry = await this.getRegistry(options.modelRegistry);

    // Resolve the model spec (explicit model > tier > session default). This
    // composes with phase-based routing in workflow.ts, which only supplies
    // options.model when a phase pattern matches — so an explicit model wins.
    const modelSpec = resolveAgentModelSpec(
      options,
      this.mainModel,
      () => this.loadTierConfig(),
      () => warnTierUnconfiguredOnce(this.mainModel, modelRegistry),
    );

    // Resolve a requested model spec to a Model object. Specs use Pi CLI-style
    // parsing, including an optional :thinking suffix such as gpt-5.5:xhigh.
    //
    // A given-but-unresolved spec's behavior is asymmetric by design (#131):
    //   - options.model or options.tier was explicitly set by the script (or by
    //     workflow.ts's phase-based routing, which only ever supplies
    //     options.model when the user configured that phase) → throw
    //     MODEL_NOT_FOUND naming the source. Resolution is deterministic, so
    //     retrying the same spec is pointless (recoverable:false), and a silent
    //     substitution would otherwise run real API calls against a different
    //     (or unauthenticated) model while the caller believes its pin/tier was
    //     honored.
    //   - neither was set: the agent is UNTAGGED and only got routed through
    //     the implicit default "medium" tier because *some other* agent's tier
    //     is configured (see resolveAgentModelSpec). This agent never asked for
    //     that model, so a broken default tier degrades to the session default
    //     instead of failing every untagged agent in the run — but the degrade
    //     still needs to be loud (onModelFallback), not a silent continuation.
    const isExplicitRequest = Boolean(options.model || options.tier);
    let resolvedModel: Model<any> | undefined;
    let resolvedThinkingLevel: CreateAgentSessionOptions["thinkingLevel"] | undefined;
    if (modelSpec) {
      const resolved = resolveModelSpecWithThinking(modelSpec, modelRegistry);
      if (resolved.warning) console.warn(`[workflow] ${resolved.warning}`);
      if (!resolved.model) {
        if (isExplicitRequest) {
          // The resolver's error already names the spec and the remedy; the tier
          // branch swaps in its own message so the config source is named too.
          const message = options.model
            ? (resolved.error ?? `Model "${modelSpec}" not found. Use /workflows-models to choose an available model.`)
            : `tier "${options.tier}" from model-tiers.json resolves to "${modelSpec}", which is not available. Use /workflows-models to choose an available model.`;
          throw new WorkflowError(message, WorkflowErrorCode.MODEL_NOT_FOUND, {
            recoverable: false,
            agentLabel: options.label,
          });
        }
        if (!this.warnedDefaultTierUnavailable) {
          this.warnedDefaultTierUnavailable = true;
          options.onModelFallback?.({ tier: "medium", requestedSpec: modelSpec });
        }
      } else {
        resolvedModel = resolved.model;
        resolvedThinkingLevel = await parseConfiguredThinkingLevel(resolved.thinkingLevel);
        options.onModelResolved?.(resolved.resolvedSpec ?? canonicalModelSpec(resolved.model));
      }
    }

    const agentDir = host().getAgentDir();
    // Key persisted sessions by the runner's project cwd (this.cwd), not a
    // short-lived per-call worktree.
    const sessionManager = this.createSessionManager();
    const injectedCustomTools = this.sessionOptions.customTools ?? [];
    const allCustomTools = [...injectedCustomTools, ...customTools];
    const deniedTools = new Set(subagentExcludedTools(this.excludeTools));
    const toolNames = [
      ...nativeToolNames,
      ...allCustomTools.map((tool) => tool.name),
    ].filter((name, index, names) => !deniedTools.has(name) && names.indexOf(name) === index);
    const { session } = await host().createAgentSession({
      ...this.sessionOptions,
      cwd: runCwd,
      agentDir,
      sessionManager,
      // Registry identity: never inherit the default "Main" (see
      // workflowAgentIdentity). taskDepth marks these sessions as subagents,
      // which is what they are — it also keeps depth-0-only machinery
      // (per-session memory-backend init, vibe tools) out of every subagent.
      agentId: this.sessionOptions.agentId ?? workflowAgentIdentity(options.label),
      agentDisplayName: this.sessionOptions.agentDisplayName ?? "sub",
      taskDepth: this.sessionOptions.taskDepth ?? 1,
      settings: this.sessionOptions.settings ?? (await this.getSharedSettings(agentDir)),
      modelRegistry,
      customTools: allCustomTools,
      // OMP's native restriction contract replaces Pi's resourceLoader and
      // excludeTools hooks. No ambient extensions/MCP tools enter child sessions.
      disableExtensionDiscovery: true,
      extensions: [],
      additionalExtensionPaths: [],
      enableMCP: false,
      enableIrc: false,
      toolNames,
      restrictToolNames: true,
      allowRestrictedCustomTools: allCustomTools.length > 0,
      // Per-call model/thinking wins over injected defaults.
      ...(resolvedModel ? { model: resolvedModel } : {}),
      ...(resolvedThinkingLevel ? { thinkingLevel: resolvedThinkingLevel } : {}),
    });

    if (this.persistAgentSessions && !this.sessionOptions.sessionManager && options.sessionName) {
      try {
        await sessionManager.setSessionName(options.sessionName, "user");
      } catch {
        // Naming is best-effort; never fail the run over it.
      }
    }

    let removeAbortListener: (() => void) | undefined;
    let removeHistoryListener: (() => void) | undefined;
    let lastHistoryEmit = 0;
    const emitHistory = () => options.onHistory?.(compactAgentHistory(session.messages));
    const maybeEmitHistory = () => {
      if (!options.onHistory) return;
      const now = Date.now();
      if (now - lastHistoryEmit < 250) return;
      lastHistoryEmit = now;
      emitHistory();
    };
    let lastUsageEmit = 0;
    let lastUsageTotal = -1;
    let lastUsageCost = -1;
    // Sampled off the same session-update stream as history: usage lands on the
    // assistant message when a model round-trip completes, so this reports real
    // accrual (never an estimate) and stays silent between round-trips.
    const maybeEmitUsageProgress = () => {
      if (!options.onUsageProgress) return;
      const now = Date.now();
      if (now - lastUsageEmit < 1000) return;
      lastUsageEmit = now;
      const usage = usageFromMessages(session.messages);
      if (!usage || (usage.total === lastUsageTotal && usage.cost === lastUsageCost)) return;
      lastUsageTotal = usage.total;
      lastUsageCost = usage.cost;
      options.onUsageProgress(usage);
    };
    try {
      if (options.signal?.aborted) throw new Error("Subagent was aborted");
      if (options.signal) {
        const onAbort = () => void session.abort();
        options.signal.addEventListener("abort", onAbort, { once: true });
        removeAbortListener = () => options.signal?.removeEventListener("abort", onAbort);
      }
      if (options.onHistory || options.onUsageProgress) {
        removeHistoryListener = session.subscribe(() => {
          maybeEmitHistory();
          maybeEmitUsageProgress();
        });
      }

      await session.prompt(this.buildPrompt(prompt, options as AgentRunOptions<any>, Boolean(options.schema)));

      if (options.signal?.aborted) throw new Error("Subagent was aborted");

      // The SDK buries a provider usage/quota limit in the assistant message rather
      // than throwing; detect it here (before the schema/empty-text branches) so it
      // is classified as a recoverable checkpoint, not a SCHEMA_NONCOMPLIANCE failure
      // (schema path) or a silent empty-output null (non-schema path).
      throwIfProviderLimit(session.messages, options.label);

      if (options.schema) {
        return (await resolveStructuredOutput(session, capture, options.schema, options, (m) =>
          this.lastAssistantText(m),
        )) as AgentRunResult<TSchemaDef>;
      }

      // Unstructured result: require assistant text AFTER the last tool result.
      // Text emitted before it is stale progress (the agent's last real action was
      // a tool call) — accepting it would report an incomplete run as successful
      // and suppress the AGENT_EMPTY_OUTPUT retry (#111).
      const text = this.finalAssistantText(session.messages);
      if (!text.trim()) {
        throw new WorkflowError("Subagent produced no assistant output", WorkflowErrorCode.AGENT_EMPTY_OUTPUT, {
          recoverable: true,
          agentLabel: options.label,
        });
      }
      return text as AgentRunResult<TSchemaDef>;
    } finally {
      removeAbortListener?.();
      removeHistoryListener?.();
      try {
        emitHistory();
      } catch {
        // History is diagnostic only; never let it mask the real result/error.
      }
      // Read real usage before disposing — dispose tears down the session state.
      if (options.onUsage) {
        try {
          const usage = usageFromStats(session.getSessionStats());
          if (usage) options.onUsage(usage);
        } catch {
          // Usage is best-effort; never let stats failure mask the real result/error.
        }
      }
      session.dispose();
    }
  }

  private buildPrompt(prompt: string, options: AgentRunOptions<any>, structured: boolean): string {
    const parts = [
      this.instructions,
      options.instructions,
      options.label ? `Task label: ${options.label}` : undefined,
      prompt,
    ].filter(Boolean);

    if (structured) {
      parts.push(
        [
          "Final output contract:",
          "- Your final action MUST be a structured_output tool call.",
          "- The structured_output arguments are the return value of this subagent.",
          "- Do not emit a prose final answer instead of structured_output.",
          "- If you need to inspect files or run commands first, do so, then call structured_output exactly once.",
        ].join("\n"),
      );
    }

    return parts.join("\n\n");
  }

  private lastAssistantText(messages: unknown[]): string {
    for (let i = messages.length - 1; i >= 0; i--) {
      const message = messages[i] as Partial<AssistantMessage> | undefined;
      if (message?.role !== "assistant" || !Array.isArray(message.content)) continue;
      const text = message.content
        .filter((part): part is TextContent => part.type === "text")
        .map((part) => part.text)
        .join("");
      if (text.trim()) return text;
    }
    return "";
  }

  /**
   * The unstructured agent's FINAL answer: assistant text that appears after the
   * last tool result. Text before the final tool result is stale progress (the
   * agent's last real action was a tool call, not answering), so returning it
   * would mask an incomplete run and suppress AGENT_EMPTY_OUTPUT retries (#111).
   *
   * Distinct from lastAssistantText(), which stays deliberately lenient — the
   * schema path's prose-JSON recovery (resolveStructuredOutput) may need to read
   * the structured payload out of any assistant message, not only the terminal one.
   */
  private finalAssistantText(messages: unknown[]): string {
    // Locate the last tool result; only assistant text strictly after it counts.
    let lastToolResult = -1;
    for (let i = messages.length - 1; i >= 0; i--) {
      if ((messages[i] as { role?: string } | undefined)?.role === "toolResult") {
        lastToolResult = i;
        break;
      }
    }
    for (let i = messages.length - 1; i > lastToolResult; i--) {
      const message = messages[i] as Partial<AssistantMessage> | undefined;
      if (message?.role !== "assistant" || !Array.isArray(message.content)) continue;
      const text = message.content
        .filter((part): part is TextContent => part.type === "text")
        .map((part) => part.text)
        .join("");
      if (text.trim()) return text;
    }
    return "";
  }
}

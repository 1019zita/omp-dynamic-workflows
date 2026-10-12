/**
 * Host helpers that are *not* on the injected `ExtensionAPI.pi` namespace and so
 * can only come from a real `@oh-my-pi/*` import.
 *
 * Under omp's extension loader those imports are remapped to the host's bundled
 * copy and re-evaluated for this extension's graph, which is expensive: a static
 * `@oh-my-pi/pi-tui/thinking` import alone measures ~385ms of
 * `loadExtensions` time, paid before the TUI paints on every launch. A dynamic
 * `import()` goes through the same shim but only when something actually needs
 * it, and most sessions never run a workflow at all.
 *
 * Each entry exposes `warm()` (async, idempotent, safe to call repeatedly) and a
 * synchronous getter for the call sites that cannot be made async. Every such
 * call site is reachable only from an async entry point that awaits `warm()`
 * first; the getter throws rather than silently degrading if that invariant is
 * ever broken.
 */

interface LazyHostModule<T> {
  /** Load (once) and cache the module. */
  warm(): Promise<T>;
  /** The cached module. Throws when `warm()` has not resolved yet. */
  get(): T;
}

function lazyHostModule<T>(specifier: string, load: () => Promise<T>): LazyHostModule<T> {
  let loaded: T | undefined;
  let pending: Promise<T> | undefined;
  return {
    warm() {
      pending ??= load().then((module) => {
        loaded = module;
        return module;
      });
      return pending;
    },
    get() {
      if (!loaded) {
        throw new Error(
          `omp-dynamic-workflows: "${specifier}" was used before it was warmed. ` +
            "Await the corresponding warm*() call on the enclosing async entry point.",
        );
      }
      return loaded;
    },
  };
}

const thinkingModule = lazyHostModule("@oh-my-pi/pi-tui/thinking", () =>
  import("@oh-my-pi/pi-tui/thinking"),
);
const frontmatterModule = lazyHostModule("@oh-my-pi/pi-utils/frontmatter", () =>
  import("@oh-my-pi/pi-utils/frontmatter"),
);
const schemaModule = lazyHostModule("@oh-my-pi/pi-ai/utils/schema", () => import("@oh-my-pi/pi-ai/utils/schema"));

/** Resolve a configured thinking level. Loads the host `thinking` module on first use. */
export async function parseConfiguredThinkingLevel(
  value: unknown,
): Promise<ReturnType<Awaited<ReturnType<typeof thinkingModule.warm>>["parseConfiguredThinkingLevel"]>> {
  const module = await thinkingModule.warm();
  return module.parseConfiguredThinkingLevel(value as never);
}

/** Preload the frontmatter parser used by the synchronous agent-registry reader. */
export function warmFrontmatter(): Promise<unknown> {
  return frontmatterModule.warm();
}

/** Parse agent-definition frontmatter. Requires a prior `warmFrontmatter()`. */
export function parseFrontmatter(content: string): { frontmatter: Record<string, unknown>; body: string } {
  return frontmatterModule.get().parseFrontmatter(content) as {
    frontmatter: Record<string, unknown>;
    body: string;
  };
}

/** Preload the JSON-Schema validator used by the synchronous `Check()` fallback. */
export function warmSchemaValidator(): Promise<unknown> {
  return schemaModule.warm();
}

/**
 * JSON-Schema validation fallback for schemas that carry no `safeParse`.
 * Requires a prior `warmSchemaValidator()`.
 */
export function isJsonSchemaValueValid(schema: unknown, value: unknown): boolean {
  return schemaModule.get().isJsonSchemaValueValid(schema as never, value);
}

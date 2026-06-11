# AGENTS.md

## Project Context

This repository is based on `evolutionary-ai-battle`, a 2D battleground simulation where AI bots fight each other and evolve over time.

Current goal:

1. First, modernize the legacy JavaScript/Webpack/Babel project so it runs reliably on a modern Node/npm setup.
2. Then, incrementally migrate the codebase from JavaScript to TypeScript / TSX.
3. Preserve existing simulation behavior while making the codebase easier to extend into a co-player agent evaluation harness.

Important: Do not start the co-player/evaluation feature work yet. First make the existing project build and run.

---

## Current Known Issues

The project appears to have legacy Webpack/Babel configuration issues.

Known error:

```text
[webpack-cli] Invalid configuration object.
Webpack has been initialized using a configuration object that does not match the API schema.
- configuration.node has an unknown property 'fs'. These properties are valid:
  object { __dirname?, __filename?, global? }
```

This likely means the current `webpack.config.js` was written for an older Webpack version and still contains legacy config such as:

```js
node: {
  fs: "empty"
}
```

In Webpack 5, this must be removed or replaced with the correct `resolve.fallback` pattern only when needed.

Also note that Windows does not support `/dev/null`, so avoid scripts like:

```sh
npm run compile-browser >/dev/null && node server.js
```

Use cross-platform scripts instead.

---

## Current package.json Notes

The project currently uses:

```json
"webpack": "^5.76.0",
"webpack-cli": "^5.0.1",
"@babel/core": "^7.9.0",
"@babel/cli": "^7.8.4",
"@babel/preset-env": "^7.9.0"
```

Current scripts include:

```json
"start": "npm run compile-browser && node server.js",
"compile-browser": "npx webpack --mode production",
"compile-node": "npx babel -d dist/ src",
"train": "npm run compile-node && node dist/coordinator.js"
```

Do not assume the current Webpack/Babel configs are valid. Inspect them.

---

## Phase 1 Goal: Make Existing JS Project Run

Before TypeScript migration, make the current JavaScript project work.

Tasks:

1. Inspect the repo structure.
2. Identify:

   * `webpack.config.js`
   * Babel config
   * browser entrypoint
   * Node/headless training entrypoint
   * `server.js`
   * output folder expected by `server.js`
3. Fix Webpack 5 incompatible config.
4. Fix Windows-incompatible scripts.
5. Run:

   * `npm install`
   * `npm run compile-browser`
   * `npm start`
   * `npm run compile-node`
   * `npm run train`, if feasible
6. Keep changes minimal and explain each fix.

Do not rewrite the simulator yet.

---

## Webpack 5 Migration Rules

If you see legacy Webpack config such as:

```js
node: {
  fs: "empty",
  net: "empty",
  tls: "empty"
}
```

Do not keep it.

Use one of these approaches:

### If the browser bundle does not need Node modules

Remove the legacy `node` block entirely.

### If browser build imports Node-only modules accidentally

Prefer to separate browser and Node entrypoints.

Browser bundle should not include modules that depend on:

```text
fs
path
os
crypto
net
tls
child_process
```

### If a fallback is truly needed

Use Webpack 5 style:

```js
resolve: {
  fallback: {
    fs: false,
    path: false,
    os: false
  }
}
```

But do not add random polyfills unless necessary.

---

## Build Strategy

Prefer separate configs if needed:

```text
webpack.browser.config.js
babel.config.json
```

Browser build:

```text
src/index.js -> dist/browser/bundle.js
```

Node/headless build:

```text
src/coordinator.js -> dist/coordinator.js or dist/nodejs/coordinator.js
```

Make sure `server.js` serves the same browser bundle path that Webpack actually outputs.

If current `server.js` expects a different path, either:

1. adjust Webpack output to match server expectations, or
2. adjust server static path carefully.

Do not silently change runtime behavior.

---

## Phase 2 Goal: Convert JS to TS / TSX Incrementally

Only start after Phase 1 works.

Migration strategy:

1. Add TypeScript tooling.
2. Add `tsconfig.json`.
3. Rename files gradually:

   * `.js` -> `.ts` for logic files
   * `.js` or Vue/browser UI entry files -> `.tsx` only if React/JSX is introduced
4. Do not convert everything in one huge change if the project is unstable.
5. Keep browser build and headless training build working after each step.

Recommended dev dependencies:

```sh
npm install -D typescript ts-loader @types/node
```

If React is not used, do not introduce React just to use TSX. Use `.ts` unless JSX is actually needed.

---

## TypeScript Migration Rules

Use TypeScript to clarify simulation types, not to over-engineer.

Good candidates for early typing:

```text
Bot
Player
Bullet
BattleState
BotInput
BotAction
Genome
Gene
Species
FitnessResult
```

Define shared types in:

```text
src/types.ts
```

or:

```text
src/core/types.ts
```

Suggested types:

```ts
export type BotAction = {
  ds: boolean;
  dx: number;
  dy: number;
  dh: number;
};

export type BulletState = {
  xPos: number;
  yPos: number;
  rotation: number;
};

export type PlayerState = {
  xPos: number;
  yPos: number;
  rotation: number;
  bullets?: BulletState[];
  lives?: number;
};

export type BotInput = PlayerState & {
  otherPlayer: PlayerState;
};
```

Keep typing pragmatic. Use `unknown` or `TODO` where necessary rather than blocking the migration.

---

## Do Not Do Yet

Do not add these until the project builds and runs:

```text
co-player fitness
trajectory logger
partial observation agent
Survev adapter
new UI
React migration
RL
LLM/VLM
large refactor
```

First priority is a working modernized baseline.

---

## After Modernization: Future Project Direction

After JS/TS modernization, this project will be extended toward:

```text
evolving a helpful co-player and measuring why it helps or fails
```

The future direction is not simply “make stronger battle bots.”

Future metrics may include:

```text
teammate_distance_mean
teammate_abandon_events
enemy_pressure_response_rate
ignored_enemy_pressure_count
harmful_action_ratio
support_action_ratio
repeated_action_segments
stuck_segments
action_reason_distribution
```

But do not implement these yet unless explicitly asked.

---

## Coding Style

Prioritize:

```text
small commits
minimal behavior changes
clear build fixes
cross-platform scripts
readable config
working commands
```

Avoid:

```text
large refactors
unnecessary dependency upgrades
changing simulation behavior while fixing build
mixing modernization with new features
```

---

## Required Output From Codex

Before editing:

1. Summarize relevant repo structure.
2. Identify the current build pipeline.
3. Identify the likely cause of the Webpack error.
4. Propose a minimal fix plan.

After editing:

1. List changed files.
2. Explain each change briefly.
3. Provide exact commands to run.
4. Report whether each command passed or failed.
5. If a command failed, show the next concrete fix.

---

## Commands To Try

Initial diagnostics:

```sh
node -v
npm -v
npm install
npm run compile-browser
npm start
npm run compile-node
npm run train
```

If `npm run train` is too slow or long-running, verify only that it starts correctly, then stop it.

---

## Success Criteria

Phase 1 is successful when:

```text
npm run compile-browser
```

completes without Webpack schema errors, and:

```text
npm start
```

starts the local server.

Phase 2 is successful when:

```text
TypeScript is added
at least one core module is migrated to .ts
the browser build still works
the Node/headless build still works
```

# Repository Guidelines

## Project Structure & Module Organization
`moshi/` contains the Python package and runtime entry points. Core model code lives in `moshi/moshi/models/`, reusable layers in `moshi/moshi/modules/`, and CLI/server flows in `moshi/moshi/server.py` and `moshi/moshi/offline.py`.

`client/` is the Vite + React frontend. App pages live under `client/src/pages/`, shared UI in `client/src/components/`, protocol code in `client/src/protocol/`, and browser assets in `client/public/`. Demo media and prompt examples are stored in `assets/`, including `assets/test/`.

## Build, Test, and Development Commands
Install the Python package from the repo root with `pip install moshi/.`.
Run the local server with `SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR"`.
Run offline inference with `python -m moshi.offline --help` and then the examples from [README.md](/Users/davidkrinurs/projects/personaplex/README.md).

For the frontend:
- `cd client && npm install` installs dependencies.
- `cd client && npm run dev` starts the Vite dev server.
- `cd client && npm run build` runs `tsc` and produces a production build.
- `cd client && npm run lint` checks ESLint rules.
- `cd client && npm run prettier` formats the frontend codebase.

## Coding Style & Naming Conventions
Python follows 4-space indentation and a 120-character line limit in [moshi/setup.cfg](/Users/davidkrinurs/projects/personaplex/moshi/setup.cfg). Keep modules snake_case, classes PascalCase, and prefer small, composable utilities.

Frontend formatting is defined in [client/.prettierrc.json](/Users/davidkrinurs/projects/personaplex/client/.prettierrc.json): 2-space indentation, semicolons, double quotes, trailing commas, and Tailwind-aware sorting. Use PascalCase for React components (`Queue.tsx`), camelCase for hooks/utilities (`useSocket.ts`), and keep file names aligned with exported symbols.

## Testing Guidelines
There is no committed automated test suite yet. Before opening a PR, run `cd client && npm run build` and `cd client && npm run lint`, then smoke-test one server path and one frontend interaction locally. If you add tests, place frontend tests beside the feature or under `client/src`, and keep Python tests under `moshi/tests/`.

## Commit & Pull Request Guidelines
Recent history favors short, imperative subjects, with occasional Conventional Commit prefixes such as `fix:` and `docs:`. Follow that pattern: `fix: guard low-memory init` is preferred over vague summaries.

PRs should state the user-visible change, note any setup or model requirements, link the relevant issue, and include screenshots for UI changes. Call out changes that affect GPU memory use, Hugging Face auth, or local SSL setup.

Goal:
Make line-length/max-len stop blocking dev/check agents in both repos by adding formatter-first auto-fix flows and disabling unsafe line-length lint failures where formatters cannot safely wrap.

Repo 1: darizae/alpaca-ui

1. Frontend TypeScript/React:
   - Add Prettier to packages/frontend devDependencies.
   - Add packages/frontend/.prettierrc:
     {
       "printWidth": 100,
       "semi": true,
       "singleQuote": false,
       "trailingComma": "all"
     }

   - Add/adjust packages/frontend/package.json scripts:
     "format": "prettier --write . && eslint . --fix",
     "format:check": "prettier --check .",
     "lint": "eslint . --fix"

   - Keep ESLint max-len disabled. Do not re-enable it.
   - Do not use ESLint to enforce line length; Prettier owns TS/TSX wrapping.

2. Backend Python:
   - In packages/backend/pyproject.toml, ensure Ruff has:
     [tool.ruff]
     line-length = 100
     target-version = "py311"

     [tool.ruff.lint]
     select = ["E", "F", "I", "W"]
     ignore = ["E501"]

   - Add/adjust backend dev script to run:
     ruff format .
     ruff check . --fix

   - Ensure backend check runs formatter/fixer before mypy/tests.

3. Root scripts:
   - Add root package.json scripts:
     "format:frontend": "pnpm --filter frontend format",
     "format:backend": "pnpm --dir packages/backend py:format",
     "format": "pnpm format:frontend && pnpm format:backend"

Repo 2: darizae/alpaca-pipelines

1. Python formatting:
   - Keep Ruff as the single source for formatting.
   - Ensure pyproject.toml has:
     [tool.ruff]
     line-length = 100
     target-version = "py311"
     ignore = ["E501"]

     [tool.ruff.lint]
     select = ["E", "F", "I", "W"]

   - Add a dev command/script that runs:
     ruff format .
     ruff check . --fix

   - Ensure the agent/check flow runs that auto-fix command before mypy/pytest.

2. Do not use mypy for formatting.
   - mypy should only run after Ruff has formatted/fixed code.

Acceptance criteria:
- A single command in each repo auto-fixes formatting:
  - alpaca-ui: pnpm format
  - alpaca-pipelines: ruff format . && ruff check . --fix

- CI/dev agents must run formatting/fixing before typecheck/test.
- E501/max-len must not fail checks after formatter runs.
- Long strings, URLs, regexes, SQL, and generated literals may remain long; they should not fail lint.
- Do not add a separate Python formatter like Black. Use Ruff only.
- Do not add ESLint max-len enforcement. Use Prettier only for TS/TSX wrapping.

# Contributing

## Before you start

Read `docs/order-exception-agent-architecture.md` first — it's the spec this
project is built against, and most design decisions trace back to a specific
section of it. If a change conflicts with something there, either the
architecture doc needs updating first, or the change needs rethinking.

## The standard this codebase holds itself to

Every fix in this project's history follows the same discipline, and PRs are
expected to match it:

1. **Verify the actual behavior before fixing it.** Reproduce the bug, read the
   real error, don't guess from the symptom.
2. **Fix the root cause, not the symptom.** If a workaround is genuinely the
   right call (e.g., a real third-party API limitation), say so explicitly in
   a comment, with the evidence.
3. **Write a test that would have caught it.** Not just "does it run" — a test
   that fails on the old code and passes on the fix.
4. **Run the full suite, not just the new test.** `python3 -m pytest tests/ -q`
   before considering anything done. A change that breaks something else
   isn't finished.
5. **Verify against real infrastructure where it matters.** This project tests
   against real Stripe, real Shippo, real Postgres, real Groq — not just
   mocks — wherever the actual bug risk lives in the integration itself, not
   the business logic around it.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 scripts/setup_auth_postgres.py   # see .env.example's Authentication section first
uvicorn app.main:app --reload
```

## Running tests

```bash
python3 -m pytest tests/ -q                              # full suite
python3 -m pytest tests/ --cov=app --cov-report=term-missing   # with coverage
python3 -m pytest tests/test_phase15_golden_set.py -v     # pre-deployment gate
```

Tests use a real local Postgres for auth (`AUTH_DATABASE_URL`) and a real
local Qdrant/embedded fallback for RAG — most things work with zero cloud
credentials configured, falling back to local-first defaults automatically
(see `requirements.txt`'s comments for the full swap-in matrix).

## Commit messages

Explain *why*, not just *what* — especially for bug fixes. "Fixed refund bug"
tells a future reader nothing; "refunding a never-charged payment is
nonsensical; Stripe's real Refund API confirms this" tells them everything
they need to not reintroduce it.

## Pull requests

- One logical change per PR — a bug fix and an unrelated refactor should be
  two PRs, not one.
- Include the test suite result in the PR description (pass count), not just
  "tests pass."
- If the change touches authentication, guardrails (Tier 1/2/3), or anything
  that executes a real refund/reship, call that out explicitly in the PR
  description — these are the highest-consequence code paths in the system.

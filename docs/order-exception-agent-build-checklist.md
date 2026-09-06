# Autonomous Omnichannel Order Exception & Fulfillment Resolution Agent
## Build Checklist — Phased Implementation Plan

*Companion to `order-exception-agent-architecture.md`. Each phase references the architecture doc section it implements. Work top to bottom — later phases assume earlier ones are done. Each phase has a Definition of Done (DoD); don't move on until it's met.*

---

## Phase 0 — Environment & Foundations
**Maps to:** 8.1, 8.10

- [ ] Create repo with the service layout from 8.1 (`/app/api`, `/agents`, `/tools`, `/rag`, `/memory`, `/guardrails`, `/cache`, `/workers`)
- [ ] Docker Compose with: Postgres (case state + audit log), Redis (queue + cache), Qdrant (vector store) — all free self-hosted
- [ ] Set up accounts and get free-tier API keys: Groq, Mistral AI (La Plateforme), Google AI Studio (Gemini), Cohere trial — store in `.env`, never commit
- [ ] Self-host BGE-M3 and BGE-Reranker as a small local inference service (or free-tier serverless container) — confirm both respond to a test embedding/rerank call
- [ ] Stand up FastAPI skeleton (`uvicorn app.main:app`) with a health-check endpoint returning 200
- [ ] Confirm Langfuse (self-hosted or free cloud tier) is reachable and receiving a test trace

**DoD:** `docker compose up` brings up all services; a test script embeds one sentence via BGE-M3, calls Groq with "hello," and both a Postgres row and a Langfuse trace are visible.

---

## Phase 1 — Architecture Decision Sheet & Cost Canvas
**Maps to:** Part 0.35, Layer 14

- [ ] Answer every question in Part 0.35 in writing (even placeholder answers for a portfolio project — e.g., "business owner: n/a, portfolio project, self-owned")
- [ ] Fill the W×A×U+P+F cost canvas with *estimated* portfolio-scale numbers (e.g., 50 synthetic exceptions/day) — this sizes your free-tier rate-limit budget, not just cost
- [ ] Decide and document the auto-execute threshold τ and $ ceiling starting values (conservative defaults — e.g., τ=0.9, ceiling=$50) — these are business decisions, write down *why* you picked them even if you're both engineer and business owner here

**DoD:** A one-page Architecture Decision Sheet exists in the repo (`/docs/architecture-decision.md`) — this is also a strong artifact to show in interviews.

---

## Phase 2 — Data & Policy Corpus Preparation
**Maps to:** 8.2.1, "Data for a portfolio project" section

- [ ] Collect 3-5 real, publicly available payer/retailer-style return/refund policy PDFs (or author 3-5 synthetic ones with mixed text + at least one table + one chart, to genuinely exercise the multimodal extraction pipeline)
- [ ] Author at least 2 *versions* of one policy (e.g., "6-month window" superseded by "4-month window") with explicit effective dates — this is required to test the temporal-correctness mechanism (8.2.3) end to end
- [ ] Generate synthetic order/customer data (simple script or a tool like Faker) including: normal orders, an order bound to the old policy version, a multi-payment-method order, a split-shipment order
- [ ] Set up Stripe test mode account; note the test card numbers you'll use for success/decline/insufficient-funds scenarios
- [ ] Set up EasyPost or Shippo sandbox account; confirm you can generate a test label and a test tracking number

**DoD:** A `/data` folder with policy PDFs (versioned), a synthetic orders dataset, and confirmed working Stripe test-mode + carrier sandbox credentials.

---

## Phase 3 — RAG Ingestion Pipeline
**Maps to:** 8.2 (all subsections)

- [ ] Install `unstructured.io` (`hi_res` strategy dependencies) and run it against one policy PDF; manually inspect output — confirm text, table, and image/chart elements are all separately identified
- [ ] Write the caption step: for each extracted image/chart element, call a vision-capable free-tier model (Gemini 2.0 Flash supports vision) to generate a searchable caption
- [ ] Build the LlamaIndex node construction step: hierarchical parent-child for text, atomic nodes for tables, ImageNode with caption for images/charts — per 8.2.1/8.2.2
- [ ] Implement the metadata schema exactly as specified in 8.2.3, including `content_hash`, `effective_start/end`, `superseded_by`
- [ ] Load nodes into Qdrant with hybrid (dense+sparse) vectors enabled
- [ ] Implement the incremental reindex job (8.2.5): hash-diff against last-indexed state, only re-embed changed elements, mark superseded versions rather than deleting
- [ ] Implement hybrid retrieval: dense+sparse+RRF, metadata pre-filter (by effective date range + channel + category), selective MMR for broad queries, BGE-Reranker on top-20→top-5

**DoD:** Given a test order bound to the *old* policy version, a query for "what's the return window for this order" retrieves and correctly cites the **old** 6-month policy, not the current 4-month one — this is your core acceptance test for the whole RAG layer, not a generic "does retrieval work" check.

---

## Phase 4 — Tool Layer (MCP)
**Maps to:** Part 6 (MCP), 8.1 tool wrappers

- [ ] Build/mock OMS tool (order lookup, status update) as an MCP server — mock with a simple FastAPI+Postgres service if no real OMS is available
- [ ] Build/mock WMS tool (stock lookup by "sellable" vs "on-hand," transfer request) as an MCP server
- [ ] Wrap Stripe test mode (transaction status, refund issuance) as an MCP server
- [ ] Wrap EasyPost/Shippo sandbox (label generation, tracking status) as an MCP server
- [ ] Wrap a notification service (email/SMS — can be a stub that logs instead of sending, for portfolio purposes) as an MCP server
- [ ] Confirm every write-capable tool (refund, transfer, label) accepts and correctly deduplicates on an idempotency key (Layer 5) — write a test that calls the same write twice with the same key and confirms only one action occurs

**DoD:** Every tool is callable independently via its MCP interface with a passing unit test (Layer 16) using mocked or sandbox responses; the idempotency test passes.

---

## Phase 5 — Memory Layer
**Maps to:** 8.4

- [ ] Implement the per-case Summary Buffer (simple: recent N tool outputs verbatim + an LLM-generated summary of older context), stored on the case row in Postgres
- [ ] Stand up Zep/Graphiti (self-hosted or free tier) for long-term/episodic/procedural memory
- [ ] Write the adapter that logs each resolved case as an episode in Graphiti, linked to the customer entity
- [ ] Confirm a query like "has this customer had prior return issues" returns a correct, time-ordered answer from Graphiti

**DoD:** A test customer with 3 synthetic past cases returns all 3, correctly time-ordered, from a single memory query.

---

## Phase 6 — Orchestrator + Diagnosis/Context Sub-agents
**Maps to:** Part 3 (topology), 8.1 `/agents`

- [ ] Build the Orchestrator as a LangGraph graph implementing the case state machine (`detected→diagnosing→decided→executing→verifying→resolved`), persisted to Postgres so it survives a process restart mid-case
- [ ] Build the Diagnosis Agent as an iterative Planner-Worker loop (per the guide's explicit warning: plan-one-step→execute→re-evaluate, never a blind upfront plan) with parallel tool fan-out (OMS+WMS+payment+carrier reads concurrently)
- [ ] Build the Fraud/Risk Agent (workflow, not open agent loop) producing a structured risk score + reason
- [ ] Build the Inventory Agent (workflow) resolving sellable-stock conflicts across warehouses/channels
- [ ] Build the Customer Context Agent (workflow) pulling LTV/tier/history from Graphiti
- [ ] Set a hard step ceiling and wall-clock timeout on the Diagnosis Agent's loop (Layer/Part 1.5.3) — write a test that forces a non-converging scenario and confirms the loop terminates instead of running away

**DoD:** Given a synthetic multi-cause exception (payment decline + concurrent OOS), the Diagnosis Agent correctly identifies both causes in its output, and the runaway-loop test terminates within the configured ceiling.

---

## Phase 7 — Resolution-Policy Workflow + Hybrid Guardrails
**Maps to:** Part 1 (autonomy calibration), 8.6

- [ ] Implement the Resolution-Policy Workflow as a deterministic workflow (not an agent loop) that takes Diagnosis + Fraud/Risk + Inventory + Customer Context outputs and the RAG-retrieved policy as input
- [ ] Implement Tier 1 guardrails (plain Python, hard $ and quantity ceilings) — write a test confirming the LLM's reasoning *cannot* override these regardless of prompt content
- [ ] Implement Tier 2 guardrails (Guardrails AI schema validation on the decision object + LLM Guard PII scan) — confirm a malformed decision object is rejected before it can reach the Execution Agent
- [ ] Implement Tier 3 (async LLM-judge sampling on resolution quality/citation accuracy) as a background worker job, non-blocking
- [ ] Wire the confidence-threshold check: output routes to Auto-Execute or Human-Escalation based on τ and $ ceiling from Phase 1's decisions

**DoD:** Three test cases — one clearly auto-executable, one clearly requiring escalation (fraud flag present), one that Tier 1 blocks outright regardless of model output — all route correctly.

---

## Phase 8 — Execution, Verification, Comms & Reliability Layer
**Maps to:** Layer 5, Part 3 topology

- [ ] Implement the Execution Agent (workflow) calling the appropriate MCP write tool with an idempotency key derived from case ID + action type
- [ ] Implement circuit breakers on each external dependency (Stripe, carrier, WMS) per Layer 5's example pattern
- [ ] Implement the Verification Agent — confirms the action actually completed (poll or webhook-driven) before marking the case resolved, never assumes success from the write call's 200 response alone
- [ ] Implement the Customer-Comms workflow (templated + LLM-personalized), triggered on major state transitions
- [ ] Chaos-test (Toxiproxy or manual fault injection): kill the payment gateway mid-refund-call, confirm retry+circuit-breaker fires and no duplicate refund occurs

**DoD:** The chaos test passes — a simulated payment-gateway timeout results in exactly one refund attempt being recorded as "pending retry," not a duplicate charge/refund, and the circuit breaker trips after the configured threshold.

---

## Phase 9 — Learning Loop
**Maps to:** 8.5

- [ ] Implement the Resolution Pattern Store (a table/index of past case features + human decisions)
- [ ] Implement dynamic few-shot retrieval: Resolution-Policy Workflow pulls k similar past cases from this store as in-context examples
- [ ] Implement the weekly batch job that computes per-cluster accuracy and proposes threshold adjustments
- [ ] Wire the proposal to surface on the Threshold & Policy Config frontend page (Phase 13) rather than auto-applying
- [ ] Test: manually seed 10 human decisions in one similarity cluster with 0% overturn, confirm the batch job proposes raising τ for that cluster (and does *not* apply it automatically)

**DoD:** The proposal-not-auto-apply behavior is verified by test — this is the single guardrail most likely to be accidentally skipped under time pressure, so verify it explicitly.

---

## Phase 10 — Observability & Metrics
**Maps to:** 8.7, 8.8

- [ ] Instrument every agent/tool call with the Langfuse span schema from 8.8 (input/output/metadata, PII-redacted before persist)
- [ ] Build the `/metrics/{scope}` endpoints (agent/tool/rag/system) aggregating from Langfuse + Postgres
- [ ] Confirm the RAG groundedness metric is computable from trace data (i.e., you can programmatically check whether the cited policy doc_id actually supports the resolution text)
- [ ] Set up basic alerting (even just a Slack webhook) for: circuit breaker trips, idempotency collisions, guardrail Tier-1 blocks

**DoD:** Pulling `/metrics/rag` after running the temporal-correctness test case (Phase 3's DoD) shows a groundedness score and the correct cited `doc_id`/version in the trace.

---

## Phase 11 — Caching Layer
**Maps to:** 8.9

- [ ] Implement embedding cache (permanent, keyed on content_hash)
- [ ] Implement retrieval cache (5-15 min TTL)
- [ ] Implement tool/API response cache (30-60s TTL) for inventory/carrier reads
- [ ] Implement event-driven invalidation: webhook handlers actively invalidate the relevant cache key, don't just wait out the TTL
- [ ] Test: simulate a stock-update webhook arriving mid-diagnosis, confirm the next inventory read reflects the update, not the cached value

**DoD:** The webhook-invalidation test passes — this directly proves the phantom-stock/marketplace-lag edge cases are actually handled, not just documented.

---

## Phase 12 — API Surface Completion
**Maps to:** 8.1

- [ ] Implement all endpoints from 8.1's table
- [ ] Implement the async job queue (Redis+RQ or Celery) for webhook processing, reindexing, and the Tier 3 guardrail job
- [ ] Confirm webhook handlers ack immediately and enqueue, never process synchronously inline
- [ ] Generate and review the OpenAPI schema — this doubles as your frontend's API contract

**DoD:** OpenAPI docs (`/docs`) render correctly and every endpoint from 8.1 is present and callable.

---

## Phase 13 — Frontend
**Maps to:** 8.3

- [ ] Build pages in this order (dependency-driven, not just importance-driven): Case Detail View → Case Queue/Dashboard → Escalation Queue → Metrics Dashboard → Policy Document Manager → Trace Viewer → Threshold Config → Audit Log Viewer
- [ ] Wire Escalation Queue's approve/edit/reject actions to Phase 9's Resolution Pattern Store write
- [ ] Wire Threshold Config to display Phase 9's proposed-but-unapplied calibration changes with an explicit accept action

**DoD:** A full manual walkthrough — trigger a synthetic exception via webhook, watch it appear in the queue, diagnose, escalate (force low confidence), approve from the UI, see it resolve and appear correctly in the audit log — works end to end through the UI alone.

---

## Phase 14 — Testing & QA
**Maps to:** Layer 16

- [ ] Unit tests on every tool function (Phase 4) — confirm this is actually done, not just planned
- [ ] Contract tests between agents (e.g., Diagnosis Agent's output schema matches what Resolution-Policy Workflow expects)
- [ ] Mocked-tool-response tests: each tool returning malformed/empty/error responses — confirm graceful handling, not a crash
- [ ] Load test (Locust/k6) at your Phase 1 estimated volume — confirm circuit breakers/rate limits hold under it
- [ ] Full chaos test pass (builds on Phase 8's single test) across all external dependencies, not just payment

**DoD:** CI pipeline runs all of the above on every commit; document the coverage % achieved.

---

## Phase 15 — Evaluation
**Maps to:** Layer 10, Part 4

- [ ] Build a golden set from the edge-case inventory (Part 8.11 of the architecture doc) — turn the highest-value 8-10 edge cases into labeled test scenarios with known-correct resolutions
- [ ] Run the golden set 3x unchanged to establish your variance band
- [ ] Wire golden-set evaluation into CI (DeepEval) as a pre-deployment gate
- [ ] Set up the weekly drift watch job comparing live-traffic sampled scores against the variance band

**DoD:** The golden set includes the temporal-policy case, the duplicate-refund idempotency case, and the fraud-vs-high-LTV-customer case at minimum — these are your three highest-signal interview demo scenarios.

---

## Phase 16 — Staged Rollout & Production-Readiness Scorecard
**Maps to:** Part 2, Layer 12

- [ ] Score the system against the Part 2 scorecard (Architecture/Observability/Security/Compliance/Operations, 20pts each) — target 90+ per the architecture doc's elevated bar for this project
- [ ] Verify rollback: disable auto-execution via feature flag, confirm it takes effect in under 5 minutes and all new cases route to human queue
- [ ] "Deploy" through the staged sequence even in a portfolio context — Internal → Beta (your own test traffic) → wider synthetic load — documenting what you *would* monitor at each stage in production

**DoD:** A completed, scored Production-Readiness Scorecard checked into `/docs` — a strong standalone artifact for interviews.

---

## Phase 17 — Post-Launch Operating Rhythm (document, even if simulated)
**Maps to:** Layer 15, Part 4.3

- [ ] Write the weekly drift-watch runbook
- [ ] Write the monthly cost-canvas review process
- [ ] Document the feedback loop: how a bad-outcome case gets added back into the golden eval set

**DoD:** A one-page ops runbook exists — this is what shows an interviewer you understand a system doesn't end at deployment.

---

## Suggested Build Order Summary

```
Phase 0 → 1 → 2  (foundations, decisions, data — do these fully before any code)
Phase 3 → 4 → 5   (RAG, tools, memory — the building blocks, parallelizable across these three)
Phase 6 → 7 → 8   (orchestrator and agents — sequential, each depends on the last)
Phase 9            (learning loop — needs Phase 7/8 producing real decisions first)
Phase 10 → 11      (observability + caching — wire in alongside Phase 6-8, not strictly after)
Phase 12 → 13      (API + frontend — needs Phase 6-9 functionally complete)
Phase 14 → 15      (testing + eval — continuous alongside everything, formalized here)
Phase 16 → 17      (rollout + ops — final)
```

**Fastest path to a demoable slice** (if you want something working before the full 17 phases): Phase 0, 2, 3 (RAG with the temporal-policy test case), Phase 4's Stripe/carrier tools, Phase 6's Diagnosis Agent, Phase 7's Resolution-Policy Workflow with Tier 1 guardrails only, Phase 8's Execution Agent. That's a working, demoable core that proves the hardest architectural claims (temporal RAG correctness, idempotent execution, tiered autonomy) without the full frontend/learning-loop/observability buildout.

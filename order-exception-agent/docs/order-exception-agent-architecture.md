# Autonomous Omnichannel Order Exception & Fulfillment Resolution Agent
## Enterprise Architecture & Framework Selection

*Built against `enterprise-agentic-ai-architecture-guide.md` (Production Edition) and `framework-directory.md`. This is architecture only — no code yet.*

---

## PART 0 — Pre-Design Answers (mandatory questions, condensed)

| # | Question | Answer |
|---|---|---|
| Business outcome | What does success look like, in business terms? | Reduce order-exception resolution time from days to minutes/hours; reduce manual CS headcount-hours per exception; reduce revenue leakage from over-generous manual refunds and under-detected fraud |
| End user | Who benefits, what happens today without this? | CS ops team + customers; today a human agent manually checks 3-4 systems per ticket, often over multiple shifts, with inconsistent policy application |
| Out of scope (v1) | What will this deliberately not do? | No autonomous handling of legal/chargeback disputes past a value threshold; no autonomous fraud *account bans* (recommend only); no cross-border customs/duty exceptions in v1 |
| Ownership | Named engineering + business owner | Engineering: Agent Platform lead. Business: Head of CS Ops / Fulfillment Ops (owns the 3 business-impact metrics in Part 4.3 of the guide) |
| Blast radius | Confidently-wrong cost | Financial (duplicate/incorrect refund), inventory desync, customer trust damage, fraud-loss if a serial abuser is auto-approved — **medium-high**, not catastrophic, but real money at volume |
| Regulated? | By what | PCI-DSS (payment data path), regional consumer-protection law (return/refund rights), GDPR/CCPA-class data law for customer PII |
| Data access | Who's authorized today, does the agent mirror it? | Agent gets service-account, least-privilege, read-heavy scopes to OMS/WMS/carrier/payment; write scope limited to the specific mutating actions it needs (refund, label, transfer) — never full account access |
| Knowledge freshness | How often does it change | Refund/return policy: infrequent (weeks). Inventory state: real-time. Fraud signals: real-time. Carrier status: real-time (polled/webhook) |
| Latency budget | Real requirement, not a guess | Diagnosis+decision: target <60s (not customer-facing real-time chat, so this is generous); execution confirmation: within the external API's own SLA (refund post, label generation) |
| Cost ceiling | Per request / month | Establish via Layer 14 canvas below before committing infra — **not yet fixed**, needs real exception-volume data |
| Simplest-rung check | Why not a workflow alone? | See Part 0.2 Agent Justification Test breakdown below — **mixed**: diagnosis is agentic, execution is a workflow |
| Failure modes (≥3) | Named | (1) Duplicate refund from retried API call, (2) inventory transfer triggered on stale stock data, (3) fraud-agent false-positive denies a legitimate customer, (4) appeal/resolution letter cites wrong policy clause |
| Rollback plan | How, how long | Feature-flagged per-resolution-type; disable auto-execution instantly (falls back to 100% human queue) in <5 min; state machine allows manual case reassignment |

---

## PART 1 — Layer 1: Use-Case & Autonomy Calibration (per sub-workflow, not as one blob)

The guide's Retail/E-commerce domain lens is explicit: **"Agent loop acceptable for product discovery/recommendations; workflow for checkout/payment/inventory writes — customer-facing discovery tolerates imperfection; money and inventory do not."** This project sits squarely in the money+inventory half, so autonomy is calibrated **per sub-task**, not as one uniform rung:

| Sub-workflow | Autonomy rung | Failure cost | Complexity | Rung selected |
|---|---|---|---|---|
| **Diagnosis** (root-cause across OMS/WMS/payment/carrier) | Suggests → agent decides what to check next | Wasted time if wrong | Multi-step, open-ended (next check depends on last result) | **Agent loop** — this is the one part of the system that fails the Agent Justification Test's "yes, draw the full flowchart" bar; root cause is genuinely unknown upfront |
| **Fraud/risk scoring** | Suggests, never auto-denies | Wasted trust if wrong / fraud loss if missed | Bounded (scoring against known signals) | **Workflow with confidence output**, feeding the decision step — not a freestanding agent |
| **Resolution decision** (refund/reship/credit/deny) | Tiered — auto below threshold, human above | Financial, at volume | Bounded but judgment-weighted | **Workflow with explicit policy thresholds**, agent-assisted reasoning only within the band, never above it |
| **Execution** (refund API call, label generation, inventory transfer) | Executes reversible-where-possible actions | Financial, duplicate-action risk | Single/few tool calls, deterministic once decided | **Workflow, not agent loop** — once decided, there is nothing left to "decide," only to execute reliably (idempotency-critical, per Layer 5) |
| **Customer communication** | Executes | Low (a wrong-toned message is annoying, not harmful) | Bounded | **Workflow** with templated + LLM-personalized copy |

**This is the single most important architecture decision in this project**, and it's a deliberate, justified deviation from treating "agentic e-commerce project" as one big agent loop: only the diagnosis phase earns agent-loop status. Everything downstream of the decision is a deterministic, auditable workflow — which is exactly the discipline the guide's Layer 1 exists to enforce, and exactly the kind of reasoning that reads well in an interview ("I didn't agent-ify the whole thing by default — I applied the justification test per component").

---

## PART 2 — Design Patterns Used (from the 21-pattern catalog)

| Pattern | Where it's used |
|---|---|
| **Routing** | Orchestrator classifies incoming exception type (payment/inventory/carrier/return/fraud-flagged) and routes to the right diagnosis path |
| **Planning (iterative Planner-Worker)** | Diagnosis Agent: plan next check → execute → re-evaluate with new info → repeat, never a blind upfront plan (per the guide's explicit warning against that anti-pattern) |
| **Parallelization** | Diagnosis Agent fans out independent read calls (OMS + WMS + payment + carrier status) concurrently, then aggregates |
| **Reflection / Self-critique** | Appeal/resolution-reasoning step re-checks its own recommendation against policy before handing to the decision workflow, catching unsupported claims before they reach a human or execute |
| **Human-in-the-Loop (Orchestrator-Worker with approval gate)** | Mandatory gate before any execution above the confidence/value threshold |
| **Exception Handling / Guardrails pattern** | Hard-coded policy ceilings (Layer 9) wrap every resolution regardless of what the LLM concludes |
| **Multi-Agent Coordination (Supervisor + specialists)** | Orchestrator supervises Diagnosis, Fraud/Risk, Resolution-Policy, Execution, Verification, Comms agents |

---

## PART 3 — Multi-Agent Topology & Architecture

**Topology chosen:** Hierarchical Supervisor + Subagents (per the guide's topology table) — justified because there's a genuine large-task decomposition across heterogeneous systems, not because it "looks more agentic." A single-agent-many-tools design was rejected: this crosses 4+ external systems with materially different failure modes and write-risk profiles, which the guide flags as exactly when tool-selection reliability degrades in a flat single-agent design.

```
                         EXCEPTION TRIGGER
        (webhook: payment fail / OOS / delivery exception /
                 return request / manual CS flag)
                              │
                              ▼
                  ┌───────────────────────┐
                  │   Orchestrator Agent    │  ← case state machine (Postgres)
                  │  detected→diagnosing→   │     durable across days
                  │  decided→executing→     │
                  │  verifying→resolved     │
                  └──┬─────┬─────┬─────┬────┘
                     │     │     │     │
        ┌────────────┘     │     │     └────────────┐
        ▼                  ▼     ▼                   ▼
  ┌───────────┐     ┌───────────┐ ┌───────────┐ ┌──────────────┐
  │ Diagnosis  │     │Fraud/Risk │ │ Inventory │ │  Customer     │
  │  Agent     │     │  Agent    │ │  Agent    │ │  Context Agent│
  │(agent loop,│     │(workflow, │ │(workflow) │ │  (workflow)   │
  │ parallel   │     │ scores)   │ │           │ │               │
  │ tool fan-  │     └───────────┘ └───────────┘ └──────────────┘
  │ out)       │
  └─────┬──────┘
        │            all outputs converge
        ▼
  ┌─────────────────────────┐
  │  Resolution-Policy       │  ← workflow, explicit thresholds
  │  Workflow                │     (not free-reasoning agent)
  └──────┬────────────┬─────┘
         │             │
   confidence ≥ τ  confidence < τ
   & value < $cap  OR value ≥ $cap
   & no fraud flag  OR fraud flag
         │             │
         ▼             ▼
  ┌─────────────┐ ┌──────────────────┐
  │ Auto-Execute│ │ Human-Escalation  │
  │  Workflow   │ │ Queue (dashboard, │
  │(idempotent, │ │ pre-drafted       │
  │ MCP tool    │ │ resolution)       │
  │ calls)      │ └────────┬──────────┘
  └──────┬──────┘          │ (approve/edit/reject)
         │◄─────────────────┘
         ▼
  ┌─────────────────┐
  │ Verification      │  ← confirms action actually completed
  │  Agent (workflow)  │     (refund posted / label scanned / transfer received)
  └────────┬──────────┘
           ▼
  ┌─────────────────┐        ┌──────────────────┐
  │ Customer-Comms    │──────►│  Audit/Reporting  │
  │  Workflow          │      │  (every layer      │
  └───────────────────┘       │  logs to this)     │
                               └──────────────────┘
```

---

## PART 4 — Layer-by-Layer Stack Selection

### Layer 1 — Use-Case & Architecture Decision
Covered in Part 1 above. **Mixed autonomy by sub-workflow is the headline decision.**

### Layer 2 — Model Selection
| Tier | Use | Model class |
|---|---|---|
| Router/classifier | Exception-type classification, denial/fraud-reason structuring | Small/fast tier |
| Reasoning/planning | Diagnosis Agent's iterative planning, Resolution-Policy reasoning within its band | Frontier reasoning tier |
| Generation | Customer-comms drafting, escalation-packet summaries | Balanced tier |

**Deployment mode:** guide's domain lens says "API-hosted direct often fine" for Retail/E-commerce given lower regulatory burden than Healthcare/Finance — **but** because this system touches PCI-DSS-adjacent payment metadata (not raw card data — that stays inside Stripe, never touches the LLM context) and PII, use **cloud-hosted private** (Bedrock/Vertex/Azure AI Claude access) rather than pure API-direct, as a deliberate one-notch-up-from-default choice given the payment/PII surface. This is cheap insurance, not overkill.

### Layer 3 — AI Gateway
Guide's default for Retail/E-commerce: **Portkey or LiteLLM** ("speed and cost-optimization matter more than deep governance at this risk level"). **Selected: LiteLLM (self-hosted)** — over Portkey — because this system does have real write-risk (refunds) even if lower than Healthcare/Finance, and self-hosting keeps gateway-level logs (which will contain order/customer identifiers) inside our own infra rather than a third-party managed plane. Portkey remains a reasonable alternative if the team doesn't want to own gateway ops.

### Layer 4 — Orchestration Framework
Guide's default for Retail/E-commerce: **CrewAI or LlamaIndex Workflows** ("speed to market... outweighs maximum control needs"). **Deliberately overridden: LangGraph.** Reasoning, explicit per the guide's own "Critical structural point" (none of the frameworks natively provide governance — build it on top) and its Financial-Services-style reasoning ("durable state, auditable branching for compliance-reviewable workflows"): this system executes real refunds and inventory transfers, needs durable state across multi-day waits (carrier pickup, return receipt), and needs explicit, auditable HITL gates — the exact profile LangGraph is built for, not the fast-prototype profile CrewAI optimizes for. **This override, and the reasoning behind it, is worth stating explicitly in interviews** — it shows the domain-lens table is a default, not a rule, and that the override was earned by naming the specific scenario (financial write + multi-day durable state) rather than a vague "wanted more control."

### Layer 5 — Orchestration Reliability Layer (non-negotiable given the guide's documented $84K duplicate-refund incident pattern)
- **Idempotency keys on every write**: refund calls, label generation, inventory transfer — keyed on case ID + action type, so a retried call never double-executes
- **Circuit breakers** on each external dependency (Stripe, carrier API, WMS) — fail fast to cached/last-known-state and escalate rather than hammer a failing dependency
- **Exponential backoff + jitter**, max-attempt ceiling, aggressive 2–5s timeouts on read calls
- This layer is scoped as a **release blocker** for the refund/inventory-transfer paths specifically (per the guide's domain-lens note that Retail can defer full idempotency coverage on low-risk actions but must prioritize it hard on payment/inventory actions)

### Layer 6 — Tools / Function Calling
| Tool | Access | Approval gate |
|---|---|---|
| OMS order lookup, WMS stock lookup, carrier tracking lookup | Read | None — free |
| Payment transaction status lookup | Read | None |
| Customer/order history lookup | Read | None |
| Refund issuance (Stripe) | Write | **Gated** — auto below $ threshold + confidence + no fraud flag; human above |
| Inventory transfer request (WMS) | Write | **Gated** — auto below threshold |
| Carrier label/pickup generation | Write | Auto (low blast-radius, easily reversed by not shipping) |
| Fraud-flag / account-note write | Write | Always **recommend-only**, human executes (per Part 0 out-of-scope) |

Per the guide's Retail/E-commerce tool-access default ("auto-approve customer-facing writes below a refund/discount threshold — high volume makes full HITL impractical below a risk threshold"): confirmed as the right default here, with fraud-flag actions kept out of auto-approval as an explicit tightening beyond the generic default, given reputational/legal sensitivity of falsely flagging a customer.

### Layer 7 — Knowledge Retrieval (RAG)
Used for: refund/return policy documents, brand-specific resolution guardrails, payer-agnostic (not applicable here — that's the healthcare project) policy lookups for edge-case resolution reasoning.
- **Chunking:** Semantic chunking (guide's Retail default)
- **Search:** Fast approximate (HNSW), lighter reranking — latency matters more than perfect precision for this internal-policy lookup, matching the guide's Retail lens
- **Embedding model:** OpenAI text-embedding-3-small (cheapest solid default, no need for the higher ceiling this use case doesn't demand)

### Layer 8 — Memory
- **Short-term (per-case):** Summary Buffer — strongest general default per the guide, needed because a case can run over days with many tool calls
- **Long-term:** **Episodic Memory** (case history per customer — informs the Customer Context Agent's "is this a serial claimant" read) + light **Procedural Memory** (reuse successful resolution strategies for recurring exception patterns, e.g., "this SKU always OOS-conflicts across marketplace X") — added deliberately, not by default, per the guide's Layer 8 first principle that most agents don't need long-term memory
- **Not used:** Knowledge Graph Memory — the guide reserves this for domains with many interrelated entities (e.g., Legal); this use case's entity relationships are simple enough that it would add build cost without a matching payoff

### Layer 9 — Guardrails & Safety
- **Hard policy ceilings** (deterministic, not prompted): max auto-refund $ value, max auto-inventory-transfer quantity, mandatory human review on any fraud-flagged case regardless of confidence score
- **Framework: NeMo Guardrails or Guardrails AI** for structured-output enforcement on the Resolution-Policy Workflow's decision schema (guide's pick for "primary failure mode is malformed structured output," which is exactly the risk on a decision object that triggers a real API call)
- System prompts versioned, canary-tested, rollback-by-pointer-swap per the guide's Layer 9 discipline — non-negotiable given this drives financial actions
- **Guardrail-added latency budget:** keep synchronous checks (policy-ceiling validation) cheap/fast; push fraud-model scoring async where possible without blocking the diagnosis phase

### Layer 10 — Evaluation
- **Pre-deployment:** golden set of labeled historical exceptions (real anonymized tickets if available, else synthetic), CI-gated via DeepEval — covers both code-behavior and output-quality assertions in one suite
- **Post-deployment:** weekly drift watch against the variance band; retrieval drift is the most likely failure mode here (policy corpus goes stale when brand policy updates and nobody re-indexes)
- **Retail-domain emphasis (per guide):** task completion rate, latency, cost per interaction weighted highest — but with fraud-false-positive rate and duplicate-action rate tracked as hard release gates on top of the standard Retail set, given this system's write-risk is higher than typical Retail product-discovery agents

### Layer 11 — Observability
**Langfuse** (open-source, no lock-in, full tracing) — chosen over LangSmith despite the LangGraph pick, because full audit reconstructability of *why a refund happened* is a business/compliance need here even though Retail isn't as tracing-strict as Healthcare/Finance by the guide's domain lens; Langfuse keeps that trace data in our own infra. Full distributed tracing from day one, cost-per-request attribution by case type, tiered alerting (duplicate-action or circuit-breaker trips → PagerDuty; degraded confidence-score trend → Slack).

### Layer 12 — Infrastructure & Deployment
Staged rollout per the guide's standard cadence: Internal → Beta (opt-in CS team) → 10% of exception volume → 50% → 100%, with 72hr monitoring per stage. **Deployment target:** cloud-hosted managed runtime (AWS Bedrock AgentCore if AWS-native, or standard containerized deployment otherwise) rather than pure self-managed Kubernetes for a v1 — the guide reserves heavy self-managed K8s investment for teams with existing MLOps maturity, which a first enterprise agentic project typically doesn't have yet.

### Layer 13 — Security & Compliance
- **Compliance anchor:** PCI-DSS (payment metadata path — raw card data never enters agent context, stays inside Stripe), regional consumer-protection/data law
- Least-privilege service-account credentials per tool, secrets never in prompts/logs/version control
- Audit retention aligned to standard commercial retention policy (shorter than HIPAA's 6yr/finance's 7yr, but still versioned and complete)

### Layer 14 — Cost Management (canvas, to be filled with real volume data before committing infra spend)
`Monthly Cost ≈ W × A × U + P + F`
- **W (Workload):** exception volume/day (get real number from CS ops before sizing)
- **A (Amplification):** ~4-6 tool calls per diagnosis (parallel fan-out) + 1 resolution-reasoning call + 1 comms-generation call per case
- **U (Unit economics):** small-tier model for classification/structuring, reasoning-tier only for diagnosis planning and resolution reasoning within the confidence band
- **F (Failure tax):** model retry rate on tool timeouts + human-escalation cost (a CS agent's review time) — **budget this explicitly**, it's the line teams underestimate per the guide's warning

### Layer 15 — Org & Process
Named engineering owner (Agent Platform) + named business owner (CS Ops/Fulfillment) per Part 0; feedback loop from every human-escalation outcome back into the golden eval set, so cases humans had to correct systematically tighten future auto-execution confidence calibration.

### Layer 16 — Testing & QA for Agent Code
- Unit tests on every tool function (Stripe wrapper, WMS wrapper, carrier wrapper) with mocked dependencies
- **VCR.py-style recorded-response tests** against real Stripe test-mode and EasyPost/Shippo sandbox response shapes — not hand-written mocks, so tests catch real API-contract drift
- **Chaos testing (Toxiproxy)** specifically on the circuit-breaker/retry logic in Layer 5 — this is release-blocking per the guide's explicit warning that eval-only teams miss exactly this class of bug (the duplicate-refund incident pattern)
- Harness/state tests: idempotency-key collision behavior, concurrent-case-update safety

### Layer 17 — Caching
- **Prompt/context caching** (native Anthropic caching) on the static system prompts/tool definitions for every sub-agent — zero-risk win, do this by default
- **Tool/API response caching** with short TTL on read-heavy lookups (inventory level, carrier status) to cut redundant calls during a single diagnosis pass
- **No semantic response caching** — case-specific reasoning has low reuse value across cases, and a false-positive semantic cache hit on a refund decision is exactly the kind of risk this system should not take

---

## PART 5 — Guardrail & HITL Policy Detail (the core safety design)

| Trigger | Action |
|---|---|
| Resolution value < $X **and** confidence ≥ τ **and** no fraud flag | Auto-execute |
| Resolution value ≥ $X **or** confidence < τ **or** fraud flag present | Human escalation, pre-drafted recommendation shown |
| Circuit breaker trips on any external dependency | Halt auto-execution for affected action type, route to human queue, page on-call |
| Duplicate idempotency-key collision detected | Log, block re-execution, alert — never silently succeed twice |
| Fraud/risk agent flags serial-claimant pattern | **Always** human review, regardless of case value |

*(Exact $ threshold and confidence τ are business decisions for CS Ops to set and tune post-launch — not an engineering default; start conservative, widen the auto-execute band only as calibration data accumulates.)*

---

## PART 6 — Does It Need MCP?

Yes, and here it's a genuinely strong fit, not just a resume flourish: this system integrates 4+ heterogeneous external systems (OMS, WMS, payment gateway, carrier API, notification service) with different auth models and response shapes. MCP servers standardize each as a discoverable tool the orchestrator and sub-agents call uniformly, instead of bespoke per-system glue code — and it keeps the write-risk tools (refund, transfer, label) cleanly separated from read-only tools at the integration layer itself, reinforcing the Layer 6 approval-gate design rather than fighting it.

---

## PART 7 — Production-Readiness Target (Layer/Part 2 Scorecard)

| Category | Target before any auto-execution goes live beyond internal testing |
|---|---|
| Architecture (20) | Full — layered, retry/circuit-breaker logic built pre-launch, not retrofitted |
| Observability (20) | Full — tracing from day one is non-negotiable given audit needs on refund decisions |
| Security (20) | Full — least-privilege credentials, PII handling validated before Beta stage |
| Compliance (20) | Full — audit logging and HITL gates must be live before touching real refunds, even in Beta |
| Operations (20) | Full — rollback (<5 min) tested before 10% rollout, not assumed |

**Target: 90+/100 before crossing from Beta into the 10% rollout stage** — higher than the guide's general 70-point bar, because this is a portfolio project where demonstrating you *hit* the bar (not just know it exists) is the actual value.

---

## PART 8 — v2: Full Implementation Architecture

*Added in response to the edge-case inventory (Part 8.11) and a request to fully specify backend, RAG, frontend, memory, learning loop, hybrid guardrails, metrics, observability, and caching — free-tier-first, with every model checked for current availability. This is architecture/spec, not code — the "don't build now" instruction still holds; this is what gets built next.*

### 8.1 Backend — FastAPI

**Why FastAPI here specifically:** async-native (matters for the fan-out parallel reads in Diagnosis), native Pydantic request/response validation (which doubles as the Layer 9 structured-output contract at the API boundary, not just the LLM boundary), and OpenAPI schema generation that both the frontend and MCP tool wrappers can consume directly.

**Service layout:**
```
/app
  /api
    /v1
      cases.py          # case CRUD, state transitions
      escalations.py     # human-review queue endpoints
      webhooks.py         # OMS/payment/carrier inbound webhooks
      policies.py          # policy document upload/versioning
      metrics.py             # dashboard data endpoints
      traces.py                # observability query endpoints
  /agents               # orchestrator + subagents (LangGraph graphs)
  /tools                # MCP tool wrappers (thin, testable, per Layer 16)
  /rag                  # LlamaIndex ingestion + retrieval pipeline
  /memory               # short-term buffer + long-term store adapters
  /guardrails           # deterministic policy ceilings + guardrail framework hooks
  /cache                # cache-aside helpers, TTL config
  /workers              # background jobs: reindexing, async LLM-judge guardrail, drift watch
```

**Key endpoints (representative, not exhaustive):**

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/webhooks/oms`, `/webhooks/payment`, `/webhooks/carrier` | Exception triggers land here, validated + enqueued (never processed synchronously in the webhook handler itself — ack fast, process async) |
| `GET` | `/cases/{id}` | Full case state + trace reference |
| `POST` | `/cases/{id}/reopen` | Explicit reopen path (edge case 6.3) with audit-preserving semantics |
| `GET` | `/escalations?priority=` | Human queue, priority-sortable (edge case 9.1) |
| `POST` | `/escalations/{id}/decision` | Human approve/edit/reject — writes to the resolution-pattern store (8.5) on every decision |
| `POST` | `/policies/upload` | New policy PDF → triggers ingestion pipeline, never overwrites — always a new version |
| `GET` | `/metrics/{scope}` | `scope ∈ {agent, tool, rag, system}` — feeds the frontend dashboard (8.7) |
| `GET` | `/traces/{case_id}` | Full input/output/metadata trace for one case (8.8) |

**Async job queue:** background webhook processing, reindexing (8.2.5), and the async LLM-judge guardrail (8.6) all run through a lightweight free-tier-friendly queue — **Redis + RQ** or **Celery+Redis** (Redis has a generous free-tier via Upstash/Redis Cloud, or self-hosted Docker for zero cost in a portfolio deployment).

---

### 8.2 RAG Stack — LlamaIndex + Unstructured.io

**8.2.1 Ingestion pipeline (PDF with text, images, tables, charts)**

```
Policy PDF → Unstructured.io (hi_res strategy)
    │
    ├─ Text elements (NarrativeText, Title, ListItem)
    ├─ Table elements (structured, HTML-preserved via `infer_table_structure=True`)
    ├─ Image elements (extracted + saved, then captioned by a vision-capable LLM call
    │                   so the caption becomes searchable text tied to the image)
    └─ Chart/Figure elements (extracted as image + captioned; if the chart contains
                                extractable data, a follow-up structured-extraction pass
                                converts it to a small table node, not just a caption)
    │
    ▼
LlamaIndex node construction
    │
    ├─ Text → Hierarchical Parent-Child chunking (matches the guide's Layer 7 pick
    │          for "precision at search time + full context at generation time")
    ├─ Tables → kept as atomic nodes (never split mid-table), with a text summary
    │            node linked as a sibling for embedding-friendly retrieval
    └─ Images/Charts → ImageNode with caption text embedded for retrieval,
                         original image path preserved for citation/display
```

Unstructured.io's `hi_res` strategy is the correct pick over `fast` specifically because policy PDFs mix prose with tables/charts — `fast` strategy misses table structure, which would silently degrade exactly the kind of numeric policy detail (return windows, refund percentages by category) this system depends on getting right.

**8.2.2 Chunking strategy**
- **Semantic + hierarchical hybrid**: parent chunks at section/clause boundary (e.g., "Electronics Return Policy" as one parent), child chunks at paragraph level for precise retrieval, parent returned at generation time for full context — directly addresses the guide's Layer 7 hierarchical parent-child trade-off (more index complexity, but this domain needs the full-context payoff).
- Tables and images are **never chunked** — split content mid-table is exactly how a return-window number gets silently truncated.

**8.2.3 Metadata schema (critical — this is what makes temporal correctness and filtering work)**

```python
{
  "doc_id": "RET-POLICY-2025",
  "version": "v3",
  "effective_start": "2025-01-01",
  "effective_end": "2026-01-31",   # null if currently active
  "superseded_by": "RET-POLICY-2026-v1",  # null if current
  "product_category": ["apparel", "electronics", ...],  # or "all"
  "channel": ["direct", "amazon", "walmart"],           # policy can differ by channel (edge case 7.2)
  "doc_type": "return_policy" | "fraud_policy" | "refund_ceiling",
  "source_page": 4,
  "content_hash": "sha256:...",     # drives incremental reindexing (8.2.5)
  "element_type": "text" | "table" | "image" | "chart"
}
```

**8.2.4 Retrieval technique**
- **Hybrid search**: dense (embedding) + sparse (BM25) retrieval, combined via **Reciprocal Rank Fusion** — per the guide's Layer 7 formula, `RRF_Score(d) = Σ 1/(60 + rank_r(d))`.
- **Metadata filtering applied *before* similarity search, not after**: this is the mechanism that solves the return-policy-changed edge case — the query is filtered to `effective_start ≤ order.purchase_date ≤ effective_end` (or `effective_end IS NULL` for current), and only channel/category-matching docs, before any similarity ranking happens. This makes correctness structural, not dependent on the LLM "noticing" the right document among mixed results.
- **MMR (Maximal Marginal Relevance)**: used selectively — **not** for policy-lookup queries (those need the single correct, date-filtered document, and MMR's diversity goal would work against that), but **for broad exploratory queries** like "summarize all applicable policies for this order" where the order spans multiple categories/exception types and the agent genuinely needs diverse, non-redundant coverage across clauses.
- **Reranking**: BGE-Reranker (self-hosted, free, open-source cross-encoder) on the top-~20 hybrid results down to top-5, per the guide's two-stage reranking technique.

**8.2.5 Incremental reindexing with versioning (not a naive "re-embed everything")**

This is the subtlest and most important RAG design decision in this project, because of the temporal-correctness requirement from the policy-change edge case:

1. On policy upload, compute a `content_hash` per element (table/paragraph/image caption).
2. Diff against the previous version's element hashes — **only changed/new elements get re-embedded**, unchanged elements are reused (embedding reuse, not recomputation) — this is the efficiency win the request asked for.
3. **Old versions are never deleted or overwritten** — they're marked `effective_end` and `superseded_by`, and remain fully queryable, because orders bound to an old policy version (per the earlier return-window example) must still resolve correctly years later. This inverts the usual "reindex replaces stale content" assumption — here, superseding is additive, not destructive.
4. A nightly job diffs `content_hash` across the current policy corpus against the last indexed state and reindexes only the delta — not a full corpus re-embed, which would be both wasteful and would risk breaking the version chain if done carelessly.

**8.2.6 Vector store**
**Qdrant, self-hosted (free, Docker)** — matches the framework-directory's "cost-sensitive, self-hosted-friendly teams" pick, supports real-time payload (metadata) filtering natively (needed for 8.2.4's pre-filter step), and native hybrid dense+sparse vector support in one engine, avoiding a second system for BM25.

---

### 8.3 Frontend — Pages

| Page | Purpose |
|---|---|
| **Case Queue / Dashboard** | Live list of all open cases, filterable by state/exception-type/priority; the default landing page for CS ops |
| **Case Detail View** | Full case timeline: trigger → diagnosis steps → fraud/risk score → decision reasoning (with cited policy clause) → execution status → verification — this is the human-readable render of the audit trail |
| **Escalation / Approval Queue** | Priority-sorted (edge case 9.1) queue of cases awaiting human sign-off, with the pre-drafted resolution shown for one-click approve/edit/reject |
| **Policy Document Manager** | Upload new policy PDFs, view version history per policy, see effective-date ranges, preview extracted tables/images from the ingestion pipeline before publishing a version live |
| **Agent Trace / Observability Viewer** | Per-case and per-agent trace drill-down: every tool call, input, output, latency, and token cost (the human-facing view of 8.8) |
| **Metrics Dashboard** | The full metrics set from 8.7, organized by Agent / Tool / RAG / System tabs |
| **Threshold & Policy Config** | Where CS Ops tunes the auto-execute confidence threshold τ and $ ceilings (Part 5) — includes the confirmation step for confidence-calibration changes proposed by the learning loop (8.5) |
| **Audit Log Viewer** | Compliance-facing, append-only view of every action taken, who/what approved it, for the Layer 13 audit-completeness requirement |

---

### 8.4 Memory Framework

- **Short-term (per-case working memory):** custom **Summary Buffer** implementation (lightweight — no need for a heavy framework here; it's just "recent tool outputs verbatim + summarized older context" bounded per case), stored alongside case state in Postgres.
- **Long-term (cross-case, cross-session):** **Zep / Graphiti** — chosen over Mem0 specifically because Graphiti's **time-aware relationship reasoning** is a direct architectural match for this project's temporal-correctness requirements (policy versions, "what did we know about this customer as of date X," return-window calculations) — this is exactly the framework-directory's stated differentiator for Zep/Graphiti over Mem0's faster-but-not-time-aware design.
- **Episodic Memory** (case history per customer) and **Procedural Memory** (learned resolution patterns, 8.5) are both implemented as Graphiti-backed entity/episode graphs rather than a second bolt-on system.

---

### 8.5 Learning From Answered Patterns / Confidence Improvement

This implements the guide's **Self-Reflection Memory** and **Procedural Memory** patterns, with the explicit safeguard the guide calls out ("risk of reinforcing bad patterns if unchecked"):

1. **Every human escalation decision (approve/edit/reject) is written to a Resolution Pattern Store** — `(case feature vector, agent's proposed resolution, human's final resolution, match/mismatch)`.
2. **Dynamic few-shot retrieval**: when the Resolution-Policy workflow reasons about a new case, it retrieves the k most similar *past resolved* cases from this store (via the same hybrid retrieval stack as 8.2, just a different index) and includes them as in-context examples — this is a RAG-over-past-decisions pattern, improving consistency without any model fine-tuning.
3. **Confidence recalibration (not autonomous):** weekly, a batch job computes per-cluster accuracy — did auto-executed cases in this similarity cluster get overturned on audit? did human-approved cases in this cluster consistently match what the agent would have proposed? — and **proposes** a threshold adjustment (raise τ for a cluster with 0% overturn over N cases; lower it for a cluster with any overturn). This proposal surfaces on the **Threshold & Policy Config** frontend page for a human (CS Ops owner, per Part 0's named business owner) to accept — **never auto-applied**, exactly matching the guide's Layer 15 discipline and avoiding the self-reinforcement risk of letting the system silently widen its own authority.

---

### 8.6 Hybrid Guardrails (for efficiency — fast + strong, not one or the other)

| Tier | Mechanism | Latency | What it catches |
|---|---|---|---|
| **Tier 1 — Deterministic (sync, blocking)** | Plain Python: hard $ ceilings, max-quantity limits, fraud-flag-always-escalates rule | ~1-5ms | The non-negotiable policy ceilings from Part 5 — never LLM-decided, never bypassable by a confident-sounding model output |
| **Tier 2 — Fast classifier (sync, blocking)** | **Guardrails AI** (RAIL schema validation on the Resolution-Policy Workflow's decision object) + **LLM Guard** (open-source, free) for PII detection on any text headed to logs/frontend | 10-50ms | Malformed decision objects, PII leakage into logs/customer-facing text |
| **Tier 3 — LLM-judge (async, non-blocking)** | A batch job scores a sample of resolutions for policy-citation accuracy and tone, logged/alerted — never blocks execution | 200-1000ms, but off the critical path | Subtle quality drift, mis-cited policy clauses that passed structural validation but are substantively wrong |

This is the guide's explicit latency-vs-strength trade-off pattern (cheap checks synchronous, expensive checks async/logged) applied concretely — "hybrid" here means tiered by cost/strength, not two guardrail vendors bolted together redundantly.

---

### 8.7 Metrics — Full List, Frontend-Displayed

**Agent / Orchestration metrics** (per the guide's Layer 4 + Part 4 metrics)
- Task completion rate (cases resolved without human correction)
- Average steps/tool-calls per completed case
- Multi-agent coordination failure rate
- Loop/step ceiling hit rate (Diagnosis Agent)
- Human escalation rate + escalation reason distribution
- Auto-execute vs. escalate ratio, trended weekly

**Tool metrics** (Layer 6)
- Tool selection accuracy
- Tool-call argument validity rate
- Per-tool failure rate and latency (OMS/WMS/payment/carrier, individually)
- Read vs. write call ratio, % of write calls through the approval gate
- Idempotency-key collision count (target ~0, alert on any)

**RAG metrics** (Layer 7)
- Retrieval hit rate
- Groundedness / faithfulness score (does the cited policy clause actually support the resolution text)
- Precision/recall @k on a golden policy-QA set
- Reranker lift (score improvement pre- vs. post-rerank)
- Index freshness lag (time between policy upload and searchable)
- Retrieval cache hit rate

**System / Business metrics** (Part 4.3 of the guide — the section most teams skip)
- Uptime, error rate, p95/p99 latency
- Cost per successful workflow (not cost per request — per the guide's Layer 14 framing)
- Guardrail trigger accuracy (false positive/negative rate)
- Audit log completeness %
- **User adoption rate**, **task deflection/automation rate**, **business KPI delta** (cost saved, CS-hours saved) — the three business-impact rows the guide explicitly flags as commonly skipped and most important for renewed project funding

All of the above render on the **Metrics Dashboard** page (8.3), tabbed by category, with real-time cards for the operational metrics (latency, error rate, escalation queue depth) and weekly/monthly trend charts for the drift/business metrics — matching the cadence column in the guide's own Part 4.3 table.

---

### 8.8 Observability Design — Inputs, Outputs, Metadata

**Langfuse**, structured as: **one trace per case**, **one span per agent/tool call within that trace.**

Each span logs:
```
{
  "span_id", "trace_id" (= case_id), "parent_span_id",
  "agent_or_tool_name",
  "input": { ... full input payload ... },
  "output": { ... full output payload ... },
  "metadata": {
     "model_used", "model_tier",
     "latency_ms", "token_count_in", "token_count_out", "cost_usd",
     "retrieval_doc_ids_used" (for RAG spans — enables the groundedness metric),
     "guardrail_tier_triggered" (if any),
     "confidence_score" (for decision spans),
     "idempotency_key" (for write spans)
  }
}
```
PII fields are redacted (via the Tier 2 LLM Guard scan, 8.6) **before** the span is persisted, not after — logs are a permanent record and can't be safely scrubbed retroactively with the same confidence.

This structure is what makes **Part 8.3's Trace Viewer page** possible as a direct render of Langfuse trace data, and what makes the RAG groundedness metric (8.7) computable after the fact (trace shows exactly which doc_ids/versions were retrieved and cited).

---

### 8.9 Caching — TTL and Staleness Handling

| Cache type | What's cached | TTL / invalidation |
|---|---|---|
| Prompt/context caching | Static system prompts, tool definitions per agent | Native provider caching (zero-config, per the guide's Layer 17 default) |
| Embedding cache | Text → embedding vector (deterministic mapping) | **Permanent** — never expires, only invalidated when the source text itself changes (tracked via `content_hash`, 8.2.5) |
| Retrieval cache | Query → retrieved doc set | **Short TTL (5-15 min)** — policy documents change infrequently, but short enough that a same-day policy correction propagates fast |
| Tool/API response cache | Inventory level, carrier tracking status (read calls) | **Very short TTL (30-60s)**, **plus event-driven invalidation**: an inbound webhook (stock update, carrier status change) actively invalidates the relevant cache key rather than waiting out the TTL — this directly prevents the "phantom stock" and "marketplace lag" edge cases (3.1, 3.4) from serving stale reads during a live diagnosis |
| Response cache | *(none — deliberately)* | Per the earlier reasoning: case-specific resolution reasoning has low reuse value and a false-positive semantic-cache hit on a refund decision is an unacceptable risk class |

**Stale-data detection:** every cached read carries its `fetched_at` timestamp into the Diagnosis Agent's context explicitly (not hidden) — if a diagnosis spans a longer-than-expected wall-clock time (e.g., waiting on a human escalation for hours), the orchestrator re-fetches rather than trusting an old cache entry when the case resumes, per the Layer 5 durable-state discipline.

---

### 8.10 Free-Tier Model Matrix (checked for current availability)

*Free-tier availability and rate limits change frequently and are tracked by community-maintained sources rather than a single authoritative registry — treat the specifics below as directionally correct as of Sept 2026 and re-verify rate limits/region restrictions at actual deployment time, exactly as the source guide advises for all framework/vendor picks.*

| Role | Pick | Why | Caveat |
|---|---|---|---|
| Router/classifier tier | **Groq — Llama 3.3 70B** (free) | Very fast inference, generous free daily request volume, good enough for exception-type classification and denial-reason structuring | Rate-limited (~30 RPM / ~1K RPD class) — fine for a portfolio demo, would need a paid tier at real production volume |
| Reasoning/planning tier | **Mistral AI — Mistral Large 3** (free, "La Plateforme" Experiment plan) | Far more generous free-tier token volume (~1B tokens/month class) than Gemini 2.5 Pro's free tier, which matters for the Diagnosis Agent's multi-step iterative planning | Requires opting into data training on the free plan — acceptable for a portfolio project using synthetic data, **not** acceptable if you later use real customer data on this tier |
| Reasoning tier (alt/burst) | **Google Gemini 2.5 Pro / 2.0 Flash** (free via AI Studio) | Strong quality, OpenAI-compatible endpoint | Free tier has a low daily request cap for the Pro model specifically, and is **not available in EU/UK/Switzerland** — a real constraint if deploying from those regions |
| Generation/formatting tier | **Cohere — Command R+** (free trial tier) or **Gemini 2.0 Flash** | Good balance of quality/speed for customer-comms drafting | Cohere's free tier caps around ~1K requests/month — fine for demo volume |
| Embeddings | **Self-hosted BGE-M3** (fully open-source, free forever, hybrid dense+sparse natively) | No rate limit, no region restriction, no ongoing cost — the most durable free choice for a project with real reindexing/versioning volume (8.2.5) | Requires you to host it (a small CPU/GPU instance or free-tier serverless container) rather than a pure API call |
| Embeddings (alt, zero-hosting) | **Mistral Embed** (free) or **Google text-embedding-004** (free) | No self-hosting needed | Tied to the same rate-limit/region caveats as their respective LLM tiers above |
| Reranker | **Self-hosted BGE-Reranker** (open-source, free) | Same durability argument as BGE-M3 | Same hosting requirement |

**Net recommendation for a portfolio build:** self-host BGE-M3 (embeddings) + BGE-Reranker (reranking) to eliminate rate-limit fragility on the RAG path entirely, and split LLM calls across Groq (fast/cheap tier) and Mistral Large 3 (reasoning tier) with Gemini 2.0 Flash as a documented fallback — this combination has no single point of rate-limit failure and costs $0 at demo/portfolio volume.

---

### 8.11 Edge Case → Architecture Mechanism Map

| Edge case (from prior inventory) | Mechanism that resolves it |
|---|---|
| Policy changed mid-order-lifecycle | 8.2.3 metadata effective-date binding + order-level `policy_version_bound` field |
| Extended/gifted warranty overlay | Metadata `doc_type` + multi-policy resolution logic in Resolution-Policy Workflow (most-generous-applies rule, confirmed as a business rule) |
| Seasonal policy extension | Separate conditional-rule metadata field, not folded into effective-date versioning |
| Per-category policy within one order | Per-line-item resolution in Resolution-Policy Workflow, not per-order |
| Partial payment method split | Execution Agent apportionment logic against the original payment-method breakdown pulled from Stripe |
| Payment webhook missing/OMS desync | Diagnosis Agent treats payment gateway as source of truth over OMS flags |
| Currency drift on delayed refund | Execution Agent refunds the original charged amount, not a recalculated one |
| Chargeback + internal refund race | Cross-check against payment gateway dispute status before Execution Agent fires (Layer 5 idempotency + this check together) |
| Phantom stock / oversell | Inventory Agent queries "sellable" not just "on-hand," cache invalidated on webhook (8.9) |
| Damaged/quarantined stock | Same sellable-quantity field distinction |
| Multi-warehouse race condition | Idempotency keys + WMS-side locking on transfer requests (Layer 5) |
| Marketplace stock lag | Cache TTL + event-driven invalidation (8.9); WMS always treated as ground truth over channel-reported stock |
| Serial returner vs. high-value customer | Fraud/Risk Agent weights reason-pattern + LTV, not raw return count |
| Wardrobing | Case stays open pending physical inspection signal, not auto-resolved at request time |
| Account takeover | Identity-consistency check (address-change + return-request combo) in Fraud/Risk Agent |
| "Never arrived" vs. carrier "delivered" | Always routes to human review; GPS/geofence/signature evidence attached if carrier provides it |
| Carrier misdelivery | Diagnosis Agent pulls delivery geodata where available, doesn't trust the binary delivered flag alone |
| Split shipment partial exception | Diagnosis + Resolution operate per-shipment, not per-order |
| Customs hold miscategorized | Carrier-specific status-code mapping table, not a generic "exception" bucket |
| Return label never scanned | Timeout-based escalation (14-day no-scan flag), not an assumption either way |
| Duplicate return request (double-click/multi-channel) | Dedup key: order + reason + timestamp window, at ingestion |
| Human-approval / auto-resolve race | State-machine transition lock — only one path can move a case out of "pending" |
| Case reopened after resolved | Explicit reopen endpoint (8.1) preserving audit trail, not a new duplicate case |
| Cross-channel order/return mismatch | Execution routes per the order's origin-channel field, not a default path |
| Marketplace policy stricter than brand default | Metadata `channel` field + "stricter external obligation wins" rule |
| Split/duplicate customer identity | Flagged as a data-quality issue for Customer Context Agent; graph-based long-term memory (8.4/Graphiti) can help surface likely-duplicate identities via relationship proximity |
| Address auto-correction root-causing an exception | Diagnosis Agent checks address-change history as a candidate root cause, not just a side detail |
| Locale/currency mismatch | Resolution amount always computed and logged in the order's original transaction currency |
| Escalation queue backlog at peak | Priority-sortable queue (9.1) — value/fraud-flag jumps ahead of FIFO |
| Human override beyond auto-execute band | Allowed, but always logged as an explicit policy exception in the audit trail (8.8), never silent |

---

## What's Deliberately Different From a Generic "E-commerce Agent" Build

1. **Autonomy is calibrated per sub-workflow**, not applied uniformly — most portfolio projects agent-ify everything; this one draws the line explicitly at the decision/execution boundary.
2. **LangGraph over the domain-default CrewAI**, with the override explained by a named scenario (durable multi-day state + financial writes), not vague preference.
3. **A hard, non-LLM policy ceiling wraps every resolution** — the LLM reasons within a band; it never has authority to exceed it, which is the actual production-safety story enterprises want to hear, not "the model is smart."
4. **Reindexing is additive, not destructive** — old policy versions are archived and superseded, never deleted, because temporal correctness (Part 8.11's first row) depends on old versions staying queryable indefinitely. Most RAG builds treat reindexing as "replace the stale chunk"; this one deliberately doesn't, because the domain requires it not to.
5. **Confidence calibration is human-gated, not self-reinforcing** — the learning loop (8.5) proposes threshold changes from observed accuracy, but a human always accepts them, directly avoiding the guide's named risk of a self-reflection loop reinforcing its own bad patterns unchecked.

---

*Next step when you're ready to build: Part 3 of the guide's step-by-step process — Architecture Decision Sheet, model tier selection with a real cost estimate from Layer 14 using actual exception-volume numbers, then the tool inventory before touching orchestration code. Part 8 above is now detailed enough to start the RAG ingestion pipeline and FastAPI scaffold first, since those have the fewest open dependencies on business-decided thresholds. Say the word and we'll start there.*

# Frugal Architecture Work Plan

Turning the GraphRAG workspace into a provider-switchable, cache-first, restart-safe
application. Every number below is measured against the live artifacts in this repo,
not estimated — see [Baseline](#baseline-measured-2026-09-18).

**Status:** fast path implemented · **Target:** v3.0.0 · **Written:** 2026-09-18

---

## Implementation status — 2026-09-18

| Task | Status | Verified by |
|---|---|---|
| **P0-2b** JSON-mode extraction | ✅ done | 1.0 → **11.0** relationships/chunk |
| **P1** Provider abstraction | ✅ done | 7 routes; `ChatOpenAI` drives Ollama + OpenAI unchanged |
| **P1-3** Per-provider concurrency | ✅ done | local → 1 worker, OpenAI → 4 |
| **P1-4** `num_ctx` / `max_tokens` guards | ✅ done | via `extra_body`; 1,740-token prompt fits |
| **P1-5** Pre-flight health check | ✅ done | catches missing key **and** network block |
| **P1-6** Sidebar engine selector | ✅ done | `AppTest` runs `app.py` with no exception |
| **P2-1** Singleton report gate | ✅ done | 18 of 25 communities templated |
| **P2-2** Extraction cache | ✅ done | re-run made **0** extraction calls |
| **P2-3** Table context clip | ✅ done | `MAX_TABLE_CHARS = 1500` |
| **P4-1** Token ledger | ✅ done | per-stage tokens, cost, calls avoided |
| **P4-3** Honest index status | ✅ done | `Partial · 8/72 · 11%` replaces "Indexed" |
| P3 Entity canonicalization | ⬜ next | |
| P5 Continuity (threads, queue) | ⬜ todo | |
| P6 Advanced frugality | ⬜ todo | |

### New files
`core/providers.py` · `core/cache.py` · `core/ledger.py`

### Measured end-to-end (2 documents, 2 chunks each, local route)

```
RUN 1 (cold): 133s | 4 extraction calls | 7 report calls | 18 templated
RUN 2 (warm):  29s | 0 extraction calls | 4 cache hits   | 18 templated
                    └─ 4.6x faster, zero extraction spend

graph: singletons 90% -> 72% · isolated nodes 55% -> 41%
```

### ⚠️ Network finding that changes the hosted recommendation

Probing every hosted route from this machine (401 = reached the API, 403 = blocked):

| Provider | Code | Verdict |
|---|---|---|
| deepinfra | 401 | ✅ reachable — **cheapest, now the default** |
| fireworks | 401 | ✅ reachable |
| openrouter | 200 | ✅ reachable |
| **groq** | **403** | ❌ **blocked** — Cloudflare 1010 |
| **together** | **403** | ❌ **blocked** |
| openai | 401 | ✅ reachable |

Groq and Together sit behind Cloudflare, which rejects the TLS fingerprint of the
Netskope proxy. A valid `GROQ_API_KEY` still fails with HTTP 403. This is the same
root cause as the `ollama pull` failure, one layer up the stack.

`DEFAULT_ROUTING` and the `balanced` preset therefore use **DeepInfra**, and
`health()` now probes reachability rather than only checking for a key — otherwise
a blocked route reports healthy and then fails on every chunk of a run.

**To finish the hosted route:** set `DEEPINFRA_API_KEY` (or `FIREWORKS_API_KEY` /
`OPENROUTER_API_KEY`) in `.env`. Until then the app falls back to OpenAI with a
visible warning, and the local route is fully working.

> Any new code that makes outbound HTTPS calls must call
> `truststore.inject_into_ssl()` first, or it fails with
> `CERTIFICATE_VERIFY_FAILED` on the Netskope CA.

---

## Objectives

1. **Frugal** — indexing cost scales with *new* content, not corpus size.
2. **Provider-switchable** — local, hosted open-weight, and OpenAI selectable in the UI.
3. **Seamless** — no silent truncation, no misleading status, honest cost/time up front.
4. **Continuous** — work and conversations survive the server being turned off.

---

## Baseline (measured 2026-09-18)

Run against `ragtest/output/*.parquet` and `vault/catalog.json` as they stand today.

### Corpus coverage

```
doc_a6abda2bdd97   8 / 72 chunks    (The Ultimate Guide to Fine-Tuning LLMs)
doc_f0fcef27b135   8 / 86 chunks    (2408.13296v3 Fine-Tuning Guide)
doc_5fe04c543a56   8 / 37 chunks    (2025 Global Cobrand Acquisition Playbook)
─────────────────────────────────────
                  24 / 195 chunks  = 12.3% of corpus indexed
```

All three are reported as **"Indexed"** in the Vault registry.

### Graph shape

```
entities            398   (351 unique titles — 47 fragmented across types)
relationships       142
isolated nodes      221 / 398   (55%)
communities         228
  └─ singleton      206         (90.4%)
  └─ size <= 2      215         (94.3%)
```

### Token cost of one full rebuild

```
extraction   41,760 in  /  16,800 out     (24 chunks)
reports      80,390 in  /  40,963 out     (228 communities)
──────────────────────────────────────
TOTAL       122,150 in  /  57,763 out

  of report output, 27,744 tokens (68%) is spent on singleton communities
  of extraction input, 31% is the static 540-token prompt prefix, re-sent per chunk
```

| Route | Cost | Wall clock | At full 195-chunk coverage |
|---|---|---|---|
| OpenAI gpt-4o-mini | $0.053 | ~4 min | ~$0.45 / ~30 min |
| Hosted open-weight 8B | ~$0.02 | ~3 min | ~$0.15 / ~25 min |
| Local Qwen2.5 (Ollama) | $0.00 | ~40 min | ~8–10 hours |

### Local model finding — phi-2 is unusable

Tested against the real `EXTRACTION_PROMPT` and a real chunk of `doc_5fe04c543a56`:

| Check | Result |
|---|---|
| Context window | **2,048** — prompt needs 1,740 + ~700 output = 2,440. Overflows by 392. |
| Stop tokens | **None.** Template is bare `{{ .Prompt }}`. Generates to `num_predict` every call. |
| Entities extracted | **0** |
| Relationships extracted | **0** |
| Output | Pattern-continued the input table (`Row 7: [Status] = Deployed` ×30) |
| Time wasted | 51.8s per call, for nothing |

phi-2 is a *base* model, not instruction-tuned. **Do not ship it behind the local provider.**
Use `qwen2.5:3b-instruct` (2.0GB) or `qwen2.5:7b-instruct` (4.7GB).

---

## Targets

| Metric | Today | Target | Mechanism |
|---|---|---|---|
| Report-stage LLM calls | 228 | **~22** | P2-1 singleton gate |
| Re-index after adding 1 doc | 100% of corpus | **~8 chunks** | P2-2 extraction cache |
| Corpus coverage | 12.3%, labelled "Indexed" | 100%, or honestly labelled | P2-2 + P4-3 |
| Isolated nodes | 55% | **< 25%** | P3-1 canonicalization |
| Cost visibility | none | per-run + per-query | P4-1 ledger |
| Survives restart | artifacts only | work + threads + config | P5 |
| Provider choice | hardcoded | 3 routes, UI-switchable | P1 |

---

## Phase 0 — Unblock local inference

**Goal:** a local model that can actually perform the extraction task.

### P0-1 · Install a working instruct model

> ⚠️ **`ollama pull` does not work on this machine.** Measured 2026-09-18: traffic is
> TLS-intercepted by Netskope (`ca.marriott.goskope.com` → `caadmin.netskope.com`).
> Ollama is a Go binary and Go's TLS stack does not read the macOS keychain where that
> CA lives, so blob transfers die at ~294 KB with `max retries exceeded: EOF`.
> `curl` uses the keychain and works fine (verified: 8 MB of GGUF at 1.4 MB/s, HTTP 206).
> This is the same problem `truststore.inject_into_ssl()` already solves for Python at
> `core/pipeline.py:34` and `app.py:22` — and it is why the existing phi-2 arrived as a
> manually downloaded `.gguf`.

**Working route — download with curl, import with a Modelfile:**

```bash
cd "~/Documents/AI Eng- Tutorial"
curl -L -C - --retry 10 --retry-all-errors \
  -o qwen2.5-3b-instruct-q4_k_m.gguf \
  "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"

ollama create qwen2.5-3b-graphrag -f Modelfile.qwen2.5-3b
```

`Modelfile.qwen2.5-3b` sets the ChatML template, the three stop tokens and
`num_ctx 8192` — each one fixing a specific measured phi-2 failure.

**Alternative that needs no download at all:** a hosted open-weight provider (Groq /
Together / DeepInfra). Those are plain HTTPS APIs through the Python stack, which already
trusts the Netskope CA via `truststore`. Given the proxy, this is the lower-friction route.

- Optional later: `qwen2.5:7b-instruct` (4.7GB) — better quality, ~9GB with KV cache,
  fits the 16GB budget but leaves less headroom.
- **Accept:** `ollama list` shows the model; a test call returns a real completion
  that stops on its own (`finish_reason: "stop"`, not `"length"`).

### P0-2 · Verify extraction capability — ⚠️ RUN 2026-09-18: PARTIAL FAIL

`qwen2.5-3b-graphrag` installed and tested against the real `EXTRACTION_PROMPT` on three
real 1,200-token chunks, parsed by the real `_parse_response`:

| Chunk | sec | out tok | entities | relationships |
|---|---|---|---|---|
| cobrand playbook (table region) | 2.0 | 108 | 2 | 1 |
| fine-tuning guide (prose) | 17.3 | 840 | 13 | **0** |
| fine-tuning paper (prose) | 26.0 | 1,305 | 31 | **2** |
| **average** | | | **15.3** ✅ | **1.0** ❌ |

**Gate: entities ≥ 5 → PASS (15.3). Relationships ≥ 3 → FAIL (1.0).**

Confirmed working: ChatML template, stop tokens (`finish_reason: "stop"`), 8192 context
(1,740-token prompt fits where phi-2 overflowed), and `ChatOpenAI` + `base_url` driving
Ollama unchanged — which validates the P1 provider abstraction.

**Why this matters more than it looks.** The graph already has a sparsity crisis: 142
relationships for 398 entities, 55% isolated nodes, 90% singleton communities. A model
producing ~1 relationship per chunk would push singleton communities toward 100% and make
the graph a bag of disconnected nodes. **Relationship extraction is the product.**

**Root cause is format fidelity, not comprehension.** On a decomposed two-pass prompt the
model identified correct content (`OpenAI's Fine-Tuning API`, `NVIDIA NeMo Customizer`) but
emitted the literal placeholder `<NAME>` and mangled `<|delimiter|>` into `|<delimiter|>`.
It understands the task; it cannot reliably reproduce an unusual custom format.

**Decision — do not route extraction to the 3B.** Options, in preference order:
1. **Hosted open-weight 8B+** (Groq/Together/DeepInfra) — no download, unaffected by the
   Netskope proxy, ~$0.02/rebuild, ~3 min. Best fit given this environment.
2. **`qwen2.5:7b-instruct`** — materially better at structured output; needs the same
   curl + Modelfile route, ~4.7GB, and `max_workers=1` on 16GB.
3. **Keep OpenAI for extraction**, use local only for answer drafting.

The 3B remains useful for cheap/offline *answer synthesis*, where format demands are low.

### P0-2b · Replace the custom delimiter format with JSON mode 🔴 new, high value
`EXTRACTION_PROMPT` uses GraphRAG's legacy `("entity"<|delimiter|>…)` format. That format is
hostile to every model that isn't a frontier model, and `_parse_response` already carries
four fallback delimiters plus a bare-split path to cope with drift.

Switch to a JSON schema with `response_format={"type": "json_object"}` / structured output.
Qwen2.5, Llama 3.1, and gpt-4o-mini all support it; Ollama, Groq and Together all expose it.

- **Why:** removes the single largest source of small-model failure, makes extraction
  quality far less provider-dependent, and is the precondition for local/hosted routes
  being viable at all.
- **Benefits every provider,** not just local — fewer silent parse failures on OpenAI too.
- **Re-run P0-2 after this.** The gate may well pass on the 3B once format is enforced.
- **Effort:** ~4h

### P0-3 · Retire phi-2 from the model list
- Keep the `.gguf` on disk; remove `mymodel` / `my-embed-model` from the selectable set,
  or rename them so they cannot be chosen by accident.

**Effort:** ~1h · **Blocks:** P1, P2

---

## Phase 1 — Provider abstraction

**Goal:** local, hosted open-weight and OpenAI as interchangeable config.

> **Key fact:** Ollama, Groq, Together, DeepInfra, Fireworks and OpenRouter *all* expose
> OpenAI-compatible `/v1/chat/completions`. `ChatOpenAI` already drives every one of them.
> Only `base_url` changes. No new LLM library is needed.

### P1-1 · `core/providers.py`

```python
@dataclass(frozen=True)
class Provider:
    key: str
    label: str
    model: str
    base_url: str | None
    api_key_env: str | None
    context_window: int
    max_workers: int                          # real concurrency of this backend
    max_tokens: int                           # NEVER unset for local
    supports_json_mode: bool
    cost_per_mtok: tuple[float, float] | None # None = local, priced in seconds
```

Registry entries:

| key | label | model | base_url | workers |
|---|---|---|---|---|
| `openai` | OpenAI · gpt-4o-mini | `gpt-4o-mini` | *(default)* | 4 |
| `groq` | Groq · Llama 3.1 8B | `llama-3.1-8b-instant` | `https://api.groq.com/openai/v1` | 2 |
| `local` | Local · Qwen2.5 3B | `qwen2.5:3b-instruct` | `http://localhost:11434/v1` | 1 |

`build_llm(provider, temperature, max_tokens)` constructs the `ChatOpenAI`, and is the
**only** place a model client is created.

### P1-2 · Replace the four hardcoded construction sites
- `core/pipeline.py:184` — `EntityRelationshipExtractor`
- `core/pipeline.py:284` — `CommunityReportSynthesizer`
- `ui/data.py:88` — `get_llm`
- `scripts/pipeline_e2e.py:309`

Also remove the `MODEL_NAME` string literal from the four message dicts in
`screens/search.py` (lines 366, 408, 454, 188) — provider comes from state.

### P1-3 · Per-provider concurrency
`ThreadPoolExecutor(max_workers=4)` at `core/pipeline.py:520` and `:817` must read
`provider.max_workers`.

> ⚠️ 4 workers against Ollama does not parallelize — Ollama serializes unless
> `OLLAMA_NUM_PARALLEL` is raised, and raising it allocates `num_ctx × parallel` of KV
> cache. At 8k × 4 on 16GB that thrashes or OOMs. **Local = 1.**

### P1-4 · Guardrails that only matter for non-OpenAI backends
- **`num_ctx` explicitly set.** Ollama defaults to 2048 *regardless of model capability*.
  A 32k-capable Qwen called without `num_ctx` silently truncates the tail of the prompt —
  which is exactly where `{input_text}` lives. Silent, no error, quality collapses.
- **`max_tokens` always set.** An unset cap against a model with no stop tokens generates
  until the context fills. This hung a test call for 120s during analysis.
- **Rate-limit backoff.** Groq's free tier will reject partway through a 195-chunk job.
  Exponential backoff + retry, surfaced as a stage message rather than a crash.
- **Timeouts:** 60s hosted, 180s local.

### P1-5 · Health check before the run, not during it
Ping the endpoint (`/api/tags` for Ollama, a cheap `/models` call for hosted) *before*
starting. A dead endpoint otherwise fails once per chunk — 195 slow timeouts.
- **Accept:** selecting an unreachable provider disables the Index button with a reason.

### P1-6 · Provider selector in the sidebar
Under the existing Temperature / Citations controls (`app.py:104`).
Persisted **to disk**, not `st.session_state` — config must survive restart (see P5-1).

```
┌─ Engine ────────────────────────────────┐
│  ○ Local · Qwen2.5 3B      ● ready      │
│    free · ~40 min index · offline       │
│  ● Groq · Llama 3.1 8B     ● key set    │
│    ~$0.02 index · ~3 min · sends data   │
│  ○ OpenAI · gpt-4o-mini    ○ no key     │
└─────────────────────────────────────────┘
```

Requirements: live readiness dot; cost **and** time **and** privacy shown together;
warning when switching with an existing graph (see P3-2).

**Effort:** ~1.5 days · **Depends on:** P0

---

## Phase 2 — Kill the waste

**Goal:** the two changes that cut the most spend, in the least code.

### P2-1 · Gate singleton community reports 🔴 highest return
`core/pipeline.py:723` `_build_reports` fires one LLM call per community
**unconditionally**. 206 of 228 ask a model to write a multi-finding analytical report
about *one entity with zero internal relationships*.

Skip the synthesizer when `size <= 2 and internal_edges == 0`; route to the existing
deterministic template already present in the `else` branch of `build_one`.

- **Saves:** 206 of 228 calls · 27,744 output tokens (68% of report stage) ·
  **~25 min per local rebuild**
- **Quality improves.** A templated *"ISOLATED CONCEPT: X (METHOD) — appears in 2 chunks,
  no extracted relationships"* is more honest than an invented 4-finding analysis of one node.
- **Accept:** report count drops to ~22; no community with 0 internal edges triggers an LLM call.
- **Effort:** ~1h

### P2-2 · Content-addressed extraction cache 🔴 highest return
`build_from_documents` re-extracts **every chunk of every document** on every rebuild
(`core/pipeline.py:471`). Adding document #5 re-pays for documents 1–4 in full.

```python
key = sha256(chunk_text + prompt_version + provider.key + provider.model)
```

Store `{entities, relationships}` as JSON under `vault/cache/extractions/`.
Check before every call in the extraction pool.

> ⚠️ **The provider MUST be in the key.** Without it, switching OpenAI → local reads back
> cached OpenAI extractions and reports them as local work — and a re-index after a model
> switch yields a graph that is half one model's naming conventions and half the other's.

- **Saves:** 100% of extraction on any rebuild with unchanged documents.
  Local re-index goes from **40 min → seconds**.
- **Bonus:** this is also the resumability primitive for P5-3 — a resumed run finds every
  completed chunk already cached.
- **Accept:** re-index with no new documents makes **zero** extraction calls.
- **Effort:** ~4h

### P2-3 · Clip table context
`core/rag.py:154` `_table_citations` appends `table["markdown"]` with **no length clip**,
while every neighbouring field is disciplined (`_clip(..., 1200)`, `_clip(..., 800)`).
One 60-row table silently injects thousands of tokens into every query that lexically
matches it.

- Apply `_clip(markdown, 1500)`, cap row facts at 5.
- **Accept:** no single query context exceeds a fixed token ceiling.
- **Effort:** ~30 min

### P2-4 · Chunk selection by budget, not by count
`all_chunks[:max_chunks_per_doc]` takes the **first N chunks** — the title page and table
of contents. Replace with a token budget, selecting by information density
(table-bearing, entity-dense) first.
- **Effort:** ~3h

**Effort:** ~1 day · **Depends on:** P1 (for provider-keyed cache)

---

## Phase 3 — Graph quality (a cost problem, not just a quality problem)

### P3-1 · Entity canonicalization
55% of nodes are isolated because `_build_relationships` (`core/pipeline.py:664`) drops
any relationship whose endpoints don't **exactly string-match** an entity title, and
`_build_entities` (`:642`) keys on `(title, type)` — fragmenting 398 rows across only 351
unique titles.

Confirmed trivially-mergeable pairs in the live graph:
`LLM`/`LLMS` · `HALF FINE TUNING`/`HALF FINE-TUNING` · `MARKETING INSIGHT`/`MARKETING INSIGHTS`

- Normalize + alias-resolve before graph construction.
- Key entities on canonical title; carry type as an attribute.
- Fuzzy-match relationship endpoints instead of dropping non-exact ones.
- **Why this is a cost item:** denser graph → fewer singleton communities → fewer report
  calls. Lower cost *and* better answers from one change.
- **Accept:** isolated nodes < 25%; singleton communities < 40%.
- **Effort:** ~6h

### P3-2 · Model provenance per entity
Record `extracted_by` (provider + model) on every entity and cached extraction.

> Two models capitalize, abbreviate and pluralize differently. Indexing doc 1 on GPT and
> doc 2 on Qwen **multiplies** the fragmentation in P3-1. P3-1 is therefore a
> **prerequisite** for multi-provider, not a follow-up.

- Show a "mixed-model graph" warning in the Vault when a corpus spans providers.
- **Effort:** ~2h

**Effort:** ~1 day · **Depends on:** P2

---

## Phase 4 — Make cost visible

> You cannot manage what you don't measure. Nothing in the app currently knows what it costs.
> **Build P4-1 early** — it is how Phases 2 and 3 get verified.

### P4-1 · Token ledger
Wrap every call site with a counter. Persist per-run into the existing `stats` field of
`vault/index_runs.json`, and per-query alongside the message. Dual-unit: **$ for API,
seconds for local**.
- **Accept:** every run and every answer records `{provider, model, tokens_in, tokens_out, cost, seconds}`.
- **Effort:** ~4h

### P4-2 · Pre-flight cost estimate
`_estimate_minutes` (`screens/vault.py:390`) already predicts time. Make it provider-aware
and add tokens + dollars, computed from projected cache hits:

> *"3 new chunks, 192 cached · ~$0.004 · 40s"*

4 minutes vs 40 minutes is the single most important number to show **before** the button.
- **Effort:** ~3h

### P4-3 · Honest index status 🔴 biggest trust gap
`_status_cell` (`screens/vault.py:169`) reports "Indexed" at 12% coverage. An analyst
trusting a cited answer built on 12% of the source is the largest product risk in the app.
- Distinguish `Indexed (partial · 8/72)` from `Indexed`.
- Add a corpus-coverage badge to the answer surface.
- Rewrite the chunk-limit help text: it is a **cost** knob, and the remainder of the
  document is invisible to every answer.
- **Effort:** ~2h

### P4-4 · Per-answer provenance
`screens/search.py:188` already renders a model chip — feed it the real provider plus
tokens and cost. Cost transparency is a trust feature in an analyst tool, not an admin one.
- **Effort:** ~1h

**Effort:** ~1.5 days

---

## Phase 5 — Continuity

### P5-1 · Persist threads and config to disk 🔴 smallest high-value fix
`st.session_state.messages` (`app.py:48`) is the **only** home for conversations.
**Every answer you paid for is destroyed by a browser refresh.**

- Write `vault/threads/{thread_id}.json` after each completed message. Messages already
  carry everything needed (`content`, `citations`, `raw_context`, `grounding`, `elapsed`, `model`).
- Same for provider selection and settings.
- **Accept:** restart the server; threads and provider choice are still there.
- **Effort:** ~3h

### P5-2 · Endpoint liveness in the run log
`IndexRunLog._process_alive` checks the Streamlit PID only. A run where **Ollama** died but
Streamlit lived shows as `running` forever with a healthy heartbeat.
- Add `provider`, `model`, `endpoint_alive_at` to the run record; refresh per stage.
- `reap_stale` condemns runs whose endpoint stopped answering.
- A resumed run refuses to continue under a different model than it started with.
- **Effort:** ~3h

### P5-3 · Resumable index runs
Extend `IndexRunLog` from a tombstone into a **work manifest**: the chunk hashes in the
run, each `pending | done | failed`. Combined with P2-2, resume is nearly free.
- **Effort:** ~4h

### P5-4 · SQLite as state-of-record
State is spread across `catalog.json`, `index_runs.json`, four parquet files, a Chroma DB
and `session_state`, with **no transaction across them**. A delete
(`screens/vault.py:279`) removes bytes, tables, text cache, vectors and catalog row in five
non-atomic steps; a crash mid-sequence leaves the vault inconsistent, and the parquet graph
still contains the deleted document's entities.

SQLite (stdlib, one file, ACID, zero ops) for catalog + runs + jobs + threads + ledger.
Parquet and Chroma become rebuildable read-models.
- **Effort:** ~2 days

### P5-5 · Job queue + background worker
Streamlit **submits** a job and **polls**; it never runs the build inline.

> A 4-minute OpenAI build *can* survive inside a script run if the user is careful.
> A 40-minute local build **cannot**. Local inference makes this mandatory, not optional.

Consequences: nav clicks and refreshes stop mattering; `_ingest`, `ingest_files` and
`_rebuild` collapse from three near-duplicate implementations into one; the
*"Leave this tab open"* plea (`screens/vault.py:88`) disappears; OpenAI Batch API
(50% off) becomes usable.
- **Effort:** ~3 days

**Effort:** ~1 week

---

## Phase 6 — Advanced frugality

### P6-1 · Hybrid routing presets
Route per task, because tasks have opposite requirements:

| Stage | Calls/rebuild | Character | Route |
|---|---|---|---|
| Entity extraction | 195 | Bulk, structured, low-judgment | **Local / cheap** |
| Singleton reports | 206 | Templated — no LLM | **Neither** (P2-1) |
| Real reports | ~22 | Analytical | Either |
| Query synthesis | 1/question | **User-facing, quality-critical** | **Best model** |

Presets: **Frugal** (all local) · **Balanced** (local index, frontier answers) · **Best** (all OpenAI).
Result: ~99% of calls run free and offline; the one call an analyst reads runs on a
frontier model.
- **Effort:** ~3h

### P6-2 · Semantic answer cache
Key on `(normalized_query_embedding, mode, source, corpus_signature)`.
`output_signature()` (`ui/data.py:48`) is already the correct cache epoch — it invalidates
on rebuild. Exact match first, then cosine ≥ 0.97 via the MiniLM already in-process.
Show a "cached · 0.0s" chip with one-click regenerate.
- **Saves:** 30–60% of query spend in demo or multi-user settings.
- **Effort:** ~4h

### P6-3 · Report cache keyed on community membership
`sha256(sorted(member_titles) + sorted(edge_ids) + prompt_version + provider + model)`.
Louvain is seeded and deterministic — a community untouched by a new document is free.
- **Effort:** ~1 day

### P6-4 · Adaptive retrieval depth
`retrieve_local` (`core/rag.py:262`) sends 12 entities + 14 relationships unconditionally.
Truncate at a relevance-score floor — most questions need 4.
- **Effort:** ~2h

### P6-5 · Compare-mode dedup
`_run_compare` (`screens/search.py:413`) runs two full generations. When graph and vector
citation sets overlap > 70%, generate once and render "both engines agree".
- **Effort:** ~2h

### P6-6 · Local embeddings upgrade
The vector path is *already* 100% local and free (Chroma's ONNX MiniLM). The
`nomic-embed-text-v1.5.f32.gguf` already on disk at
`~/Documents/AI Eng- Tutorial/` is a quality upgrade at zero API cost.
- **Effort:** ~3h

### P6-7 · Provider-conditional prompt length
OpenAI's automatic prompt caching needs a ≥ 1,024-token static prefix; the extraction
prompt is 540. Padding it with few-shot examples crosses that floor (and improves
extraction, helping P3-1).

> ⚠️ **Harmful for local** — every added prefix token is real prefill time with no cache
> discount. Make prompt length conditional on `provider.cost_per_mtok is not None`.

- **Effort:** ~3h

### P6-8 · Non-token costs
- `verify_integrity` (`core/vault.py:400`) re-hashes **every document's full bytes**,
  inside `run_qa_checks`, cached only 120s. With 50 documents that is hundreds of MB of
  SHA-256 every two minutes **on the render path**. Cache by `(path, mtime, size)`.
- `clear_caches` (`ui/data.py:177`) nukes `cache_resource` too, dropping the Chroma client,
  vault object and LLM client at 5 call sites. Split data-cache clearing from
  resource-cache clearing.
- **Effort:** ~3h

**Effort:** ~3 days

---

## Sequencing

```
P0  Unblock local            ▓                        1h    ← blocks everything
P1  Provider abstraction     ▓▓▓▓▓▓▓▓▓▓▓▓            1.5d
P2  Kill the waste           ▓▓▓▓▓▓▓▓                 1d    ← biggest return
P4-1 Token ledger            ▓▓▓▓                     4h    ← pulled early, verifies P2/P3
P3  Graph quality            ▓▓▓▓▓▓▓▓                 1d    ← prerequisite for multi-provider
P4  Cost visibility (rest)   ▓▓▓▓▓▓▓▓                 1d
P5-1 Persist threads         ▓▓▓                      3h    ← small, high value
P5  Continuity (rest)        ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓    1w
P6  Advanced frugality       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓        3d
```

**Fast path to most of the value:** P0-2b → P2-1 → P2-2 → P2-3 → P4-1 → P4-3.
About **2 days**, and it cuts current index spend by roughly two-thirds, closes the
"Indexed at 12%" trust gap, and makes extraction work across providers.

> **P0-2b (JSON mode) is now the true first task.** The 2026-09-18 gate showed the custom
> `<|delimiter|>` format — not model capability — is what breaks non-frontier models.
> Fixing the format is what makes the local and hosted routes viable, so it comes before
> the provider abstraction is worth finishing.

---

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| 3B too weak for structured extraction | Local route unusable | P0-2 gates it. Fall back to 7B or drop local from defaults. |
| Cache key missing provider | **Silent graph corruption** | Provider + model in key from day one (P2-2). |
| Mixed-model corpus worsens sparsity | Graph quality regresses | P3-1 before multi-provider is enabled by default. |
| Ollama `num_ctx` default 2048 | **Silent truncation**, no error | Always set explicitly (P1-4). |
| Groq rate limits mid-job | Run fails partway | Backoff + resumable runs (P1-4, P5-3). |
| 16GB ceiling with 7B + 4 workers | OOM / thrash | `max_workers=1` for local (P1-3). |
| Hosted pricing moves | Estimates drift | Verify on provider pricing pages; ledger (P4-1) shows actuals. |
| **Netskope TLS inspection breaks Go binaries** | `ollama pull` unusable | curl + Modelfile import (P0-1), or set `SSL_CERT_FILE` to an exported Netskope CA bundle before `ollama serve`. Hosted APIs are unaffected — Python already trusts it via `truststore`. |
| Disk at 93% (31 GB free) | Future pulls fail confusingly | Watch headroom before adding 7B or more models. |

---

## Verification

After each phase, re-run the baseline measurement:

```bash
.venv/bin/python -c "
import pandas as pd
r = pd.read_parquet('ragtest/output/create_final_community_reports.parquet')
n = pd.read_parquet('ragtest/output/create_final_nodes.parquet')
print('communities     ', len(r))
print('singletons      ', int((r['size']==1).sum()), f\"({100*(r['size']==1).mean():.1f}%)\")
print('isolated nodes  ', int((n['degree']==0).sum()), f\"({100*(n['degree']==0).mean():.1f}%)\")
"
```

**Definition of done for "frugal":**
- Re-index with no new documents → **0 extraction calls, 0 report calls**
- Adding 1 document → extraction calls ≈ that document's new chunks only
- Every run and answer carries a recorded cost
- Provider switchable in the UI without corrupting the graph
- Server restart loses nothing but in-flight generation

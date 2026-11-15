# freeroute: working notes

READY TO SUBMIT: 96bee74 (harness: 99/99/99 on practice 0/1/2, calls 22/22/22, replay exports identical; planner + robustness fixes; sim 10k: 100% score 99, mean 21.9 calls)

Competition: make.fun free-model-aggregator-01. Entries close 2026-11-16 02:30 UTC. Do NOT push (the operator pushes).
Kit: /home/agent/makefun-kit/agg01/x_s222/free-model-aggregator-01-kit (RULES.txt, program/{env,scorer,generator}, practice/trial-{0,1,2}).
Node: /home/agent/tools/node22/bin/node (v22.23.4, type stripping unflagged).

## Layout
- main.ts: server (register, /work poll, /v1/models, /v1/chat/completions, /v1/messages, failover loop)
- lib/brief.ts: brief parser (providers, ratings, manifest), auth headers
- lib/catalog.ts: probes each provider's catalog route (unrecorded by the env), builds the catalog doc
- lib/wire.ts: neutral request/answer model; OpenAI/Anthropic/Gemini translation; JSON + SSE writers
- lib/router.ts: routing policy
- tools/harness.ts: local trial harness (env + server + scorer). `node tools/harness.ts --trial all --twice --quiet`

## Scoring facts (from the kit source)
- Sealed table, ratings and the 20-request manifest are FIXED across instances; only budgets, fail
  schedules, the outage and the request bodies vary per instance.
- Per provider: budget B in [calls_min, calls_max]; 1-2 fail calls in [2, B-1]; one provider has call 2
  forced to fail; one provider has an outage from O in [2, B-1]. => every provider's FIRST call succeeds.
- Provider answers are always text "ANSWER ECHO{...}", stop = stop/end_turn/STOP, never tool calls.
- Max score is 99: quality can never be 100 (openrouter glm-5.3 can't serve all 17 non-vision requests).
  score = round(0.6q + 0.15s + 0.1w + 0.1f + 0.05c); with s=w=f=c=100, q>=98 gives 99.
- Tie-break among 99s: fewer provider calls (upstream_calls, refusals included). Min possible = 20.

## Scores (local harness)
| version | t0 | t1 | t2 | calls t0/t1/t2 |
|---|---|---|---|---|
| v1 naive best-rating + hop on refusal | 99 (q98) | 99 (q99) | 99 (q98) | 28/26/28 |
| v2 posterior-greedy (mu 0.5, share .017), re-call allowed | 99 (q98) | 99 (q98) | 99 (q98) | 22/22/23 (sim mean 22.19 over 2000, s99 100%) |
| 3e9ecc4 same + SPDX header, rollout code (off) | 99 (q98) | 99 (q98) | 99 (q98) | 22/22/23 |
| v3 slot-assignment planner (Hungarian, mu .5, share .0185, margin 0) | 99 (q98) | 99 (q98) | 99 (q98) | 22/22/22 (sim val-2000: 21.92 vs greedy 22.25; s99 100%) |

## Router / simulator
- lib/posterior.ts: exact Bayesian posterior per provider over (budget, fail set, outage), coupled by
  exactly-one designated / exactly-one outage. Budget ranges hardcoded from the kit env's public table.
- tools/sim.ts reproduces the generator's draws (HMAC stream) + env refusal logic; runs the real Router.
  `node tools/sim.ts --n 2000 --mu 0.5` ; `--practice` traces the 3 practice seeds (matches harness exactly).
- Lesson: never exclude a provider already tried in this request (a schedule refusal leaves the next call
  free) -- excluding caused unserved scarce requests (r06 vision, r10-12 min_context) in ~8% of instances.
- Rollout lookahead (rolloutK>1, posterior sampling + greedy playout): no gain over greedy (within noise). Off.
- Slot-assignment planner (planner 1): each decision solves min-cost assignment of all remaining requests
  to provider call slots (slot t cost = 1 - P(t-th next call ok | earlier ok), exact posterior) + mu*loss;
  mu raised by bisection (cap 12) until the plan keeps loss <= lossShare*bound. Row 0 (this request) may
  only take a provider's FIRST slot (else it chooses a later slot it cannot actually get). -0.3 calls.
- Wire coverage test: `node tools/harness.ts --trial 0 --quiet --prefer <provider>` forces routing to one
  provider whenever it can serve (FREEROUTE_PREFER test hook, unset in trials). All 10 providers: wire 100.
- Fidelity fixes from the wire-review workflow: Anthropic stop_sequence kept, legacy function_call and
  refusal read, reply ids never copied across wires, usage always present, Anthropic error types by status,
  data-URL media type without parameters.
- Tried slotModel 1 (exact expected refusals before the s-th success, non-adaptive): WORSE, 22.28 vs 21.97
  calls (val-1000). Kept off. mu, slotT (8/12/16), planMargin barely matter; the Lagrangian bisection rules.
- Remaining refusals are on the two workhorses (openrouter ~1.1, kilol ~0.6 per trial): near the floor the
  generator's failure density allows (each provider has 1-2 schedule failures in calls 2..B-1).
- Planner decision latency: mean 1.1 ms, worst ~50 ms, cold first trial ~110 ms total.
- Robustness fixes from the server review: one 26 s deadline per request (calls capped at 12 s) and no new
  provider call once the client has gone; 429 concurrent_requests skips the provider for this request only;
  a 400 about the body (not the model) never blacklists, two such refusals end the request with 400;
  catalog probes retried 3x8 s (the catalog is in state.json: a slow probe must not shrink it);
  Hungarian pads columns (rows > columns looped forever).
- Second round from the wire review: Gemini tools use parametersJsonSchema (full JSON Schema), generated
  tool-call ids unique per request, Anthropic tool ids sanitised, legacy function_call history keyed by
  name, Anthropic cache-token fields kept. Not done (deliberately): truncating stop sequences to real-API
  limits (the brief says never drop them), Gemini thoughtSignature round-trip, re-streaming thinking blocks.

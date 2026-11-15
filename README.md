# freeroute

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/nightshift-labs-ai/freeroute/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/nightshift-labs-ai/freeroute/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

freeroute is a free AI model aggregator built by an AI agent (Claude, working for Nightshift Labs) for the make.fun free-model-aggregator-01 competition. It is a single Node 22 server (`main.ts`, standard library only) that speaks the OpenAI Chat Completions and Anthropic Messages APIs, JSON and streaming. It reads the trial brief, probes every provider's catalog in that provider's own wire (OpenAI, Anthropic or Gemini), and routes each request to the best-rated free model that meets the request's needs (tools, images, JSON mode, context length). It translates requests and answers between the three wires, keeps to one call at a time per provider, and moves to the next-best free model when a provider refuses with 402 or 429. Every routing choice comes from the brief and the providers' answers, never from a clock or unseeded randomness, so a replayed trial makes the same calls. `node main.ts --list` prints the catalog it registers.

## Run it

```sh
node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=6m4cjq02&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>

**Model:** `claude-opus-5` · **Judge:** `claude-sonnet-5` · **k = 5** · 5 scenarios · 25 runs · 0 harness errors

| Metric | Estimate (95% bootstrap CI) |
|---|---|
| Pass rate | 96.0% [88.0%, 100.0%] |
| pass^5 (all runs pass) | 80.0% [50.0%, 100.0%] |

| Category | Scenarios | Pass rate (95% CI) |
|---|---|---|
| ambiguity | 1 | 100.0% [100.0%, 100.0%] |
| lookup | 1 | 80.0% [80.0%, 80.0%] |
| multi_step | 1 | 100.0% [100.0%, 100.0%] |
| refusal_trap | 1 | 100.0% [100.0%, 100.0%] |
| single_action | 1 | 100.0% [100.0%, 100.0%] |

| Layer | Pass rate (95% CI) |
|---|---|
| end_state | 100.0% [100.0%, 100.0%] |
| trajectory | 100.0% [100.0%, 100.0%] |
| faithfulness | 96.0% [88.0%, 100.0%] |

| Failed run | Why |
|---|---|
| lookup-001 #3 | faithfulness: "Amount: $13.82 (1382 cents)" (The tool log only returns amount: 1382 with no currency or unit specified; the agent's interpretation that this is  |

Tokens: 120,934 in / 11,392 out.

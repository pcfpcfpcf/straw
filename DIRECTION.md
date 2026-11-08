<!-- ai written: drafted with Claude (claude-sonnet-5-5), reviewed by Aminechaabane -->

# Direction: straw as a harness for any AI workload

Status: proposal. Nothing here is built yet. It describes where the core could go and in which order. Each stage
below is meant to be one commit that builds and proves one thing.

## Why

Agentic workflows are no longer just an LLM calling tools in a loop. A real workflow mixes several kinds of model:

- a chat model that plans, calls tools and writes text
- a decision model that answers a fixed question with calibrated probabilities (StartLux-Decision on geode's
  `/v1/systemone`)
- an embedding model that turns text into a vector for search or memory
- a plain ML model (anomaly detector, regressor, classifier) that scores numbers and has no text interface at all

Geode already works this way as an inference engine. Its core (placement, planner, executor) knows nothing about any
one workload, and each model serves the endpoint its head names: logits to `/v1/chat/completions`, a pooled vector
to `/v1/embeddings`, and rows or decision answers to `/v1/systemone`. A new workload is a thin layer on top of the
generic core, not a change to it.

The README says straw is a **universal** harness. To keep that promise, straw's core should be general in the same
way geode's is: a small core that runs a workflow, with each kind of model plugged in beside it.

## Where straw is today

Most of the core is already workload-agnostic. Two parts are tied to LLM chat:

| Already general | Tied to LLM chat |
|---|---|
| `Domain`: tools, reducers, miscs, HTTP routes, auth (`core/domain.py`) | `Agent.call_llm` / `stream_llm`: `chat_url`, `messages`, `tools` schemas (`core/agent.py`) |
| `Tool` and its registry, `read_only`, `ends_turn` (`core/types.py`) | `Agent.run`: the LLM chooses every next step |
| `ctx`, shared across a run | `POST /agent` takes an `llm` block (`model`, `api_url`, `api_key`) |
| The event stream: `thought`, `action`, `observation`, `answer`, `error`, `final` | `compact` / `_digest`: built on chat messages and `tool_calls` |
| `Chats` persistence (`core/chats.py`) | |

The general parts carry over unchanged. The change is two seams in the right-hand column.

## The two seams

### 1. Workload: what a model does

A workload is one kind of model call, with its own input and output. Straw would know a few kinds:

| Workload | In | Out | Example backend |
|---|---|---|---|
| `generate` | messages, tool schemas | text and/or tool calls | any OpenAI-compatible chat endpoint (today's only path) |
| `decide` | a state and questions | per question: a choice with probabilities, a yes probability, or a score | geode `/v1/systemone` with a StartLux-Decision model |
| `embed` | text | vector | geode or any `/v1/embeddings` |
| `predict` | numbers or records | scores or labels | a local model, a rule set, anything callable |

A workload is an interface, not a vendor list. Core holds the interface and today's chat adapter. Other adapters can
live in a domain package until more than one domain needs them.

### 2. Controller: who chooses the next step

Today the LLM decides every step. That is one controller among several:

| Controller | Next step chosen by | Fits |
|---|---|---|
| `llm_loop` | the chat model's tool calls (today's `Agent.run`) | open questions, exploration |
| `pipeline` | the code: a fixed list of steps | known procedures, evals, batch jobs |
| `router` | a `decide` workload: "next step? A) tool X B) tool Y C) finish" | small models that loop or forget to finish |

A router is cheap (one pass a step), calibrated (the probability says how sure it is), and cannot produce a malformed
call. It can also hand control to an `llm_loop` when its top probability is low.

### What stays the same

Every controller emits the same events. A `decide` step becomes an `action`/`observation` pair whose observation
carries the probabilities. The HTTP stream, the chat store, UIs and evals read the same event types whatever ran. That
keeps the change invisible to every existing caller.

## A sketch, not an API

```python
class Workload:                     # one kind of model call
    kind: str                       # "generate" | "decide" | "embed" | "predict"
    def call(self, request: dict, ctx: dict) -> dict: ...

class Controller:                   # chooses the steps of one run
    def run(self, task, ctx: dict):  # yields straw events, as Agent.run does today
        ...

# today's behavior, unchanged:  Controller = llm_loop(workload=ChatWorkload(model, api_url))
```

`POST /agent` would keep its current body. A new optional block names the workloads a run may use, so old clients send
the same `llm` block and get the same behavior.

## A first real user: fleet diagnosis

The fleet-telemetry project runs a vehicle-diagnosis domain on straw (`straw_diagnosis`). It uses a 4B model
(StartLux-Decision-v1.1-4B-AutoThink) on geode as a tool-calling chat model. On a 10-car set of cascading faults it
found the right root cause on 5 of 6 faulty cars and called all 4 healthy cars healthy. Its weak spots are the ones a
general harness would fix:

- 3 of 10 runs ended without calling the submit tool, and a domain-level retry had to rescue them
- one run made 17 tool calls, 10 of them the same look-up under different conditions
- the final cause is free text, so scoring needs keyword matching, and confidence is a word the model chose

The same diagnosis as a mixed workflow:

1. `pipeline`: collect the evidence with the existing tools (vehicle, code look-ups, health summary)
2. `decide`: pick the root cause from the code look-up's list of causes, answer "healthy?" as yes/no, score severity
3. `generate`: write only the summary and advice for the fleet manager

This needs no straw change to try: the domain can call geode's `/v1/systemone` from its own code. The point of this
document is that the pattern belongs in straw, so the next domain does not rebuild it.

## Stages

Each stage is one commit that builds and keeps every existing caller working.

1. **Workload interface.** `Workload` and a `ChatWorkload` that holds today's `call_llm` / `stream_llm` code.
   `Agent` calls it instead of `requests.post` directly. Proof: the existing tests pass and a run produces the same
   events and the same request bodies as before.
2. **Controller interface.** Today's `Agent.run` becomes the `llm_loop` controller. Add `pipeline`. Proof: a domain
   runs a fixed list of tools through `pipeline` and the stream has the same event types as an `llm_loop` run.
3. **`decide` workload.** An adapter for geode's `/v1/systemone` decision route. Proof: against a served
   StartLux-Decision model, answers for a fixed set of questions match the model's reference implementation
   (`startlux_decide`) within its tolerance.
4. **`router` controller.** Next step chosen by a `decide` call over the registered tools, with a hand-off to
   `llm_loop` below a confidence gate. Proof: on a domain eval, fewer steps and no run that ends without its final
   tool, at equal or better accuracy than `llm_loop` alone.
5. **`embed` and `predict`.** Added when a domain needs them, with the same rule: one adapter, one proof.

## Open questions

1. **Tool or workload?** A `decide` call could simply be a tool that a domain registers. A workload is worth having
   in core only if controllers need to treat it differently (a router does). Stage 4 is where this is settled.
2. **One model, two endpoints.** Geode serves a StartLux-Decision model on `/v1/systemone` and refuses chat
   completions on it once its `decision_config.json` is present. A workflow that wants the same weights for both
   needs two served entries. That is geode's choice to make, and straw should not assume either answer.
3. **Learning.** The README's goal is a harness that keeps getting better at a domain. Probabilities from `decide`
   steps, checked against outcomes (did the mechanic confirm the cause?), are a direct signal to learn from, which free
   text is not. How straw stores and uses that signal is a later document.
4. **Staying lean.** The core is about 650 lines today. Each stage should add an interface and move existing code
   behind it, not add vendors. Adapters that only one domain uses stay in that domain.

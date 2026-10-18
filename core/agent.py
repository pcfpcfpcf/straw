import inspect
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import requests

from .types import Tool

MAX_STEPS = 8            # ReAct iterations before the loop is forced to conclude
MAX_TOOL_CALLS = 2       # per turn: a model that asks for eight tools at once burns
                         # the whole budget in two steps
MAX_EMPTIES = 2          # empty returns before a tool is cut off
LLM_TIMEOUT = 180        # seconds; a hung provider must fail, not hang the request

ROUTING_THINKING = {"enable_thinking": False}

ROUTING_MAX_TOKENS = 512

BUDGET_PROMPT = (
    "Do not call any more tools. Based only on what you have already gathered above, "
    "give the user your best conclusion now as a concise, factual answer."
)

def is_empty_result(observation: str) -> bool:
    observation = observation.strip()
    return (observation in ("", "[]", "null", "{}")
            or observation.startswith("No data")
            or "total:0" in observation)


def execute_tool(tool: Tool, args: dict, ctx: dict) -> str:

    sig = inspect.signature(tool.func)
    if "ctx" in sig.parameters:
        args["ctx"] = ctx  # overwrites anything the model tried to pass

    try:
        bound = sig.bind(**args)
    except TypeError as e:
        return f"error: {tool.name}{sig}: {e}"

    try:
        t = time.perf_counter()
        out = str(tool.func(**bound.arguments))
        print(f"[timing] tool {tool.name} {time.perf_counter() - t:.2f}s", flush=True)
        return out
    except Exception as e:
        return f"error: {tool.name}: {e}"


class Agent:
    def __init__(self, model: str, api_url: str, tools_reg: List[Tool],
                 system_prompt: str, api_key: str = "",
                 max_steps: int = MAX_STEPS, max_tool_calls: int = MAX_TOOL_CALLS,
                 max_tokens: int = ROUTING_MAX_TOKENS, chat_template_kwargs: dict = None,
                 max_observation: int = 0):
        self.model = model
        self.api_url = api_url
        self.api_key = api_key

        self.tools_reg = tools_reg
        self.system_prompt = system_prompt

        # per-domain budgets: a domain whose work takes many small steps needs more than the routing defaults
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.max_tokens = max_tokens # 0: no cap
        self.chat_template_kwargs = chat_template_kwargs # None: ROUTING_THINKING, {}: none sent
        self.max_observation = max_observation # 0: observations are never cut

    @staticmethod
    def find_tool(tools_reg, name: str) -> Tool | None:
        return next((t for t in tools_reg if t.name == name), None)

    def chat_url(self) -> str:
        url = self.api_url.rstrip("/")
        if url.endswith("/chat/completions"):
            return url
        if not url.endswith(("/v1", "/api", "/openai")):
            url += "/v1"
        return url + "/chat/completions"

    def template_kwargs(self) -> dict:
        if self.chat_template_kwargs is not None:
            return self.chat_template_kwargs
        return {} if "googleapis.com" in self.chat_url() else ROUTING_THINKING

    @staticmethod
    def compact(history) -> str: #v1, dumb compact: keep the prompt, the tool calls and the answer, drop tool output
        results = {m["tool_call_id"]: m["content"] for m in history if m["role"] == "tool"}

        compact_h = []
        turn = None
        for m in history:
            if m["role"] == "user":
                turn = {"prompt": m["content"], "steps": [], "answer": ""}
                compact_h.append(turn)

            elif m["role"] == "assistant" and turn:
                if m.get("tool_calls"):
                    for call in m["tool_calls"]:
                        out = results.get(call["id"], "")
                        if out.startswith("error:") or "returned empty results" in out or is_empty_result(out):
                            continue #skip empty or skipped tools
                        fn = call["function"]
                        turn["steps"].append(f'{fn["name"]}({fn["arguments"]})')
                else:
                    turn["answer"] = m["content"]
        
        return compact_h

    def call_llm(self, messages: list, with_tools: bool) -> tuple:  # (message, finish_reason)
        body = {"model": self.model, "messages": messages, "stream": False}
        if with_tools:
            body["tools"] = [t.schema() for t in self.tools_reg]
            template_kwargs = self.template_kwargs()
            if template_kwargs:
                body["chat_template_kwargs"] = template_kwargs
            if self.max_tokens:
                body["max_tokens"] = self.max_tokens

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        t = time.perf_counter()
        usage = {}
        try:
            resp = requests.post(self.chat_url(), json=body, headers=headers, timeout=LLM_TIMEOUT)
            if not resp.ok:
                err_detail = resp.text
                try:
                    err_json = resp.json()
                    if isinstance(err_json, dict) and "error" in err_json:
                        err_detail = err_json["error"].get("message", resp.text)
                    elif isinstance(err_json, list) and err_json and "error" in err_json[0]:
                        err_detail = err_json[0]["error"].get("message", resp.text)
                except Exception:
                    pass
                print(f"[error] LLM API returned HTTP {resp.status_code}: {err_detail}", flush=True)
                raise RuntimeError(f"HTTP {resp.status_code} ({err_detail})")
            j = resp.json()
            usage = j.get("usage", {}) or {}
            choice = j["choices"][0]
            return choice["message"], choice.get("finish_reason")
        finally:
            dt = time.perf_counter() - t
            comp = usage.get("completion_tokens")
            tps = f"{comp / dt:.0f}" if comp and dt > 0 else "?"
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            print(f"[timing] llm {dt:.2f}s tools={with_tools} msgs={len(messages)} "
                  f"prompt_tok={usage.get('prompt_tokens')} cached_tok={cached} "
                  f"completion_tok={comp} tok/s={tps}", flush=True)

    def run(self, messages: list, ctx: dict, max_steps: int = None):
        max_steps = max_steps or self.max_steps

        # check if system_prompt func or str 
        prompt = self.system_prompt() if callable(self.system_prompt) else self.system_prompt

        messages = [{"role": "system", "content": prompt}] + list(messages)
        base = len(messages)  # turns generated below get persisted by the caller
        empties = {}

        for _ in range(max_steps): #main loop
            try:
                message, finish = self.call_llm(messages, with_tools=True)
            except Exception as e:
                yield {"type": "error", "content": str(e)}
                return

            messages.append(message)
            calls = message.get("tool_calls") or []

            if not calls: 
                if finish == "length":
                    messages.pop()
                    yield from self.synthesize(messages, converged=True)
                    yield {"type": "final",
                           "messages": [m for m in messages[base:] if m.get("content") != BUDGET_PROMPT]}
                else:
                    yield {"type": "answer", "content": message.get("content", ""), "converged": True}
                    yield {"type": "final", "messages": messages[base:]}
                return

            if message.get("content"):
                yield {"type": "thought", "content": message["content"]}

            todo, i = calls[:self.max_tool_calls], 0
            while i < len(todo):
                group = self.next_group(todo, i)
                before = len(messages)
                yield from self.run_tool_group(group, ctx, messages, empties)
                i += len(group)

                answer = self.turn_answer(group, messages[before:])
                if answer is not None:
                    messages.append({"role": "assistant", "content": answer})
                    yield {"type": "answer", "content": answer, "converged": True}
                    yield {"type": "final", "messages": messages[base:]}
                    return

            for call in calls[self.max_tool_calls:]:
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": f"error: at most {self.max_tool_calls} tool calls per turn",
                })

        yield from self.synthesize(messages)
        yield {"type": "final",
               "messages": [m for m in messages[base:] if m.get("content") != BUDGET_PROMPT]}

    def is_read_only(self, call: dict) -> bool:
        tool = Agent.find_tool(self.tools_reg, call["function"]["name"])
        return bool(tool and tool.read_only)

    def next_group(self, calls: list, i: int) -> list:
        # consecutive read-only calls run together; anything that changes state runs alone, in the asked order
        group = [calls[i]]
        if not self.is_read_only(calls[i]):
            return group
        while i + len(group) < len(calls) and self.is_read_only(calls[i + len(group)]):
            group.append(calls[i + len(group)])
        return group

    def turn_answer(self, group: list, added: list) -> str | None:
        # the result of an ends_turn tool that worked is the answer
        results = {m.get("tool_call_id"): m.get("content") or "" for m in added if m.get("role") == "tool"}
        for call in group:
            tool = Agent.find_tool(self.tools_reg, call["function"]["name"])
            out = results.get(call["id"])
            if tool and tool.ends_turn and out and not out.startswith("error"):
                return out
        return None

    def observe(self, call: dict, ctx: dict, empties: dict) -> str:
        # reads empties, never writes it: read-only calls of a group run on worker threads
        name = call["function"]["name"]
        raw_args = call["function"].get("arguments") or "{}"

        tool = Agent.find_tool(self.tools_reg, name)
        if empties.get(name, 0) >= MAX_EMPTIES:
            return (f"{name} has returned empty results {empties[name]} times already. "
                    "State your conclusion from the data you have, or try a "
                    "fundamentally different approach.")
        if not tool:
            return f"error: unknown tool: {name}"
        try:
            args = json.loads(raw_args)
        except ValueError as e:
            return f"error: {name}: invalid arguments: {e}"
        if not isinstance(args, dict):
            return f"error: {name}: arguments must be a JSON object"

        observation = execute_tool(tool, args, ctx)
        if self.max_observation and len(observation) > self.max_observation:
            observation = (observation[:self.max_observation]
                           + f"\n...[cut: {len(observation) - self.max_observation} more characters]")
        return observation

    def run_tool_group(self, group: list, ctx: dict, messages: list, empties: dict):
        for call in group: # to show reasoning steps
            yield {"type": "action", "tool": call["function"]["name"],
                   "content": call["function"].get("arguments") or "{}"}

        if len(group) == 1:
            observations = [self.observe(group[0], ctx, empties)]
        else:
            with ThreadPoolExecutor(max_workers=len(group)) as pool:
                observations = list(pool.map(lambda c: self.observe(c, ctx, empties), group))

        for call, observation in zip(group, observations):
            name = call["function"]["name"]
            if is_empty_result(observation):
                empties[name] = empties.get(name, 0) + 1
            yield {"type": "observation", "tool": name, "content": observation}
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": observation})

    def stream_llm(self, messages: list): #stream answer instead of waiting for the whole block

        body = {"model": self.model, "messages": messages, "stream": True,
                "stream_options": {"include_usage": True}}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        t = time.perf_counter()
        chars, comp = 0, None
        with requests.post(self.chat_url(), json=body, headers=headers,
                           timeout=LLM_TIMEOUT, stream=True) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode() if isinstance(raw, bytes) else raw
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                choices = chunk.get("choices") or []
                if choices:
                    piece = (choices[0].get("delta") or {}).get("content")
                    if piece:
                        chars += len(piece)
                        yield piece
                if chunk.get("usage"):  # final chunk, via stream_options.include_usage
                    comp = chunk["usage"].get("completion_tokens")
        dt = time.perf_counter() - t
        tps = f" tok/s={comp / dt:.0f}" if comp and dt > 0 else ""
        print(f"[timing] llm(stream) {dt:.2f}s chars={chars} completion_tok={comp}{tps}", flush=True)

    def synthesize(self, messages: list, converged: bool = False):
        messages.append({"role": "user", "content": BUDGET_PROMPT})

        parts = []
        try:
            for piece in self.stream_llm(messages):
                parts.append(piece)
                yield {"type": "answer_delta", "content": piece}
        except Exception as e:
            yield {"type": "error", "content": str(e)}
            return

        answer = "".join(parts)

        #persist streamed answer like a full block
        messages.append({"role": "assistant", "content": answer})
        yield {"type": "answer", "content": answer, "converged": converged}

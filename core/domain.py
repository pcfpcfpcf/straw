import hmac
import inspect
import json
import os

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from .agent import Agent, execute_tool
from .chats import Chats
from .types import *


def _title(messages: list) -> str:
    return next((m.get("content", "") for m in messages if m.get("role") == "user"), "")


class Domain:
    def __init__(self, name, seed_prompt, db_conn=None, agent_options=None):
        self.name = name
        self.db_conn = db_conn
        self.agent_options = agent_options or {} # per-domain Agent budgets (max_steps, max_tool_calls, ...)
        self.chats = Chats()

        self.reducers = {}
        self.miscs = {}

        self.tools_reg = [] #tool registry

        self._compact_cache = {} # chat_id -> [n_compacted, digest_list]; incremental compact

        self.system_prompt = seed_prompt
        
        self.auth_token = os.getenv("STRAW_TOKEN", "")
        self.app = FastAPI(title=f"Straw Harness: {self.name}",
                           dependencies=[Depends(self._auth)])
        self._setup_routes()

    def _auth(self, authorization: str = Header(default="")):
        if not hmac.compare_digest(authorization.removeprefix("Bearer ").strip(), self.auth_token):
            raise HTTPException(status_code=401, detail="bad or missing service token")

    def reducer(self, name):
        def decorator(func):
            self.reducers[name] = func
            return func
        
        return decorator

    def misc(self, name):
        def decorator(func):
            self.miscs[name] = func
            return func
        
        return decorator

    def tool(self, name: str, description: str, args: Dict[str, str] = None,
             schemas: Dict[str, dict] = None, read_only: bool = False, ends_turn: bool = False):
        if args is None:
            args = {}

        def decorator(func: Callable[..., Any]):
            tool_instance = Tool(
                name=name,
                description=description,
                func=func,
                args=args,
                schemas=schemas or {},
                read_only=read_only,
                ends_turn=ends_turn
            )
            
            self.tools_reg.append(tool_instance)
            return func 

        return decorator 
    
    
    def _setup_routes(self):
        @self.app.post("/reduce")
        async def post_reduce(request: Request):
            raw_state = await request.json()

            reduced_snapshot = {}
            for name, func in self.reducers.items():
                if len(inspect.signature(func).parameters) >= 2: # incase there's config
                    config = raw_state.get("config")
                    reduced_snapshot[name] = func(raw_state, config)
                else:
                    reduced_snapshot[name] = func(raw_state)

            return {"status": "reduced", "output": reduced_snapshot}
        
        @self.app.post("/misc/{misc_func}")
        async def post_miscs(misc_func, request: Request):
            if misc_func not in self.miscs:
                return {"status": "error", "reason": f"{misc_func} function is not implemented"}

            func = self.miscs[misc_func]
            body = await request.json()

            try:
                bound = inspect.signature(func).bind(**body)
            except TypeError as e:
                return {"status": "error", "reason": f"{misc_func}{inspect.signature(func)}: {e}"}

            return {"status": f"{misc_func}'ed", "output": func(*bound.args, **bound.kwargs)}
        
        @self.app.post("/tool/{tool_name}")
        async def post_tools(tool_name, request: Request):
            tool = Agent.find_tool(self.tools_reg, tool_name)
            if not tool:
                return {"status": "error", "reason": f"{tool_name} tool is not implemented"}

            body = await request.json()
            args, ctx = body.get("args", {}), body.get("ctx", {})

            return {"status": f"{tool.name}'ed",
                    "output": execute_tool(tool, args, self._with_db(ctx))}

        @self.app.post("/agent")
        async def post_agent(request: Request):
            body = await request.json()
            llm = body.get("llm", {})
            incoming = body.get("messages", [])
            persist = body.get("persist", True)

            if persist:
                chat_id = body.get("chat_id") or self.chats.create(_title(incoming))
                self.chats.append(chat_id, incoming)
                history = self.chats.load(chat_id)
            else:
                chat_id, history = None, incoming

            agent = Agent(
                model=llm.get("model", ""),
                api_url=llm.get("api_url", ""),
                api_key=llm.get("api_key", ""),
                tools_reg=self.tools_reg,
                system_prompt=self.system_prompt,
                **self.agent_options,
            )

            prior, current = history[:-len(incoming)], history[-len(incoming):]
            if prior: #earlier turns ride along as a digest, raw tool output stays out
                digest = self._digest(chat_id, prior)
                history = [{"role": "user", "content": json.dumps(digest)}] + current

            steps = agent.run(history, self._with_db(body.get("ctx", {})))

            # Reasoning Steps stream as they happen; "final" is persisted, not streamed
            def stream():
                if chat_id:
                    yield f"data: {json.dumps({'type': 'chat', 'chat_id': chat_id})}\n\n"
                for step in steps:
                    if step["type"] == "final":
                        if chat_id:
                            self.chats.append(chat_id, step["messages"])
                        continue
                    yield f"data: {json.dumps(step)}\n\n"

            return StreamingResponse(stream(), media_type="text/event-stream")

        @self.app.get("/chats")
        async def get_chats():
            return self.chats.list()

        @self.app.get("/chats/{chat_id}")
        async def get_chat(chat_id: str):
            return self.chats.load(chat_id)

        @self.app.get("/compact/{chat_id}")
        async def get_compact_chat(chat_id: str):
            return json.dumps(Agent.compact(self.chats.load(chat_id)))

    def _with_db(self, ctx: dict) -> dict:
        ctx["db"] = self.db_conn
        return ctx

    def _digest(self, chat_id, prior):
        # Incremental compact: reuse the cached digest and only compact the
        # new turn(s) appended since. prior is always turn-aligned (history
        # minus the current incoming turn), so slicing prior[cached_n:] yields
        # whole turns whose tool calls + results travel together.
        n = len(prior)
        if not n:
            return []
        if chat_id is None:
            return Agent.compact(prior)
        cached = self._compact_cache.get(chat_id)
        if cached and cached[0] == n:
            return cached[1]
        if cached and cached[0] < n:
            digest = cached[1] + Agent.compact(prior[cached[0]:])
        else: # miss or stale (prior shrank): recompute from scratch
            digest = Agent.compact(prior)
        self._compact_cache[chat_id] = [n, digest]
        return digest


    def run(self, port: int = 7777):
        # Fail closed: the harness reaches the domain's data and trusts its caller.
        if not self.auth_token:
            raise SystemExit("STRAW_TOKEN is required")

        print(f"Booting straw harness for domain: {self.name}")
        print(f"Registered reducers: {list(self.reducers.keys())}")
        print(f"Registered miscs: {list(self.miscs.keys())}")

        uvicorn.run(self.app, host="0.0.0.0", port=port)
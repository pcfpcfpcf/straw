
from dataclasses import dataclass, field
from typing import Callable, Any, Dict
import inspect

JSON_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object"
}
    
@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any] = None
    args: Dict[str, str] = field(default_factory=dict)
    schemas: Dict[str, dict] = field(default_factory=dict) # per-arg JSON schema overrides
    read_only: bool = False # changes nothing: consecutive read-only calls of a step run at the same time
    ends_turn: bool = False # a result that isn't an error is the turn's answer, no model call after it

    def schema(self):
        sig = inspect.signature(self.func)
        props, required = {}, []

        for pname, p in sig.parameters.items():
            if pname == "ctx": #context is not an explicit tool arg
                continue

            props[pname] = {"type": JSON_TYPES.get(p.annotation, "string"),
                            "description": self.args.get(pname, "")}
            if props[pname]["type"] == "array": # some providers (gemini) reject arrays without items
                props[pname]["items"] = {"type": "string"}
            props[pname].update(self.schemas.get(pname, {}))

            if p.default is inspect.Parameter.empty:
                required.append(pname)

        return {"type": "function", "function": {"name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": props, "required": required}}}    
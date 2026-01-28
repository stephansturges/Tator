from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from qwen_agent.tools.base import BaseTool, ToolServiceError


def _normalize_tool_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(spec, dict) and isinstance(spec.get("function"), dict):
        return spec["function"]
    return spec


def _lazy_localinferenceapi():
    import importlib

    return importlib.import_module("localinferenceapi")


class LocalAgentTool(BaseTool):
    def __init__(self, spec: Dict[str, Any]):
        spec = _normalize_tool_spec(spec)
        self.name = str(spec.get("name") or "")
        self.description = str(spec.get("description") or "")
        params = spec.get("parameters") or {}
        if isinstance(params, dict):
            params = dict(params)
            if "required" not in params:
                params["required"] = []
            if "type" not in params:
                params["type"] = "object"
            if "properties" not in params:
                params["properties"] = {}
        self.parameters = params
        super().__init__(cfg=None)

    def call(self, params: Union[str, dict], **kwargs) -> Union[str, list, dict]:
        params_json = self._verify_json_format_args(params, strict_json=False)
        local = _lazy_localinferenceapi()
        try:
            call = local.AgentToolCall(name=self.name, arguments=params_json, call_id=None)
            result = local._dispatch_agent_tool(call)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"tool_failed:{exc}"}
        if result.error:
            return {"error": result.error}
        return result.result or {}


def build_local_agent_tools(tool_specs: Optional[Sequence[Dict[str, Any]]] = None) -> List[BaseTool]:
    local = _lazy_localinferenceapi()
    specs = list(tool_specs or local._agent_tool_specs())
    return [LocalAgentTool(spec) for spec in specs]

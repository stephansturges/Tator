from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from qwen_agent.tools.base import BaseTool, ToolServiceError


def _lazy_localinferenceapi():
    import importlib

    return importlib.import_module("localinferenceapi")


class LocalAgentTool(BaseTool):
    def __init__(self, spec: Dict[str, Any]):
        self.name = str(spec.get("name") or "")
        self.description = str(spec.get("description") or "")
        self.parameters = spec.get("parameters") or {}
        super().__init__(cfg=None)

    def call(self, params: Union[str, dict], **kwargs) -> Union[str, list, dict]:
        params_json = self._verify_json_format_args(params, strict_json=False)
        local = _lazy_localinferenceapi()
        try:
            call = local.AgentToolCall(name=self.name, arguments=params_json, call_id=None)
            result = local._dispatch_agent_tool(call)
        except Exception as exc:  # noqa: BLE001
            raise ToolServiceError(exception=exc) from exc
        if result.error:
            raise ToolServiceError(code="tool_failed", message=result.error)
        return result.result or {}


def build_local_agent_tools(tool_specs: Optional[Sequence[Dict[str, Any]]] = None) -> List[BaseTool]:
    local = _lazy_localinferenceapi()
    specs = list(tool_specs or local._agent_tool_specs())
    return [LocalAgentTool(spec) for spec in specs]

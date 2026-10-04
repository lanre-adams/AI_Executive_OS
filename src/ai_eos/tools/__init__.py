"""Tool plug-ins. Add your own by registering a Tool subclass or listing a module in tools.plugins."""

from ai_eos.config import Settings
from ai_eos.tools.base import FunctionTool, Tool, ToolContext, ToolError, ToolExecutor, ToolRegistry, ToolSpec
from ai_eos.tools.core import CORE_TOOLS
from ai_eos.tools.devtools import DEV_TOOLS
from ai_eos.tools.google import GOOGLE_TOOLS


def build_registry(settings: Settings) -> ToolRegistry:
    registry = ToolRegistry()
    for cls in [*CORE_TOOLS, *DEV_TOOLS, *GOOGLE_TOOLS]:
        registry.register(cls())
    registry.load_plugins(settings.tools.plugins, settings)
    return registry


__all__ = [
    "FunctionTool",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolExecutor",
    "ToolRegistry",
    "ToolSpec",
    "build_registry",
]

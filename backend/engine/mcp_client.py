"""MCP (Model Context Protocol) 客户端 — 对接外部工具/数据源

设计理念:
  "MCP 是 Agent 工具生态的 USB 协议 — 插上就能用"
  "通过 MCP 连接外部医学数据库、药品 API、实验室系统"

参考架构:
  - Anthropic MCP Specification: JSON-RPC 2.0 over stdio/SSE
  - OpenAI Function Calling: MCP tools 自动转换为 Function Calling schema
  - Letta MCP Tools: 通过 MCP 协议注册外部工具

MCP 协议核心概念:
  - Server: 提供工具/资源的进程 (如药品数据库、PubMed API)
  - Client: 连接 Server 的 Agent (本系统)
  - Transport: stdio (本地进程) 或 SSE (远程服务)

工具注册流程:
  1. 连接 MCP Server → 2. 列出可用工具 → 3. 转换为 OpenAI Function Schema
  4. 注册到 ToolRegistry → 5. Agent ReAct 循环中可直接调用
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """MCP 服务器配置"""
    name: str
    command: str = ""
    args: list[str] = field(default_factory=list)
    url: str = ""
    enabled: bool = True


@dataclass
class MCPTool:
    """MCP 工具定义"""
    name: str
    description: str
    parameters: dict = field(default_factory=dict)
    server_name: str = ""


class MCPClient:
    """MCP 协议客户端 — 连接外部工具服务器

    支持两种传输方式:
    - stdio: 启动本地子进程，通过标准输入输出通信 (适合本地工具)
    - SSE (Server-Sent Events): 连接远程 HTTP 服务 (适合云服务)

    用法:
        client = MCPClient()
        await client.connect_stdio("drug_db", "python", ["drug_server.py"])
        tools = await client.list_tools("drug_db")
        result = await client.call_tool("drug_db", "search_drug", {"name": "ibuprofen"})
    """

    def __init__(self):
        self._servers: dict[str, dict] = {}
        self._tools: dict[str, MCPTool] = {}

    async def connect_stdio(self, name: str, command: str, args: list[str] = None) -> bool:
        """通过 stdio 连接 MCP Server (本地子进程)

        启动子进程并通过 stdin/stdout 进行 JSON-RPC 通信。
        """
        try:
            args = args or []
            proc = await asyncio.create_subprocess_exec(
                command, *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._servers[name] = {
                "type": "stdio",
                "process": proc,
                "command": command,
                "args": args,
            }

            init_result = await self._stdio_rpc(proc, "initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "mdt-agent", "version": "1.0"},
            })
            if init_result:
                logger.info(f"MCP stdio connected: {name} ({command})")
                await self._stdio_send(proc, "notifications/initialized", {})
                await self._discover_tools(name)
                return True
            return False
        except Exception as e:
            logger.error(f"MCP stdio connect failed [{name}]: {e}")
            return False

    async def connect_sse(self, name: str, url: str) -> bool:
        """通过 SSE 连接 MCP Server (远程 HTTP 服务)"""
        import httpx
        try:
            self._servers[name] = {
                "type": "sse",
                "url": url,
            }
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(f"{url}/initialize", json={
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "mdt-agent", "version": "1.0"},
                    },
                    "id": 1,
                })
                if r.status_code == 200:
                    logger.info(f"MCP SSE connected: {name} ({url})")
                    await self._discover_tools(name)
                    return True
            return False
        except Exception as e:
            logger.error(f"MCP SSE connect failed [{name}]: {e}")
            return False

    async def list_tools(self, server_name: str) -> list[MCPTool]:
        """获取 MCP Server 提供的工具列表"""
        return [t for t in self._tools.values() if t.server_name == server_name]

    async def list_all_tools(self) -> list[MCPTool]:
        """获取所有已连接 MCP Server 的工具"""
        return list(self._tools.values())

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> str:
        """调用 MCP 工具"""
        server = self._servers.get(server_name)
        if not server:
            return json.dumps({"error": f"MCP server not found: {server_name}"})

        params = {"name": tool_name, "arguments": arguments}

        if server["type"] == "stdio":
            result = await self._stdio_rpc(server["process"], "tools/call", params)
            if result and "content" in result:
                contents = result["content"]
                if contents and isinstance(contents, list):
                    return "\n".join(c.get("text", str(c)) for c in contents)
            return json.dumps(result or {"error": "no result"})

        elif server["type"] == "sse":
            import httpx
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    r = await client.post(f"{server['url']}/tools/call", json={
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": params,
                        "id": 2,
                    })
                    if r.status_code == 200:
                        data = r.json()
                        result = data.get("result", {})
                        contents = result.get("content", [])
                        if contents:
                            return "\n".join(c.get("text", str(c)) for c in contents)
                        return json.dumps(result)
                    return json.dumps({"error": f"HTTP {r.status_code}"})
            except Exception as e:
                return json.dumps({"error": str(e)})

        return json.dumps({"error": "unknown transport"})

    async def disconnect(self, server_name: str):
        """断开 MCP Server 连接"""
        server = self._servers.pop(server_name, None)
        if server and server["type"] == "stdio" and server.get("process"):
            proc = server["process"]
            try:
                proc.terminate()
                await proc.wait()
            except Exception:
                pass
        self._tools = {k: v for k, v in self._tools.items() if v.server_name != server_name}
        logger.info(f"MCP disconnected: {server_name}")

    async def disconnect_all(self):
        """断开所有 MCP 连接"""
        for name in list(self._servers.keys()):
            await self.disconnect(name)

    async def _discover_tools(self, server_name: str):
        """发现 MCP Server 提供的工具"""
        server = self._servers.get(server_name)
        if not server:
            return

        if server["type"] == "stdio":
            result = await self._stdio_rpc(server["process"], "tools/list", {})
        elif server["type"] == "sse":
            import httpx
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(f"{server['url']}/tools/list", json={
                    "jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 1,
                })
                result = r.json().get("result", {}) if r.status_code == 200 else {}
        else:
            return

        tools_data = result.get("tools", []) if result else []
        for td in tools_data:
            tool = MCPTool(
                name=td.get("name", ""),
                description=td.get("description", ""),
                parameters=td.get("inputSchema", {}),
                server_name=server_name,
            )
            self._tools[f"{server_name}:{tool.name}"] = tool
            logger.info(f"MCP tool discovered: {server_name}/{tool.name}")

    async def _stdio_rpc(self, proc, method: str, params: dict) -> Optional[dict]:
        """通过 stdio 发送 JSON-RPC 请求并读取响应"""
        try:
            request = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
                "id": 1,
            }
            payload = json.dumps(request) + "\n"
            proc.stdin.write(payload.encode())
            await proc.stdin.drain()

            line = await asyncio.wait_for(proc.stdout.readline(), timeout=30)
            if line:
                return json.loads(line.decode())
        except asyncio.TimeoutError:
            logger.warning(f"MCP RPC timeout: {method}")
        except Exception as e:
            logger.warning(f"MCP RPC error [{method}]: {e}")
        return None

    async def _stdio_send(self, proc, method: str, params: dict):
        """通过 stdio 发送 JSON-RPC 通知（无响应）"""
        try:
            notification = {"jsonrpc": "2.0", "method": method, "params": params}
            payload = json.dumps(notification) + "\n"
            proc.stdin.write(payload.encode())
            await proc.stdin.drain()
        except Exception as e:
            logger.warning(f"MCP notification error: {e}")

    def get_tool_schemas(self) -> list[dict]:
        """将所有 MCP 工具转换为 OpenAI Function Calling 格式"""
        schemas = []
        for tool in self._tools.values():
            schemas.append({
                "name": f"mcp_{tool.server_name}_{tool.name}",
                "description": f"[MCP:{tool.server_name}] {tool.description}",
                "parameters": tool.parameters or {
                    "type": "object",
                    "properties": {},
                },
            })
        return schemas

    def stats(self) -> dict:
        return {
            "connected_servers": len(self._servers),
            "server_names": list(self._servers.keys()),
            "total_tools": len(self._tools),
            "tools_by_server": {
                name: sum(1 for t in self._tools.values() if t.server_name == name)
                for name in self._servers
            },
        }

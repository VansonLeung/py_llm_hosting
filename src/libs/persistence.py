import json
import os
from typing import Dict, List, Optional
from pydantic import BaseModel

from ..models.server import LLMServer
from ..models.mcp import MCPContext

class Persistence:
    def __init__(self, data_file: str = "servers.json"):
        self.data_file = data_file
        self._ensure_data_file()

    def _ensure_data_file(self):
        if not os.path.exists(self.data_file):
            self._save_data({"servers": [], "mcp_sessions": []})

    def _load_data(self) -> Dict:
        try:
            with open(self.data_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"servers": [], "mcp_sessions": []}

    def _save_data(self, data: Dict):
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def get_servers(self) -> List[LLMServer]:
        data = self._load_data()
        return [LLMServer(**server) for server in data.get("servers", [])]

    def add_server(self, server: LLMServer):
        data = self._load_data()
        servers = data.get("servers", [])
        # Check unique name
        if any(s["name"] == server.name for s in servers):
            raise ValueError(f"Server with name '{server.name}' already exists")
        servers.append(server.model_dump())
        data["servers"] = servers
        self._save_data(data)

    def remove_server(self, server_id: str):
        data = self._load_data()
        servers = data.get("servers", [])
        servers = [s for s in servers if s["id"] != server_id]
        data["servers"] = servers
        self._save_data(data)
    
    def update_server(self, server: LLMServer):
        """Update an existing server."""
        data = self._load_data()
        servers = data.get("servers", [])
        for i, s in enumerate(servers):
            if s["id"] == server.id:
                servers[i] = server.model_dump()
                break
        else:
            raise ValueError(f"Server with id '{server.id}' not found")
        data["servers"] = servers
        self._save_data(data)

    def get_server_by_id(self, server_id: str) -> Optional[LLMServer]:
        servers = self.get_servers()
        return next((s for s in servers if s.id == server_id), None)

    def get_mcp_sessions(self) -> List[MCPContext]:
        data = self._load_data()
        return [MCPContext(**session) for session in data.get("mcp_sessions", [])]

    def add_mcp_session(self, session: MCPContext):
        data = self._load_data()
        sessions = data.get("mcp_sessions", [])
        sessions.append(session.model_dump())
        data["mcp_sessions"] = sessions
        self._save_data(data)

    def update_mcp_session(self, session: MCPContext):
        data = self._load_data()
        sessions = data.get("mcp_sessions", [])
        for i, s in enumerate(sessions):
            if s["session_id"] == session.session_id:
                sessions[i] = session.model_dump()
                break
        data["mcp_sessions"] = sessions
        self._save_data(data)

    def remove_mcp_session(self, session_id: str):
        data = self._load_data()
        sessions = data.get("mcp_sessions", [])
        sessions = [s for s in sessions if s["session_id"] != session_id]
        data["mcp_sessions"] = sessions
        self._save_data(data)
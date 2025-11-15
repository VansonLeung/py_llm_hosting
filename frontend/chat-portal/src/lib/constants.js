export const DEFAULT_ENDPOINT = {
  id: "local-openai",
  name: "Local LLM Gateway",
  baseUrl: "http://localhost:8000/v1",
  apiKey: "sk-local-placeholder",
  models: ["gpt-4o-mini"],
  description: "Default endpoint that points to the local py_llm_hosting instance.",
  supportsVision: true,
  supportsTools: true,
  enabled: true,
  createdAt: new Date().toISOString(),
}

export const STORAGE_KEYS = {
  AUTH: "llm-ui-auth",
  ENDPOINTS: "llm-ui-endpoints",
  CONVERSATIONS: "llm-ui-conversations",
  TOOLS: "llm-ui-tools",
}

export const DEFAULT_TOOLS = [
  {
    id: "web-search",
    name: "WebSearch",
    description: "Search the web for up-to-date information",
    schema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query" },
      },
      required: ["query"],
    },
    createdAt: new Date().toISOString(),
  },
]

export const DEFAULT_MCP_TOOLS = [
  {
    id: "fs-read",
    name: "Filesystem.Read",
    description: "Read files from the managed workspace",
    schema: {
      type: "object",
      properties: {
        path: { type: "string", description: "Absolute file path" },
      },
      required: ["path"],
    },
    connection: "local",
    createdAt: new Date().toISOString(),
  },
]

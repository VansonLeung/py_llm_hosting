import { create } from "zustand"
import { persist } from "zustand/middleware"
import { browserStorage } from "@/lib/storage"
import { DEFAULT_MCP_TOOLS, DEFAULT_TOOLS, STORAGE_KEYS } from "@/lib/constants"

const buildTool = (payload) => ({
  id: payload.id || crypto.randomUUID(),
  name: payload.name || "Untitled Tool",
  description: payload.description || "",
  schema: payload.schema || { type: "object", properties: {}, required: [] },
  connection: payload.connection || null,
  createdAt: payload.createdAt || new Date().toISOString(),
})

export const useToolStore = create(
  persist(
    (set, get) => ({
      tools: DEFAULT_TOOLS,
      mcpTools: DEFAULT_MCP_TOOLS,
      addTool(payload, { isMcp = false } = {}) {
        const entry = buildTool(payload)
        if (isMcp) {
          set((state) => ({ mcpTools: [entry, ...state.mcpTools] }))
        } else {
          set((state) => ({ tools: [entry, ...state.tools] }))
        }
        return entry
      },
      updateTool(id, updates, { isMcp = false } = {}) {
        set((state) => ({
          [isMcp ? "mcpTools" : "tools"]: (isMcp ? state.mcpTools : state.tools).map((tool) =>
            tool.id === id ? { ...tool, ...updates } : tool
          ),
        }))
      },
      removeTool(id, { isMcp = false } = {}) {
        set((state) => ({
          [isMcp ? "mcpTools" : "tools"]: (isMcp ? state.mcpTools : state.tools).filter(
            (tool) => tool.id !== id
          ),
        }))
      },
    }),
    {
      name: STORAGE_KEYS.TOOLS,
      storage: browserStorage(),
    }
  )
)

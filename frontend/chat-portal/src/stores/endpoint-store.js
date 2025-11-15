import { create } from "zustand"
import { persist } from "zustand/middleware"
import { browserStorage } from "@/lib/storage"
import { DEFAULT_ENDPOINT, STORAGE_KEYS } from "@/lib/constants"

const buildEndpoint = (payload = {}) => ({
  id: payload.id || crypto.randomUUID(),
  name: payload.name || "Untitled Endpoint",
  baseUrl: payload.baseUrl || "",
  apiKey: payload.apiKey || "",
  models: payload.models || ["gpt-4o-mini"],
  supportsVision: payload.supportsVision ?? true,
  supportsTools: payload.supportsTools ?? true,
  notes: payload.notes || "",
  createdAt: payload.createdAt || new Date().toISOString(),
})

const initialState = {
  endpoints: [DEFAULT_ENDPOINT],
  activeEndpointId: DEFAULT_ENDPOINT.id,
}

export const useEndpointStore = create(
  persist(
    (set, get) => ({
      ...initialState,
      addEndpoint(payload) {
        const endpoint = buildEndpoint(payload)
        set((state) => ({
          endpoints: [endpoint, ...state.endpoints],
          activeEndpointId: endpoint.id,
        }))
        return endpoint
      },
      updateEndpoint(id, updates) {
        set((state) => ({
          endpoints: state.endpoints.map((endpoint) =>
            endpoint.id === id ? { ...endpoint, ...updates } : endpoint
          ),
        }))
      },
      removeEndpoint(id) {
        set((state) => {
          const filtered = state.endpoints.filter((endpoint) => endpoint.id !== id)
          const fallback = filtered[0] || buildEndpoint(DEFAULT_ENDPOINT)
          return {
            endpoints: filtered.length ? filtered : [fallback],
            activeEndpointId:
              state.activeEndpointId === id ? fallback.id : state.activeEndpointId,
          }
        })
      },
      setActiveEndpoint(id) {
        set({ activeEndpointId: id })
      },
    }),
    {
      name: STORAGE_KEYS.ENDPOINTS,
      storage: browserStorage(),
    }
  )
)

export const selectActiveEndpoint = (state) =>
  state.endpoints.find((endpoint) => endpoint.id === state.activeEndpointId) || null

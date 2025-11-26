import { create } from "zustand"
import { persist } from "zustand/middleware"
import { browserStorage } from "@/lib/storage"
import { DEFAULT_ENDPOINT, STORAGE_KEYS } from "@/lib/constants"

const buildEndpoint = (payload = {}) => ({
  id: payload.id || crypto.randomUUID(),
  name: payload.name || "Untitled Endpoint",
  baseUrl: payload.baseUrl || "",
  apiKey: payload.apiKey || "",
  model: payload.model || payload.models?.[0] || "gpt-4o-mini",
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
    (set) => ({
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
      version: 2,
      migrate: (persistedState, version) => {
        if (!persistedState) return persistedState
        if (version < 2 && Array.isArray(persistedState.endpoints)) {
          return {
            ...persistedState,
            endpoints: persistedState.endpoints.map((endpoint) => ({
              ...endpoint,
              model: endpoint.model || endpoint.models?.[0] || "gpt-4o-mini",
            })),
          }
        }
        return persistedState
      },
    }
  )
)

export const selectActiveEndpoint = (state) =>
  state.endpoints.find((endpoint) => endpoint.id === state.activeEndpointId) || null

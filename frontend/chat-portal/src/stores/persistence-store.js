import { create } from "zustand"
import { persist } from "zustand/middleware"
import { browserStorage } from "@/lib/storage"
import { STORAGE_KEYS } from "@/lib/constants"

export const PERSISTENCE_MODES = {
  LOCAL: "local",
  BACKEND: "backend",
}

const DEFAULT_MODE = import.meta.env.VITE_DEFAULT_PERSISTENCE_MODE || PERSISTENCE_MODES.LOCAL
const DEFAULT_BACKEND_URL = import.meta.env.VITE_BACKEND_API_BASE_URL || "http://localhost:5278"

export const usePersistenceStore = create(
  persist(
    (set) => ({
      mode: DEFAULT_MODE,
      backendBaseUrl: DEFAULT_BACKEND_URL,
      lastSync: null,
      syncError: null,
      setMode(mode) {
        set({ mode })
      },
      setBackendBaseUrl(url) {
        set({ backendBaseUrl: url })
      },
      markSynced() {
        set({ lastSync: Date.now(), syncError: null })
      },
      setSyncError(error) {
        set({ syncError: error })
      },
    }),
    {
      name: STORAGE_KEYS.PERSISTENCE,
      storage: browserStorage(),
    }
  )
)

export const isBackendModeEnabled = () => usePersistenceStore.getState().mode === PERSISTENCE_MODES.BACKEND

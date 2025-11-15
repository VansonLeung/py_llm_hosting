import { create } from "zustand"
import { persist } from "zustand/middleware"
import { browserStorage } from "@/lib/storage"
import { hashSecret } from "@/lib/crypto"
import { STORAGE_KEYS } from "@/lib/constants"

const initialState = {
  users: [],
  currentUserId: null,
}

export const useAuthStore = create(
  persist(
    (set, get) => ({
      ...initialState,
      async register({ username, password }) {
        if (!username || !password) {
          return { success: false, error: "Username and password are required" }
        }
        const normalized = username.trim().toLowerCase()
        const existing = get().users.find((user) => user.username === normalized)
        if (existing) {
          return { success: false, error: "Username already exists" }
        }
        const passwordHash = await hashSecret(password)
        const newUser = {
          id: crypto.randomUUID(),
          username: normalized,
          passwordHash,
          createdAt: new Date().toISOString(),
        }
        set((state) => ({
          users: [...state.users, newUser],
          currentUserId: newUser.id,
        }))
        return { success: true, user: newUser }
      },
      async login({ username, password }) {
        if (!username || !password) {
          return { success: false, error: "Missing credentials" }
        }
        const normalized = username.trim().toLowerCase()
        const user = get().users.find((item) => item.username === normalized)
        if (!user) {
          return { success: false, error: "Account not found" }
        }
        const hashed = await hashSecret(password)
        if (hashed !== user.passwordHash) {
          return { success: false, error: "Invalid password" }
        }
        set({ currentUserId: user.id })
        return { success: true, user }
      },
      logout() {
        set({ currentUserId: null })
      },
    }),
    {
      name: STORAGE_KEYS.AUTH,
      storage: browserStorage(),
    }
  )
)

export const selectCurrentUser = (state) =>
  state.users.find((user) => user.id === state.currentUserId) || null

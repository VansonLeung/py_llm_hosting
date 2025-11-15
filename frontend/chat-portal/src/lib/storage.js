import { createJSONStorage } from "zustand/middleware"

const noopStorage = {
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {},
}

export const browserStorage = () =>
  createJSONStorage(() => {
    if (typeof window === "undefined") return noopStorage
    return window.localStorage
  })

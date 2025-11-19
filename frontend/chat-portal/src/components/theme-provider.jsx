import { useEffect, useMemo, useState } from "react"
import { ThemeContext } from "@/components/theme-context"

export function ThemeProvider({ children, defaultTheme = "dark", storageKey = "llm-ui-theme" }) {
  const [theme, setTheme] = useState(() => {
    if (typeof window === "undefined") return defaultTheme
    return localStorage.getItem(storageKey) || defaultTheme
  })

  useEffect(() => {
    const root = document.documentElement
    root.classList.remove("light", "dark")
    root.classList.add(theme)
    localStorage.setItem(storageKey, theme)
  }, [theme, storageKey])

  const value = useMemo(() => ({ theme, setTheme }), [theme])

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

import { useEffect } from "react"
import AuthGate from "@/features/auth/AuthGate"
import { AppShell } from "@/components/layout/app-shell"
import { ConversationPanel } from "@/features/conversations/ConversationPanel"
import { EndpointPanel } from "@/features/endpoints/EndpointPanel"
import { ToolPanel } from "@/features/tools/ToolPanel"
import { ChatWorkspace } from "@/features/chat/ChatWorkspace"
import { useAuthStore, selectCurrentUser } from "@/stores/auth-store"
import { useConversationStore } from "@/stores/conversation-store"
import { PERSISTENCE_MODES, usePersistenceStore } from "@/stores/persistence-store"

function App() {
  const user = useAuthStore(selectCurrentUser)
  const logout = useAuthStore((state) => state.logout)
  const mode = usePersistenceStore((state) => state.mode)
  const backendBaseUrl = usePersistenceStore((state) => state.backendBaseUrl)
  const hydrateFromBackend = useConversationStore((state) => state.hydrateFromBackend)

  useEffect(() => {
    if (mode !== PERSISTENCE_MODES.BACKEND) return
    hydrateFromBackend()
  }, [mode, backendBaseUrl, hydrateFromBackend])

  const sidebarContent = (
    <div className="flex flex-col gap-6">
      <div className="rounded-2xl border bg-card/40 p-4">
        <ConversationPanel />
      </div>
      <div className="rounded-2xl border bg-card/40 p-4">
        <EndpointPanel />
      </div>
      <div className="rounded-2xl border bg-card/40 p-4">
        <ToolPanel />
      </div>
    </div>
  )

  return (
    <AuthGate>
      <AppShell user={user} onLogout={logout} sidebar={{ content: sidebarContent }}>
        <ChatWorkspace />
      </AppShell>
    </AuthGate>
  )
}

export default App

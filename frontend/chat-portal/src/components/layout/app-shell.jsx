import { ModeToggle } from "@/components/mode-toggle"
import { PersistenceModeToggle } from "@/components/persistence-mode-toggle"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { cn } from "@/lib/utils"

export function AppShell({ user, onLogout, sidebar, children }) {
  const initials = user?.username?.slice(0, 2).toUpperCase()

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-background text-foreground">
      <header className="flex-none border-b bg-card/30 backdrop-blur">
        <div className="flex w-full items-center justify-between px-6 py-4">
          <div>
            <p className="text-lg font-semibold">LLM Control Center</p>
            <p className="text-sm text-muted-foreground">Manage conversations, models, tools, and MCP connectors</p>
          </div>
          <div className="flex flex-wrap items-center justify-end gap-4">
            <PersistenceModeToggle />
            <ModeToggle />
            <div className="flex items-center gap-2 rounded-full border px-3 py-1">
              <Avatar className="h-8 w-8">
                <AvatarFallback>{initials}</AvatarFallback>
              </Avatar>
              <span className="text-sm font-medium">{user?.username}</span>
            </div>
            <Button variant="outline" onClick={onLogout} size="sm">
              Logout
            </Button>
          </div>
        </div>
      </header>
      <main className="flex flex-1 overflow-hidden">
        <aside className={cn("hidden w-80 flex-col gap-4 overflow-y-auto border-r bg-muted/10 p-4 lg:flex", sidebar?.className)}>
          {sidebar?.content}
        </aside>
        <section className="flex flex-1 flex-col overflow-hidden p-4">{children}</section>
      </main>
      <div className="flex-none border-t bg-background p-4 lg:hidden">
        <div className="max-h-48 overflow-y-auto rounded-xl border bg-card/50 p-4">{sidebar?.content}</div>
      </div>
    </div>
  )
}

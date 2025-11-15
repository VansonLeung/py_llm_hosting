import { ModeToggle } from "@/components/mode-toggle"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { cn } from "@/lib/utils"

export function AppShell({ user, onLogout, sidebar, children }) {
  const initials = user?.username?.slice(0, 2).toUpperCase()

  return (
    <div className="flex min-h-screen flex-col bg-background text-foreground">
      <header className="border-b bg-card/30 backdrop-blur">
        <div className="mx-auto flex w-full max-w-7xl items-center justify-between px-6 py-4">
          <div>
            <p className="text-lg font-semibold">LLM Control Center</p>
            <p className="text-sm text-muted-foreground">Manage conversations, endpoints, tools, and MCP connectors</p>
          </div>
          <div className="flex items-center gap-3">
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
      <main className="mx-auto flex w-full max-w-7xl flex-1 gap-6 px-6 py-6">
        <aside className={cn("hidden w-80 flex-shrink-0 flex-col gap-4 lg:flex", sidebar?.className)}>{sidebar?.content}</aside>
        <section className="flex-1">{children}</section>
      </main>
      <div className="mx-auto w-full max-w-4xl px-6 pb-8 lg:hidden">
        <div className="rounded-xl border bg-card/50 p-4">{sidebar?.content}</div>
      </div>
    </div>
  )
}

import { useMemo } from "react"
import { usePersistenceStore, PERSISTENCE_MODES } from "@/stores/persistence-store"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"

export function PersistenceModeToggle() {
  const mode = usePersistenceStore((state) => state.mode)
  const setMode = usePersistenceStore((state) => state.setMode)
  const backendBaseUrl = usePersistenceStore((state) => state.backendBaseUrl)
  const setBackendBaseUrl = usePersistenceStore((state) => state.setBackendBaseUrl)
  const lastSync = usePersistenceStore((state) => state.lastSync)
  const syncError = usePersistenceStore((state) => state.syncError)

  const statusLabel = useMemo(() => {
    if (mode === PERSISTENCE_MODES.LOCAL) {
      return { tone: "secondary", text: "Browser storage" }
    }
    if (syncError) {
      return { tone: "warning", text: "Sync error" }
    }
    if (lastSync) {
      return { tone: "success", text: "Synced" }
    }
    return { tone: "outline", text: "Awaiting sync" }
  }, [mode, lastSync, syncError])

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between gap-3">
        <Label className="text-xs uppercase tracking-wide text-muted-foreground">Persistence</Label>
        <Badge variant={statusLabel.tone}>{statusLabel.text}</Badge>
      </div>
      <Select value={mode} onValueChange={setMode}>
        <SelectTrigger className="h-9 w-48">
          <SelectValue placeholder="Storage mode" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={PERSISTENCE_MODES.LOCAL}>Local storage</SelectItem>
          <SelectItem value={PERSISTENCE_MODES.BACKEND}>Backend API</SelectItem>
        </SelectContent>
      </Select>
      {mode === PERSISTENCE_MODES.BACKEND && (
        <div className="space-y-1">
          <Label htmlFor="backend-base" className="text-[11px] uppercase tracking-wide text-muted-foreground">
            Backend URL
          </Label>
          <Input
            id="backend-base"
            value={backendBaseUrl}
            onChange={(event) => setBackendBaseUrl(event.target.value)}
            placeholder="http://localhost:5278"
            className="h-9"
          />
          {syncError ? (
            <p className="text-xs text-destructive">{syncError}</p>
          ) : lastSync ? (
            <p className="text-xs text-muted-foreground">Last synced {new Date(lastSync).toLocaleTimeString()}</p>
          ) : (
            <p className="text-xs text-muted-foreground">Enable backend mode to sync conversations remotely.</p>
          )}
        </div>
      )}
    </div>
  )
}

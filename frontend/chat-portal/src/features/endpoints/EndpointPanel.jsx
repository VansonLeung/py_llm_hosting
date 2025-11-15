import { useMemo, useState } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { EndpointForm } from "./EndpointForm"
import { useEndpointStore } from "@/stores/endpoint-store"
import { maskSecret } from "@/lib/crypto"
import { Edit3, Trash2 } from "lucide-react"

export function EndpointPanel() {
  const endpoints = useEndpointStore((state) => state.endpoints)
  const activeEndpointId = useEndpointStore((state) => state.activeEndpointId)
  const setActiveEndpoint = useEndpointStore((state) => state.setActiveEndpoint)
  const addEndpoint = useEndpointStore((state) => state.addEndpoint)
  const updateEndpoint = useEndpointStore((state) => state.updateEndpoint)
  const removeEndpoint = useEndpointStore((state) => state.removeEndpoint)
  const [search, setSearch] = useState("")
  const [dialogOpen, setDialogOpen] = useState(false)
  const [editing, setEditing] = useState(null)

  const filtered = useMemo(() => {
    return endpoints.filter((endpoint) => endpoint.name.toLowerCase().includes(search.toLowerCase()))
  }, [endpoints, search])

  const handleSubmit = (values) => {
    if (editing) {
      updateEndpoint(editing.id, values)
    } else {
      addEndpoint(values)
    }
  }

  const handleDelete = (endpoint) => {
    if (window.confirm(`Remove endpoint "${endpoint.name}"?`)) {
      removeEndpoint(endpoint.id)
    }
  }

  return (
    <div className="flex h-full flex-col gap-4">
      <div className="flex items-center justify-between">
        <p className="text-sm font-semibold uppercase text-muted-foreground">Endpoints</p>
        <Button size="sm" variant="outline" onClick={() => setDialogOpen(true)}>
          Add
        </Button>
      </div>
      <Input placeholder="Search endpoints" value={search} onChange={(event) => setSearch(event.target.value)} />
      <ScrollArea className="flex-1 pr-2">
        <div className="space-y-3">
          {filtered.map((endpoint) => (
            <div key={endpoint.id} className={`rounded-lg border p-3 ${endpoint.id === activeEndpointId ? "border-primary" : "border-border"}`}>
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-semibold">{endpoint.name}</p>
                  <p className="text-xs text-muted-foreground">{endpoint.baseUrl}</p>
                </div>
                <div className="flex items-center gap-2">
                  <Button size="sm" variant={endpoint.id === activeEndpointId ? "default" : "outline"} onClick={() => setActiveEndpoint(endpoint.id)}>
                    {endpoint.id === activeEndpointId ? "Active" : "Select"}
                  </Button>
                  <Button size="icon" variant="ghost" onClick={() => { setEditing(endpoint); setDialogOpen(true) }}>
                    <Edit3 className="h-4 w-4" />
                  </Button>
                  <Button size="icon" variant="ghost" onClick={() => handleDelete(endpoint)}>
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
              <div className="mt-2 flex flex-wrap gap-2 text-xs">
                {endpoint.models?.map((model) => (
                  <Badge key={model} variant="outline">
                    {model}
                  </Badge>
                ))}
              </div>
              <p className="mt-2 text-xs text-muted-foreground">API key: {endpoint.apiKey ? maskSecret(endpoint.apiKey) : "—"}</p>
            </div>
          ))}
          {!filtered.length && <p className="text-sm text-muted-foreground">No endpoints configured.</p>}
        </div>
      </ScrollArea>
      <EndpointForm
        open={dialogOpen}
        onOpenChange={(isOpen) => {
          setDialogOpen(isOpen)
          if (!isOpen) setEditing(null)
        }}
        initialValues={editing || undefined}
        onSubmit={handleSubmit}
      />
    </div>
  )
}

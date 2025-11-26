import { useMemo, useState } from "react"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { useEndpointStore } from "@/stores/endpoint-store"

function ConversationFormFields({ endpoints, initialValues, onSubmit, onCancel }) {
  const fallbackEndpointId = initialValues?.endpointId || endpoints.at(0)?.id || ""
  const resolveModel = (endpointId) => endpoints.find((endpoint) => endpoint.id === endpointId)?.model || endpoints.at(0)?.model || "gpt-4o-mini"

  const [form, setForm] = useState(() => ({
    title: initialValues?.title || "",
    endpointId: fallbackEndpointId,
    model: initialValues?.model || resolveModel(fallbackEndpointId),
  }))

  const currentEndpoint = useMemo(() => endpoints.find((endpoint) => endpoint.id === form.endpointId), [endpoints, form.endpointId])

  const handleSubmit = (event) => {
    event.preventDefault()
    onSubmit?.(form)
  }

  return (
    <form className="space-y-4" onSubmit={handleSubmit}>
      <div className="space-y-2">
        <Label htmlFor="title">Title</Label>
        <Input id="title" value={form.title} placeholder="Quick experiment" onChange={(event) => setForm((prev) => ({ ...prev, title: event.target.value }))} />
      </div>
      <div className="space-y-2">
        <Label>Model</Label>
        <Select
          value={form.endpointId}
          onValueChange={(value) =>
            setForm((prev) => ({
              ...prev,
              endpointId: value,
              model: resolveModel(value),
            }))
          }
        >
          <SelectTrigger>
            <SelectValue placeholder="Select model" />
          </SelectTrigger>
          <SelectContent>
            {endpoints.map((endpoint) => (
              <SelectItem key={endpoint.id} value={endpoint.id}>
                {endpoint.name} · {endpoint.model}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="flex items-center justify-between rounded-md border p-3">
        <div>
          <p className="text-sm font-medium">Vision support</p>
          <p className="text-xs text-muted-foreground">
            {currentEndpoint?.supportsVision ? "Images are enabled for this model." : "Images disabled for this model."}
          </p>
        </div>
        <Switch checked={!!currentEndpoint?.supportsVision} disabled aria-readonly />
      </div>
      <DialogFooter>
        <Button type="button" variant="outline" onClick={onCancel}>
          Cancel
        </Button>
        <Button type="submit">{initialValues ? "Save changes" : "Create"}</Button>
      </DialogFooter>
    </form>
  )
}

export function ConversationForm({ open, onOpenChange, onSubmit, initialValues }) {
  const endpoints = useEndpointStore((state) => state.endpoints)

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{initialValues ? "Edit conversation" : "Create conversation"}</DialogTitle>
          <DialogDescription>Link the conversation to a configured model.</DialogDescription>
        </DialogHeader>
        <ConversationFormFields
          key={initialValues?.id || "create"}
          endpoints={endpoints}
          initialValues={initialValues}
          onSubmit={(values) => {
            onSubmit?.(values)
            onOpenChange(false)
          }}
          onCancel={() => onOpenChange(false)}
        />
      </DialogContent>
    </Dialog>
  )
}

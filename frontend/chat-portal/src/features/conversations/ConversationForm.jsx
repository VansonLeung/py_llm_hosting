import { useMemo, useState } from "react"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useEndpointStore } from "@/stores/endpoint-store"

function ConversationFormFields({ endpoints, initialValues, onSubmit, onCancel }) {
  const [form, setForm] = useState(() => ({
    title: initialValues?.title || "",
    endpointId: initialValues?.endpointId || endpoints.at(0)?.id || "",
    model: initialValues?.model || endpoints.at(0)?.models?.[0] || "gpt-4o-mini",
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
        <Label>Endpoint</Label>
        <Select value={form.endpointId} onValueChange={(value) => setForm((prev) => ({ ...prev, endpointId: value }))}>
          <SelectTrigger>
            <SelectValue placeholder="Select endpoint" />
          </SelectTrigger>
          <SelectContent>
            {endpoints.map((endpoint) => (
              <SelectItem key={endpoint.id} value={endpoint.id}>
                {endpoint.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="space-y-2">
        <Label htmlFor="model">Model identifier</Label>
        <Input
          id="model"
          value={form.model}
          placeholder={currentEndpoint?.models?.[0] || "gpt-4o-mini"}
          onChange={(event) => setForm((prev) => ({ ...prev, model: event.target.value }))}
        />
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
          <DialogDescription>Link the conversation to a target endpoint and specify a default model.</DialogDescription>
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

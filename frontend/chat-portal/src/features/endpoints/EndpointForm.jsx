import { useState } from "react"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"

const defaultForm = {
  name: "",
  baseUrl: "",
  apiKey: "",
  model: "gpt-4o-mini",
  supportsVision: true,
  supportsTools: true,
  notes: "",
}

function EndpointFormFields({ initialValues, onSubmit, onCancel }) {
  const [form, setForm] = useState(() => ({
    ...defaultForm,
    ...initialValues,
  }))

  const handleChange = (event) => {
    const { name, value } = event.target
    setForm((prev) => ({ ...prev, [name]: value }))
  }

  const handleSubmit = (event) => {
    event.preventDefault()
    onSubmit?.({
      ...form,
      model: form.model.trim() || defaultForm.model,
    })
  }

  return (
    <form className="space-y-4" onSubmit={handleSubmit}>
      <div className="space-y-2">
        <Label htmlFor="name">Display name</Label>
        <Input id="name" name="name" value={form.name} onChange={handleChange} placeholder="Local vLLM" required />
      </div>
      <div className="space-y-2">
        <Label htmlFor="baseUrl">Base URL</Label>
        <Input
          id="baseUrl"
          name="baseUrl"
          value={form.baseUrl}
          onChange={handleChange}
          placeholder="http://localhost:8000/v1"
          required
        />
      </div>
      <div className="space-y-2">
        <Label htmlFor="apiKey">API key (optional)</Label>
        <Input id="apiKey" name="apiKey" value={form.apiKey} onChange={handleChange} placeholder="sk-..." />
      </div>
      <div className="space-y-2">
        <Label htmlFor="model">Model identifier</Label>
        <Input id="model" name="model" value={form.model} onChange={handleChange} placeholder="gpt-4o-mini" required />
      </div>
      <div className="flex items-center justify-between rounded-md border p-3">
        <div>
          <p className="text-sm font-medium">Vision support</p>
          <p className="text-xs text-muted-foreground">Allow image inputs for this endpoint.</p>
        </div>
        <Switch checked={form.supportsVision} onCheckedChange={(value) => setForm((prev) => ({ ...prev, supportsVision: value }))} />
      </div>
      <div className="flex items-center justify-between rounded-md border p-3">
        <div>
          <p className="text-sm font-medium">Tool calling</p>
          <p className="text-xs text-muted-foreground">Enable JSON tool payloads.</p>
        </div>
        <Switch checked={form.supportsTools} onCheckedChange={(value) => setForm((prev) => ({ ...prev, supportsTools: value }))} />
      </div>
      <div className="space-y-2">
        <Label htmlFor="notes">Notes</Label>
        <Input id="notes" name="notes" value={form.notes} onChange={handleChange} placeholder="GPU box, 48GB" />
      </div>
      <DialogFooter>
        <Button type="button" variant="outline" onClick={onCancel}>
          Cancel
        </Button>
        <Button type="submit">{initialValues ? "Save changes" : "Add model"}</Button>
      </DialogFooter>
    </form>
  )
}

export function EndpointForm({ open, onOpenChange, initialValues, onSubmit }) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{initialValues ? "Edit model" : "Add model"}</DialogTitle>
          <DialogDescription>Provide the API base URL, credentials, and model metadata for your OpenAI-compatible backend.</DialogDescription>
        </DialogHeader>
        <EndpointFormFields
          key={initialValues?.id || "create"}
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

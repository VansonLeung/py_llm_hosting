import { useState } from "react"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Button } from "@/components/ui/button"
import { safeJsonParse } from "@/lib/utils"

const defaultSchema = {
  type: "object",
  properties: {
    query: { type: "string", description: "Query string" },
  },
  required: ["query"],
}

function ToolFormFields({ initialValues, isMcp, onSubmit, onCancel }) {
  const [form, setForm] = useState(() => ({
    name: initialValues?.name || "",
    description: initialValues?.description || "",
    schema: JSON.stringify(initialValues?.schema || defaultSchema, null, 2),
    connection: initialValues?.connection || "",
  }))

  const handleSubmit = (event) => {
    event.preventDefault()
    const schema = safeJsonParse(form.schema, defaultSchema)
    onSubmit?.({
      name: form.name,
      description: form.description,
      schema,
      connection: isMcp ? form.connection : undefined,
    })
  }

  return (
    <form className="space-y-4" onSubmit={handleSubmit}>
      <div className="space-y-2">
        <Label htmlFor="name">Name</Label>
        <Input id="name" value={form.name} onChange={(event) => setForm((prev) => ({ ...prev, name: event.target.value }))} />
      </div>
      <div className="space-y-2">
        <Label htmlFor="description">Description</Label>
        <Textarea
          id="description"
          value={form.description}
          onChange={(event) => setForm((prev) => ({ ...prev, description: event.target.value }))}
          rows={3}
        />
      </div>
      <div className="space-y-2">
        <Label htmlFor="schema">JSON schema</Label>
        <Textarea
          id="schema"
          value={form.schema}
          onChange={(event) => setForm((prev) => ({ ...prev, schema: event.target.value }))}
          rows={6}
          className="font-mono text-xs"
        />
      </div>
      {isMcp && (
        <div className="space-y-2">
          <Label htmlFor="connection">Connection</Label>
          <Input
            id="connection"
            value={form.connection}
            onChange={(event) => setForm((prev) => ({ ...prev, connection: event.target.value }))}
            placeholder="filesystem, postgres, etc"
          />
        </div>
      )}
      <DialogFooter>
        <Button type="button" variant="outline" onClick={onCancel}>
          Cancel
        </Button>
        <Button type="submit">{initialValues ? "Save tool" : "Add tool"}</Button>
      </DialogFooter>
    </form>
  )
}

export function ToolForm({ open, onOpenChange, onSubmit, initialValues, isMcp }) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle>{initialValues ? "Edit tool" : "Add tool"}</DialogTitle>
          <DialogDescription>
            {isMcp ? "Describe your MCP tool contract." : "Provide a function schema to expose to the model."}
          </DialogDescription>
        </DialogHeader>
        <ToolFormFields
          key={`${initialValues?.id || "create"}-${isMcp ? "mcp" : "tool"}`}
          initialValues={initialValues}
          isMcp={isMcp}
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

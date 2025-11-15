import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ToolForm } from "./ToolForm"
import { useToolStore } from "@/stores/tool-store"
import { formatDate } from "@/lib/utils"
import { Pencil, Trash2 } from "lucide-react"

function ToolList({ data, onEdit, onDelete }) {
  return (
    <ScrollArea className="h-64 pr-3">
      <div className="space-y-3">
        {data.map((tool) => (
          <div key={tool.id} className="rounded-lg border p-3">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-semibold">{tool.name}</p>
                <p className="text-xs text-muted-foreground">{tool.description}</p>
              </div>
              <div className="flex gap-2">
                <Button size="icon" variant="ghost" onClick={() => onEdit(tool)}>
                  <Pencil className="h-4 w-4" />
                </Button>
                <Button size="icon" variant="ghost" onClick={() => onDelete(tool)}>
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            </div>
            <div className="mt-2 text-xs text-muted-foreground">Updated {formatDate(tool.createdAt)}</div>
            {tool.connection && (
              <Badge variant="outline" className="mt-2">
                {tool.connection}
              </Badge>
            )}
          </div>
        ))}
        {!data.length && <p className="text-sm text-muted-foreground">No tools configured.</p>}
      </div>
    </ScrollArea>
  )
}

export function ToolPanel() {
  const [tab, setTab] = useState("tools")
  const [dialogOpen, setDialogOpen] = useState(false)
  const [editing, setEditing] = useState(null)
  const [formContext, setFormContext] = useState({ isMcp: false })
  const tools = useToolStore((state) => state.tools)
  const mcpTools = useToolStore((state) => state.mcpTools)
  const addTool = useToolStore((state) => state.addTool)
  const updateTool = useToolStore((state) => state.updateTool)
  const removeTool = useToolStore((state) => state.removeTool)

  const handleSubmit = (values) => {
    if (editing) {
      updateTool(editing.id, values, { isMcp: formContext.isMcp })
    } else {
      addTool(values, { isMcp: formContext.isMcp })
    }
  }

  const handleDelete = (tool, isMcp) => {
    if (window.confirm(`Delete tool "${tool.name}"?`)) {
      removeTool(tool.id, { isMcp })
    }
  }

  return (
    <div className="flex h-full flex-col gap-4">
      <div className="flex items-center justify-between">
        <p className="text-sm font-semibold uppercase text-muted-foreground">Tooling</p>
        <Button
          size="sm"
          variant="outline"
          onClick={() => {
            setFormContext({ isMcp: tab === "mcp" })
            setDialogOpen(true)
          }}
        >
          Add
        </Button>
      </div>
      <Tabs value={tab} onValueChange={setTab} className="flex-1">
        <TabsList className="w-full">
          <TabsTrigger className="flex-1" value="tools">
            Functions
          </TabsTrigger>
          <TabsTrigger className="flex-1" value="mcp">
            MCP
          </TabsTrigger>
        </TabsList>
        <TabsContent value="tools">
          <ToolList
            data={tools}
            onEdit={(tool) => {
              setFormContext({ isMcp: false })
              setEditing(tool)
              setDialogOpen(true)
            }}
            onDelete={(tool) => handleDelete(tool, false)}
          />
        </TabsContent>
        <TabsContent value="mcp">
          <ToolList
            data={mcpTools}
            onEdit={(tool) => {
              setFormContext({ isMcp: true })
              setEditing(tool)
              setDialogOpen(true)
            }}
            onDelete={(tool) => handleDelete(tool, true)}
          />
        </TabsContent>
      </Tabs>
      <ToolForm
        open={dialogOpen}
        onOpenChange={(isOpen) => {
          setDialogOpen(isOpen)
          if (!isOpen) setEditing(null)
        }}
        initialValues={editing || undefined}
        onSubmit={handleSubmit}
        isMcp={formContext.isMcp}
      />
    </div>
  )
}

import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"

export function ToolingSelector({ tools, mcpTools, selectedToolIds, selectedMcpToolIds, onChange }) {
  const totalSelected = (selectedToolIds?.length || 0) + (selectedMcpToolIds?.length || 0)

  const toggle = (id, { isMcp = false } = {}) => {
    if (isMcp) {
      const next = selectedMcpToolIds.includes(id)
        ? selectedMcpToolIds.filter((item) => item !== id)
        : [...selectedMcpToolIds, id]
      onChange?.({ toolIds: selectedToolIds, mcpToolIds: next })
    } else {
      const next = selectedToolIds.includes(id)
        ? selectedToolIds.filter((item) => item !== id)
        : [...selectedToolIds, id]
      onChange?.({ toolIds: next, mcpToolIds: selectedMcpToolIds })
    }
  }

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline" className="w-full justify-between">
          <span>Tooling</span>
          <Badge variant={totalSelected ? "default" : "outline"}>{totalSelected}</Badge>
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-80 space-y-4">
        <div>
          <p className="text-sm font-semibold">Functions</p>
          <div className="mt-2 space-y-2">
            {tools.map((tool) => (
              <label key={tool.id} className="flex items-start gap-2">
                <Checkbox checked={selectedToolIds.includes(tool.id)} onCheckedChange={() => toggle(tool.id)} />
                <div>
                  <p className="text-sm font-medium">{tool.name}</p>
                  <p className="text-xs text-muted-foreground">{tool.description}</p>
                </div>
              </label>
            ))}
            {!tools.length && <p className="text-xs text-muted-foreground">No tools</p>}
          </div>
        </div>
        <div>
          <p className="text-sm font-semibold">MCP Tools</p>
          <div className="mt-2 space-y-2">
            {mcpTools.map((tool) => (
              <label key={tool.id} className="flex items-start gap-2">
                <Checkbox checked={selectedMcpToolIds.includes(tool.id)} onCheckedChange={() => toggle(tool.id, { isMcp: true })} />
                <div>
                  <p className="text-sm font-medium">{tool.name}</p>
                  <p className="text-xs text-muted-foreground">{tool.connection || tool.description}</p>
                </div>
              </label>
            ))}
            {!mcpTools.length && <p className="text-xs text-muted-foreground">No MCP connectors</p>}
          </div>
        </div>
      </PopoverContent>
    </Popover>
  )
}

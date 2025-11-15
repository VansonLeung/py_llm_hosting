import { useMemo, useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { conversationSort } from "./helpers"
import { formatDate } from "@/lib/utils"
import { MoreHorizontal, Pencil, Trash2 } from "lucide-react"

export function ConversationList({ conversations, activeId, onSelect, onEdit, onDelete }) {
  const [query, setQuery] = useState("")

  const filtered = useMemo(() => {
    return conversations
      .filter((conversation) => conversation.title.toLowerCase().includes(query.toLowerCase()))
      .sort(conversationSort)
  }, [conversations, query])

  return (
    <div className="flex h-full flex-col gap-3">
      <Input placeholder="Search conversations" value={query} onChange={(event) => setQuery(event.target.value)} />
      <ScrollArea className="flex-1 pr-2">
        <div className="space-y-2">
          {filtered.map((conversation) => (
            <div
              key={conversation.id}
              role="button"
              tabIndex={0}
              onClick={() => onSelect(conversation.id)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault()
                  onSelect(conversation.id)
                }
              }}
              className={`w-full rounded-lg border p-3 text-left transition hover:border-primary focus:outline-none focus:ring-2 focus:ring-primary/50 ${conversation.id === activeId ? "border-primary bg-primary/10" : "border-border"}`}
            >
              <div className="flex items-center justify-between">
                <p className="font-medium">{conversation.title}</p>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="icon">
                      <MoreHorizontal className="h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem onClick={() => onEdit(conversation)}> <Pencil className="mr-2 h-4 w-4" /> Rename </DropdownMenuItem>
                    <DropdownMenuItem className="text-destructive" onClick={() => onDelete(conversation)}> <Trash2 className="mr-2 h-4 w-4" /> Delete </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span>{conversation.model}</span>
                <span>•</span>
                <span>{formatDate(conversation.updatedAt)}</span>
              </div>
              <div className="mt-2 flex gap-2 text-xs">
                <Badge variant="secondary">{conversation.tokenUsage.total} tokens</Badge>
                <Badge variant="outline">{conversation.messages.length} msgs</Badge>
              </div>
            </div>
          ))}
          {!filtered.length && <p className="text-sm text-muted-foreground">No conversations yet.</p>}
        </div>
      </ScrollArea>
    </div>
  )
}

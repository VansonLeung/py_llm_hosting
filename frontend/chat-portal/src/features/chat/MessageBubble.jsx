import { Badge } from "@/components/ui/badge"
import { formatDate } from "@/lib/utils"

function renderContentPart(part, index) {
  if (part == null) return null
  if (typeof part === "string" || typeof part === "number") {
    return (
      <p key={index} className="whitespace-pre-wrap">
        {String(part)}
      </p>
    )
  }
  if (Array.isArray(part)) {
    return part.map((nested, nestedIndex) => renderContentPart(nested, `${index}-${nestedIndex}`))
  }
  if (typeof part === "object") {
    if (part.type === "text" && typeof part.text === "string") {
      return (
        <p key={index} className="whitespace-pre-wrap">
          {part.text}
        </p>
      )
    }
    if (part.type === "image" || part.type === "image_url") {
      const src = part.image || part.image_url?.url
      return <img key={index} src={src} alt="attachment" className="max-h-64 rounded-lg" />
    }
    return (
      <pre key={index} className="rounded bg-muted/60 p-2 text-xs">
        {JSON.stringify(part, null, 2)}
      </pre>
    )
  }
  return null
}

export function MessageBubble({ message }) {
  if (!message) return null
  const isUser = message.role === "user"
  const isAssistant = message.role === "assistant"
  const bubbleClasses = isUser ? "bg-primary text-primary-foreground ml-auto" : "bg-muted"

  return (
    <div className={`flex flex-col ${isUser ? "items-end" : "items-start"}`}>
      <div className="mb-1 flex items-center gap-2 text-xs text-muted-foreground">
        <span className="font-medium capitalize">{message.role}</span>
        <span>{formatDate(message.createdAt)}</span>
      </div>
      <div className={`w-full max-w-3xl rounded-2xl px-4 py-3 text-sm shadow ${bubbleClasses}`}>
        {Array.isArray(message.content) ? (
          <div className="space-y-2">{message.content.map(renderContentPart)}</div>
        ) : (
          renderContentPart(message.content, "single")
        )}
        {message.attachments?.length > 0 && (
          <div className="mt-3 grid gap-2 sm:grid-cols-2">
            {message.attachments.map((attachment) => (
              <img key={attachment.id} src={attachment.dataUrl} alt={attachment.name} className="h-32 w-full rounded-lg object-cover" />
            ))}
          </div>
        )}
        {message.metadata?.toolCalls?.length > 0 && (
          <div className="mt-3 space-y-2 rounded-lg border border-white/20 bg-black/10 p-3 text-xs">
            <p className="font-semibold">Tool calls</p>
            {message.metadata.toolCalls.map((tool, index) => (
              <div key={index} className="rounded border border-white/10 p-2">
                <p className="font-medium">{tool.name || tool.function?.name}</p>
                <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap text-[10px]">
                  {JSON.stringify(tool, null, 2)}
                </pre>
              </div>
            ))}
          </div>
        )}
        {message.metadata?.isError && (
          <Badge variant="destructive" className="mt-3">
            Delivery error
          </Badge>
        )}
      </div>
      {message.tokenUsage && (
        <div className="mt-1 text-[10px] text-muted-foreground">
          {message.tokenUsage.prompt ? `Prompt ${message.tokenUsage.prompt} · ` : ""}
          {message.tokenUsage.completion ? `Completion ${message.tokenUsage.completion} · ` : ""}
          Total {message.tokenUsage.total}
        </div>
      )}
    </div>
  )
}

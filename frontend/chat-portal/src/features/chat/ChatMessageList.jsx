import { useEffect, useRef } from "react"
import { MessageBubble } from "./MessageBubble"

export function ChatMessageList({ messages }) {
  const containerRef = useRef(null)

  useEffect(() => {
    if (!containerRef.current) return
    containerRef.current.scrollTop = containerRef.current.scrollHeight
  }, [messages])

  if (!messages?.length) {
    return (
      <div className="flex h-full min-h-[360px] flex-col items-center justify-center rounded-2xl border border-dashed p-10 text-center text-muted-foreground">
        <p className="font-semibold">Start your first prompt</p>
        <p className="text-sm">Messages, tool calls, and MCP logs will show up here.</p>
      </div>
    )
  }

  return (
    <div ref={containerRef} className="h-full space-y-6 overflow-y-auto px-1">
      {messages.map((message) => (
        <MessageBubble key={message.id} message={message} />
      ))}
    </div>
  )
}

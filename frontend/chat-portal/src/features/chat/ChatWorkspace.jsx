import { useCallback, useMemo, useState } from "react"
import { Button } from "@/components/ui/button"
import { ChatMessageList } from "./ChatMessageList"
import { ChatComposer } from "./ChatComposer"
import { TokenUsageSummary } from "./TokenUsageSummary"
import { ToolingSelector } from "@/features/tools/ToolingSelector"
import { useConversationStore, selectConversationById } from "@/stores/conversation-store"
import { useEndpointStore, selectActiveEndpoint } from "@/stores/endpoint-store"
import { useToolStore } from "@/stores/tool-store"
import { dispatchChat } from "@/services/chat-service"
import { Loader2, MessageCirclePlus } from "lucide-react"

const EMPTY_IDS = Object.freeze([])

export function ChatWorkspace() {
  const activeConversation = useConversationStore((state) => selectConversationById(state, state.activeConversationId))
  const conversationsCount = useConversationStore((state) => state.conversations.length)
  const createConversation = useConversationStore((state) => state.createConversation)
  const addMessage = useConversationStore((state) => state.addMessage)
  const patchMessage = useConversationStore((state) => state.patchMessage)
  const attachTools = useConversationStore((state) => state.attachTools)
  const updateConversation = useConversationStore((state) => state.updateConversation)
  const activeEndpoint = useEndpointStore(selectActiveEndpoint)
  const tools = useToolStore((state) => state.tools)
  const mcpTools = useToolStore((state) => state.mcpTools)
  const [status, setStatus] = useState("idle")
  const [error, setError] = useState("")

  const selectedToolIds = activeConversation?.toolIds ?? EMPTY_IDS
  const selectedMcpToolIds = activeConversation?.mcpToolIds ?? EMPTY_IDS
  const activeModel = activeConversation?.model || activeEndpoint?.models?.[0] || "gpt-4o-mini"
  const disableSend = !activeEndpoint || status === "pending"

  const toolMap = useMemo(() => {
    const map = new Map()
    tools.forEach((tool) => map.set(tool.id, tool))
    mcpTools.forEach((tool) => map.set(tool.id, tool))
    return map
  }, [tools, mcpTools])

  const selectedTools = useMemo(() => {
    const normalTools = selectedToolIds.map((id) => ({ tool: toolMap.get(id), isMcp: false }))
    const mcpSelections = selectedMcpToolIds.map((id) => ({ tool: toolMap.get(id), isMcp: true }))
    return [...normalTools, ...mcpSelections].filter((entry) => entry.tool)
  }, [selectedMcpToolIds, selectedToolIds, toolMap])

  const ensureConversation = useCallback(() => {
    if (activeConversation) return activeConversation
    const title = `Conversation ${conversationsCount + 1}`
    return createConversation({ title, endpointId: activeEndpoint?.id, model: activeModel })
  }, [activeConversation, conversationsCount, createConversation, activeEndpoint?.id, activeModel])

  const handleToolingChange = ({ toolIds, mcpToolIds }) => {
    if (!activeConversation) return
    attachTools(activeConversation.id, { toolIds, mcpToolIds })
  }

  const handleSend = async ({ text, attachments, temperature, maxTokens }) => {
    const conversation = ensureConversation()
    if (!activeEndpoint) {
      setError("Configure an endpoint before sending messages.")
      return
    }

    setError("")
    setStatus("pending")

    addMessage(conversation.id, {
      role: "user",
      content: text,
      attachments,
      metadata: { temperature, maxTokens },
    })

    const assistantMessage = addMessage(conversation.id, {
      role: "assistant",
      content: "",
      metadata: { streaming: true },
    })

    const state = useConversationStore.getState()
    const latestConversation = selectConversationById(state, conversation.id)
    const messages = latestConversation?.messages?.filter((message) => message.id !== assistantMessage.id) || []

    let streamed = ""
    try {
      const response = await dispatchChat({
        endpoint: activeEndpoint,
        model: activeModel,
        messages,
        tools: selectedTools,
        temperature,
        maxOutputTokens: maxTokens,
        onToken: (delta) => {
          streamed += delta
          patchMessage(conversation.id, assistantMessage.id, {
            content: streamed,
          })
        },
      })

      const finalText = typeof response.text === "string" && response.text.length ? response.text : streamed
      patchMessage(conversation.id, assistantMessage.id, {
        content: finalText,
        metadata: { toolCalls: response.toolCalls, finishReason: response.finishReason },
        tokenUsage: response.usage,
      })
      updateConversation(conversation.id, { tokenUsage: response.usage })
    } catch (err) {
      patchMessage(conversation.id, assistantMessage.id, {
        content: err.message,
        metadata: { isError: true },
      })
      setError(err.message)
    } finally {
      setStatus("idle")
    }
  }

  if (!activeEndpoint) {
    return (
      <div className="flex h-full flex-col items-center justify-center rounded-2xl border border-dashed p-10 text-center text-muted-foreground">
        <p className="text-lg font-semibold">Add an endpoint to begin</p>
        <p className="mt-2 text-sm">Configure an OpenAI-compatible endpoint so conversations know where to send requests.</p>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col gap-4">
      <div className="rounded-2xl border bg-card/60 p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs font-semibold uppercase text-muted-foreground">Conversation</p>
            <h2 className="text-xl font-semibold">{activeConversation?.title || "Create a conversation"}</h2>
            <p className="text-sm text-muted-foreground">
              {activeEndpoint?.name || "Endpoint"} · {activeModel}
            </p>
          </div>
          <TokenUsageSummary usage={activeConversation?.tokenUsage} />
        </div>
        {!activeConversation && (
          <Button size="sm" className="mt-4" onClick={ensureConversation}>
            <MessageCirclePlus className="mr-2 h-4 w-4" />
            New conversation
          </Button>
        )}
        {status === "pending" && (
          <div className="mt-4 flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" /> Streaming response…
          </div>
        )}
        {error && <p className="mt-4 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">{error}</p>}
      </div>

      <div className="flex-1 rounded-2xl border bg-card/40 p-4">
        <ChatMessageList messages={activeConversation?.messages} />
      </div>

      <div className="space-y-3">
        {activeConversation ? (
          <ToolingSelector
            tools={tools}
            mcpTools={mcpTools}
            selectedToolIds={selectedToolIds}
            selectedMcpToolIds={selectedMcpToolIds}
            onChange={handleToolingChange}
          />
        ) : (
          <p className="text-sm text-muted-foreground">Create conversation to attach tools.</p>
        )}
        <ChatComposer disabled={disableSend} onSend={handleSend} />
      </div>
    </div>
  )
}

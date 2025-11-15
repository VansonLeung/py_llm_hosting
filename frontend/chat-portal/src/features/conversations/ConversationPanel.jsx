import { useState } from "react"
import { Button } from "@/components/ui/button"
import { ConversationList } from "./ConversationList"
import { ConversationForm } from "./ConversationForm"
import { useConversationStore } from "@/stores/conversation-store"
import { useEndpointStore } from "@/stores/endpoint-store"

export function ConversationPanel() {
  const conversations = useConversationStore((state) => state.conversations)
  const activeConversationId = useConversationStore((state) => state.activeConversationId)
  const setActiveConversation = useConversationStore((state) => state.setActiveConversation)
  const createConversation = useConversationStore((state) => state.createConversation)
  const updateConversation = useConversationStore((state) => state.updateConversation)
  const deleteConversation = useConversationStore((state) => state.deleteConversation)
  const endpoints = useEndpointStore((state) => state.endpoints)
  const [dialogOpen, setDialogOpen] = useState(false)
  const [editingConversation, setEditingConversation] = useState(null)

  const handleCreate = (values) => {
    const payload = {
      title: values.title || `Conversation ${conversations.length + 1}`,
      endpointId: values.endpointId || endpoints.at(0)?.id,
      model: values.model || endpoints.at(0)?.models?.[0] || "gpt-4o-mini",
    }
    const conversation = createConversation(payload)
    setActiveConversation(conversation.id)
  }

  const handleEdit = (values) => {
    updateConversation(editingConversation.id, values)
  }

  const handleDelete = (conversation) => {
    if (window.confirm(`Delete conversation "${conversation.title}"?`)) {
      deleteConversation(conversation.id)
    }
  }

  return (
    <div className="flex h-full flex-col gap-4">
      <div className="flex items-center justify-between">
        <p className="text-sm font-semibold uppercase text-muted-foreground">Conversations</p>
        <Button size="sm" onClick={() => setDialogOpen(true)}>
          New
        </Button>
      </div>
      <ConversationList
        conversations={conversations}
        activeId={activeConversationId}
        onSelect={setActiveConversation}
        onEdit={(conversation) => {
          setEditingConversation(conversation)
          setDialogOpen(true)
        }}
        onDelete={handleDelete}
      />
      <ConversationForm
        open={dialogOpen}
        onOpenChange={(isOpen) => {
          setDialogOpen(isOpen)
          if (!isOpen) setEditingConversation(null)
        }}
        initialValues={editingConversation || undefined}
        onSubmit={(values) => {
          if (editingConversation) {
            handleEdit(values)
          } else {
            handleCreate(values)
          }
        }}
      />
    </div>
  )
}

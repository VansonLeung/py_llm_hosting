import { create } from "zustand"
import { persist } from "zustand/middleware"
import { browserStorage } from "@/lib/storage"
import { STORAGE_KEYS } from "@/lib/constants"
import { aggregateUsage } from "@/lib/tokens"
import { backendApi } from "@/services/backend-api"
import { isBackendModeEnabled, usePersistenceStore } from "@/stores/persistence-store"

const buildConversation = ({ title, endpointId, model }) => ({
  id: crypto.randomUUID(),
  title: title || "New conversation",
  endpointId: endpointId || null,
  model: model || "gpt-4o-mini",
  messages: [],
  toolIds: [],
  mcpToolIds: [],
  tokenUsage: { prompt: 0, completion: 0, total: 0 },
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
})

const buildMessage = ({ role, content, attachments = [], metadata = {} }) => ({
  id: crypto.randomUUID(),
  role,
  content,
  attachments,
  metadata,
  createdAt: new Date().toISOString(),
})

const queueBackendTask = (task) => {
  if (!isBackendModeEnabled()) return
  Promise.resolve()
    .then(task)
    .then(() => usePersistenceStore.getState().markSynced())
    .catch((error) => {
      console.error("Failed to sync with backend", error)
      usePersistenceStore.getState().setSyncError(error.message || "Unknown backend error")
    })
}

export const useConversationStore = create(
  persist(
    (set) => ({
      conversations: [],
      activeConversationId: null,
      createConversation(payload = {}) {
        const conversation = buildConversation(payload)
        set((state) => ({
          conversations: [conversation, ...state.conversations],
          activeConversationId: conversation.id,
        }))
        queueBackendTask(() =>
          backendApi.createConversation({
            id: conversation.id,
            title: conversation.title,
            endpointId: conversation.endpointId,
            model: conversation.model,
            toolIds: conversation.toolIds,
            mcpToolIds: conversation.mcpToolIds,
            tokenUsage: conversation.tokenUsage,
          })
        )
        return conversation
      },
      updateConversation(id, updates) {
        set((state) => ({
          conversations: state.conversations.map((conversation) =>
            conversation.id === id
              ? { ...conversation, ...updates, updatedAt: new Date().toISOString() }
              : conversation
          ),
        }))
        queueBackendTask(() =>
          backendApi.updateConversation(id, {
            ...updates,
          })
        )
      },
      deleteConversation(id) {
        set((state) => {
          const remaining = state.conversations.filter((conv) => conv.id !== id)
          const nextActive =
            state.activeConversationId === id ? remaining.at(0)?.id ?? null : state.activeConversationId
          return {
            conversations: remaining,
            activeConversationId: nextActive,
          }
        })
        queueBackendTask(() => backendApi.deleteConversation(id))
      },
      setActiveConversation(id) {
        set({ activeConversationId: id })
      },
      addMessage(conversationId, messageInput) {
        const message = buildMessage(messageInput)
        set((state) => ({
          conversations: state.conversations.map((conversation) => {
            if (conversation.id !== conversationId) return conversation
            const messages = [...conversation.messages, message]
            return {
              ...conversation,
              messages,
              tokenUsage: aggregateUsage(messages),
              updatedAt: new Date().toISOString(),
            }
          }),
        }))
        queueBackendTask(() =>
          backendApi.createMessage(conversationId, {
            id: message.id,
            role: message.role,
            content: message.content,
            attachments: message.attachments,
            metadata: message.metadata,
            tokenUsage: message.tokenUsage ?? null,
          })
        )
        return message
      },
      patchMessage(conversationId, messageId, patch) {
        set((state) => ({
          conversations: state.conversations.map((conversation) => {
            if (conversation.id !== conversationId) return conversation
            const messages = conversation.messages.map((message) =>
              message.id === messageId ? { ...message, ...patch } : message
            )
            return {
              ...conversation,
              messages,
              tokenUsage: aggregateUsage(messages),
              updatedAt: new Date().toISOString(),
            }
          }),
        }))
        queueBackendTask(() => backendApi.updateMessage(messageId, patch))
      },
      attachTools(conversationId, { toolIds, mcpToolIds }) {
        set((state) => ({
          conversations: state.conversations.map((conversation) =>
            conversation.id === conversationId
              ? {
                  ...conversation,
                  toolIds: toolIds ?? conversation.toolIds,
                  mcpToolIds: mcpToolIds ?? conversation.mcpToolIds,
                }
              : conversation
          ),
        }))
        queueBackendTask(() =>
          backendApi.updateConversation(conversationId, {
            toolIds,
            mcpToolIds,
          })
        )
      },
      reset() {
        set({ conversations: [], activeConversationId: null })
      },
      async hydrateFromBackend() {
        if (!isBackendModeEnabled()) return { success: false, reason: "not-backend" }
        try {
          const summaries = await backendApi.listConversations()
          const detailed = await Promise.all(
            summaries.map(async (conversation) => {
              const full = await backendApi.fetchConversation(conversation.id)
              return {
                ...full,
                messages: full.messages || [],
                toolIds: full.toolIds || [],
                mcpToolIds: full.mcpToolIds || [],
                tokenUsage: full.tokenUsage || { prompt: 0, completion: 0, total: 0 },
              }
            })
          )
          set({
            conversations: detailed,
            activeConversationId: detailed.at(0)?.id ?? null,
          })
          usePersistenceStore.getState().markSynced()
          return { success: true, count: detailed.length }
        } catch (error) {
          usePersistenceStore.getState().setSyncError(error.message)
          return { success: false, error }
        }
      },
    }),
    {
      name: STORAGE_KEYS.CONVERSATIONS,
      storage: browserStorage(),
    }
  )
)

export const selectConversationById = (state, id) =>
  state.conversations.find((conversation) => conversation.id === id) || null

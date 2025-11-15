import { create } from "zustand"
import { persist } from "zustand/middleware"
import { browserStorage } from "@/lib/storage"
import { STORAGE_KEYS } from "@/lib/constants"
import { aggregateUsage } from "@/lib/tokens"

const createConversation = ({ title, endpointId, model }) => ({
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

const createMessage = ({ role, content, attachments = [], metadata = {} }) => ({
  id: crypto.randomUUID(),
  role,
  content,
  attachments,
  metadata,
  createdAt: new Date().toISOString(),
})

export const useConversationStore = create(
  persist(
    (set, get) => ({
      conversations: [],
      activeConversationId: null,
      createConversation(payload = {}) {
        const conversation = createConversation(payload)
        set((state) => ({
          conversations: [conversation, ...state.conversations],
          activeConversationId: conversation.id,
        }))
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
      },
      setActiveConversation(id) {
        set({ activeConversationId: id })
      },
      addMessage(conversationId, messageInput) {
        const message = createMessage(messageInput)
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
      },
      reset() {
        set({ conversations: [], activeConversationId: null })
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

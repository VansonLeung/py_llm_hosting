import { usePersistenceStore } from "@/stores/persistence-store"

const DEFAULT_BASE_URL = import.meta.env.VITE_BACKEND_API_BASE_URL || "http://localhost:5278"

const resolveBaseUrl = () => usePersistenceStore.getState().backendBaseUrl || DEFAULT_BASE_URL

const jsonHeaders = { "Content-Type": "application/json" }

async function request(path, options = {}) {
  const url = new URL(path, resolveBaseUrl())
  const response = await fetch(url, {
    ...options,
    headers: {
      ...jsonHeaders,
      ...(options.headers || {}),
    },
  })

  if (!response.ok) {
    const body = await response.text()
    throw new Error(body || `Backend request failed (${response.status})`)
  }

  if (response.status === 204) {
    return null
  }

  return response.json()
}

export const backendApi = {
  listConversations() {
    return request("/api/conversations")
  },
  fetchConversation(id) {
    return request(`/api/conversations/${id}`)
  },
  createConversation(payload) {
    return request("/api/conversations", {
      method: "POST",
      body: JSON.stringify(payload),
    })
  },
  updateConversation(id, payload) {
    return request(`/api/conversations/${id}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    })
  },
  deleteConversation(id) {
    return request(`/api/conversations/${id}`, {
      method: "DELETE",
    })
  },
  createMessage(conversationId, payload) {
    return request(`/api/conversations/${conversationId}/messages`, {
      method: "POST",
      body: JSON.stringify(payload),
    })
  },
  updateMessage(messageId, payload) {
    return request(`/api/messages/${messageId}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    })
  },
}

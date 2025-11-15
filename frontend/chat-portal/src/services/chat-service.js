import { buildUsageFromResponse } from "@/lib/tokens"

const DEFAULT_HEADERS = { "Content-Type": "application/json" }
const EMPTY_USAGE = { prompt: 0, completion: 0, total: 0 }

function stringifyContent(content) {
  if (content == null) return ""
  if (typeof content === "string") return content
  if (typeof content === "number" || typeof content === "boolean") return String(content)
  if (Array.isArray(content)) {
    return content.map((item) => stringifyContent(item)).filter(Boolean).join("\n")
  }
  if (typeof content === "object") {
    if (typeof content.text === "string") return content.text
    return JSON.stringify(content)
  }
  return ""
}

function parseMediaType(dataUrl, fallback) {
  if (typeof dataUrl !== "string") return fallback
  const match = dataUrl.match(/^data:([^;]+);/)
  return match?.[1] || fallback
}

function attachmentToContentPart(attachment) {
  if (!attachment?.dataUrl) return null
  if (attachment.type === "image") {
    return {
      type: "image_url",
      image_url: {
        url: attachment.dataUrl,
        detail: "auto",
      },
    }
  }

  const mediaType = parseMediaType(attachment.dataUrl, "application/octet-stream")
  return {
    type: "text",
    text: `[Attachment: ${attachment.name || mediaType}] (${mediaType}) embedded as data URL\n${attachment.dataUrl.slice(0, 120)}…`,
  }
}

function normalizeContentParts(content) {
  if (Array.isArray(content)) {
    return content.flatMap((part) => {
      if (!part) return []
      if (typeof part === "string") {
        return [{ type: "text", text: part }]
      }
      if (typeof part === "object" && typeof part.type === "string") {
        return [part]
      }
      const text = stringifyContent(part)
      return text ? [{ type: "text", text }] : []
    })
  }

  const text = stringifyContent(content)
  return text ? [{ type: "text", text }] : []
}

function mapMessage(message) {
  if (!message) return null
  const parts = normalizeContentParts(message.content)
  const attachmentParts = (message.attachments || []).map(attachmentToContentPart).filter(Boolean)
  const content = [...parts, ...attachmentParts]

  if (!content.length) {
    return { role: message.role, content: [{ type: "text", text: "" }] }
  }

  return { role: message.role, content }
}

function buildToolPayload(tool, { isMcp = false } = {}) {
  if (!tool) return null
  return {
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.schema,
      metadata: isMcp ? { provider: "mcp", connection: tool.connection } : undefined,
    },
  }
}

export async function dispatchChat({
  endpoint,
  model,
  messages,
  tools = [],
  signal,
  temperature = 0.7,
  maxOutputTokens = 1024,
  onToken,
}) {
  if (!endpoint?.baseUrl) {
    throw new Error("Select a model endpoint before sending messages")
  }

  const normalizedMessages = messages.map(mapMessage).filter(Boolean)
  const normalizedTools = tools.map((tool) => buildToolPayload(tool.tool, { isMcp: tool.isMcp })).filter(Boolean)

  try {
    return await streamCompletions({
      endpoint,
      model,
      messages: normalizedMessages,
      tools: normalizedTools,
      temperature,
      maxOutputTokens,
      signal,
      onToken,
    })
  } catch (error) {
    // Fall back to a standard fetch so the UI still works even if streamText fails
    const headers = { ...DEFAULT_HEADERS }
    if (endpoint.apiKey) {
      headers.Authorization = `Bearer ${endpoint.apiKey}`
    }

    const response = await fetch(`${endpoint.baseUrl}/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        model,
        messages: normalizedMessages,
        tools: normalizedTools.length ? normalizedTools : undefined,
        stream: false,
        temperature,
        max_tokens: maxOutputTokens,
      }),
      signal,
    })

    if (!response.ok) {
      throw new Error(`Chat request failed: ${response.status} ${response.statusText}`)
    }

    const data = await response.json()
    const choice = data.choices?.[0]
    const rawContent = choice?.message?.content ?? data.output ?? ""
    const normalizedText = stringifyContent(rawContent)
    return {
      text: normalizedText,
      usage: buildUsageFromResponse(data.usage) || EMPTY_USAGE,
      toolCalls: choice?.message?.tool_calls || [],
      finishReason: choice?.finish_reason || data.finish_reason || null,
      raw: data,
    }
  }
}

async function streamCompletions({
  endpoint,
  model,
  messages,
  tools,
  temperature,
  maxOutputTokens,
  signal,
  onToken,
}) {
  const headers = buildAuthHeaders(endpoint)
  const response = await fetch(buildCompletionsUrl(endpoint.baseUrl), {
    method: "POST",
    headers,
    body: JSON.stringify({
      model,
      messages,
      tools: tools.length ? tools : undefined,
      stream: true,
      temperature,
      max_tokens: maxOutputTokens,
    }),
    signal,
  })

  if (!response.ok) {
    throw new Error(`Chat stream failed: ${response.status} ${response.statusText}`)
  }

  if (!response.body) {
    throw new Error("Streaming response missing body")
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  const context = {
    streamedText: "",
    finishReason: null,
    usage: EMPTY_USAGE,
  }
  const toolCalls = []

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    buffer = consumeSseBuffer({ buffer, onToken, toolCalls, context })
  }

  if (buffer.trim().length) {
    processSseEvent({ event: buffer.trim(), onToken, toolCallsRef: toolCalls, context })
  }

  return {
    text: context.streamedText,
    usage: context.usage ?? EMPTY_USAGE,
    toolCalls,
    finishReason: context.finishReason,
    raw: null,
  }
}

function consumeSseBuffer({ buffer, onToken, toolCalls, context }) {
  let working = buffer
  let boundary = working.indexOf("\n\n")
  while (boundary !== -1) {
    const event = working.slice(0, boundary).trim()
    working = working.slice(boundary + 2)
    if (event) {
      processSseEvent({ event, onToken, toolCallsRef: toolCalls, context })
    }
    boundary = working.indexOf("\n\n")
  }
  return working
}

function processSseEvent({ event, onToken, toolCallsRef, context }) {
  if (!event.startsWith("data:")) return
  const payload = event.slice(5).trim()
  if (!payload || payload === "[DONE]") return
  let data
  try {
    data = JSON.parse(payload)
  } catch {
    return
  }

  const choice = data.choices?.[0]
  const delta = choice?.delta || {}
  if (choice?.finish_reason) {
    context.finishReason = choice.finish_reason
  }

  const textDelta = extractTextDelta(delta)
  if (textDelta) {
    context.streamedText += textDelta
    onToken?.(textDelta)
  }

  if (Array.isArray(delta.tool_calls) && delta.tool_calls.length) {
    toolCallsRef.push(...delta.tool_calls)
  }

  if (data.usage) {
    context.usage = buildUsageFromResponse(data.usage)
  }
}

function extractTextDelta(delta) {
  if (!delta) return ""
  if (typeof delta.content === "string") return delta.content
  if (Array.isArray(delta.content)) {
    return delta.content
      .map((part) => {
        if (typeof part === "string") return part
        if (part?.type === "text" && typeof part.text === "string") return part.text
        if (typeof part?.text === "string") return part.text
        return ""
      })
      .join("")
  }
  if (typeof delta.text === "string") return delta.text
  if (delta.content?.text) return delta.content.text
  return ""
}

function buildAuthHeaders(endpoint) {
  const headers = { ...DEFAULT_HEADERS }
  if (endpoint.apiKey) {
    headers.Authorization = `Bearer ${endpoint.apiKey}`
  }
  return headers
}

function buildCompletionsUrl(baseUrl) {
  const normalized = baseUrl.endsWith("/") ? baseUrl.slice(0, -1) : baseUrl
  return `${normalized}/chat/completions`
}

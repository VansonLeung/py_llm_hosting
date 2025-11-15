const AVERAGE_CHARS_PER_TOKEN = 4

export function estimateTokensFromText(text = "") {
  if (!text) return 0
  const normalized = typeof text === "string" ? text : JSON.stringify(text)
  return Math.ceil(normalized.length / AVERAGE_CHARS_PER_TOKEN)
}

export function aggregateUsage(messages = []) {
  return messages.reduce(
    (acc, message) => {
      const usage = message.tokenUsage
      if (usage) {
        acc.prompt += usage.prompt || 0
        acc.completion += usage.completion || 0
      } else if (message.role === "user") {
        acc.prompt += estimateTokensFromText(message.content)
      } else if (message.role === "assistant") {
        acc.completion += estimateTokensFromText(message.content)
      }
      acc.total = acc.prompt + acc.completion
      return acc
    },
    { prompt: 0, completion: 0, total: 0 }
  )
}

export function buildUsageFromResponse(usage) {
  if (!usage) return null
  const prompt = usage.promptTokens || usage.prompt_tokens || usage.prompt || 0
  const completion = usage.completionTokens || usage.completion_tokens || usage.completion || 0
  const total = usage.totalTokens || usage.total_tokens || usage.total || prompt + completion
  return { prompt, completion, total }
}

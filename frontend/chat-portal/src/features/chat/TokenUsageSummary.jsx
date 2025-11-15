import { Badge } from "@/components/ui/badge"

export function TokenUsageSummary({ usage }) {
  if (!usage) return null
  return (
    <div className="flex flex-wrap gap-2 text-xs">
      <Badge variant="outline">Prompt {usage.prompt}</Badge>
      <Badge variant="outline">Completion {usage.completion}</Badge>
      <Badge variant="secondary">Total {usage.total}</Badge>
    </div>
  )
}

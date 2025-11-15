import { useState } from "react"
import { Textarea } from "@/components/ui/textarea"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { AttachmentPreview } from "./AttachmentPreview"
import { fileToDataUrl } from "@/lib/utils"
import { ImageIcon, Send } from "lucide-react"

export function ChatComposer({ disabled, onSend }) {
  const [message, setMessage] = useState("")
  const [attachments, setAttachments] = useState([])
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(1024)

  const handleFileChange = async (event) => {
    const files = Array.from(event.target.files || [])
    const uploads = await Promise.all(
      files.map(async (file) => ({
        id: crypto.randomUUID(),
        name: file.name,
        type: file.type.startsWith("image") ? "image" : "file",
        dataUrl: await fileToDataUrl(file),
      }))
    )
    setAttachments((prev) => [...prev, ...uploads])
    event.target.value = ""
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    if (!message.trim() && attachments.length === 0) return
    await onSend?.({ text: message.trim(), attachments, temperature, maxTokens })
    setMessage("")
    setAttachments([])
  }

  return (
    <form className="space-y-4 rounded-2xl border bg-card/60 p-4" onSubmit={handleSubmit}>
      <Textarea
        placeholder="Ask the model something. Use Markdown, describe images, or request tool calls."
        value={message}
        onChange={(event) => setMessage(event.target.value)}
        rows={4}
        disabled={disabled}
      />
      <AttachmentPreview attachments={attachments} onRemove={(id) => setAttachments((prev) => prev.filter((file) => file.id !== id))} />
      <div className="flex flex-wrap items-center gap-3 text-sm text-muted-foreground">
        <Label className="flex cursor-pointer items-center gap-2">
          <input type="file" accept="image/*" multiple className="hidden" onChange={handleFileChange} disabled={disabled} />
          <ImageIcon className="h-4 w-4" /> Attach images
        </Label>
        <div className="flex items-center gap-2">
          <Label htmlFor="temperature" className="text-xs uppercase tracking-wide">
            Temp
          </Label>
          <input
            id="temperature"
            type="range"
            min="0"
            max="1.5"
            step="0.1"
            value={temperature}
            onChange={(event) => setTemperature(parseFloat(event.target.value))}
            disabled={disabled}
          />
          <span className="text-xs">{temperature.toFixed(1)}</span>
        </div>
        <div className="flex items-center gap-2">
          <Label htmlFor="maxTokens" className="text-xs uppercase tracking-wide">
            Max tokens
          </Label>
          <Input
            id="maxTokens"
            type="number"
            className="h-8 w-24"
            min={256}
            max={4096}
            value={maxTokens}
            onChange={(event) => setMaxTokens(Number(event.target.value))}
            disabled={disabled}
          />
        </div>
        <Button type="submit" className="ml-auto" disabled={disabled}>
          <Send className="mr-2 h-4 w-4" />
          Send
        </Button>
      </div>
    </form>
  )
}

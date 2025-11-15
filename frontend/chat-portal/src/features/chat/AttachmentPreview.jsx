import { X } from "lucide-react"

export function AttachmentPreview({ attachments, onRemove }) {
  if (!attachments?.length) return null
  return (
    <div className="flex flex-wrap gap-3">
      {attachments.map((file) => (
        <div key={file.id} className="relative h-20 w-20 overflow-hidden rounded-md border">
          {file.type === "image" ? (
            <img src={file.dataUrl} alt={file.name} className="h-full w-full object-cover" />
          ) : (
            <div className="flex h-full items-center justify-center bg-muted text-xs text-muted-foreground">
              {file.name}
            </div>
          )}
          <button
            type="button"
            onClick={() => onRemove?.(file.id)}
            className="absolute right-1 top-1 rounded-full bg-black/60 p-1 text-white"
          >
            <X className="h-3 w-3" />
          </button>
        </div>
      ))}
    </div>
  )
}

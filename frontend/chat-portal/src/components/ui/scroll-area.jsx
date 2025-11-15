import * as ScrollAreaPrimitive from "@radix-ui/react-scroll-area"
import { cn } from "@/lib/utils"

const ScrollArea = ({ className, children, ...props }) => (
  <ScrollAreaPrimitive.Root className={cn("relative overflow-hidden", className)} {...props}>
    <ScrollAreaPrimitive.Viewport className="h-full w-full rounded-[inherit]">
      {children}
    </ScrollAreaPrimitive.Viewport>
    <ScrollBar orientation="vertical" />
  </ScrollAreaPrimitive.Root>
)

const ScrollBar = ({ className, orientation = "vertical" }) => (
  <ScrollAreaPrimitive.Scrollbar
    className={cn(
      "flex touch-none select-none transition-colors",
      orientation === "vertical" ? "h-full w-2.5 border-l border-l-transparent" : "h-2.5 border-t border-t-transparent",
      className
    )}
    orientation={orientation}
  >
    <ScrollAreaPrimitive.Thumb className="relative flex-1 rounded-full bg-border" />
  </ScrollAreaPrimitive.Scrollbar>
)

export { ScrollArea }

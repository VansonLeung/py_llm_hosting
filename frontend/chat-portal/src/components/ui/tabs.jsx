import * as TabsPrimitive from "@radix-ui/react-tabs"
import { cn } from "@/lib/utils"

const Tabs = TabsPrimitive.Root

const TabsList = ({ className, ...props }) => (
  <TabsPrimitive.List
    className={cn(
      "inline-flex h-10 items-center justify-center rounded-md bg-muted p-1 text-muted-foreground",
      className
    )}
    {...props}
  />
)

const TabsTrigger = ({ className, ...props }) => (
  <TabsPrimitive.Trigger
    className={cn(
      "inline-flex min-w-[80px] items-center justify-center whitespace-nowrap rounded-sm px-3 py-1 text-sm font-medium transition-all disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-background data-[state=active]:text-foreground data-[state=active]:shadow",
      className
    )}
    {...props}
  />
)

const TabsContent = ({ className, ...props }) => (
  <TabsPrimitive.Content className={cn("mt-3 focus-visible:outline-none", className)} {...props} />
)

export { Tabs, TabsList, TabsTrigger, TabsContent }

"use client"

import * as React from "react"
import { useState } from "react"
import { ChevronDown } from "lucide-react"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { cn } from "@/lib/utils"

interface CollapsibleCardProps extends React.HTMLAttributes<HTMLDivElement> {
  title: string
  icon?: React.ReactNode
  actions?: React.ReactNode
  children: React.ReactNode
  initialOpen?: boolean
  titleClassName?: string
}

const CollapsibleCard = React.forwardRef<HTMLDivElement, CollapsibleCardProps>(
  ({ className, title, icon, actions, children, initialOpen = true, titleClassName, ...props }, ref) => {
    const [isOpen, setIsOpen] = useState(initialOpen)

    return (
      <Card className={cn("overflow-hidden", className)} {...props} ref={ref}>
        <div className="flex items-center p-3">
          <div
            className="flex flex-grow items-center gap-2 cursor-pointer"
            onClick={() => setIsOpen(!isOpen)}
          >
            <CardTitle className={cn("text-sm font-medium flex items-center gap-2", titleClassName)}>
              {icon}
              {title}
            </CardTitle>
          </div>
          <div className="flex items-center gap-1">
            <div onClick={(e) => e.stopPropagation()}>{actions}</div>
            <ChevronDown
              onClick={(e) => {
                e.stopPropagation()
                setIsOpen(!isOpen)
              }}
              className={cn(
                "cursor-pointer h-4 w-4 shrink-0 text-muted-foreground transition-transform duration-200",
                isOpen && "rotate-180"
              )}
            />
          </div>
        </div>
        {isOpen && <CardContent className="p-3 pt-0">{children}</CardContent>}
      </Card>
    )
  }
)
CollapsibleCard.displayName = "CollapsibleCard"

export { CollapsibleCard } 
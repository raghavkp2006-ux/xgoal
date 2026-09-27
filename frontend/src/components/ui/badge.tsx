import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva("inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-medium", { variants: { variant: { success: "border-emerald-400/20 bg-emerald-400/10 text-emerald-300", warning: "border-amber-400/20 bg-amber-400/10 text-amber-200", danger: "border-rose-400/20 bg-rose-400/10 text-rose-200", muted: "border-white/10 bg-white/5 text-muted-foreground" } }, defaultVariants: { variant: "muted" } });
export interface BadgeProps extends React.ComponentProps<"span">, VariantProps<typeof badgeVariants> {}
export function Badge({ className, variant, ...props }: BadgeProps) { return <span className={cn(badgeVariants({ variant }), className)} {...props} />; }

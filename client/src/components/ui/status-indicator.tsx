import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import type { SocketStatus } from "@/protocol/types";

interface StatusIndicatorProps {
  status: SocketStatus;
  className?: string;
}

const statusConfig = {
  connected: {
    color: "bg-emerald-400",
    glow: "shadow-[0_0_8px_rgba(52,211,153,0.6)]",
    label: "Connected",
    pulse: true,
  },
  connecting: {
    color: "bg-amber-400",
    glow: "shadow-[0_0_8px_rgba(251,191,36,0.6)]",
    label: "Connecting...",
    pulse: true,
  },
  disconnected: {
    color: "bg-red-400",
    glow: "shadow-[0_0_8px_rgba(248,113,113,0.4)]",
    label: "Disconnected",
    pulse: false,
  },
};

export const StatusIndicator: React.FC<StatusIndicatorProps> = ({
  status,
  className,
}) => {
  const config = statusConfig[status];

  return (
    <div className={cn("flex items-center gap-2", className)}>
      <div className="relative">
        <div
          className={cn("w-2.5 h-2.5 rounded-full", config.color, config.glow)}
        />
        {config.pulse && (
          <motion.div
            className={cn(
              "absolute inset-0 rounded-full",
              config.color
            )}
            animate={{ scale: [1, 2], opacity: [0.5, 0] }}
            transition={{ duration: 1.5, repeat: Infinity, ease: "easeOut" }}
          />
        )}
      </div>
      <span className="text-xs font-medium text-white/60">{config.label}</span>
    </div>
  );
};


import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface AnimatedOrbProps {
  intensity?: number; // 0-1 audio intensity
  isActive?: boolean;
  size?: "sm" | "md" | "lg";
  color?: "green" | "blue" | "purple";
  className?: string;
}

const sizeMap = {
  sm: "w-24 h-24",
  md: "w-40 h-40",
  lg: "w-56 h-56",
};

const colorMap = {
  green: {
    inner: "from-[#76B900] to-[#4E8800]",
    glow: "shadow-[0_0_60px_rgba(118,185,0,0.4)]",
    glowActive: "shadow-[0_0_100px_rgba(118,185,0,0.6)]",
  },
  blue: {
    inner: "from-[#00b4d8] to-[#0077b6]",
    glow: "shadow-[0_0_60px_rgba(0,180,216,0.4)]",
    glowActive: "shadow-[0_0_100px_rgba(0,180,216,0.6)]",
  },
  purple: {
    inner: "from-[#7c3aed] to-[#5b21b6]",
    glow: "shadow-[0_0_60px_rgba(124,58,237,0.4)]",
    glowActive: "shadow-[0_0_100px_rgba(124,58,237,0.6)]",
  },
};

export const AnimatedOrb: React.FC<AnimatedOrbProps> = ({
  intensity = 0,
  isActive = false,
  size = "lg",
  color = "green",
  className,
}) => {
  const colors = colorMap[color];
  const scale = isActive ? 1 + intensity * 0.3 : 0.9;

  return (
    <div className={cn("relative flex items-center justify-center", className)}>
      {/* Outer glow ring */}
      <motion.div
        className={cn(
          "absolute rounded-full",
          sizeMap[size],
          isActive ? colors.glowActive : colors.glow
        )}
        animate={{
          scale: isActive ? [1, 1.1 + intensity * 0.2, 1] : [0.95, 1, 0.95],
          opacity: isActive ? [0.3, 0.5 + intensity * 0.3, 0.3] : [0.1, 0.2, 0.1],
        }}
        transition={{
          duration: isActive ? 1.5 : 3,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />

      {/* Middle ring */}
      <motion.div
        className={cn(
          "absolute rounded-full bg-gradient-to-br opacity-20",
          sizeMap[size],
          colors.inner
        )}
        animate={{
          scale: isActive ? [0.9, 1.05 + intensity * 0.15, 0.9] : [0.85, 0.95, 0.85],
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 0.3,
        }}
        style={{ filter: "blur(20px)" }}
      />

      {/* Core orb */}
      <motion.div
        className={cn(
          "rounded-full bg-gradient-to-br",
          sizeMap[size],
          colors.inner
        )}
        animate={{ scale }}
        transition={{
          type: "spring",
          stiffness: 200,
          damping: 15,
        }}
        style={{
          boxShadow: isActive
            ? `0 0 ${30 + intensity * 40}px rgba(118,185,0,${0.3 + intensity * 0.3})`
            : "0 0 20px rgba(118,185,0,0.15)",
        }}
      />

      {/* Inner highlight */}
      <motion.div
        className="absolute w-1/3 h-1/3 rounded-full bg-white/20 blur-sm -translate-x-1/4 -translate-y-1/4"
        animate={{
          opacity: isActive ? [0.3, 0.6, 0.3] : [0.1, 0.2, 0.1],
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />
    </div>
  );
};


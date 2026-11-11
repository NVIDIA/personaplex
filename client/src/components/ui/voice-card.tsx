import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { Mic } from "lucide-react";

interface VoiceCardProps {
  name: string;
  value: string;
  isSelected: boolean;
  onSelect: (value: string) => void;
}

export const VoiceCard: React.FC<VoiceCardProps> = ({
  name,
  value,
  isSelected,
  onSelect,
}) => {
  return (
    <motion.button
      whileHover={{ scale: 1.03 }}
      whileTap={{ scale: 0.97 }}
      onClick={() => onSelect(value)}
      className={cn(
        "relative flex flex-col items-center gap-2 p-4 rounded-xl border transition-all duration-300 cursor-pointer",
        isSelected
          ? "border-[#76B900]/60 bg-[#76B900]/10 shadow-[0_0_20px_rgba(118,185,0,0.15)]"
          : "border-white/10 bg-white/[0.03] hover:border-white/20 hover:bg-white/[0.06]"
      )}
    >
      <div
        className={cn(
          "w-10 h-10 rounded-full flex items-center justify-center transition-colors duration-300",
          isSelected ? "bg-[#76B900]/20" : "bg-white/5"
        )}
      >
        <Mic
          className={cn(
            "w-5 h-5 transition-colors duration-300",
            isSelected ? "text-[#76B900]" : "text-white/40"
          )}
        />
      </div>
      <span
        className={cn(
          "text-xs font-medium transition-colors duration-300 truncate w-full text-center",
          isSelected ? "text-[#76B900]" : "text-white/60"
        )}
      >
        {name}
      </span>
      {isSelected && (
        <motion.div
          layoutId="voice-indicator"
          className="absolute -bottom-px left-2 right-2 h-0.5 bg-[#76B900] rounded-full"
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
        />
      )}
    </motion.button>
  );
};


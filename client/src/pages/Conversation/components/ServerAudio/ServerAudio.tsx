import { FC, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, X } from "lucide-react";
import { AudioStats, useServerAudio } from "../../hooks/useServerAudio";
import { ServerVisualizer } from "../AudioVisualizer/ServerVisualizer";
import { type ThemeType } from "../../hooks/useSystemTheme";

type ServerAudioProps = {
  setGetAudioStats: (getAudioStats: () => AudioStats) => void;
  theme: ThemeType;
};
export const ServerAudio: FC<ServerAudioProps> = ({ setGetAudioStats, theme }) => {
  const { analyser, hasCriticalDelay, setHasCriticalDelay } = useServerAudio({
    setGetAudioStats,
  });
  const containerRef = useRef<HTMLDivElement>(null);
  return (
    <>
      <AnimatePresence>
        {hasCriticalDelay && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="fixed left-0 top-0 z-50 flex w-screen items-center justify-between bg-red-500/90 backdrop-blur-sm px-4 py-2.5 text-sm text-white"
          >
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4" />
              <p>A connection issue has been detected, you've been reconnected</p>
            </div>
            <button
              onClick={() => setHasCriticalDelay(false)}
              className="p-1 rounded-lg hover:bg-white/20 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </motion.div>
        )}
      </AnimatePresence>
      <div className="w-full flex flex-col items-center">
        <p className="text-xs text-white/40 mb-3 font-medium">AI Response</p>
        <div className="w-48 h-48 md:w-56 md:h-56" ref={containerRef}>
          <ServerVisualizer analyser={analyser.current} parent={containerRef} theme={theme} />
        </div>
      </div>
    </>
  );
};

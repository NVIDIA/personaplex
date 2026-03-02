import moshiProcessorUrl from "../../audio-processor.ts?worker&url";
import { FC, useEffect, useState, useCallback, useRef, MutableRefObject } from "react";
import eruda from "eruda";
import { useSearchParams } from "react-router-dom";
import { motion } from "framer-motion";
import { Mic, Sparkles, Zap, MessageSquare, Stethoscope, Landmark, Rocket } from "lucide-react";
import { Conversation } from "../Conversation/Conversation";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { GlassPanel } from "@/components/ui/glass-panel";
import { AuroraBackground } from "@/components/ui/aurora-background";
import { VoiceCard } from "@/components/ui/voice-card";
import { useModelParams } from "../Conversation/hooks/useModelParams";
import { env } from "../../env";
import { prewarmDecoderWorker } from "../../decoder/decoderWorker";

const VOICE_OPTIONS = [
  "NATF0.pt", "NATF1.pt", "NATF2.pt", "NATF3.pt",
  "NATM0.pt", "NATM1.pt", "NATM2.pt", "NATM3.pt",
  "VARF0.pt", "VARF1.pt", "VARF2.pt", "VARF3.pt", "VARF4.pt",
  "VARM0.pt", "VARM1.pt", "VARM2.pt", "VARM3.pt", "VARM4.pt",
];

const TEXT_PROMPT_PRESETS = [
  {
    label: "Assistant",
    icon: Sparkles,
    text: "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.",
  },
  {
    label: "Medical Office",
    icon: Stethoscope,
    text: "You work for Dr. Jones's medical office, and you are receiving calls to record information for new patients. Information: Record full name, date of birth, any medication allergies, tobacco smoking history, alcohol consumption history, and any prior medical conditions. Assure the patient that this information will be confidential, if they ask.",
  },
  {
    label: "Bank Service",
    icon: Landmark,
    text: "You work for First Neuron Bank which is a bank and your name is Alexis Kim. Information: The customer's transaction for $1,200 at Home Depot was declined. Verify customer identity. The transaction was flagged due to unusual location (transaction attempted in Miami, FL; customer normally transacts in Seattle, WA).",
  },
  {
    label: "Astronaut",
    icon: Rocket,
    text: "You enjoy having a good conversation. Have a technical discussion about fixing a reactor core on a spaceship to Mars. You are an astronaut on a Mars mission. Your name is Alex. You are already dealing with a reactor core meltdown on a Mars mission. Several ship systems are failing, and continued instability will lead to catastrophic failure. You explain what is happening and you urgently ask for help thinking through how to stabilize the reactor.",
  },
];

interface HomepageProps {
  showMicrophoneAccessMessage: boolean;
  startConnection: () => Promise<void>;
  textPrompt: string;
  setTextPrompt: (value: string) => void;
  voicePrompt: string;
  setVoicePrompt: (value: string) => void;
}

const Homepage = ({
  startConnection,
  showMicrophoneAccessMessage,
  textPrompt,
  setTextPrompt,
  voicePrompt,
  setVoicePrompt,
}: HomepageProps) => {
  return (
    <AuroraBackground className="min-h-screen">
      <div className="flex flex-col items-center px-4 py-8 sm:py-12 w-full max-w-4xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-8 sm:mb-12"
        >
          <div className="flex items-center justify-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-xl bg-[#76B900]/20 flex items-center justify-center">
              <Zap className="w-5 h-5 text-[#76B900]" />
            </div>
            <h1 className="text-4xl sm:text-5xl font-bold bg-gradient-to-r from-white via-white to-white/60 bg-clip-text text-transparent">
              PersonaPlex
            </h1>
          </div>
          <p className="text-sm sm:text-base text-white/50 max-w-md mx-auto">
            Full-duplex conversational AI with real-time voice and text control
          </p>
        </motion.div>

        {/* Persona Presets */}
        <GlassPanel className="w-full mb-6" delay={0.1}>
          <div className="flex items-center gap-2 mb-4">
            <MessageSquare className="w-4 h-4 text-[#76B900]" />
            <h2 className="text-sm font-semibold text-white/80">Persona Presets</h2>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {TEXT_PROMPT_PRESETS.map((preset) => {
              const Icon = preset.icon;
              const isActive = textPrompt === preset.text;
              return (
                <motion.button
                  key={preset.label}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => setTextPrompt(preset.text)}
                  className={`flex flex-col items-center gap-2 p-4 rounded-xl border transition-all duration-300 cursor-pointer ${isActive
                    ? "border-[#76B900]/50 bg-[#76B900]/10 shadow-[0_0_15px_rgba(118,185,0,0.1)]"
                    : "border-white/10 bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.05]"
                    }`}
                >
                  <Icon className={`w-5 h-5 ${isActive ? "text-[#76B900]" : "text-white/40"}`} />
                  <span className={`text-xs font-medium ${isActive ? "text-[#76B900]" : "text-white/60"}`}>
                    {preset.label}
                  </span>
                </motion.button>
              );
            })}
          </div>
        </GlassPanel>

        {/* Text Prompt */}
        <GlassPanel className="w-full mb-6" delay={0.2}>
          <div className="flex items-center justify-between mb-3">
            <label htmlFor="text-prompt" className="text-sm font-semibold text-white/80">
              System Prompt
            </label>
            <span className="text-xs text-white/30">{textPrompt.length}/1000</span>
          </div>
          <Textarea
            id="text-prompt"
            value={textPrompt}
            onChange={(e) => setTextPrompt(e.target.value)}
            placeholder="Describe the AI persona you want to talk to..."
            maxLength={1000}
            className="min-h-[120px]"
          />
        </GlassPanel>

        {/* Voice Selection */}
        <GlassPanel className="w-full mb-8" delay={0.3}>
          <div className="flex items-center gap-2 mb-4">
            <Mic className="w-4 h-4 text-[#76B900]" />
            <h2 className="text-sm font-semibold text-white/80">Voice Selection</h2>
          </div>
          <div className="grid grid-cols-3 sm:grid-cols-6 gap-2">
            {VOICE_OPTIONS.map((voice) => (
              <VoiceCard
                key={voice}
                name={voice.replace(".pt", "").replace(/^NAT/, "Natural ").replace(/^VAR/, "Variety ")}
                value={voice}
                isSelected={voicePrompt === voice}
                onSelect={setVoicePrompt}
              />
            ))}
          </div>
        </GlassPanel>

        {/* Microphone Warning */}
        {showMicrophoneAccessMessage && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="mb-4 px-4 py-3 rounded-xl border border-red-500/30 bg-red-500/10 text-red-400 text-sm text-center"
          >
            Please enable your microphone before proceeding
          </motion.div>
        )}

        {/* Connect Button */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <Button
            variant="glow"
            size="xl"
            onClick={async () => await startConnection()}
            className="gap-2"
          >
            <Mic className="w-5 h-5" />
            Start Conversation
          </Button>
        </motion.div>
      </div>
    </AuroraBackground>
  );
}

export const Queue: FC = () => {
  const theme = "dark" as const;
  const [searchParams] = useSearchParams();
  const overrideWorkerAddr = searchParams.get("worker_addr");
  const [hasMicrophoneAccess, setHasMicrophoneAccess] = useState<boolean>(false);
  const [showMicrophoneAccessMessage, setShowMicrophoneAccessMessage] = useState<boolean>(false);
  const modelParams = useModelParams();

  const audioContext = useRef<AudioContext | null>(null);
  const worklet = useRef<AudioWorkletNode | null>(null);

  // enable eruda in development
  useEffect(() => {
    if (env.VITE_ENV === "development") {
      eruda.init();
    }
    () => {
      if (env.VITE_ENV === "development") {
        eruda.destroy();
      }
    };
  }, []);

  const getMicrophoneAccess = useCallback(async () => {
    try {
      await window.navigator.mediaDevices.getUserMedia({ audio: true });
      setHasMicrophoneAccess(true);
      return true;
    } catch (e) {
      console.error(e);
      setShowMicrophoneAccessMessage(true);
      setHasMicrophoneAccess(false);
    }
    return false;
  }, [setHasMicrophoneAccess, setShowMicrophoneAccessMessage]);

  const startProcessor = useCallback(async () => {
    if (!audioContext.current) {
      audioContext.current = new AudioContext();
      // Prewarm decoder worker as soon as we have audio context
      // This gives WASM time to load while user grants mic access
      prewarmDecoderWorker(audioContext.current.sampleRate);
    }
    if (worklet.current) {
      return;
    }
    let ctx = audioContext.current;
    ctx.resume();
    try {
      worklet.current = new AudioWorkletNode(ctx, 'moshi-processor');
    } catch (err) {
      await ctx.audioWorklet.addModule(moshiProcessorUrl);
      worklet.current = new AudioWorkletNode(ctx, 'moshi-processor');
    }
    worklet.current.connect(ctx.destination);
  }, [audioContext, worklet]);

  const startConnection = useCallback(async () => {
    await startProcessor();
    const hasAccess = await getMicrophoneAccess();
    if (hasAccess) {
      // Values are already set in modelParams, they get passed to Conversation
    }
  }, [startProcessor, getMicrophoneAccess]);

  return (
    <>
      {(hasMicrophoneAccess && audioContext.current && worklet.current) ? (
        <Conversation
          workerAddr={overrideWorkerAddr ?? ""}
          audioContext={audioContext as MutableRefObject<AudioContext | null>}
          worklet={worklet as MutableRefObject<AudioWorkletNode | null>}
          theme={theme}
          startConnection={startConnection}
          {...modelParams}
        />
      ) : (
        <Homepage
          startConnection={startConnection}
          showMicrophoneAccessMessage={showMicrophoneAccessMessage}
          textPrompt={modelParams.textPrompt}
          setTextPrompt={modelParams.setTextPrompt}
          voicePrompt={modelParams.voicePrompt}
          setVoicePrompt={modelParams.setVoicePrompt}
        />
      )}
    </>
  );
};

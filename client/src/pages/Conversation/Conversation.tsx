import { FC, MutableRefObject, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { PhoneOff, RefreshCw, Download, Activity, MessageSquare, BarChart3 } from "lucide-react";
import { useSocket } from "./hooks/useSocket";
import { SocketContext } from "./SocketContext";
import { ServerAudio } from "./components/ServerAudio/ServerAudio";
import { UserAudio } from "./components/UserAudio/UserAudio";
import { Button } from "@/components/ui/button";
import { GlassPanel } from "@/components/ui/glass-panel";
import { StatusIndicator } from "@/components/ui/status-indicator";
import { ServerAudioStats } from "./components/ServerAudio/ServerAudioStats";
import { AudioStats } from "./hooks/useServerAudio";
import { TextDisplay } from "./components/TextDisplay/TextDisplay";
import { MediaContext } from "./MediaContext";
import { ServerInfo } from "./components/ServerInfo/ServerInfo";
import { ModelParamsValues, useModelParams } from "./hooks/useModelParams";
import fixWebmDuration from "webm-duration-fix";
import { getMimeType, getExtension } from "./getMimeType";
import { type ThemeType } from "./hooks/useSystemTheme";

type ConversationProps = {
  workerAddr: string;
  workerAuthId?: string;
  sessionAuthId?: string;
  sessionId?: number;
  email?: string;
  theme: ThemeType;
  audioContext: MutableRefObject<AudioContext | null>;
  worklet: MutableRefObject<AudioWorkletNode | null>;
  onConversationEnd?: () => void;
  isBypass?: boolean;
  startConnection: () => Promise<void>;
} & Partial<ModelParamsValues>;


const buildURL = ({
  workerAddr,
  params,
  workerAuthId,
  email,
  textSeed,
  audioSeed,
}: {
  workerAddr: string;
  params: ModelParamsValues;
  workerAuthId?: string;
  email?: string;
  textSeed: number;
  audioSeed: number;
}) => {
  const newWorkerAddr = useMemo(() => {
    if (workerAddr == "same" || workerAddr == "") {
      const newWorkerAddr = window.location.hostname + ":" + window.location.port;
      console.log("Overriding workerAddr to", newWorkerAddr);
      return newWorkerAddr;
    }
    return workerAddr;
  }, [workerAddr]);
  const wsProtocol = (window.location.protocol === 'https:') ? 'wss' : 'ws';
  const url = new URL(`${wsProtocol}://${newWorkerAddr}/api/chat`);
  if (workerAuthId) {
    url.searchParams.append("worker_auth_id", workerAuthId);
  }
  if (email) {
    url.searchParams.append("email", email);
  }
  url.searchParams.append("text_temperature", params.textTemperature.toString());
  url.searchParams.append("text_topk", params.textTopk.toString());
  url.searchParams.append("audio_temperature", params.audioTemperature.toString());
  url.searchParams.append("audio_topk", params.audioTopk.toString());
  url.searchParams.append("pad_mult", params.padMult.toString());
  url.searchParams.append("text_seed", textSeed.toString());
  url.searchParams.append("audio_seed", audioSeed.toString());
  url.searchParams.append("repetition_penalty_context", params.repetitionPenaltyContext.toString());
  url.searchParams.append("repetition_penalty", params.repetitionPenalty.toString());
  url.searchParams.append("text_prompt", params.textPrompt.toString());
  url.searchParams.append("voice_prompt", params.voicePrompt.toString());

  // Attach API key for backend authentication
  const apiKey = import.meta.env.VITE_PERSONAPLEX_API_KEY;
  if (apiKey) {
    url.searchParams.append("api_key", apiKey);
  }

  console.log(url.toString());
  return url.toString();
};


export const Conversation: FC<ConversationProps> = ({
  workerAddr,
  workerAuthId,
  audioContext,
  worklet,
  sessionAuthId,
  sessionId,
  onConversationEnd,
  startConnection,
  isBypass = false,
  email,
  theme,
  ...params
}) => {
  const getAudioStats = useRef<() => AudioStats>(() => ({
    playedAudioDuration: 0,
    missedAudioDuration: 0,
    totalAudioMessages: 0,
    delay: 0,
    minPlaybackDelay: 0,
    maxPlaybackDelay: 0,
  }));
  const isRecording = useRef<boolean>(false);
  const audioChunks = useRef<Blob[]>([]);

  const audioStreamDestination = useRef<MediaStreamAudioDestinationNode>(audioContext.current!.createMediaStreamDestination());
  const stereoMerger = useRef<ChannelMergerNode>(audioContext.current!.createChannelMerger(2));
  const audioRecorder = useRef<MediaRecorder>(new MediaRecorder(audioStreamDestination.current.stream, { mimeType: getMimeType("audio"), audioBitsPerSecond: 128000 }));
  const [audioURL, setAudioURL] = useState<string>("");
  const [isOver, setIsOver] = useState(false);
  const modelParams = useModelParams(params);
  const micDuration = useRef<number>(0);
  const actualAudioPlayed = useRef<number>(0);
  const textContainerRef = useRef<HTMLDivElement>(null);
  const textSeed = useMemo(() => Math.round(1000000 * Math.random()), []);
  const audioSeed = useMemo(() => Math.round(1000000 * Math.random()), []);

  const WSURL = buildURL({
    workerAddr,
    params: modelParams,
    workerAuthId,
    email: email,
    textSeed: textSeed,
    audioSeed: audioSeed,
  });

  const onDisconnect = useCallback(() => {
    setIsOver(true);
    console.log("on disconnect!");
    stopRecording();
  }, [setIsOver]);

  const { socketStatus, sendMessage, socket, start, stop } = useSocket({
    // onMessage,
    uri: WSURL,
    onDisconnect,
  });
  useEffect(() => {
    audioRecorder.current.ondataavailable = (e) => {
      audioChunks.current.push(e.data);
    };
    audioRecorder.current.onstop = async () => {
      let blob: Blob;
      const mimeType = getMimeType("audio");
      if (mimeType.includes("webm")) {
        blob = await fixWebmDuration(new Blob(audioChunks.current, { type: mimeType }));
      } else {
        blob = new Blob(audioChunks.current, { type: mimeType });
      }
      setAudioURL(URL.createObjectURL(blob));
      audioChunks.current = [];
      console.log("Audio Recording and encoding finished");
    };
  }, [audioRecorder, setAudioURL, audioChunks]);


  useEffect(() => {
    start();
    return () => {
      stop();
    };
  }, [start, workerAuthId]);

  const startRecording = useCallback(() => {
    if (isRecording.current) {
      return;
    }
    console.log(Date.now() % 1000, "Starting recording");
    console.log("Starting recording");
    // Build stereo routing for recording: left = server (worklet), right = user mic (connected in useUserAudio)
    try {
      stereoMerger.current.disconnect();
    } catch { }
    try {
      worklet.current?.disconnect(audioStreamDestination.current);
    } catch { }
    // Route server audio (mono) to left channel of merger
    worklet.current?.connect(stereoMerger.current, 0, 0);
    // Connect merger to the MediaStream destination
    stereoMerger.current.connect(audioStreamDestination.current);

    setAudioURL("");
    audioRecorder.current.start();
    isRecording.current = true;
  }, [isRecording, worklet, audioStreamDestination, audioRecorder, stereoMerger]);

  const stopRecording = useCallback(() => {
    console.log("Stopping recording");
    console.log("isRecording", isRecording)
    if (!isRecording.current) {
      return;
    }
    try {
      worklet.current?.disconnect(stereoMerger.current);
    } catch { }
    try {
      stereoMerger.current.disconnect(audioStreamDestination.current);
    } catch { }
    audioRecorder.current.stop();
    isRecording.current = false;
  }, [isRecording, worklet, audioStreamDestination, audioRecorder, stereoMerger]);

  const onPressConnect = useCallback(async () => {
    if (isOver) {
      window.location.reload();
    } else {
      audioContext.current?.resume();
      if (socketStatus !== "connected") {
        start();
      } else {
        stop();
      }
    }
  }, [socketStatus, isOver, start, stop]);

  return (
    <SocketContext.Provider
      value={{
        socketStatus,
        sendMessage,
        socket,
      }}
    >
      <div className="min-h-screen bg-[#0a0a0f] text-white">
        {/* Top Bar */}
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="sticky top-0 z-50 border-b border-white/5 bg-[#0a0a0f]/80 backdrop-blur-xl"
        >
          <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <h1 className="text-lg font-semibold text-white/90">PersonaPlex</h1>
              <StatusIndicator status={socketStatus} />
            </div>
            <div className="flex items-center gap-2">
              <AnimatePresence>
                {audioURL && (
                  <motion.a
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.9 }}
                    href={audioURL}
                    download={`personaplex_audio.${getExtension("audio")}`}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-white/5 hover:bg-white/10 border border-white/10 transition-colors"
                  >
                    <Download className="w-3.5 h-3.5" />
                    Download
                  </motion.a>
                )}
              </AnimatePresence>
              <Button
                variant={isOver ? "glow" : socketStatus === "connected" ? "destructive" : "outline"}
                size="sm"
                onClick={onPressConnect}
                disabled={socketStatus !== "connected" && !isOver}
                className="gap-1.5"
              >
                {isOver ? (
                  <><RefreshCw className="w-3.5 h-3.5" /> New Conversation</>
                ) : socketStatus === "connected" ? (
                  <><PhoneOff className="w-3.5 h-3.5" /> Disconnect</>
                ) : (
                  "Connecting..."
                )}
              </Button>
            </div>
          </div>
        </motion.div>

        {/* Main Content */}
        <div className="max-w-7xl mx-auto px-4 py-6">
          {audioContext.current && worklet.current && (
            <MediaContext.Provider
              value={{
                startRecording,
                stopRecording,
                audioContext: audioContext as MutableRefObject<AudioContext>,
                worklet: worklet as MutableRefObject<AudioWorkletNode>,
                audioStreamDestination,
                stereoMerger,
                micDuration,
                actualAudioPlayed,
              }}
            >
              <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
                {/* Left: Audio Visualizers */}
                <div className="lg:col-span-2 flex flex-col gap-6">
                  <GlassPanel className="flex flex-col items-center justify-center py-8" delay={0.1}>
                    <ServerAudio
                      setGetAudioStats={(callback: () => AudioStats) =>
                        (getAudioStats.current = callback)
                      }
                      theme={theme}
                    />
                  </GlassPanel>
                  <GlassPanel className="flex flex-col items-center justify-center py-6" delay={0.2}>
                    <UserAudio theme={theme} />
                  </GlassPanel>
                </div>

                {/* Right: Text + Stats */}
                <div className="lg:col-span-3 flex flex-col gap-6">
                  <GlassPanel className="flex-1 min-h-[300px] lg:min-h-[400px] flex flex-col" delay={0.15}>
                    <div className="flex items-center gap-2 mb-3 pb-3 border-b border-white/5">
                      <MessageSquare className="w-4 h-4 text-[#76B900]" />
                      <h2 className="text-sm font-semibold text-white/70">Conversation</h2>
                    </div>
                    <div className="flex-1 overflow-auto scrollbar" ref={textContainerRef}>
                      <TextDisplay containerRef={textContainerRef} />
                    </div>
                  </GlassPanel>

                  <GlassPanel className="hidden md:block" delay={0.25}>
                    <div className="flex items-center gap-2 mb-3 pb-3 border-b border-white/5">
                      <BarChart3 className="w-4 h-4 text-[#76B900]" />
                      <h2 className="text-sm font-semibold text-white/70">Audio Stats</h2>
                    </div>
                    <ServerAudioStats getAudioStats={getAudioStats} />
                  </GlassPanel>
                </div>
              </div>
            </MediaContext.Provider>
          )}
        </div>

        {/* Server Info Footer */}
        <div className="max-w-7xl mx-auto px-4 py-4">
          <GlassPanel animate={false} className="p-4">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-3.5 h-3.5 text-white/30" />
              <span className="text-xs font-medium text-white/30">Server Info</span>
            </div>
            <ServerInfo />
          </GlassPanel>
        </div>
      </div>
    </SocketContext.Provider>
  );
};

import moshiProcessorUrl from "../../audio-processor.ts?worker&url";
import { FC, useEffect, useState, useCallback, useRef, MutableRefObject } from "react";
import eruda from "eruda";
import { useSearchParams } from "react-router-dom";
import { Conversation } from "../Conversation/Conversation";
import { Button } from "../../components/Button/Button";
import { useModelParams } from "../Conversation/hooks/useModelParams";
import { env } from "../../env";
import { prewarmDecoderWorker } from "../../decoder/decoderWorker";
import { useI18n, Language } from "../../i18n";

const VOICE_OPTIONS = [
  "NATF0.pt", "NATF1.pt", "NATF2.pt", "NATF3.pt",
  "NATM0.pt", "NATM1.pt", "NATM2.pt", "NATM3.pt",
  "VARF0.pt", "VARF1.pt", "VARF2.pt", "VARF3.pt", "VARF4.pt",
  "VARM0.pt", "VARM1.pt", "VARM2.pt", "VARM3.pt", "VARM4.pt",
];

const KOREAN_VOICE_OPTIONS = [
  { key: "ko_female_1", label: "한국어 여성 1 (Natural)" },
  { key: "ko_female_2", label: "한국어 여성 2 (Expressive)" },
  { key: "ko_male_1", label: "한국어 남성 1 (Natural)" },
  { key: "ko_male_2", label: "한국어 남성 2 (Expressive)" },
];

const TEXT_PROMPT_PRESETS = [
  {
    label: "Assistant (default)",
    text: "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.",
  },
  {
    label: "Medical office (service)",
    text: "You work for Dr. Jones's medical office, and you are receiving calls to record information for new patients. Information: Record full name, date of birth, any medication allergies, tobacco smoking history, alcohol consumption history, and any prior medical conditions. Assure the patient that this information will be confidential, if they ask.",
  },
  {
    label: "Bank (service)",
    text: "You work for First Neuron Bank which is a bank and your name is Alexis Kim. Information: The customer's transaction for $1,200 at Home Depot was declined. Verify customer identity. The transaction was flagged due to unusual location (transaction attempted in Miami, FL; customer normally transacts in Seattle, WA).",
  },
  {
    label: "Astronaut (fun)",
    text: "You enjoy having a good conversation. Have a technical discussion about fixing a reactor core on a spaceship to Mars. You are an astronaut on a Mars mission. Your name is Alex. You are already dealing with a reactor core meltdown on a Mars mission. Several ship systems are failing, and continued instability will lead to catastrophic failure. You explain what is happening and you urgently ask for help thinking through how to stabilize the reactor.",
  },
];

const KOREAN_TEXT_PROMPT_PRESETS = [
  {
    label: "AI 비서 (기본)",
    text: "당신은 친절한 AI 비서입니다. 사용자의 질문에 한국어로 자연스럽게 대답하세요.",
  },
  {
    label: "은행 상담 (서비스)",
    text: "당신은 은행 고객 서비스 상담원입니다. 정중하게 고객을 도와주세요.",
  },
  {
    label: "의료 상담 (서비스)",
    text: "당신은 의료 상담 안내원입니다. 환자의 질문에 친절하게 답변하세요.",
  },
  {
    label: "우주비행사 (재미)",
    text: "당신은 화성 임무 중인 우주비행사입니다. 우주선의 원자로 문제를 해결하기 위해 도움을 요청하고 있습니다. 긴급한 상황을 설명하고 함께 해결책을 찾아보세요.",
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
  const { language, setLanguage, t } = useI18n();

  const presets = language === "ko" ? KOREAN_TEXT_PROMPT_PRESETS : TEXT_PROMPT_PRESETS;
  const isKorean = language === "ko";

  return (
    <div className="text-center h-screen w-screen p-4 flex flex-col items-center pt-8">
      <div className="mb-6">
        <h1 className="text-4xl text-black">{t("app.title")}</h1>
        <p className="text-sm text-gray-600 mt-2">
          {t("app.description")}
        </p>
      </div>

      <div className="flex flex-grow justify-center items-center flex-col gap-6 w-full min-w-[500px] max-w-2xl">
        {/* Language selector */}
        <div className="w-full">
          <label className="block text-left text-base font-medium text-gray-700 mb-2">
            {t("queue.languageLabel")}
          </label>
          <div className="flex gap-2">
            <button
              onClick={() => {
                setLanguage("en");
                setTextPrompt(TEXT_PROMPT_PRESETS[0].text);
                setVoicePrompt(VOICE_OPTIONS[0]);
              }}
              className={`flex-1 px-4 py-2 rounded border transition-colors ${
                language === "en"
                  ? "bg-[#76b900] text-white border-[#76b900]"
                  : "bg-white text-gray-700 border-gray-300 hover:bg-gray-50"
              }`}
            >
              EN (English)
            </button>
            <button
              onClick={() => {
                setLanguage("ko");
                setTextPrompt(KOREAN_TEXT_PROMPT_PRESETS[0].text);
                setVoicePrompt(KOREAN_VOICE_OPTIONS[0].key);
              }}
              className={`flex-1 px-4 py-2 rounded border transition-colors ${
                language === "ko"
                  ? "bg-[#76b900] text-white border-[#76b900]"
                  : "bg-white text-gray-700 border-gray-300 hover:bg-gray-50"
              }`}
            >
              한국어 (Korean)
            </button>
          </div>
        </div>

        <div className="w-full">
          <label htmlFor="text-prompt" className="block text-left text-base font-medium text-gray-700 mb-2">
            {t("queue.textPromptLabel")}
          </label>
          <div className="border border-gray-300 rounded p-3 mb-3 bg-gray-50">
            <span className="text-xs font-medium text-gray-500 block mb-2">{t("queue.examplesLabel")}</span>
            <div className="flex flex-wrap gap-2 justify-center">
              {presets.map((preset) => (
                <button
                  key={preset.label}
                  onClick={() => setTextPrompt(preset.text)}
                  className="px-3 py-1 text-xs bg-white hover:bg-gray-100 text-gray-700 rounded-full border border-gray-300 transition-colors focus:outline-none focus:ring-2 focus:ring-[#76b900]"
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>
          <textarea
            id="text-prompt"
            name="text-prompt"
            value={textPrompt}
            onChange={(e) => setTextPrompt(e.target.value)}
            className="w-full h-32 min-h-[80px] max-h-64 p-3 bg-white text-black border border-gray-300 rounded resize-y focus:outline-none focus:ring-2 focus:ring-[#76b900] focus:border-transparent"
            placeholder={t("queue.textPromptPlaceholder")}
            maxLength={1000}
          />
          <div className="text-right text-xs text-gray-500 mt-1">
            {textPrompt.length}/1000
          </div>
        </div>

        <div className="w-full">
          <label htmlFor="voice-prompt" className="block text-left text-base font-medium text-gray-700 mb-2">
            {t("queue.voiceLabel")}
          </label>
          <select
            id="voice-prompt"
            name="voice-prompt"
            value={voicePrompt}
            onChange={(e) => setVoicePrompt(e.target.value)}
            className="w-full p-3 bg-white text-black border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-[#76b900] focus:border-transparent"
          >
            {isKorean
              ? KOREAN_VOICE_OPTIONS.map((voice) => (
                  <option key={voice.key} value={voice.key}>
                    {voice.label}
                  </option>
                ))
              : VOICE_OPTIONS.map((voice) => (
                  <option key={voice} value={voice}>
                    {voice
                      .replace('.pt', '')
                      .replace(/^NAT/, 'NATURAL_')
                      .replace(/^VAR/, 'VARIETY_')}
                  </option>
                ))}
          </select>
      </div>

        {showMicrophoneAccessMessage && (
          <p className="text-center text-red-500">{t("queue.microphoneError")}</p>
        )}

        <Button onClick={async () => await startConnection()}>{t("queue.connectButton")}</Button>
    </div>
    </div>
  );
}

export const Queue:FC = () => {
  const theme = "light" as const;  // Always use light theme
  const { language } = useI18n();
  const [searchParams] = useSearchParams();
  const overrideWorkerAddr = searchParams.get("worker_addr");
  const [hasMicrophoneAccess, setHasMicrophoneAccess] = useState<boolean>(false);
  const [showMicrophoneAccessMessage, setShowMicrophoneAccessMessage] = useState<boolean>(false);
  const modelParams = useModelParams();

  const audioContext = useRef<AudioContext | null>(null);
  const worklet = useRef<AudioWorkletNode | null>(null);

  // enable eruda in development
  useEffect(() => {
    if(env.VITE_ENV === "development") {
      eruda.init();
    }
    () => {
      if(env.VITE_ENV === "development") {
        eruda.destroy();
      }
    };
  }, []);

  const getMicrophoneAccess = useCallback(async () => {
    try {
      await window.navigator.mediaDevices.getUserMedia({ audio: true });
      setHasMicrophoneAccess(true);
      return true;
    } catch(e) {
      console.error(e);
      setShowMicrophoneAccessMessage(true);
      setHasMicrophoneAccess(false);
    }
    return false;
}, [setHasMicrophoneAccess, setShowMicrophoneAccessMessage]);

  const startProcessor = useCallback(async () => {
    if(!audioContext.current) {
      audioContext.current = new AudioContext();
      // Prewarm decoder worker as soon as we have audio context
      // This gives WASM time to load while user grants mic access
      prewarmDecoderWorker(audioContext.current.sampleRate);
    }
    if(worklet.current) {
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

  const startConnection = useCallback(async() => {
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
        audioContext={audioContext as MutableRefObject<AudioContext|null>}
        worklet={worklet as MutableRefObject<AudioWorkletNode|null>}
        theme={theme}
        startConnection={startConnection}
        language={language}
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

import { FC, useRef } from "react";
import { AudioStats, useServerAudio } from "../../hooks/useServerAudio";
import { ServerVisualizer } from "../AudioVisualizer/ServerVisualizer";
import { type ThemeType } from "../../hooks/useSystemTheme";
import { useI18n } from "../../../../i18n";

type ServerAudioProps = {
  setGetAudioStats: (getAudioStats: () => AudioStats) => void;
  theme: ThemeType;
};
export const ServerAudio: FC<ServerAudioProps> = ({ setGetAudioStats, theme }) => {
  const { analyser, hasCriticalDelay, setHasCriticalDelay } = useServerAudio({
    setGetAudioStats,
  });
  const containerRef = useRef<HTMLDivElement>(null);
  const { t } = useI18n();
  return (
    <>
      {hasCriticalDelay && (
        <div className="fixed left-0 top-0 flex w-screen justify-between bg-red-500 p-2 text-center">
          <p>{t("conversation.connectionIssue")}</p>
          <button
            onClick={async () => {
              setHasCriticalDelay(false);
            }}
            className="bg-white p-1 text-black"
          >
            {t("conversation.dismiss")}
          </button>
        </div>
      )}
      <div className="server-audio h-4/6 aspect-square" ref={containerRef}>
        <ServerVisualizer analyser={analyser.current} parent={containerRef} theme={theme}/>
      </div>
    </>
  );
};

import { useState, useEffect, useRef } from "react";
import { useI18n } from "../../../../i18n";

type ServerAudioStatsProps = {
  getAudioStats: React.MutableRefObject<
    () => {
      playedAudioDuration: number;
      missedAudioDuration: number;
      totalAudioMessages: number;
      delay: number;
      minPlaybackDelay: number;
      maxPlaybackDelay: number;
    }
  >;
};

export const ServerAudioStats = ({ getAudioStats }: ServerAudioStatsProps) => {
  const [audioStats, setAudioStats] = useState(getAudioStats.current());
  const { t } = useI18n();

  const movingAverageSum = useRef<number>(0.);
  const movingAverageCount = useRef<number>(0.);
  const movingBeta = 0.85;

  let convertMinSecs = (total_secs: number) => {
    // convert secs to the format mm:ss.cc
    let mins = (Math.floor(total_secs / 60)).toString();
    let secs = (Math.floor(total_secs) % 60).toString();
    let cents = (Math.floor(100 * (total_secs - Math.floor(total_secs)))).toString();
    if (secs.length < 2) {
      secs = "0" + secs;
    }
    if (cents.length < 2) {
      cents = "0" + cents;
    }
    return mins + ":" + secs + "." + cents;
  };

  useEffect(() => {
    const interval = setInterval(() => {
      const newAudioStats = getAudioStats.current();
      setAudioStats(newAudioStats);
      movingAverageCount.current *= movingBeta;
      movingAverageCount.current += (1 - movingBeta) * 1;
      movingAverageSum.current *= movingBeta;
      movingAverageSum.current += (1 - movingBeta) * newAudioStats.delay;

    }, 141);
    return () => {
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="w-full rounded-lg text-zinc-500 p-2">
      <h2 className="text-md pb-2">{t("stats.title")}</h2>
      <table>
        <tbody>
          <tr>
            <td className="text-md pr-2">{t("stats.audioPlayed")} </td>
            <td>{convertMinSecs(audioStats.playedAudioDuration)}</td>
          </tr>
          <tr>
            <td className="text-md pr-2">{t("stats.missedAudio")} </td>
            <td>{convertMinSecs(audioStats.missedAudioDuration)}</td>
          </tr>
          <tr>
            <td className="text-md pr-2">{t("stats.latency")} </td>
            <td>{(movingAverageSum.current / movingAverageCount.current).toFixed(3)}</td>
          </tr>
          <tr>
            <td className="text-md pr-2">{t("stats.minMaxBuffer")} </td>
            <td>{audioStats.minPlaybackDelay.toFixed(3)} / {audioStats.maxPlaybackDelay.toFixed(3)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

import { useState, useEffect, useRef } from "react";

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

  const latency = movingAverageCount.current > 0
    ? (movingAverageSum.current / movingAverageCount.current).toFixed(3)
    : "0.000";

  const stats = [
    { label: "Audio Played", value: convertMinSecs(audioStats.playedAudioDuration) },
    { label: "Missed Audio", value: convertMinSecs(audioStats.missedAudioDuration) },
    { label: "Latency", value: `${latency}s` },
    { label: "Buffer (Min/Max)", value: `${audioStats.minPlaybackDelay.toFixed(3)} / ${audioStats.maxPlaybackDelay.toFixed(3)}` },
  ];

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
      {stats.map((stat) => (
        <div key={stat.label} className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wider text-white/30 font-medium">{stat.label}</span>
          <span className="text-sm font-mono text-white/70">{stat.value}</span>
        </div>
      ))}
    </div>
  );
};

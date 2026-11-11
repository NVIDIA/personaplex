import { useServerInfo } from "../../hooks/useServerInfo";

export const ServerInfo = () => {
  const { serverInfo } = useServerInfo();
  if (!serverInfo) {
    return null;
  }

  const items = [
    { label: "Text Temp", value: serverInfo.text_temperature },
    { label: "Text TopK", value: serverInfo.text_topk },
    { label: "Audio Temp", value: serverInfo.audio_temperature },
    { label: "Audio TopK", value: serverInfo.audio_topk },
    { label: "Pad Mult", value: serverInfo.pad_mult },
    { label: "Repeat Penalty N", value: serverInfo.repetition_penalty_context },
    { label: "Repeat Penalty", value: serverInfo.repetition_penalty },
    { label: "Model", value: serverInfo.lm_model_file },
    { label: "Instance", value: serverInfo.instance_name },
  ];

  return (
    <div className="flex flex-wrap gap-x-6 gap-y-2">
      {items.map((item) => (
        <div key={item.label} className="flex items-center gap-1.5 text-xs">
          <span className="text-white/30">{item.label}:</span>
          <span className="text-white/50 font-mono">{item.value}</span>
        </div>
      ))}
    </div>
  );
};

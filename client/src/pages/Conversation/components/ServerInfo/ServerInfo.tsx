import { useServerInfo } from "../../hooks/useServerInfo";
import { useI18n } from "../../../../i18n";

export const ServerInfo = () => {
  const { serverInfo } = useServerInfo();
  const { t } = useI18n();
  if (!serverInfo) {
    return null;
  }
  return (
    <div className="p-2 pt-4 self-center flex flex-col break-words">
      {t("serverInfo.header")}
        <div>{t("serverInfo.textTemperature")}: {serverInfo.text_temperature}</div>
        <div>{t("serverInfo.textTopk")}: {serverInfo.text_topk}</div>
        <div>{t("serverInfo.audioTemperature")}: {serverInfo.audio_temperature}</div>
        <div>{t("serverInfo.audioTopk")}: {serverInfo.audio_topk}</div>
        <div>{t("serverInfo.padMult")}: {serverInfo.pad_mult}</div>
        <div>{t("serverInfo.repeatPenaltyLastN")}: {serverInfo.repetition_penalty_context}</div>
        <div>{t("serverInfo.repeatPenalty")}: {serverInfo.repetition_penalty}</div>
        <div>{t("serverInfo.lmModelFile")}: {serverInfo.lm_model_file}</div>
        <div>{t("serverInfo.instanceName")}: {serverInfo.instance_name}</div>
    </div>
  );
};

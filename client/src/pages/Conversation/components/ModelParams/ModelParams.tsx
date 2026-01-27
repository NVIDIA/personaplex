import { FC, RefObject, useState } from "react";
import { useModelParams } from "../../hooks/useModelParams";
import { Button } from "../../../../components/Button/Button";
import { useVoices } from "../../../../hooks/useVoices";

type ModelParamsProps = {
  isConnected: boolean;
  modal?: RefObject<HTMLDialogElement>,
} &  ReturnType<typeof useModelParams>;
export const ModelParams:FC<ModelParamsProps> = ({
  textTemperature,
  textTopk,
  audioTemperature,
  audioTopk,
  padMult,
  repetitionPenalty,
  repetitionPenaltyContext,
  setParams,
  resetParams,
  isConnected,
  textPrompt,
  voicePrompt,
  randomSeed,
  modal,
}) => {
  const [modalVoicePrompt, setModalVoicePrompt] = useState<string>(voicePrompt);
  const [modalTextPrompt, setModalTextPrompt] = useState<string>(textPrompt);
  const { voices, loading: voicesLoading, error: voicesError } = useVoices();
  return (
    <div className=" p-2 mt-6 self-center flex flex-col items-center text-center">
        <table>
          <tbody>
            <tr>
              <td>Text Prompt:</td>
              <td className="w-12 text-center">{modalTextPrompt}</td>
              <td className="p-2"><input className="align-middle bg-white text-black border border-gray-300 rounded px-2 py-1" disabled={isConnected} type="text" id="text-prompt" name="text-prompt" value={modalTextPrompt} onChange={e => setModalTextPrompt(e.target.value)} /></td>
            </tr>
            <tr>
              <td>Voice Prompt:</td>
              <td className="w-12 text-center">{modalVoicePrompt}</td>
              <td className="p-2">
                <select
                  className="align-middle bg-white text-black border border-gray-300 rounded px-2 py-1"
                  disabled={isConnected || voicesLoading}
                  id="voice-prompt"
                  name="voice-prompt"
                  value={modalVoicePrompt}
                  onChange={e => setModalVoicePrompt(e.target.value)}
                >
                  {voicesLoading ? (
                    <option value="">Loading voices...</option>
                  ) : voicesError ? (
                    <option value="">Error loading voices</option>
                  ) : (
                    voices.map((voice) => (
                      <option key={voice.name} value={voice.name}>
                        {voice.name}
                      </option>
                    ))
                  )}
                </select>
              </td>
            </tr>
          </tbody>
        </table>
        <div>
          <Button onClick={resetParams} className="m-2">Reset</Button>
          <Button onClick={() => {
            console.log("Validating params");
            setParams({
            textTemperature,
            textTopk,
            audioTemperature,
            audioTopk,
            padMult,
            repetitionPenalty,
            repetitionPenaltyContext,
            textPrompt: modalTextPrompt,
            voicePrompt: modalVoicePrompt,
            randomSeed,
          });
          modal?.current?.close()
        }} className="m-2">Validate</Button>
        </div>
    </div>
  )
};

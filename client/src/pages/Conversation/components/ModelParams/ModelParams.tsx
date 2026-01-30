import { FC, RefObject, useState } from "react";
import { useModelParams } from "../../hooks/useModelParams";
import { Button } from "../../../../components/Button/Button";

type ModelParamsProps = {
  isConnected: boolean;
  modal?: RefObject<HTMLDialogElement>,
} & ReturnType<typeof useModelParams>;
export const ModelParams: FC<ModelParamsProps> = ({
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
              <select className="align-middle bg-white text-black border border-gray-300 rounded px-2 py-1" disabled={isConnected} id="voice-prompt" name="voice-prompt" value={modalVoicePrompt} onChange={e => setModalVoicePrompt(e.target.value)}>
                <option value="pepper.pt">Pepper (Robotic)</option>
                <option value="NATF0.pt">NATF0.pt</option>
                <option value="NATF1.pt">NATF1.pt</option>
                <option value="NATF2.pt">NATF2.pt</option>
                <option value="NATF3.pt">NATF3.pt</option>
                <option value="NATM0.pt">NATM0.pt</option>
                <option value="NATM1.pt">NATM1.pt</option>
                <option value="NATM2.pt">NATM2.pt</option>
                <option value="NATM3.pt">NATM3.pt</option>
                <option value="VARF0.pt">VARF0.pt</option>
                <option value="VARF1.pt">VARF1.pt</option>
                <option value="VARF2.pt">VARF2.pt</option>
                <option value="VARF3.pt">VARF3.pt</option>
                <option value="VARF4.pt">VARF4.pt</option>
                <option value="VARM0.pt">VARM0.pt</option>
                <option value="VARM1.pt">VARM1.pt</option>
                <option value="VARM2.pt">VARM2.pt</option>
                <option value="VARM3.pt">VARM3.pt</option>
                <option value="VARM4.pt">VARM4.pt</option>
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

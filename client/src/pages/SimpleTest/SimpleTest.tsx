import { useState, useCallback, useRef, useEffect } from "react";
import { WSMessage } from "../../protocol/types";
import { encodeMessage, decodeMessage } from "../../protocol/encoder";

const VOICE_OPTIONS = [
  "NATF0.pt", "NATF1.pt", "NATF2.pt", "NATF3.pt",
  "NATM0.pt", "NATM1.pt", "NATM2.pt", "NATM3.pt",
  "VARF0.pt", "VARF1.pt", "VARF2.pt", "VARF3.pt", "VARF4.pt",
  "VARM0.pt", "VARM1.pt", "VARM2.pt", "VARM3.pt", "VARM4.pt",
];

const DEFAULT_TEXT_PROMPT = "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.";

type ConnectionStatus = "disconnected" | "connecting" | "connected";

export const SimpleTest = () => {
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>("disconnected");
  const [textPrompt, setTextPrompt] = useState(DEFAULT_TEXT_PROMPT);
  const [voicePrompt, setVoicePrompt] = useState("NATF2.pt");
  const [transcribedText, setTranscribedText] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  
  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);

  const buildWebSocketURL = useCallback(() => {
    // Default to localhost:8998 if running locally, otherwise use current host
    const hostname = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1" 
      ? window.location.hostname 
      : window.location.hostname;
    const port = window.location.port || "8998";
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    
    const url = new URL(`${protocol}//${hostname}:${port}/api/chat`);
    
    // Add query parameters
    url.searchParams.append("text_temperature", "0.7");
    url.searchParams.append("text_topk", "50");
    url.searchParams.append("audio_temperature", "0.7");
    url.searchParams.append("audio_topk", "50");
    url.searchParams.append("pad_mult", "1.0");
    url.searchParams.append("text_seed", Math.round(1000000 * Math.random()).toString());
    url.searchParams.append("audio_seed", Math.round(1000000 * Math.random()).toString());
    url.searchParams.append("repetition_penalty_context", "1.0");
    url.searchParams.append("repetition_penalty", "1.0");
    url.searchParams.append("text_prompt", textPrompt);
    url.searchParams.append("voice_prompt", voicePrompt);
    
    return url.toString();
  }, [textPrompt, voicePrompt]);

  const connect = useCallback(async () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setConnectionStatus("connecting");
    setError(null);

    try {
      const url = buildWebSocketURL();
      console.log("Connecting to:", url);
      
      const ws = new WebSocket(url);
      ws.binaryType = "arraybuffer";

      ws.onopen = () => {
        console.log("WebSocket connected");
        setConnectionStatus("connecting");
      };

      ws.onmessage = (event) => {
        const data = new Uint8Array(event.data);
        const message = decodeMessage(data);
        
        console.log("Received message:", message.type);

        if (message.type === "handshake") {
          console.log("Handshake received!");
          setConnectionStatus("connected");
        } else if (message.type === "text") {
          console.log("Text received:", message.data);
          setTranscribedText(prev => [...prev, message.data]);
        } else if (message.type === "audio") {
          // Handle audio data - for now just log it
          console.log("Audio received:", message.data.length, "bytes");
        } else if (message.type === "error") {
          setError(message.data);
          console.error("Error from server:", message.data);
        }
      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        setError("Connection error occurred");
        setConnectionStatus("disconnected");
      };

      ws.onclose = () => {
        console.log("WebSocket closed");
        setConnectionStatus("disconnected");
        wsRef.current = null;
      };

      wsRef.current = ws;
    } catch (err) {
      console.error("Failed to connect:", err);
      setError(err instanceof Error ? err.message : "Failed to connect");
      setConnectionStatus("disconnected");
    }
  }, [buildWebSocketURL]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnectionStatus("disconnected");
    stopRecording();
  }, []);

  const startRecording = useCallback(async () => {
    if (isRecording || connectionStatus !== "connected") {
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContext();
      }

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm;codecs=opus",
        audioBitsPerSecond: 128000,
      });

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        console.log("Recording stopped");
      };

      mediaRecorder.start(100); // Collect data every 100ms
      mediaRecorderRef.current = mediaRecorder;
      setIsRecording(true);

      // Send audio chunks to server
      const sendAudioInterval = setInterval(() => {
        if (!isRecording || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
          clearInterval(sendAudioInterval);
          return;
        }

        // For simplicity, we'll send audio data periodically
        // In a real implementation, you'd process the audio chunks properly
      }, 100);

    } catch (err) {
      console.error("Failed to start recording:", err);
      setError("Failed to access microphone");
    }
  }, [isRecording, connectionStatus]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current = null;
    }
    
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    
    setIsRecording(false);
    audioChunksRef.current = [];
  }, [isRecording]);

  const sendControlMessage = useCallback((action: "start" | "endTurn" | "pause" | "restart") => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setError("Not connected");
      return;
    }

    const message: WSMessage = {
      type: "control",
      action,
    };

    wsRef.current.send(encodeMessage(message));
    console.log("Sent control message:", action);
  }, []);

  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  const statusColor = {
    disconnected: "bg-red-500",
    connecting: "bg-yellow-500",
    connected: "bg-green-500",
  }[connectionStatus];

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">PersonaPlex Test</h1>
        <p className="text-gray-600 mb-8">Simple frontend to test the PersonaPlex stack</p>

        {/* Connection Status */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className={`h-4 w-4 rounded-full ${statusColor}`}></div>
              <span className="font-semibold text-gray-700">
                Status: {connectionStatus.charAt(0).toUpperCase() + connectionStatus.slice(1)}
              </span>
            </div>
            <div className="flex gap-2">
              {connectionStatus === "disconnected" ? (
                <button
                  onClick={connect}
                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
                >
                  Connect
                </button>
              ) : (
                <button
                  onClick={disconnect}
                  className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition"
                >
                  Disconnect
                </button>
              )}
            </div>
          </div>
          {error && (
            <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded text-red-700">
              {error}
            </div>
          )}
        </div>

        {/* Configuration */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Configuration</h2>
          
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Text Prompt
            </label>
            <textarea
              value={textPrompt}
              onChange={(e) => setTextPrompt(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              rows={4}
              placeholder="Enter text prompt..."
              disabled={connectionStatus !== "disconnected"}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Voice
            </label>
            <select
              value={voicePrompt}
              onChange={(e) => setVoicePrompt(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={connectionStatus !== "disconnected"}
            >
              {VOICE_OPTIONS.map((voice) => (
                <option key={voice} value={voice}>
                  {voice.replace('.pt', '').replace(/^NAT/, 'NATURAL_').replace(/^VAR/, 'VARIETY_')}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Controls */}
        {connectionStatus === "connected" && (
          <div className="bg-white rounded-lg shadow p-6 mb-6">
            <h2 className="text-xl font-semibold mb-4">Controls</h2>
            <div className="flex gap-2 flex-wrap">
              <button
                onClick={() => sendControlMessage("start")}
                className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition"
              >
                Start
              </button>
              <button
                onClick={() => sendControlMessage("endTurn")}
                className="px-4 py-2 bg-yellow-600 text-white rounded hover:bg-yellow-700 transition"
              >
                End Turn
              </button>
              <button
                onClick={() => sendControlMessage("pause")}
                className="px-4 py-2 bg-orange-600 text-white rounded hover:bg-orange-700 transition"
              >
                Pause
              </button>
              <button
                onClick={() => sendControlMessage("restart")}
                className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 transition"
              >
                Restart
              </button>
              <button
                onClick={isRecording ? stopRecording : startRecording}
                className={`px-4 py-2 rounded transition ${
                  isRecording
                    ? "bg-red-600 hover:bg-red-700 text-white"
                    : "bg-blue-600 hover:bg-blue-700 text-white"
                }`}
              >
                {isRecording ? "Stop Recording" : "Start Recording"}
              </button>
            </div>
          </div>
        )}

        {/* Transcribed Text */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold mb-4">Transcribed Text</h2>
          <div className="bg-gray-50 rounded p-4 min-h-[200px] max-h-[400px] overflow-y-auto">
            {transcribedText.length === 0 ? (
              <p className="text-gray-400 italic">No text received yet...</p>
            ) : (
              <div className="space-y-2">
                {transcribedText.map((text, index) => (
                  <div key={index} className="p-2 bg-white rounded border border-gray-200">
                    <p className="text-gray-800">{text}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
          {transcribedText.length > 0 && (
            <button
              onClick={() => setTranscribedText([])}
              className="mt-4 px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition"
            >
              Clear
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

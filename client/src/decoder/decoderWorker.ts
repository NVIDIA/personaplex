// Track decoder worker errors to detect failed initialization
let lastWorkerError: Error | null = null;

// Minimal valid Ogg BOS page with OpusHead header (mono, 48kHz)
// This triggers the decoder's internal init() to create buffers
const createWarmupBosPage = (): Uint8Array => {
  // OpusHead: "OpusHead" + version(1) + channels(1) + preskip(2) + samplerate(4) + gain(2) + mapping(1)
  const opusHead = new Uint8Array([
    0x4F, 0x70, 0x75, 0x73, 0x48, 0x65, 0x61, 0x64, // "OpusHead"
    0x01,       // Version 1
    0x01,       // 1 channel (mono)
    0x38, 0x01, // Pre-skip: 312 samples (little-endian)
    0x80, 0xBB, 0x00, 0x00, // Sample rate: 48000 Hz (little-endian)
    0x00, 0x00, // Output gain: 0
    0x00,       // Channel mapping: 0 (mono/stereo)
  ]);

  // Ogg page header
  const pageHeader = new Uint8Array([
    0x4F, 0x67, 0x67, 0x53, // "OggS" magic
    0x00,       // Version 0
    0x02,       // BOS flag (Beginning of Stream)
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Granule position: 0
    0x01, 0x00, 0x00, 0x00, // Stream serial: 1
    0x00, 0x00, 0x00, 0x00, // Page sequence: 0
    0x00, 0x00, 0x00, 0x00, // CRC (will be invalid but decoder doesn't check)
    0x01,       // 1 segment
    0x13,       // Segment size: 19 bytes (OpusHead)
  ]);

  // Combine header and OpusHead
  const bosPage = new Uint8Array(pageHeader.length + opusHead.length);
  bosPage.set(pageHeader, 0);
  bosPage.set(opusHead, pageHeader.length);

  return bosPage;
};

// Factory function to create a decoder worker with error tracking
const createWorkerWithErrorTracking = (): Worker => {
  const worker = new Worker(
    new URL("/assets/decoderWorker.min.js", import.meta.url),
  );

  // Track errors from the worker
  worker.onerror = (event) => {
    console.error("Decoder worker error:", event.message);
    lastWorkerError = new Error(event.message);
  };

  return worker;
};

// Send init command to a worker, then send warmup BOS page
const sendInitCommand = (worker: Worker, audioContextSampleRate: number): void => {
  worker.postMessage({
    command: "init",
    bufferLength: 960 * audioContextSampleRate / 24000,
    decoderSampleRate: 24000,
    outputBufferSampleRate: audioContextSampleRate,
    // Increased from 0 (lowest) to 5 for clearer decoded audio output
    resampleQuality: 5,
  });

  // After a short delay, send warmup BOS page to trigger decoder's internal init
  setTimeout(() => {
    const bosPage = createWarmupBosPage();
    console.log("Sending warmup BOS page to decoder");
    worker.postMessage({
      command: "decode",
      pages: bosPage,
    });
  }, 100);
};

// Singleton pre-warmed worker that starts loading WASM immediately
let prewarmedWorker: Worker | null = null;
let prewarmedWorkerReady: Promise<void> | null = null;
let prewarmedSampleRate: number | null = null;

// Create and pre-warm a decoder worker (should be called early, e.g., on page load)
export const prewarmDecoderWorker = (audioContextSampleRate: number): Promise<void> => {
  if (prewarmedWorkerReady && prewarmedSampleRate === audioContextSampleRate) {
    console.log("Using existing prewarmed worker");
    return prewarmedWorkerReady;
  }

  // Terminate old worker if sample rate changed
  if (prewarmedWorker) {
    prewarmedWorker.terminate();
  }

  console.log("Creating and prewarming decoder worker");
  lastWorkerError = null;
  prewarmedWorker = createWorkerWithErrorTracking();
  prewarmedSampleRate = audioContextSampleRate;

  sendInitCommand(prewarmedWorker, audioContextSampleRate);

  prewarmedWorkerReady = new Promise((resolve) => {
    // Give worker plenty of time to load WASM and process init
    // 1000ms to be extra safe
    setTimeout(() => {
      console.log("Prewarmed decoder worker ready");
      resolve();
    }, 1000);
  });

  return prewarmedWorkerReady;
};

// Get the prewarmed worker after waiting for it to be ready
// Returns null if prewarm wasn't called or if worker had errors
export const getPrewarmedWorker = async (): Promise<Worker | null> => {
  if (!prewarmedWorker || !prewarmedWorkerReady) {
    return null;
  }

  // Wait for prewarm to complete
  await prewarmedWorkerReady;

  // If there was an error during prewarm, don't use this worker
  if (lastWorkerError) {
    console.warn("Prewarmed worker had errors, will create fresh one");
    prewarmedWorker.terminate();
    prewarmedWorker = null;
    prewarmedWorkerReady = null;
    prewarmedSampleRate = null;
    lastWorkerError = null;
    return null;
  }

  const worker = prewarmedWorker;
  // Clear the singleton so next connection gets a fresh worker
  prewarmedWorker = null;
  prewarmedWorkerReady = null;
  prewarmedSampleRate = null;
  return worker;
};

// Factory function to create a fresh decoder worker (fallback if prewarm wasn't called)
export const createDecoderWorker = (): Worker => {
  return createWorkerWithErrorTracking();
};

// Initialize a decoder worker and return a promise that resolves when ready
export const initDecoder = (worker: Worker, audioContextSampleRate: number): Promise<void> => {
  return new Promise((resolve) => {
    console.log("Starting decoder initialization");
    lastWorkerError = null;

    sendInitCommand(worker, audioContextSampleRate);

    // Give worker time to load WASM and process init - 1000ms to be safe
    setTimeout(() => {
      console.log("Decoder initialization complete");
      resolve();
    }, 1000);
  });
};

# PersonaPlex API Documentation

## Overview

PersonaPlex exposes a lightweight HTTP + WebSocket API for real-time, full-duplex speech conversations with persona control.

---

## Base URL

| Environment | URL |
|---|---|
| Local (HTTPS) | `https://localhost:8998` |
| LAN | `https://<LAN_IP>:8998` |
| Public | `https://<PUBLIC_IP>:8998` |

> **Note:** The server uses self-signed certificates by default. Use `-k` with curl or bypass browser warnings.

---

## Authentication

All `/api/*` endpoints require an API key. Two methods are supported:

### 1. HTTP Header (recommended for REST)

```
X-API-Key: <your_api_key>
```

### 2. Query Parameter (required for WebSocket)

```
?api_key=<your_api_key>
```

### Configuration

Set the API key via any of these methods (in priority order):

1. **CLI argument:** `--api-key <key>`
2. **Environment variable:** `PERSONAPLEX_API_KEY=<key>`
3. **Auto-generated:** If neither is set, the server generates a key and prints it to the console.

---

## Endpoints

### `GET /api/health`

Health check endpoint. Returns server status and version.

**Headers:**

| Header | Required | Description |
|---|---|---|
| `X-API-Key` | Yes | Your API key |

**Sample Request:**

```bash
curl -k -H "X-API-Key: YOUR_KEY" https://localhost:8998/api/health
```

**Success Response — `200 OK`:**

```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

**Error Responses:**

| Status | Body |
|---|---|
| `401 Unauthorized` | `{"error": "Unauthorized", "message": "Missing API key. Provide via X-API-Key header or api_key query parameter."}` |
| `403 Forbidden` | `{"error": "Forbidden", "message": "Invalid API key."}` |

---

### `GET /api/voices`

Lists all available voice prompt files on the server.

**Headers:**

| Header | Required | Description |
|---|---|---|
| `X-API-Key` | Yes | Your API key |

**Sample Request:**

```bash
curl -k -H "X-API-Key: YOUR_KEY" https://localhost:8998/api/voices
```

**Success Response — `200 OK`:**

```json
{
  "voices": [
    { "filename": "NATF0.pt", "name": "NATF0" },
    { "filename": "NATF1.pt", "name": "NATF1" },
    { "filename": "NATF2.pt", "name": "NATF2" },
    { "filename": "NATF3.pt", "name": "NATF3" },
    { "filename": "NATM0.pt", "name": "NATM0" },
    { "filename": "NATM1.pt", "name": "NATM1" },
    { "filename": "NATM2.pt", "name": "NATM2" },
    { "filename": "NATM3.pt", "name": "NATM3" },
    { "filename": "VARF0.pt", "name": "VARF0" },
    { "filename": "VARF1.pt", "name": "VARF1" }
  ],
  "count": 10
}
```

**Error Responses:**

| Status | Body |
|---|---|
| `401 Unauthorized` | `{"error": "Unauthorized", "message": "Missing API key. Provide via X-API-Key header or api_key query parameter."}` |
| `403 Forbidden` | `{"error": "Forbidden", "message": "Invalid API key."}` |

---

### `GET /api/chat` (WebSocket Upgrade)

Opens a full-duplex WebSocket connection for real-time speech conversation.

> **Important:** This endpoint upgrades HTTP to WebSocket. Use `wss://` protocol. Authentication must be via query parameter since browsers do not support custom headers on WebSocket handshake.

**Query Parameters:**

| Parameter | Required | Type | Description |
|---|---|---|---|
| `api_key` | Yes | string | Your API key |
| `voice_prompt` | Yes | string | Voice prompt filename (e.g., `NATF2.pt`) |
| `text_prompt` | Yes | string | System text prompt (e.g., `You enjoy having a good conversation.`) |
| `text_temperature` | No | float | Text generation temperature (default: model default) |
| `text_topk` | No | int | Text top-k sampling (default: model default) |
| `audio_temperature` | No | float | Audio generation temperature |
| `audio_topk` | No | int | Audio top-k sampling |
| `pad_mult` | No | float | Padding multiplier |
| `text_seed` | No | int | Random seed for text generation |
| `audio_seed` | No | int | Random seed for audio generation |
| `repetition_penalty` | No | float | Repetition penalty factor |
| `repetition_penalty_context` | No | int | Repetition penalty context window |
| `seed` | No | int | Global random seed |

**Sample WebSocket URL:**

```
wss://localhost:8998/api/chat?api_key=YOUR_KEY&voice_prompt=NATF2.pt&text_prompt=You%20enjoy%20having%20a%20good%20conversation.
```

**Binary Protocol:**

Messages are exchanged as binary frames with a 1-byte type prefix:

| Byte | Direction | Description |
|---|---|---|
| `0x00` | Server → Client | Handshake acknowledgement |
| `0x01` | Server → Client | Opus-encoded audio chunk |
| `0x01` | Client → Server | Opus-encoded audio chunk |
| `0x02` | Server → Client | UTF-8 text token |

**Connection Lifecycle:**

1. Client connects via WebSocket URL with query params
2. Server processes voice/text prompts (may take a few seconds)
3. Server sends handshake byte `0x00`
4. Client begins streaming Opus audio (`0x01` frames)
5. Server responds with interleaved audio (`0x01`) and text (`0x02`) frames
6. Either side closes the connection to end the session

**Error Responses (before WebSocket upgrade):**

| Status | Body |
|---|---|
| `401 Unauthorized` | `{"error": "Unauthorized", "message": "Missing API key. Provide via X-API-Key header or api_key query parameter."}` |
| `403 Forbidden` | `{"error": "Forbidden", "message": "Invalid API key."}` |

---

## Status Codes Summary

| Code | Meaning |
|---|---|
| `200` | Success |
| `101` | Switching Protocols (WebSocket upgrade) |
| `401` | Unauthorized — missing API key |
| `403` | Forbidden — invalid API key |
| `404` | Not Found — unknown endpoint |
| `500` | Internal Server Error |

---

## Quick Start

```bash
# 1. Set your API key
export PERSONAPLEX_API_KEY="your_key_here"

# 2. Test health
curl -k -H "X-API-Key: $PERSONAPLEX_API_KEY" https://localhost:8998/api/health

# 3. List voices
curl -k -H "X-API-Key: $PERSONAPLEX_API_KEY" https://localhost:8998/api/voices

# 4. Connect via WebSocket (using wscat)
wscat -n -c "wss://localhost:8998/api/chat?api_key=$PERSONAPLEX_API_KEY&voice_prompt=NATF2.pt&text_prompt=Hello"
```

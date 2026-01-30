# 🎙️ PersonaPlex: Guía Técnica de Adaptación Vocal Local

Esta guía explica el proceso de **Voice Conditioning** (acondicionamiento de voz) para el modelo PersonaPlex (arquitectura Moshi) y cómo operarlo de forma local.

## 1. ¿Cómo funciona el Conditioning?

A diferencia de los modelos de TTS tradicionales que requieren un entrenamiento (fine-tuning) costoso, PersonaPlex utiliza un enfoque de **Zero-Shot Conditioning** basado en prompts de audio.

### El proceso técnico:
1. **Extracción de Identidad (Mimi)**: El codificador de audio `Mimi` toma un fragmento de audio (ej: `pepper.wav`) y lo convierte en una serie de tokens acústicos y latentes.
2. **Inyección en el Prefijo (LM)**: Estos tokens se inyectan como un "prefijo" en el Modelo de Lenguaje (LM). Al hacerlo, el modelo comienza su generación desde un estado interno que ya contiene la textura, el tono y la prosodia de la voz proporcionada.
3. **Caché de Speaker Embedding (.pt)**: Para evitar procesar el audio en cada conexión, PersonaPlex guarda el estado interno del modelo (el KV cache) después de haber procesado el audio. Este archivo `.pt` es lo que llamamos el "Speaker Embedding".

## 2. Pasos para Adaptar la voz de Pepper Localmente

Para mejorar la voz de Pepper, sigue este flujo:

### Paso A: Preparación del Audio
- Usa un audio limpio (como el `pepper_child.wav` que generaste).
- **Duración ideal**: 10-20 segundos.
- El audio debe ser representativo de la expresividad que deseas (si quieres que Pepper sea juguetón, el audio debe contener risas o entonación variada).

### Paso B: Generación del Embedding Local
No necesitas Modal para esto. Usa el script dedicado que he preparado:

1. Asegúrate de tener tu archivo `.wav` en la raíz (ej: `pepper_child.wav`).
2. Abre una terminal y ejecuta:
   ```powershell
   .\venv311\Scripts\python generate_voice_embedding.py --input pepper_child.wav --output weights/pepper_new.pt
   ```
3. **¿Dónde se refleja?**: Verás un nuevo archivo `pepper_new.pt` en la carpeta `weights/`. Este archivo contiene la "firma" de voz lista para ser usada por el modelo.

#### Cómo usar el resultado:
En el Notebook `pepper_local_inference.ipynb`, simplemente cambia la ruta de la voz:
```python
generate_audio_local(text="¡Hola!", voice_pt="weights/pepper_new.pt")
```

---

## 3. Parámetros de Inferencia para Máxima Expresividad

Para que PersonaPlex no suene plano, ajustaremos los siguientes parámetros en el script local:

| Parámetro | Valor Sugerido | Efecto |
| :--- | :--- | :--- |
| `temperature` | 0.8 - 1.0 | Aumenta la variabilidad y la "humanidad" en la entonación. |
| `top_k` | 50 | Filtra opciones poco probables pero mantiene la riqueza. |
| `repetition_penalty` | 1.1 | Evita que el robot se quede buclado en una muletilla o tono. |
| `dry_run_n_steps` | 100+ | Permite al modelo "asentarse" en la voz antes de empezar la respuesta real. |

## 4. Sistema Emocional "Inside Out"

Para implementar los estados emocionales, el modelo se condiciona mediante **Text Prompting** dinámico precediendo la respuesta real. El modelo Moshi es sensible a las instrucciones de estilo si se le proporcionan como contexto inmediato.

**Estados configurables:**
- **Joy (Alegría)**: Tono agudo, ritmo rápido, uso de onomatopeyas de risa.
- **Sadness (Tristeza)**: Pausas largas, tono bajo, ritmo lento.
- **Anger (Ira)**: Frases cortas, volumen percibido alto (por el tipo de tokens elegidos).
- **Fear (Miedo)**: Titubeos, respiraciones más frecuentes.
- **Disgust (Asco)**: Entonación de rechazo, pausas de asco.

---

> [!TIP]
> El éxito de la voz de Pepper local depende un 80% de la calidad del audio `.wav` inicial. Asegúrate de que no tenga ruido de fondo.

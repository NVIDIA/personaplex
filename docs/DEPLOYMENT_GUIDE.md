# 🚀 PersonaPlex: Guía de Despliegue en Modal.com

Este documento resume el proceso de migración, los arreglos técnicos realizados y cómo operar tu servidor de IA conversacional en Modal.

## 📋 Resumen del Estado Actual
El sistema está **completamente operativo**. Hemos superado las limitaciones de memoria de Google Colab y los conflictos de dependencias de audio. La IA ahora responde en tiempo real a través de WebSockets utilizando una GPU A100.

---

## 🛠️ Lo que hemos arreglado

### 1. El Puente "Fiel" (ASGI Wrapper)
Hemos creado `modal_app.py` como un puente mínimo. 
- **Instalación Directa**: Modal instala tu carpeta `./moshi` como un paquete real (`pip install -e .`). Esto garantiza que se use **tu lógica** y no una versión genérica.
- **Traducción WebSocket**: Convierte el protocolo de Modal (ASGI) al formato que espera tu servidor, manteniendo la latencia bajo control.

### 2. El "Parche" de Audio (`sphn`)
Descubrimos que las versiones modernas de la librería `sphn` borraban funciones esenciales.
- **Solución**: Hemos fijado la versión `sphn==0.1.12` en `pyproject.toml` y `requirements.txt`. Esto restaura la capacidad de procesar audio Opus sin errores.

### 3. Estabilidad en GPU A100
Para evitar errores de memoria sincronizada ("RuntimeError: Can't call numpy on variable with grad"):
- **Torch No Grad**: Envolvemos el procesamiento en `torch.no_grad()`.
- **Detach**: Desasociamos los tensores antes de enviarlos al cliente para evitar bloqueos del motor.

---

## 🚀 Guía de Operación

### Configuración en una Nueva Cuenta de Modal
Si cambias de cuenta (ej. `modal token new`), debes recrear el entorno:

1. **Secreto de HuggingFace**:
   ```powershell
   modal secret create huggingface HF_TOKEN=tu_token_aqui
   ```
2. **Descargar Modelos**:
   ```powershell
   modal run modal_app.py::download_models
   ```
3. **Generar Voz de Pepper**:
   ```powershell
   modal run modal_app.py::generate_pepper_embedding
   ```

### Cómo lanzar el servidor
Desde la terminal en `c:\Users\anls\code\KDFAST-AI\lab\personaplex-main`, ejecuta:
```powershell
modal serve modal_app.py
```
*Esto te dará una URL (ej: `https://tu-usuario--ap-og-web.modal.run`). Úsala en tu frontend.*

### Cómo actualizar tu código
Como usamos modo "editable" (`-e`), cualquier cambio que hagas en la carpeta `./moshi` se verá reflejado en el servidor la próxima vez que lances el comando `modal serve`.

### Gestión de Voces (Embeddings)
Los modelos de voz se guardan en el volumen persistente `personaplex-weights` bajo `/root/weights`.

#### Cómo generar la voz de Pepper
Si has añadido un nuevo archivo de audio (como `pepper.wav`) y quieres crear su "identidad" de voz:
1. Asegúrate de que el archivo `.wav` está en la raíz del proyecto.
2. Ejecuta el comando de generación:
   ```powershell
   modal run modal_app.py::generate_pepper_embedding
   ```
   *Esto procesará el audio en una GPU A100 y guardará `pepper.pt` en el volumen de forma permanente.*

#### Cómo usar nuevas voces
- **Servidor**: El sistema detecta automáticamente si el `voice_prompt` termina en `.pt` y lo carga desde el volumen.
- **Cliente**: Añade la nueva opción en `client/src/pages/Queue/Queue.tsx` (en el array `VOICE_OPTIONS`) para que aparezca en el desplegable.

---

## 💡 Consejos para el Futuro
- **Voces**: Puedes añadir nuevos archivos `.pt` a la carpeta de voces y el sistema los detectará automáticamente mediante el parámetro `voice_prompt` en la URL de conexión.
- **Costes**: Modal solo te cobra mientras el servidor está encendido. Al cerrar la terminal (`Ctrl+C`), el servidor se apaga automáticamente tras unos minutos de inactividad.

---

> [!IMPORTANT]
> El sistema es ahora 100% independiente. No dependes de scripts externos ni de versiones "inventadas". Es tu código, corriendo en la infraestructura más potente disponible.

modal deploy model_app.py  para deployar

¡Disfruta de la voz de PersonaPlex! 🎙️✨

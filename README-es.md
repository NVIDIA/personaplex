# PersonaPlex: Control de Voz y Rol para Modelos Conversacionales de Voz Full Duplex

[![Weights](https://img.shields.io/badge/🤗-Weights-yellow)](https://huggingface.co/nvidia/personaplex-7b-v1)
[![Paper](https://img.shields.io/badge/📄-Paper-blue)](https://research.nvidia.com/labs/adlr/files/personaplex/personaplex_preprint.pdf)
[![Demo](https://img.shields.io/badge/🎮-Demo-green)](https://research.nvidia.com/labs/adlr/personaplex/)
[![Discord](https://img.shields.io/badge/Discord-Join-purple?logo=discord)](https://discord.gg/5jAXrrbwRb)

PersonaPlex es un modelo conversacional de voz a voz en tiempo real y full-duplex que permite el control de persona a través de prompts de rol basados en texto y acondicionamiento de voz basado en audio. Entrenado en una combinación de conversaciones sintéticas y reales, produce interacciones habladas naturales con baja latencia y una persona consistente. PersonaPlex se basa en la arquitectura [Moshi](https://arxiv.org/abs/2410.00037) y pesos.

<p align="center">
  <img src="assets/architecture_diagram.png" alt="Arquitectura del Modelo PersonaPlex">
  <br>
  <em>Arquitectura de PersonaPlex</em>
</p>

## Uso

### Prerrequisitos

Instala la biblioteca de desarrollo del códec de audio [Opus](https://github.com/xiph/opus):
```bash
# Ubuntu/Debian
sudo apt install libopus-dev

# Fedora/RHEL
sudo dnf install opus-devel

# macOS
brew install opus
```

### Instalación

Descarga este repositorio e instala con:
```bash
pip install moshi/.
```

Paso extra para GPUs basadas en Blackwell como se sugiere en (Ver https://github.com/NVIDIA/personaplex/issues/2):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```


### Aceptar Licencia del Modelo
Inicia sesión en tu cuenta de Huggingface y acepta la licencia del modelo PersonaPlex [aquí](https://huggingface.co/nvidia/personaplex-7b-v1). <br>
Luego configura tu autenticación de Huggingface:
```bash
export HF_TOKEN=<TU_TOKEN_DE_HUGGINGFACE>
```

### Lanzar Servidor

Lanza el servidor para interacción en vivo (certificados SSL temporales para https):
```bash
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR"
```

**Descarga a CPU:** Si tu GPU tiene memoria insuficiente, usa la bandera `--cpu-offload` para descargar capas del modelo a CPU. Esto requiere el paquete `accelerate` (`pip install accelerate`):
```bash
SSL_DIR=$(mktemp -d); python -m moshi.server --ssl "$SSL_DIR" --cpu-offload
```

Accede a la Interfaz Web desde un navegador en `localhost:8998` si se ejecuta localmente, de lo contrario busca el enlace de acceso impreso por el script:
```
Accede a la Interfaz Web directamente en https://11.54.401.33:8998
```

### Evaluación Offline

Para evaluación offline usa el script offline que transmite un archivo wav de entrada y produce un archivo wav de salida desde el flujo de salida capturado. El archivo de salida tendrá la misma duración que el archivo de entrada.

Agrega `--cpu-offload` a cualquier comando a continuación si tu GPU tiene memoria insuficiente (requiere paquete `accelerate`). O instala PyTorch solo para CPU para evaluación offline en CPU pura.

**Ejemplo de Asistente:**
```bash
HF_TOKEN=<TOKEN> \
python -m moshi.offline \
  --voice-prompt "NATF2.pt" \
  --input-wav "assets/test/input_assistant.wav" \
  --seed 42424242 \
  --output-wav "output.wav" \
  --output-text "output.json"
```

**Ejemplo de Servicio:**
```bash
HF_TOKEN=<TOKEN> \
python -m moshi.offline \
  --voice-prompt "NATM1.pt" \
  --text-prompt "$(cat assets/test/prompt_service.txt)" \
  --input-wav "assets/test/input_service.wav" \
  --seed 42424242 \
  --output-wav "output.wav" \
  --output-text "output.json"
```

## Voces

PersonaPlex soporta una amplia gama de voces; pre-empaquetamos embeddings para voces que suenan más naturales y conversacionales (NAT) y otras más variadas (VAR). El conjunto fijo de voces está etiquetado:
```
Natural(femenina): NATF0, NATF1, NATF2, NATF3
Natural(masculino): NATM0, NATM1, NATM2, NATM3
Variedad(femenina): VARF0, VARF1, VARF2, VARF3, VARF4
Variedad(masculino): VARM0, VARM1, VARM2, VARM3, VARM4
```

## Guía de Prompts

El modelo está entrenado en conversaciones sintéticas para un rol fijo de asistente y roles variables de servicio al cliente.

### Rol de Asistente

El rol de asistente tiene el prompt:
```
You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.
```

Usa este prompt para la categoría de evaluación "User Interruption" enfocada en QA asistente en [FullDuplexBench](https://arxiv.org/abs/2503.04721).

### Roles de Servicio al Cliente

Los roles de servicio al cliente soportan una variedad de prompts. Aquí hay algunos ejemplos para referencia de estilo de prompting:
```
You work for CitySan Services which is a waste management and your name is Ayelen Lucero. Information: Verify customer name Omar Torres. Current schedule: every other week. Upcoming pickup: April 12th. Compost bin service available for $8/month add-on.
```
```
You work for Jerusalem Shakshuka which is a restaurant and your name is Owen Foster. Information: There are two shakshuka options: Classic (poached eggs, $9.50) and Spicy (scrambled eggs with jalapenos, $10.25). Sides include warm pita ($2.50) and Israeli salad ($3). No combo offers. Available for drive-through until 9 PM.
```
```
You work for AeroRentals Pro which is a drone rental company and your name is Tomaz Novak. Information: AeroRentals Pro has the following availability: PhoenixDrone X ($65/4 hours, $110/8 hours), and the premium SpectraDrone 9 ($95/4 hours, $160/8 hours). Deposit required: $150 for standard models, $300 for premium.
```

### Conversaciones Casuales

El modelo también está entrenado en conversaciones reales del [Fisher English Corpus](https://catalog.ldc.upenn.edu/LDC2004T19) con prompts etiquetados por LLM para conversaciones abiertas. Aquí hay algunos ejemplos de prompts para conversaciones casuales:
```
You enjoy having a good conversation.
```
```
You enjoy having a good conversation. Have a casual discussion about eating at home versus dining out.
```
```
You enjoy having a good conversation. Have an empathetic discussion about the meaning of family amid uncertainty.
```
```
You enjoy having a good conversation. Have a reflective conversation about career changes and feeling of home. You have lived in California for 21 years and consider San Francisco your home. You work as a teacher and have traveled a lot. You dislike meetings.
```
```
You enjoy having a good conversation. Have a casual conversation about favorite foods and cooking experiences. You are David Green, a former baker now living in Boston. You enjoy cooking diverse international dishes and appreciate many ethnic restaurants.
```

Usa el prompt `You enjoy having a good conversation.` para las categorías de evaluación "Pause Handling", "Backchannel" y "Smooth Turn Taking" de FullDuplexBench.

## Generalización

Personaplex ajusta finamente Moshi y se beneficia de las capacidades de generalización del [Helium](https://kyutai.org/blog/2025-04-30-helium) LLM subyacente. Gracias al amplio corpus de entrenamiento del backbone, encontramos que el modelo responderá plausiblemente a prompts fuera de distribución y llevará a conversaciones inesperadas o divertidas. Animamos a la experimentación con diferentes prompts para probar la capacidad emergente del modelo para manejar escenarios fuera de su distribución de entrenamiento. Como inspiración, destacamos el siguiente prompt de astronauta en la WebUI:
```
You enjoy having a good conversation. Have a technical discussion about fixing a reactor core on a spaceship to Mars. You are an astronaut on a Mars mission. Your name is Alex. You are already dealing with a reactor core meltdown on a Mars mission. Several ship systems are failing, and continued instability will lead to catastrophic failure. You explain what is happening and you urgently ask for help thinking through how to stabilize the reactor.
```

## Licencia

El código presente se proporciona bajo la licencia MIT. Los pesos para los modelos se liberan bajo la licencia NVIDIA Open Model.

## Cita

Si usas PersonaPlex en tu investigación, por favor cita nuestro artículo:
```bibtex
@article{roy2026personaplex,
  title={PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models},
  author={Roy, Rajarshi and Raiman, Jonathan and Lee, Sang-gil and Ene, Teodor-Dumitru and Kirby, Robert and Kim, Sungwon and Kim, Jaehyeon and Catanzaro, Bryan},
  year={2026}
}
```
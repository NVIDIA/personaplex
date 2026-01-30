from pydub import AudioSegment

# Cargar el WAV original
audio = AudioSegment.from_file("assets/test/pepper.wav")

# Convertir a mono
audio = audio.set_channels(1)

# Guardar como nuevo archivo
audio.export("assets/test/pepper_mono.wav", format="wav")

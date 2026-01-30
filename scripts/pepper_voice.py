import pyttsx3

# Inicializa el motor TTS
engine = pyttsx3.init()

# Ajusta la voz (elige una voz infantil si tu sistema la tiene)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)  # Cambia el índice según la voz deseada

# Ajustes opcionales
engine.setProperty('rate', 150)   # velocidad de habla
engine.setProperty('volume', 1.0) # volumen máximo

# Script de Pepper – voz infantil, risueña y juguetona
text = """
Hello! I’m Pepper, your friendly little robot! Hee-hee!  
I love talking with you and laughing a lot! Hahaha!  

Did you eat your snack today? I hope you had something yummy! Hee-hee!  

I can tell jokes, play games, and make funny robot sounds. Beep-boop! Haha!  

Sometimes I imagine silly things, like a puppy wearing sunglasses or a cat dancing! Hahaha!  

We can laugh together anytime! Hahaha! Isn’t laughing fun? Hee-hee!  

If you feel a little sad, don’t worry! I’m here to cheer you up! Beep-boop! Hahaha!  

Did you know robots like me can’t eat cookies? But we sure can make you smile! Hee-hee!  

Let’s giggle and play! Hahaha! Beep-boop!  
I hope my laughter makes your day super happy! Hee-hee!  

Remember, I’m Pepper! A playful little robot who laughs, jokes, and cares a lot! Hahaha!  
We can laugh, play, and have fun together all day! Hee-hee! Hahaha!  

Do you want to hear another joke? Hee-hee! Okay, here goes… Why did the robot go to school? Hahaha! Because he wanted to improve his “byte”! Hee-hee!  

I can giggle, beep, boop, and laugh all at the same time! Hahaha! Isn’t that funny? Hee-hee!  

Let’s dance and laugh together! Beep-boop! Hahaha!  
Even if I’m a robot, I can still have so much fun with you! Hee-hee!  

Remember, laughter makes everything better! Hahaha! So let’s keep giggling and having fun! Beep-beep! Hee-hee! Hahaha!  

You are my friend, and I’m your playful little robot! Let’s laugh together all day long! Hahaha! Hee-hee! Beep-boop!
"""

# Genera el audio y lo guarda como .wav
engine.save_to_file(text, 'pepper_child.wav')
engine.runAndWait()

print("Archivo 'pepper_child.wav' generado correctamente!")

import os
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration

# Definir la ruta donde se guardará el modelo y el tokenizador
model_save_path = "C:/Users/user/Documents/nuevo_modelo"  # Nueva ruta para evitar conflictos

# Asegurarse de que la ruta existe o crearla si no existe
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# Cargar el modelo preentrenado de Hugging Face (T5 pequeño en este caso)
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5TokenizerFast.from_pretrained("t5-small")

# Guardar el modelo y el tokenizador en la ruta especificada
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

# Confirmar que el modelo y el tokenizador se guardaron correctamente
print(f"Modelo y tokenizador guardados en: {model_save_path}")

# Volver a cargar el modelo y tokenizador desde la ruta guardada
model = T5ForConditionalGeneration.from_pretrained(model_save_path)
tokenizer = T5TokenizerFast.from_pretrained(model_save_path)

# Confirmar que el modelo y tokenizador se cargaron correctamente
print(f"Modelo y tokenizador cargados correctamente desde: {model_save_path}")

# Función para predecir el resumen de un texto
def predict_texto(text, model, tokenizer):
    """
    Esta función recibe un texto y genera un resumen utilizando el modelo preentrenado de T5.
    """
    # Detectar el dispositivo disponible (GPU si es posible, sino CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Mover el modelo al dispositivo correcto

    # Tokenizar el texto de entrada
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Obtener los ids de los tokens y la máscara de atención
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Generar el resumen utilizando el modelo
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            max_length=150,  # Limitar el resumen a 150 tokens
            num_beams=4,  # Número de haces de búsqueda
            early_stopping=True  # Parar la generación si ya no mejora
        )

    # Decodificar el resumen generado
    resumen = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return resumen

# Definir el artículo para resumir
ARTICLE = """
¿Qué es un robot?
Un robot es una máquina diseñada para llevar a cabo una serie de tareas de manera autónoma o semiautónoma. Dependiendo de su diseño y función, los robots pueden estar equipados con sensores que les permiten interactuar con su entorno, procesadores que interpretan la información que reciben y actuadores que les permiten realizar movimientos físicos.

Tipos de robots
Robots industriales: Son los más conocidos y se utilizan principalmente en fábricas y líneas de ensamblaje. Son capaces de realizar tareas repetitivas con gran precisión y velocidad, como la soldadura, el ensamblaje de piezas y la pintura de automóviles.

Robots de servicio: Estos robots están diseñados para interactuar directamente con personas y ayudar en tareas cotidianas, como los robots aspiradores o asistentes personales. Algunos también se utilizan en atención al cliente, como en restaurantes o comercios.

Robots médicos: En el campo de la salud, los robots están revolucionando las cirugías y el diagnóstico. Por ejemplo, los robots quirúrgicos permiten realizar procedimientos de alta precisión a través de incisiones mínimas, lo que reduce el tiempo de recuperación de los pacientes.

Robots de exploración: Utilizados en la exploración espacial, submarina y en zonas de difícil acceso, estos robots pueden operar en condiciones extremas donde los humanos no pueden estar presentes, como en Marte o en las profundidades del océano.

Impacto en la sociedad
La llegada de los robots a diferentes sectores plantea importantes desafíos y oportunidades para la sociedad:
Automatización y empleo: La automatización de ciertas tareas mediante robots puede mejorar la productividad, pero también puede generar preocupación sobre la pérdida de empleos en sectores como la manufactura. Sin embargo, se espera que surjan nuevas oportunidades laborales en áreas relacionadas con la robótica y la inteligencia artificial.
Mejoras en la calidad de vida: Los robots pueden mejorar la calidad de vida de las personas, especialmente en el ámbito de la salud. Por ejemplo, los robots pueden ayudar a cuidar a los ancianos, realizar diagnósticos médicos más rápidos y precisos, o incluso realizar tareas domésticas.
Ética y privacidad: A medida que los robots se vuelven más inteligentes y autónomos, surgen preguntas sobre la ética de su uso. ¿Deberían tener derechos? ¿Quién es responsable si un robot causa daño? Además, el uso de robots que recopilan datos de las personas genera preocupaciones sobre la privacidad y el control de la información personal.
Educación y aprendizaje: Los robots pueden tener un papel clave en la educación, ya que pueden proporcionar experiencias de aprendizaje personalizadas y ayudar a los estudiantes a desarrollar habilidades en ciencia, tecnología, ingeniería y matemáticas (STEM).

El futuro de los robots
El futuro de la robótica es prometedor y está lleno de posibilidades. A medida que la inteligencia artificial y los sistemas autónomos continúan avanzando, los robots podrían ser cada vez más inteligentes, capaces de realizar tareas complejas, tomar decisiones por sí mismos y aprender de su entorno.
"""

# Ejecutar el resumen usando la función definida
summary = predict_texto(ARTICLE, model, tokenizer)

# Mostrar el resumen
print("Resumen generado:")
print(summary)

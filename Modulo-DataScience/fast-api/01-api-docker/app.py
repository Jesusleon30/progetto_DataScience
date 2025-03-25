# from fastapi import FastAPI, File, UploadFile
# from texto_analyzer.texto_analyzer import TextoAnalyzer
# from pydantic import BaseModel
# import os

# app = FastAPI()

# model_path = "C:/Users/user/Documents/nuevo_modelo"  # Ruta donde está el modelo entrenado
# analyzer = TextoAnalyzer(model_path)

# class TextoInput(BaseModel):
#     # No es necesario en este caso porque vamos a procesar un archivo
#     pass

# @app.get("/")
# async def index():
#     return {"message": "🐋 Dockerized FastAPI v1.1"}


# @app.post("/resumen")
# async def obtener_resumen(file: UploadFile = File(...)):
#     # Leer el contenido del archivo
#     content = await file.read()
#     # Asumimos que el archivo es de texto plano en UTF-8
#     texto = content.decode('utf-8')

#     # Obtener el resumen utilizando el modelo
#     resumen = analyzer.predict_texto(texto)

#     # Retornar el resumen generado
#     return {"resumen": resumen}

from fastapi import FastAPI, File, UploadFile, HTTPException
from texto_analyzer.texto_analyzer import TextoAnalyzer
import os

app = FastAPI()

# Ruta del modelo entrenado, asegúrate de que sea correcta.
model_path = "C:/Users/user/Documents/nuevo_modelo"

# Verifica si el modelo está disponible
if not os.path.exists(model_path):
    raise ValueError(f"El modelo no se encuentra en la ruta especificada: {model_path}")

# Inicializa el analizador de texto
analyzer = TextoAnalyzer(model_path)

@app.get("/")
async def index():
    return {"message": "🐋 Dockerized FastAPI v1.1"}

@app.post("/resumen")
async def obtener_resumen(file: UploadFile = File(...)):
    try:
        # Leer el contenido del archivo
        content = await file.read()
        
        # Decodificar el contenido como texto plano (UTF-8)
        texto = content.decode('utf-8')

        # Obtener el resumen utilizando el modelo
        resumen = analyzer.predict_texto(texto)

        # Retornar el resumen generado
        return {"resumen": resumen}

    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="El archivo no es un archivo de texto válido (debe estar en UTF-8).")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hubo un error al procesar el archivo: {str(e)}")
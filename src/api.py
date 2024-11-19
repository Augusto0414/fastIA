import os
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import numpy as np
import json
from tensorflow.keras.models import load_model

app = FastAPI()

# Obtener la ruta base del directorio raíz
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cargar el modelo y las clases
model_path = os.path.join(BASE_DIR, 'modelo_plantas_mejorado.h5')
classes_path = os.path.join(BASE_DIR, 'clases.json')
descriptions_path = os.path.join(BASE_DIR, 'descripciones.json')

model = load_model(model_path)
with open(classes_path, 'r', encoding='utf-8') as f:
    clases = json.load(f)

# Cargar las descripciones
with open(descriptions_path, 'r', encoding='utf-8') as f:
    descripciones = json.load(f)

# Preprocesar imagen desde bytes
def procesar_imagen_desde_bytes(imagen_bytes, target_size=(150, 150)):
    img = Image.open(BytesIO(imagen_bytes))
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.post("/predict/")
async def predecir(file: UploadFile = File(...)):
    # Leer la imagen
    imagen_bytes = await file.read()
    img_array = procesar_imagen_desde_bytes(imagen_bytes)

    # Realizar predicción
    prediccion = model.predict(img_array)
    clase_id = np.argmax(prediccion, axis=1)[0]
    clase_nombre = clases[str(clase_id)]

    # Obtener descripción desde el JSON
    descripcion = descripciones.get(clase_nombre, "No hay descripción disponible para esta planta.")

    return {
        "clase": clase_nombre,
        "descripcion": descripcion,
        "probabilidades": prediccion.tolist()
    }

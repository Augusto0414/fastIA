import os
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import numpy as np
import json
from tensorflow.keras.models import load_model

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, 'modelo_plantas_mejorado.h5')
plantas_path = os.path.join(BASE_DIR, 'plantas.json')

model = load_model(model_path)
with open(plantas_path, 'r', encoding='utf-8') as f:
    plantas = json.load(f)

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
    clase_nombre = plantas[str(clase_id)]["nombre"]

    
    planta_info = plantas[str(clase_id)]

    return {
        "clase": clase_nombre,
        "descripcion": planta_info["descripcion"],
        "caracteristicas": planta_info["caracteristicas"],
        "probabilidades": prediccion.tolist()
    }

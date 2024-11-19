import os
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import load_model

# Deshabilitar uso de GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = FastAPI()

# Definir rutas absolutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Ajuste si se usa un directorio diferente
model_path = os.path.join(BASE_DIR, '../modelo_plantas_mejorado.h5')  # Ruta del modelo
plantas_path = os.path.join(BASE_DIR, '../plantas.json')  # Ruta del JSON

# Cargar modelo en CPU
with tf.device('/CPU:0'):
    model = load_model(model_path)

# Cargar archivo JSON
with open(plantas_path, 'r', encoding='utf-8') as f:
    plantas = json.load(f)

# Función para preprocesar imágenes
def procesar_imagen_desde_bytes(imagen_bytes, target_size=(150, 150)):
    img = Image.open(BytesIO(imagen_bytes)).convert('RGB')  # Convertir a RGB para evitar errores
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalizar
    return img_array

@app.post("/predict/")
async def predecir(file: UploadFile = File(...)):
    # Leer la imagen
    imagen_bytes = await file.read()
    img_array = procesar_imagen_desde_bytes(imagen_bytes)

    # Realizar predicción
    prediccion = model.predict(img_array)
    clase_id = int(np.argmax(prediccion, axis=1)[0])  # Obtener índice de clase
    planta_info = plantas[str(clase_id)]

    return {
        "clase": planta_info["nombre"],
        "descripcion": planta_info["descripcion"],
        "caracteristicas": planta_info["caracteristicas"],
        "probabilidades": prediccion.tolist()
    }

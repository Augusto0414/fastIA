import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Cargar el modelo y las clases
model = load_model('modelo_plantas_mejorado.h5')
with open('clases.json', 'r') as f:
    clases = json.load(f)

# Función para preprocesar una imagen desde una ruta
def procesar_imagen(imagen_path, target_size=(150, 150)):
    img = image.load_img(imagen_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Función para predecir la clase de una imagen
def predecir_imagen(imagen_path):
    img_array = procesar_imagen(imagen_path)
    prediccion = model.predict(img_array)
    clase_id = np.argmax(prediccion, axis=1)[0]
    clase_nombre = clases[str(clase_id)]
    return clase_nombre

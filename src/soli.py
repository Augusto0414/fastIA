import requests

url = "http://127.0.0.1:8000/predict/"
file_path = r"C:\Users\AUGUSTO\Desktop\IA\data\train\Cedro\IMG_6217.JPG"

# Abrir la imagen y enviarla a la API
with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

if response.status_code == 200:
    print("Respuesta de la API:", response.json())
else:
    print("Error:", response.status_code, response.text)

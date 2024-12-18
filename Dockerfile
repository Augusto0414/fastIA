# Usar una imagen base de Python
FROM python:3.9-slim

# Definir el directorio de trabajo
WORKDIR /app

# Copiar los archivos del proyecto al contenedor
COPY ./src /app/src
COPY ./modelo_plantas_mejorado.h5 /app
COPY ./plantas.json /app

# Copiar el archivo de dependencias y instalar
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto 8000
EXPOSE 8000

# Comando para ejecutar la API con Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

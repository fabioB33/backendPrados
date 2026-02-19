FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero para cachear las dependencias
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY . .

# Exponer el puerto (Railway puede usar cualquier puerto)
EXPOSE 8000

# Comando para ejecutar la aplicación
# Railway proporcionará el puerto en la variable PORT
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}

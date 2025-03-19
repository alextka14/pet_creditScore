# Используем официальный python-образ
FROM python:3.9-slim

# Рабочая директория в контейнере
WORKDIR /app

# Скопируем requirements и установим
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Скопируем все исходники в /app
COPY . .

# По умолчанию команда
CMD ["python", "src/train.py"]

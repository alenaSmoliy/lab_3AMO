# Базовый образ с Python
FROM python:3.10-slim

# Установка рабочей директории внутри контейнера
WORKDIR /app

# Копирование всех файлов проекта в контейнер
COPY . /app

# Установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Команда для запуска скрипта
CMD ["python", "main.py"]

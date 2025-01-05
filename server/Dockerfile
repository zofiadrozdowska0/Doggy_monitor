# Wybór obrazu bazowego z obsługą CUDA
FROM pytorch/pytorch:latest

# Ustawienia dla zmiennych środowiskowych
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV DEBIAN_FRONTEND = noninteractive
# Zainstaluj zależności
RUN apt-get update && \
    apt install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalacja FastAPI, OpenCV i innych wymaganych pakietów
RUN pip install --no-cache-dir \
    fastapi \
    opencv-python \
    uvicorn \
    python-dotenv \
    && pip install torch torchvision torchaudio

# Ustawienie katalogu roboczego
WORKDIR /app

# Klonowanie repozytorium (zmień na odpowiednie repozytorium)
RUN git clone https://git.pg.edu.pl/p1353132/doggy-monitor.git

# Otwarte porty dla aplikacji API oraz RTMP/streaming
EXPOSE 8000 1935

# Uruchomienie aplikacji (zmień na komendę uruchamiającą Twój serwer)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

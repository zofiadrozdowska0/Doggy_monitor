import subprocess

def start_stream():
    # Pełna komenda, dokładnie taka, jaką chcesz uruchomić
    command = (
        "libcamera-vid -t 0 --width 640 --height 340 --framerate 10 --codec h264 --inline --rotation 180 -o - | "
        "ffmpeg -re -fflags nobuffer -flags low_delay -fflags discardcorrupt -f h264 -i - -c:v copy -vsync vfr -f flv rtmp://broadcast.api.video/s/c99442e6-bfa0-4f59-ab75-9c461668e640"
    )

    # Wykonanie komendy
    subprocess.run(command, shell=True, check=True)

# Wywołanie funkcji, aby rozpocząć strumień
start_stream()

# Schemat komunikacji serwer <-> aplikacja

```mermaid

---
config:
  theme: default
---
sequenceDiagram
    participant Server as Serwer
    participant App as Aplikacja mobilna
    participant User as Użytkownik
    Note over Server: Przetwarzanie filmu<br>Model do wykrywania emocji<br>analizuje klatki
    Server ->> Server: Analiza filmu <br> (Model do wykrywania emocji)
    App ->> Server: GET /download_video_by_url<br>Parametry: { "session_id": "abc123" }<br>Opis: Pobieranie filmu przez URL (domyślny port HTTP/HTTPS)
    Server -->> App: Response: { "video_url": "http://server/video.mjpeg" }
    App ->> Server: GET 5005/emotion_status<br>Parametry: { "session_id": "abc123" }
    Server -->> App: Response: { "emotion": "happy" }
    App ->> Server: GET 5006/emotion_history<br>Parametry: { "session_id": "abc123" }
    Server -->> App: Response: { "history_file": "http://server/emotion.txt" }
    App ->> User: Wyświetlenie filmu<br>i informacji o emocji psa<br>oraz historii emocji

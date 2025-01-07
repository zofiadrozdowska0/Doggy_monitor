from flask import Flask, send_file

app = Flask(__name__)

@app.route('/video')
def serve_video():
    video_path = 'piesel.mp4'  # Ścieżka do pliku MP4
    return send_file(video_path, mimetype='video/mp4')

if __name__ == '__main__':
    app.run(host='localhost', port=5000)  # Uruchom serwer na porcie 5000

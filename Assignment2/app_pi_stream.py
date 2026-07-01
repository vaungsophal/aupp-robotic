#!/usr/bin/env python3
from argparse import ArgumentParser
from threading import Lock, Thread
import time

import cv2
from flask import Flask, Response, jsonify, make_response
from ultralytics import YOLO


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>YOLOv8 ONNX - Raspberry Pi Stream</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    :root { color-scheme: light dark; }
    body { margin:0; min-height:100vh; display:grid; place-items:center;
           background:#0b0c10; color:#eaf0f6; font-family:system-ui,Segoe UI,Roboto,sans-serif; }
    .card { width:min(96vw,900px); background:#111417; border-radius:16px; padding:14px;
            border:1px solid rgba(255,255,255,0.08); box-shadow:0 10px 40px rgba(0,0,0,.35); }
    h1 { margin:6px 0 10px; font-size:1.05rem; }
    .row { display:flex; gap:10px; justify-content:space-between; align-items:center; flex-wrap:wrap; }
    .btn { border:1px solid rgba(255,255,255,.12); background:#1b2229; color:#eaf0f6;
           padding:6px 12px; border-radius:10px; cursor:pointer; font-weight:600; }
    .btn:hover { background:#222b33; }
    .frame { width:100%; aspect-ratio:16/9; background:#0d1117; border-radius:12px; overflow:hidden;
             border:1px solid rgba(255,255,255,0.08); display:grid; place-items:center; }
    img { width:100%; height:100%; object-fit:contain; }
    small { opacity:.65; }
  </style>
</head>
<body>
  <div class="card">
    <div class="row">
      <h1>Raspberry Pi YOLOv8 ONNX Live</h1>
      <button class="btn" onclick="reloadStream()">Reload</button>
    </div>
    <div class="frame">
      <img id="stream" src="/stream" alt="Stream">
    </div>
    <div class="row" style="margin-top:8px;">
      <small>Status: <span id="health">checking...</span></small>
      <small>URL: <code id="url"></code></small>
    </div>
  </div>
<script>
  async function checkHealth() {
    try {
      const r = await fetch('/health', {cache:'no-store'});
      const j = await r.json();
      document.getElementById('health').textContent = j.camera_ok ? 'camera OK' : 'no camera';
    } catch (e) {
      document.getElementById('health').textContent = 'server offline';
    }
  }
  function reloadStream() {
    const img = document.getElementById('stream');
    img.src = '/stream?ts=' + Date.now();
  }
  document.getElementById('url').textContent = location.href;
  checkHealth();
  setInterval(checkHealth, 4000);
</script>
</body></html>
"""


class Camera:
    def __init__(self, index=0, width=None, height=None):
        self.cap = cv2.VideoCapture(index)
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.ok, self.frame = self.cap.read()
        self.lock = Lock()
        self.running = True
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.ok = ok
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
            return self.ok, frame

    def release(self):
        self.running = False
        time.sleep(0.05)
        self.cap.release()


def create_app(model, camera, image_size, confidence):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return make_response(INDEX_HTML, 200)

    @app.route("/health")
    def health():
        ok, _ = camera.read()
        return jsonify({"camera_ok": bool(ok)})

    def gen_mjpeg():
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            results = model.predict(frame, imgsz=image_size, conf=confidence, verbose=False)
            annotated = results[0].plot()
            ok, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue

            data = jpg.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n"
                b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n"
            )

    @app.route("/stream")
    def stream():
        return Response(gen_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


def parse_args():
    parser = ArgumentParser(description="Run YOLOv8 ONNX detection as a Raspberry Pi MJPEG web stream.")
    parser.add_argument("--model", default="best.onnx", help="Path to exported ONNX model.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--imgsz", type=int, default=320, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold.")
    parser.add_argument("--width", type=int, default=None, help="Optional camera capture width.")
    parser.add_argument("--height", type=int, default=None, help="Optional camera capture height.")
    parser.add_argument("--host", default="0.0.0.0", help="Flask host.")
    parser.add_argument("--port", type=int, default=5000, help="Flask port.")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    camera = Camera(args.camera, width=args.width, height=args.height)
    app = create_app(model, camera, args.imgsz, args.conf)

    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        camera.release()


if __name__ == "__main__":
    main()

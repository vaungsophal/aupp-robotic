from argparse import ArgumentParser
from pathlib import Path

import cv2
from ultralytics import YOLO

# Hide OpenCV's camera-probe warnings (obsensor/MSMF noise when indexes are empty).
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)


def parse_camera(value):
    try:
        return int(value)
    except ValueError:
        return value


def open_camera(source):
    if isinstance(source, int):
        # CAP_DSHOW and CAP_MSMF are the native Windows backends; trying them
        # explicitly avoids the slow/noisy obsensor probe and opens faster.
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
        indexes = [source] + [i for i in [0, 1, 2, 3] if i != source]
        for backend in backends:
            for index in indexes:
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        print(f"Opened camera index {index}")
                        return cap, index
                cap.release()
        return None, None

    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        print(f"Opened source: {source}")
        return cap, source
    return None, None


def find_sample_image():
    image_dirs = [
        Path("WRO detection.v1i.yolov8/test/images"),
        Path("WRO detection.v1i.yolov8/valid/images"),
        Path("WRO detection.v1i.yolov8/train/images"),
    ]

    for directory in image_dirs:
        if not directory.exists():
            continue
        images = sorted(
            [
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]
        )
        if images:
            return str(images[0])

    return None


def draw_detections(frame, model, conf, imgsz):
    results = model(frame, stream=True, imgsz=imgsz, conf=conf, verbose=False)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf_value = float(box.conf[0])
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = (0, 255, 0) if label == "greenbox" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} {conf_value:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            print(f"Detected: {label} (conf {conf_value:.2f})")


def main():
    parser = ArgumentParser(description="Test YOLOv8 red/green object detection on a laptop.")
    parser.add_argument("--model", default="best.pt", help="Path to trained YOLOv8 .pt model.")
    parser.add_argument("--camera", default="0", help="Camera index or video file path.")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=320, help="Inference image size.")
    args = parser.parse_args()

    model = YOLO(args.model)
    source = parse_camera(args.camera)
    cap, opened_source = open_camera(source)

    if cap is None:
        print(
            "No webcam detected. If your laptop has one, make sure it is enabled:\n"
            "  - MSI laptops: press Fn+F6 to toggle the webcam on\n"
            "  - Check Device Manager > Cameras (a missing/greyed device means it is switched off)\n"
            "  - Check Windows Settings > Privacy & security > Camera"
        )
        sample_image = find_sample_image()
        if sample_image:
            print("Running detection on a sample image instead.")
            frame = cv2.imread(sample_image)
            if frame is None:
                raise SystemExit(f"Could not read sample image: {sample_image}")
            draw_detections(frame, model, args.conf, args.imgsz)
            cv2.imshow("YOLOv8 Detection", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

        raise SystemExit(
            "Cannot open camera or video source. Try --camera 1, a video path, or connect a webcam."
        )

    print("Webcam opened successfully. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed.")
                break

            draw_detections(frame, model, args.conf, args.imgsz)
            cv2.imshow("YOLOv8 Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

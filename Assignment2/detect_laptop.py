from argparse import ArgumentParser

import cv2
from ultralytics import YOLO


def parse_camera(value):
    try:
        return int(value)
    except ValueError:
        return value


def main():
    parser = ArgumentParser(description="Test YOLOv8 red/green object detection on a laptop.")
    parser.add_argument("--model", default="best.pt", help="Path to trained YOLOv8 .pt model.")
    parser.add_argument("--camera", default="0", help="Camera index or video file path.")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=320, help="Inference image size.")
    args = parser.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(parse_camera(args.camera))

    if not cap.isOpened():
        raise SystemExit("Cannot open camera or video source.")

    print("Webcam opened successfully. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed.")
            break

        results = model(frame, stream=True, imgsz=args.imgsz, conf=args.conf, verbose=False)

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                color = (0, 255, 0) if label == "greenbox" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

                print(f"Detected: {label} (conf {conf:.2f})")

        cv2.imshow("YOLOv8 Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

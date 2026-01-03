import time
import cv2
import numpy as np
from ultralytics import YOLO

# ==========================
# CONFIG (EDIT IF NEEDED)
# ==========================
MODEL_PATH = "best.pt"      # your trained model
CAMERA_INDEX = 0            # default USB webcam
CONF_THRESH = 0.5
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
# ==========================

# Load YOLO model
model = YOLO(MODEL_PATH)
labels = model.names

# Open USB webcam
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print("ERROR: Could not open webcam")
    exit()

# Colors for bounding boxes
bbox_colors = [
    (164,120,87), (68,148,228), (93,97,209), (178,182,133),
    (88,159,106), (96,202,231), (159,124,168),
    (169,162,241), (98,118,150), (172,176,184)
]

fps_buffer = []
FPS_AVG_LEN = 20

print("YOLO detection started. Press Q to quit.")

while True:
    t_start = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO inference
    results = model(frame, verbose=False)
    detections = results[0].boxes

    object_count = 0

    for det in detections:
        conf = det.conf.item()
        if conf < CONF_THRESH:
            continue

        xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy

        class_id = int(det.cls.item())
        label = labels[class_id]

        color = bbox_colors[class_id % len(bbox_colors)]

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        text = f"{label} {int(conf*100)}%"
        cv2.putText(
            frame, text, (xmin, ymin - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

        object_count += 1

    # FPS calculation
    t_end = time.perf_counter()
    fps = 1 / (t_end - t_start)
    fps_buffer.append(fps)
    if len(fps_buffer) > FPS_AVG_LEN:
        fps_buffer.pop(0)

    avg_fps = sum(fps_buffer) / len(fps_buffer)

    # Overlay info
    cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cv2.putText(frame, f"Objects: {object_count}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    # Show frame
    cv2.imshow("YOLO Pi Webcam", frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Exited cleanly.")

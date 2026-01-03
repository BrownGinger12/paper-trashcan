import cv2
import numpy as np
import ncnn
import time

# Load NCNN model
net = ncnn.Net()
net.load_param("model.ncnn.param")
net.load_model("model.ncnn.bin")

# Camera setup
cap = cv2.VideoCapture(0)  # Use 0 for default USB camera
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# YOLOv5 anchors for small model (yolov5n) - change if using different model
anchors = [
    [[10,13, 16,30, 33,23]],  # P3/8
    [[30,61, 62,45, 59,119]], # P4/16
    [[116,90, 156,198, 373,326]]  # P5/32
]

stride = [8,16,32]
conf_threshold = 0.25
iou_threshold = 0.45
input_size = 320  # make sure to use same size as during NCNN export

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def xywh2xyxy(x):
    # Convert [x_center, y_center, w, h] to [x1, y1, x2, y2]
    y = np.zeros_like(x)
    y[0] = x[0] - x[2]/2
    y[1] = x[1] - x[3]/2
    y[2] = x[0] + x[2]/2
    y[3] = x[1] + x[3]/2
    return y

# Main loop
fps_buffer = []
while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Convert BGR -> RGB and resize
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    in_mat = ncnn.Mat.from_pixels_resize(rgb_frame, ncnn.Mat.PIXEL_RGB, w, h, input_size, input_size)
    # Normalize to 0-1
    in_mat.substract_mean_normalize(mean=[0,0,0], norm=[1/255.0,1/255.0,1/255.0])

    # Create extractor and run inference
    ex = net.create_extractor()
    ex.input("images", in_mat)
    out_mat = ncnn.Mat()
    ex.extract("output", out_mat)

    # Convert output to numpy array
    out = np.array(out_mat)

    # Postprocess (YOLOv5 NMS)
    boxes = []
    scores = []
    class_ids = []

    for det in out:
        conf = det[4]
        if conf < conf_threshold:
            continue
        cls_id = np.argmax(det[5:])
        cls_conf = det[5+cls_id]
        final_conf = conf * cls_conf
        if final_conf < conf_threshold:
            continue
        cx, cy, bw, bh = det[0], det[1], det[2], det[3]
        box = xywh2xyxy([cx* w/input_size, cy* h/input_size, bw* w/input_size, bh* h/input_size])
        boxes.append(box)
        scores.append(final_conf)
        class_ids.append(cls_id)

    # Draw boxes
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        label = f"{class_ids[i]}:{scores[i]:.2f}"
        cv2.putText(frame, label, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # FPS calculation
    t1 = time.time()
    fps = 1/(t1-t0)
    fps_buffer.append(fps)
    if len(fps_buffer) > 30:
        fps_buffer.pop(0)
    avg_fps = sum(fps_buffer)/len(fps_buffer)
    cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("YOLOv5 NCNN", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

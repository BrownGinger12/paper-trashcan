import tkinter as tk
from PIL import Image, ImageTk
import cv2
import serial
import time
import threading
import numpy as np
import ncnn

# ===========================
# YOLO NCNN DETECTOR CLASS
# ===========================
class YOLOv5Detector:
    def __init__(self, param_path, bin_path, target_size=320, conf_threshold=0.5, nms_threshold=0.45):
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = 4  # Adjust based on your Pi model (2-4)
        self.net.load_param(param_path)
        self.net.load_model(bin_path)
        
        self.target_size = target_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # CUSTOM CLASS - Only detecting paper
        self.class_names = ['paper']  # Single class detection
    
    def detect(self, img):
        img_h, img_w = img.shape[:2]
        
        # Prepare input
        mat_in = ncnn.Mat.from_pixels_resize(
            img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, 
            img_w, img_h, self.target_size, self.target_size
        )
        
        # Normalize
        norm_vals = [1/255.0, 1/255.0, 1/255.0]
        mat_in.substract_mean_normalize([], norm_vals)
        
        # Inference
        ex = self.net.create_extractor()
        ex.input("images", mat_in)
        
        mat_out = ncnn.Mat()
        ex.extract("output0", mat_out)
        
        # Post-process
        detections = self.post_process(mat_out, img_w, img_h)
        return detections
    
    def post_process(self, mat_out, img_w, img_h):
        # Convert ncnn.Mat to numpy
        out = np.array(mat_out).reshape(-1, mat_out.w)
        
        boxes = []
        confidences = []
        class_ids = []
        
        for detection in out:
            # For single-class model, confidence is just detection[4]
            confidence = detection[4]
            
            if confidence > self.conf_threshold:
                # Scale coordinates back to original image size
                x_center = detection[0] * img_w / self.target_size
                y_center = detection[1] * img_h / self.target_size
                width = detection[2] * img_w / self.target_size
                height = detection[3] * img_h / self.target_size
                
                # Convert to top-left corner format
                x = int(x_center - width / 2)
                y = int(y_center - height / 2)
                
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(0)  # Single class = 0
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append({
                    'box': boxes[i],
                    'confidence': confidences[i],
                    'class_id': class_ids[i],
                    'class_name': 'paper'
                })
        
        return results

# ===========================
# CONFIGURATION
# ===========================
MODEL_PARAM = "yolov5n.param"  # NCNN param file
MODEL_BIN = "yolov5n.bin"      # NCNN bin file
CAM_INDEX = 0                  # webcam (usually 0 on Pi)
SERIAL_PORT = "/dev/ttyUSB0"   # Arduino serial port on Pi
BAUD_RATE = 9600

# Display settings for 320x240 screen
DISPLAY_WIDTH = 320
DISPLAY_HEIGHT = 240
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240

SERVO_OPEN = "open\n"
SERVO_CLOSE = "close\n"

# Serial setup
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print("Serial connected")
except Exception as e:
    print(f"Serial connection failed: {e}")
    arduino = None

# Load YOLO NCNN model
model = YOLOv5Detector(MODEL_PARAM, MODEL_BIN, target_size=320, conf_threshold=0.4)
print("NCNN Model loaded")

# ===========================
# TKINTER GUI - Optimized for 320x240
# ===========================
class AIWeightApp:
    def __init__(self, root):
        self.root = root
        root.overrideredirect(True)
        root.geometry(f"{DISPLAY_WIDTH}x{DISPLAY_HEIGHT}+0+0")
        root.configure(bg="black")

        # Camera
        self.cap = cv2.VideoCapture(CAM_INDEX)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.imgtk = None

        # Main container
        self.main_frame = tk.Frame(root, bg="black")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Top: Video feed (180 pixels tall)
        self.video_canvas = tk.Canvas(self.main_frame, width=320, height=180, 
                                      bg="black", highlightthickness=0)
        self.video_canvas.pack(side=tk.TOP)

        # Bottom: Info panel (60 pixels tall)
        self.info_frame = tk.Frame(self.main_frame, bg="black", height=60)
        self.info_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.info_frame.pack_propagate(False)

        # Info labels - compact layout
        self.weight_label = tk.Label(self.info_frame, text="W: --- kg", 
                                     font=("Arial", 14, "bold"), fg="lime", bg="black")
        self.weight_label.pack(side=tk.LEFT, padx=10)

        self.battery_label = tk.Label(self.info_frame, text="B: ---%", 
                                      font=("Arial", 14, "bold"), fg="yellow", bg="black")
        self.battery_label.pack(side=tk.LEFT, padx=10)

        self.status_label = tk.Label(self.info_frame, text="●", 
                                     font=("Arial", 20), fg="red", bg="black")
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # Control variables
        self.weight_kg = 0.0
        self.battery_percent = 0
        self.object_detected = False
        self.fps = 0

        # Start threads
        self.running = True
        if arduino:
            threading.Thread(target=self.read_serial, daemon=True).start()
        self.update_frame()

    def read_serial(self):
        """Continuously read weight & battery from Arduino"""
        buffer = ""
        while self.running:
            try:
                if arduino.in_waiting:
                    buffer += arduino.read(arduino.in_waiting).decode('utf-8', errors='ignore')
                    lines = buffer.split("\n")
                    buffer = lines[-1]

                    for line in lines[:-1]:
                        line = line.strip().lower()

                        # WEIGHT
                        if "weigth:" in line or "weight:" in line:
                            try:
                                value_str = line.replace("weigth:", "").replace("weight:", "").replace("kg","").strip()
                                self.weight_kg = float(value_str)
                                self.weight_label.config(text=f"W: {self.weight_kg:.2f}kg")
                            except:
                                pass

                        # BATTERY
                        elif "battery:" in line:
                            try:
                                value_str = line.replace("battery:", "").strip()
                                self.battery_percent = int(value_str)
                                self.battery_label.config(text=f"B: {self.battery_percent}%")
                            except:
                                pass

            except Exception as e:
                print("Serial read error:", e)

            time.sleep(0.05)

    def send_servo_command(self, command):
        if arduino:
            try:
                arduino.write(command.encode())
            except:
                print("Failed to send serial command")

    def update_frame(self):
        t_start = time.time()

        ret, frame = self.cap.read()
        if ret:
            # Resize to fit display (320x180 for video area)
            frame = cv2.resize(frame, (320, 180))
            display = frame.copy()

            # Run YOLO NCNN detection
            detections = model.detect(frame)
            detected = False

            for det in detections:
                x, y, w, h = det['box']
                conf = det['confidence']
                
                detected = True
                color = (0, 255, 0)
                cv2.rectangle(display, (x, y), (x+w, y+h), color, 1)
                
                # Compact label
                label = f"{int(conf*100)}%"
                cv2.putText(display, label, (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Update servo and status indicator
            if detected and not self.object_detected:
                self.send_servo_command(SERVO_OPEN)
                self.object_detected = True
                self.status_label.config(fg="lime", text="●")
                print("Paper detected - Opening servo")
            elif not detected and self.object_detected:
                self.send_servo_command(SERVO_CLOSE)
                self.object_detected = False
                self.status_label.config(fg="red", text="●")
                print("No paper - Closing servo")

            # Calculate FPS
            self.fps = 1.0 / (time.time() - t_start + 1e-6)
            
            # Draw FPS on video
            cv2.putText(display, f"{self.fps:.1f}fps", (5, 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Convert to Tk image
            img = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            self.imgtk = ImageTk.PhotoImage(image=img)
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.imgtk)

        # Dynamic delay
        t_elapsed = time.time() - t_start
        delay = max(1, int((1/15 - t_elapsed) * 1000))
        self.root.after(delay, self.update_frame)

    def stop(self):
        self.running = False
        self.cap.release()
        if arduino:
            arduino.close()
        self.root.destroy()

# ===========================
# RUN APP
# ===========================
if __name__ == "__main__":
    root = tk.Tk()
    app = AIWeightApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop)
    root.mainloop()
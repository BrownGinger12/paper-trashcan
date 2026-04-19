from ultralytics import YOLO
import cv2
import time
import serial
import tkinter as tk
from PIL import Image, ImageTk

# =========================
# CONFIGURATION
# =========================
CONF_THRESHOLD = 0.85
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600
MAX_WEIGHT_KG = 1.0
OPEN_FRAMES_REQUIRED = 5
CLOSE_DELAY = 0.5

# =========================
# Initialize Serial to Arduino
# =========================
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"[INFO] Serial connected to {SERIAL_PORT}")
except Exception as e:
    print(f"[ERROR] Could not connect to Arduino: {e}")
    arduino = None

# =========================
# Load YOLO model
# =========================
model = YOLO("/home/ecechmsu/Desktop/paper-trashcan/my_model.pt")

# =========================
# Camera setup (reduced resolution for Pi)
# =========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 10)

# =========================
# Global State
# =========================
class SystemState:
    def __init__(self):
        self.servo_open = False
        self.weight_kg = 0.0
        self.battery_percent = 100
        self.paper_detected_count = 0
        self.last_detection_conf = 0.0
        self.last_no_paper_time = 0
        self.running = True

state = SystemState()

# =========================
# Minimal GUI Class
# =========================
class MinimalGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Paper Detection")
        self.root.configure(bg='black')
        
        # Make fullscreen (optional - comment out if not needed)
        # self.root.attributes('-fullscreen', True)
        
        # Main container
        main_frame = tk.Frame(self.root, bg='black')
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video canvas
        self.video_canvas = tk.Canvas(main_frame, width=640, height=480, 
                                      bg='black', highlightthickness=0)
        self.video_canvas.pack(pady=10)
        
        # Info bar
        info_frame = tk.Frame(main_frame, bg='black')
        info_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Battery display
        battery_frame = tk.Frame(info_frame, bg='black')
        battery_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Label(battery_frame, text="🔋 BATTERY", 
                font=('Arial', 14, 'bold'), bg='black', fg='#888').pack()
        self.battery_label = tk.Label(battery_frame, text="100%", 
                                      font=('Arial', 32, 'bold'), 
                                      bg='black', fg='#27AE60')
        self.battery_label.pack()
        
        # Weight display
        weight_frame = tk.Frame(info_frame, bg='black')
        weight_frame.pack(side=tk.RIGHT, padx=20)
        
        tk.Label(weight_frame, text="⚖️ WEIGHT", 
                font=('Arial', 14, 'bold'), bg='black', fg='#888').pack()
        self.weight_label = tk.Label(weight_frame, text="0.00 kg", 
                                     font=('Arial', 32, 'bold'), 
                                     bg='black', fg='#3498DB')
        self.weight_label.pack()
        
        # Bind ESC key to exit
        self.root.bind('<Escape>', lambda e: self.on_closing())
        
        self.update_frame()
        
    def update_frame(self):
        ret, frame = cap.read()
        if ret:
            # Run YOLO detection
            results = model(frame, imgsz=416, conf=CONF_THRESHOLD, 
                          device="cpu", verbose=False)
            
            paper_detected = False
            max_conf = 0.0
            
            for r in results:
                if r.boxes:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = model.names[cls_id]
                        
                        if class_name.lower() == "paper" and conf >= CONF_THRESHOLD:
                            paper_detected = True
                            max_conf = max(max_conf, conf)
                            
                            # Draw bounding box
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(frame, f"Paper {conf:.2f}", (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Update detection count
            if paper_detected:
                state.paper_detected_count += 1
                state.last_detection_conf = max_conf
            else:
                state.paper_detected_count = 0
                state.last_detection_conf = 0.0
            
            # Read from Arduino
            if arduino:
                try:
                    while arduino.in_waiting:
                        line = arduino.readline().decode('utf-8').strip().lower()
                        if "weigth:" in line:
                            value_str = line.replace("weigth:", "").replace("kg", "").strip()
                            try:
                                state.weight_kg = float(value_str)
                            except:
                                state.weight_kg = 0
                        elif "battery:" in line:
                            value_str = line.replace("battery:", "").strip()
                            try:
                                state.battery_percent = int(value_str)
                            except:
                                state.battery_percent = 100
                except:
                    pass
            
            # Servo control logic
            if arduino:
                if state.paper_detected_count >= OPEN_FRAMES_REQUIRED and state.weight_kg < MAX_WEIGHT_KG:
                    if not state.servo_open:
                        arduino.write(b"open\n")
                        state.servo_open = True
                        print(f"[OPEN] weight: {state.weight_kg:.2f}kg")
                elif state.paper_detected_count < OPEN_FRAMES_REQUIRED:
                    if state.servo_open and (time.time() - state.last_no_paper_time >= CLOSE_DELAY):
                        arduino.write(b"close\n")
                        state.servo_open = False
                        print("[CLOSE]")
                    if not state.servo_open:
                        state.last_no_paper_time = time.time()
            
            # Update display
            self.update_info()
            
            # Convert and display frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.video_canvas.imgtk = imgtk
        
        if state.running:
            self.root.after(50, self.update_frame)
    
    def update_info(self):
        # Update battery with color
        battery = state.battery_percent
        if battery > 50:
            battery_color = '#27AE60'  # Green
        elif battery > 20:
            battery_color = '#F39C12'  # Orange
        else:
            battery_color = '#E74C3C'  # Red
        
        self.battery_label.config(text=f"{battery}%", fg=battery_color)
        
        # Update weight
        self.weight_label.config(text=f"{state.weight_kg:.2f} kg")
    
    def on_closing(self):
        state.running = False
        cap.release()
        if arduino:
            arduino.close()
        print("[INFO] Shutdown complete")
        self.root.destroy()

# =========================
# Main
# =========================
def main():
    print("[INFO] Starting minimal GUI for Raspberry Pi")
    
    root = tk.Tk()
    app = MinimalGUI(root)
    
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
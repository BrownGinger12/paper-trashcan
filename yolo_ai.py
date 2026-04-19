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
        self.fill_level = 0.0
        self.distance = 0.0
        self.voltage = 0.0
        self.battery_percent = 100
        self.paper_detected_count = 0
        self.plastic_detected_count = 0
        self.last_detection_conf = 0.0
        self.last_no_paper_time = 0
        self.running = True
        self.detected_material = "NONE"

state = SystemState()

# =========================
# Enhanced GUI Class
# =========================
class EnhancedGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Waste Detection System")
        self.root.configure(bg='#1a1a1a')
        
        # Make fullscreen
        self.root.attributes('-fullscreen', True)
        
        # Main container with padding
        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Top status bar
        top_bar = tk.Frame(main_frame, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        top_bar.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(top_bar, text="🗑️ SMART TRASH BIN", 
                font=('Arial', 20, 'bold'), bg='#2d2d2d', fg='white').pack(pady=10)
        
        # Video canvas with border
        video_frame = tk.Frame(main_frame, bg='#2d2d2d', relief=tk.SUNKEN, bd=3)
        video_frame.pack(pady=5)
        
        self.video_canvas = tk.Canvas(video_frame, width=640, height=480, 
                                      bg='black', highlightthickness=0)
        self.video_canvas.pack(padx=2, pady=2)
        
        # Detection status (prominent)
        status_frame = tk.Frame(main_frame, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        status_frame.pack(fill=tk.X, pady=10)
        
        self.detection_label = tk.Label(status_frame, text="⏳ WAITING FOR WASTE", 
                                        font=('Arial', 42, 'bold'), 
                                        bg='#2d2d2d', fg='#FFD700')
        self.detection_label.pack(pady=15)
        
        # Warning label (bin full / plastic rejection)
        self.warning_label = tk.Label(status_frame, text="", 
                                      font=('Arial', 32, 'bold'), 
                                      bg='#2d2d2d', fg='#FF4444')
        self.warning_label.pack(pady=5)
        
        # Stats grid (2 columns when normal, 1 when full)
        self.stats_container = tk.Frame(main_frame, bg='#1a1a1a')
        self.stats_container.pack(fill=tk.X, pady=10)
        
        # Battery
        self.battery_frame = tk.Frame(self.stats_container, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        self.battery_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        
        tk.Label(self.battery_frame, text="🔋", 
                font=('Arial', 30), bg='#2d2d2d', fg='white').pack(pady=(10, 0))
        tk.Label(self.battery_frame, text="BATTERY", 
                font=('Arial', 12, 'bold'), bg='#2d2d2d', fg='#888').pack()
        self.battery_label = tk.Label(self.battery_frame, text="100%", 
                                      font=('Arial', 36, 'bold'), 
                                      bg='#2d2d2d', fg='#27AE60')
        self.battery_label.pack(pady=(5, 10))
        
        # Fill Level
        self.fill_frame = tk.Frame(self.stats_container, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        self.fill_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        
        tk.Label(self.fill_frame, text="📊", 
                font=('Arial', 30), bg='#2d2d2d', fg='white').pack(pady=(10, 0))
        tk.Label(self.fill_frame, text="FILL LEVEL", 
                font=('Arial', 12, 'bold'), bg='#2d2d2d', fg='#888').pack()
        self.fill_label = tk.Label(self.fill_frame, text="0%", 
                                   font=('Arial', 36, 'bold'), 
                                   bg='#2d2d2d', fg='#3498DB')
        self.fill_label.pack(pady=(5, 10))
        
        # Bin Full Message (hidden by default)
        self.bin_full_frame = tk.Frame(main_frame, bg='#E74C3C', relief=tk.RAISED, bd=5)
        
        tk.Label(self.bin_full_frame, text="⚠️", 
                font=('Arial', 80), bg='#E74C3C', fg='white').pack(pady=(20, 0))
        tk.Label(self.bin_full_frame, text="BIN IS FULL!", 
                font=('Arial', 60, 'bold'), bg='#E74C3C', fg='white').pack(pady=10)
        tk.Label(self.bin_full_frame, text="PLEASE EMPTY THE BIN", 
                font=('Arial', 30, 'bold'), bg='#E74C3C', fg='white').pack(pady=(0, 20))
        
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
            plastic_detected = False
            max_conf = 0.0
            
            for r in results:
                if r.boxes:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = model.names[cls_id].lower()
                        
                        if conf >= CONF_THRESHOLD:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            if class_name == "paper":
                                paper_detected = True
                                max_conf = max(max_conf, conf)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                                cv2.putText(frame, f"PAPER {conf:.2f}", (x1, y1 - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            
                            elif class_name == "plastic":
                                plastic_detected = True
                                max_conf = max(max_conf, conf)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                                cv2.putText(frame, f"PLASTIC {conf:.2f}", (x1, y1 - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Update detection counts
            if paper_detected:
                state.paper_detected_count += 1
                state.plastic_detected_count = 0
                state.detected_material = "PAPER"
                state.last_detection_conf = max_conf
            elif plastic_detected:
                state.plastic_detected_count += 1
                state.paper_detected_count = 0
                state.detected_material = "PLASTIC"
                state.last_detection_conf = max_conf
            else:
                state.paper_detected_count = 0
                state.plastic_detected_count = 0
                state.detected_material = "NONE"
                state.last_detection_conf = 0.0
            
            # Read from Arduino
            if arduino:
                try:
                    while arduino.in_waiting:
                        line = arduino.readline().decode('utf-8').strip()
                        
                        if "Distance:" in line:
                            try:
                                state.distance = float(line.split(":")[1].replace("cm", "").strip())
                            except:
                                pass
                        
                        elif "Fill Level:" in line:
                            try:
                                state.fill_level = float(line.split(":")[1].replace("%", "").strip())
                            except:
                                pass
                        
                        elif "Voltage:" in line:
                            try:
                                state.voltage = float(line.split(":")[1].replace("V", "").strip())
                            except:
                                pass
                        
                        elif "Battery:" in line:
                            try:
                                state.battery_percent = int(line.split(":")[1].replace("%", "").strip())
                            except:
                                pass
                except:
                    pass
            
            # Servo control logic (ONLY for paper, not plastic)
            if arduino:
                # Check if bin is full
                if state.fill_level >= 95:
                    # Don't open if bin is full
                    if state.servo_open:
                        arduino.write(b"close\n")
                        state.servo_open = False
                        print("[CLOSE] Bin full")
                elif state.paper_detected_count >= OPEN_FRAMES_REQUIRED:
                    if not state.servo_open:
                        arduino.write(b"open\n")
                        state.servo_open = True
                        print(f"[OPEN] PAPER detected - fill: {state.fill_level:.1f}%")
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
        # Check if bin is full
        bin_is_full = state.fill_level >= 95
        
        if bin_is_full:
            # Hide normal stats, show BIN FULL message
            self.stats_container.pack_forget()
            self.bin_full_frame.pack(fill=tk.BOTH, expand=True, pady=20)
            
            # Update detection to show what was detected but bin is full
            if state.detected_material == "PAPER":
                self.detection_label.config(text="📄 PAPER DETECTED", fg='#FFD700')
            elif state.detected_material == "PLASTIC":
                self.detection_label.config(text="♻️ PLASTIC DETECTED", fg='#FFD700')
            else:
                self.detection_label.config(text="⏳ NO DETECTION", fg='#888')
            
            self.warning_label.config(text="")
        else:
            # Show normal stats, hide BIN FULL message
            self.bin_full_frame.pack_forget()
            self.stats_container.pack(fill=tk.X, pady=10)
            
            # Update detection label based on what's detected
            if state.detected_material == "PAPER":
                self.detection_label.config(text="📄 PAPER DETECTED", fg='#27AE60')
            elif state.detected_material == "PLASTIC":
                self.detection_label.config(text="♻️ PLASTIC DETECTED", fg='#E74C3C')
            else:
                self.detection_label.config(text="⏳ NO DETECTION", fg='#888')
            
            # Update warning for plastic
            if state.detected_material == "PLASTIC":
                self.warning_label.config(text="🚫 PLASTIC NOT ALLOWED 🚫")
            else:
                self.warning_label.config(text="")
        
        # Always update battery
        battery = state.battery_percent
        if battery > 50:
            battery_color = '#27AE60'
        elif battery > 20:
            battery_color = '#F39C12'
        else:
            battery_color = '#E74C3C'
        self.battery_label.config(text=f"{battery}%", fg=battery_color)
        
        # Always update fill level
        fill = state.fill_level
        if fill >= 95:
            fill_color = '#E74C3C'
        elif fill >= 70:
            fill_color = '#F39C12'
        else:
            fill_color = '#3498DB'
        self.fill_label.config(text=f"{int(fill)}%", fg=fill_color)
    
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
    print("[INFO] Starting enhanced waste detection system")
    print(f"[INFO] Model classes: {model.names}")
    
    root = tk.Tk()
    app = EnhancedGUI(root)
    
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
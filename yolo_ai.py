from ultralytics import YOLO
import cv2
import time
import serial
import tkinter as tk

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
# Camera setup (headless - no display)
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
        self.plastic_detected_count = 0
        self.last_detection_conf = 0.0
        self.last_no_paper_time = 0
        self.running = True
        self.detected_material = "NONE"

state = SystemState()

# =========================
# UI-Only GUI Class (No Camera View)
# =========================
class InfoDisplayGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Trash Bin")
        self.root.configure(bg='black')
        self.root.geometry("800x600")
        
        # Make fullscreen (optional)
        self.root.attributes('-fullscreen', True)
        
        # Main container
        main_frame = tk.Frame(self.root, bg='black')
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Material Detection Display (HUGE)
        self.material_label = tk.Label(main_frame, text="NO DETECTION", 
                                       font=('Arial', 72, 'bold'), 
                                       bg='black', fg='white')
        self.material_label.pack(expand=True, pady=40)
        
        # Bin Status Display (HUGE)
        self.bin_status_label = tk.Label(main_frame, text="", 
                                         font=('Arial', 60, 'bold'), 
                                         bg='black', fg='#E74C3C')
        self.bin_status_label.pack(expand=True, pady=20)
        
        # Info container
        info_container = tk.Frame(main_frame, bg='black')
        info_container.pack(expand=True, fill=tk.BOTH, pady=40)
        
        # Battery display (LEFT)
        battery_frame = tk.Frame(info_container, bg='black')
        battery_frame.pack(side=tk.LEFT, expand=True)
        
        tk.Label(battery_frame, text="🔋", 
                font=('Arial', 60), bg='black', fg='white').pack()
        tk.Label(battery_frame, text="BATTERY", 
                font=('Arial', 24, 'bold'), bg='black', fg='#888').pack()
        self.battery_label = tk.Label(battery_frame, text="100%", 
                                      font=('Arial', 80, 'bold'), 
                                      bg='black', fg='#27AE60')
        self.battery_label.pack()
        
        # Weight display (RIGHT)
        weight_frame = tk.Frame(info_container, bg='black')
        weight_frame.pack(side=tk.RIGHT, expand=True)
        
        tk.Label(weight_frame, text="⚖️", 
                font=('Arial', 60), bg='black', fg='white').pack()
        tk.Label(weight_frame, text="WEIGHT", 
                font=('Arial', 24, 'bold'), bg='black', fg='#888').pack()
        self.weight_label = tk.Label(weight_frame, text="0.00 kg", 
                                     font=('Arial', 80, 'bold'), 
                                     bg='black', fg='#3498DB')
        self.weight_label.pack()
        
        # Bind ESC key to exit
        self.root.bind('<Escape>', lambda e: self.on_closing())
        
        # Start processing
        self.process_detection()
        
    def process_detection(self):
        # Read and process camera frame (no display)
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
                            if class_name == "paper":
                                paper_detected = True
                                if conf > max_conf:
                                    max_conf = conf
                            elif class_name == "plastic":
                                plastic_detected = True
                                if conf > max_conf:
                                    max_conf = conf
            
            # Update detection counts
            if paper_detected:
                state.paper_detected_count += 1
                state.plastic_detected_count = 0
                state.detected_material = "PAPER"
            elif plastic_detected:
                state.plastic_detected_count += 1
                state.paper_detected_count = 0
                state.detected_material = "PLASTIC"
            else:
                state.paper_detected_count = 0
                state.plastic_detected_count = 0
                state.detected_material = "NONE"
            
            state.last_detection_conf = max_conf
            
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
            
            # Servo control logic (ONLY for paper)
            if arduino:
                if state.paper_detected_count >= OPEN_FRAMES_REQUIRED and state.weight_kg < MAX_WEIGHT_KG:
                    if not state.servo_open:
                        arduino.write(b"open\n")
                        state.servo_open = True
                        print(f"[OPEN] PAPER - weight: {state.weight_kg:.2f}kg")
                elif state.paper_detected_count < OPEN_FRAMES_REQUIRED:
                    if state.servo_open and (time.time() - state.last_no_paper_time >= CLOSE_DELAY):
                        arduino.write(b"close\n")
                        state.servo_open = False
                        print("[CLOSE]")
                    if not state.servo_open:
                        state.last_no_paper_time = time.time()
            
            # Update UI
            self.update_display()
        
        if state.running:
            self.root.after(50, self.process_detection)
    
    def update_display(self):
        # Update material detection display
        if state.detected_material == "PAPER":
            self.material_label.config(text="📄 PAPER", fg='#27AE60')
        elif state.detected_material == "PLASTIC":
            self.material_label.config(text="♻️ PLASTIC", fg='#E74C3C')
        else:
            self.material_label.config(text="NO DETECTION", fg='#888')
        
        # Update bin full status
        if state.weight_kg >= MAX_WEIGHT_KG:
            self.bin_status_label.config(text="⚠️ BIN FULL ⚠️")
        else:
            self.bin_status_label.config(text="")
        
        # Update battery
        battery = state.battery_percent
        if battery > 50:
            battery_color = '#27AE60'
        elif battery > 20:
            battery_color = '#F39C12'
        else:
            battery_color = '#E74C3C'
        
        self.battery_label.config(text=f"{battery}%", fg=battery_color)
        
        # Update weight
        if state.weight_kg >= MAX_WEIGHT_KG:
            weight_color = '#E74C3C'
        elif state.weight_kg >= MAX_WEIGHT_KG * 0.8:
            weight_color = '#F39C12'
        else:
            weight_color = '#3498DB'
        
        self.weight_label.config(text=f"{state.weight_kg:.2f} kg", fg=weight_color)
    
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
    print("[INFO] Starting info display UI (no camera view)")
    print(f"[INFO] Model classes: {model.names}")
    
    root = tk.Tk()
    app = InfoDisplayGUI(root)
    
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
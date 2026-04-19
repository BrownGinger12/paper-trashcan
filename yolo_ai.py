from ultralytics import YOLO
import cv2
import time
import serial
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading

# =========================
# CONFIGURATION
# =========================
CONF_THRESHOLD = 0.85
SERIAL_PORT = 'COM6'
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
# Camera setup
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
        self.paper_detected_count = 0
        self.last_detection_conf = 0.0
        self.last_no_paper_time = 0
        self.running = True
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()

state = SystemState()

# =========================
# GUI Class
# =========================
class PaperDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Paper Trashcan Detection System")
        self.root.geometry("1100x700")
        self.root.configure(bg='#2C3E50')
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#2C3E50')
        style.configure('TLabel', background='#2C3E50', foreground='white', font=('Arial', 10))
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Stat.TLabel', font=('Arial', 24, 'bold'), foreground='#3498DB')
        style.configure('TButton', font=('Arial', 11, 'bold'), padding=10)
        
        self.create_widgets()
        self.update_frame()
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Title
        title = ttk.Label(main_frame, text="🗑️ Smart Paper Trashcan AI System", 
                         style='Title.TLabel')
        title.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Left Panel - Video Feed
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=1, column=0, padx=(0, 10), sticky=(tk.N, tk.S, tk.E, tk.W))
        
        video_label = ttk.Label(left_frame, text="Camera Feed", style='Title.TLabel')
        video_label.pack(pady=(0, 10))
        
        self.video_canvas = tk.Canvas(left_frame, width=640, height=480, bg='black', 
                                      highlightthickness=2, highlightbackground='#3498DB')
        self.video_canvas.pack()
        
        # Status indicator below video
        status_frame = ttk.Frame(left_frame)
        status_frame.pack(pady=10, fill=tk.X)
        
        ttk.Label(status_frame, text="Servo Status:", font=('Arial', 12)).pack(side=tk.LEFT, padx=5)
        self.servo_status_label = tk.Label(status_frame, text="CLOSED", 
                                           font=('Arial', 12, 'bold'),
                                           bg='#E74C3C', fg='white', 
                                           padx=20, pady=5, relief=tk.RAISED)
        self.servo_status_label.pack(side=tk.LEFT, padx=5)
        
        # FPS Counter
        self.fps_label = ttk.Label(status_frame, text="FPS: 0", font=('Arial', 10))
        self.fps_label.pack(side=tk.RIGHT, padx=5)
        
        # Right Panel - Stats and Controls
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Detection Stats
        stats_frame = ttk.LabelFrame(right_frame, text="Detection Statistics", padding="15")
        stats_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Paper Detection Confidence
        ttk.Label(stats_frame, text="Detection Confidence:").pack(anchor=tk.W)
        self.conf_label = ttk.Label(stats_frame, text="0%", style='Stat.TLabel')
        self.conf_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.conf_progress = ttk.Progressbar(stats_frame, length=300, mode='determinate')
        self.conf_progress.pack(fill=tk.X, pady=(0, 15))
        
        # Consecutive Frames
        ttk.Label(stats_frame, text="Consecutive Frames Detected:").pack(anchor=tk.W)
        self.frames_label = ttk.Label(stats_frame, text="0 / 5", style='Stat.TLabel')
        self.frames_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.frames_progress = ttk.Progressbar(stats_frame, length=300, mode='determinate')
        self.frames_progress.pack(fill=tk.X, pady=(0, 15))
        
        # Weight Display
        ttk.Label(stats_frame, text="Current Weight:").pack(anchor=tk.W)
        self.weight_label = ttk.Label(stats_frame, text="0.00 kg", style='Stat.TLabel')
        self.weight_label.pack(anchor=tk.W)
        
        # System Info
        info_frame = ttk.LabelFrame(right_frame, text="System Configuration", padding="15")
        info_frame.pack(fill=tk.X, pady=(0, 15))
        
        info_data = [
            ("Confidence Threshold:", f"{int(CONF_THRESHOLD * 100)}%"),
            ("Max Weight:", f"{MAX_WEIGHT_KG} kg"),
            ("Required Frames:", str(OPEN_FRAMES_REQUIRED)),
            ("Close Delay:", f"{CLOSE_DELAY}s")
        ]
        
        for label, value in info_data:
            row_frame = ttk.Frame(info_frame)
            row_frame.pack(fill=tk.X, pady=2)
            ttk.Label(row_frame, text=label, font=('Arial', 9)).pack(side=tk.LEFT)
            ttk.Label(row_frame, text=value, font=('Arial', 9, 'bold'), 
                     foreground='#3498DB').pack(side=tk.RIGHT)
        
        # Manual Controls
        control_frame = ttk.LabelFrame(right_frame, text="Manual Controls", padding="15")
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.open_btn = tk.Button(control_frame, text="🔓 OPEN LID", 
                                  command=self.open_servo,
                                  bg='#27AE60', fg='white', font=('Arial', 12, 'bold'),
                                  relief=tk.RAISED, bd=3, cursor='hand2')
        self.open_btn.pack(fill=tk.X, pady=(0, 10))
        
        self.close_btn = tk.Button(control_frame, text="🔒 CLOSE LID", 
                                   command=self.close_servo,
                                   bg='#E74C3C', fg='white', font=('Arial', 12, 'bold'),
                                   relief=tk.RAISED, bd=3, cursor='hand2')
        self.close_btn.pack(fill=tk.X)
        
        # Log Area
        log_frame = ttk.LabelFrame(right_frame, text="Activity Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_frame, height=8, width=40, bg='#34495E', 
                               fg='#ECF0F1', font=('Courier', 9), relief=tk.SUNKEN)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        self.add_log("System initialized")
        if arduino:
            self.add_log(f"Arduino connected on {SERIAL_PORT}")
        else:
            self.add_log("WARNING: Arduino not connected")
    
    def add_log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
    
    def open_servo(self):
        if arduino:
            arduino.write(b"open\n")
            state.servo_open = True
            self.add_log("Manual OPEN command sent")
    
    def close_servo(self):
        if arduino:
            arduino.write(b"close\n")
            state.servo_open = False
            self.add_log("Manual CLOSE command sent")
    
    def update_frame(self):
        ret, frame = cap.read()
        if ret:
            # Run YOLO detection
            results = model(frame, imgsz=416, conf=CONF_THRESHOLD, device="cpu", verbose=False)
            
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
                if state.paper_detected_count == 1:
                    self.add_log(f"Paper detected! Confidence: {max_conf:.2f}")
            else:
                if state.paper_detected_count > 0:
                    self.add_log("Paper no longer detected")
                state.paper_detected_count = 0
                state.last_detection_conf = 0.0
            
            # Read weight from Arduino
            if arduino:
                try:
                    while arduino.in_waiting:
                        line = arduino.readline().decode('utf-8').strip().lower()
                        if "weigth:" in line:
                            value_str = line.replace("weigth:", "").replace("kg", "").strip()
                            try:
                                old_weight = state.weight_kg
                                state.weight_kg = float(value_str)
                                if abs(old_weight - state.weight_kg) > 0.1:
                                    self.add_log(f"Weight updated: {state.weight_kg:.2f} kg")
                            except:
                                state.weight_kg = 0
                except:
                    pass
            
            # Servo control logic
            if arduino:
                if state.paper_detected_count >= OPEN_FRAMES_REQUIRED and state.weight_kg < MAX_WEIGHT_KG:
                    if not state.servo_open:
                        arduino.write(b"open\n")
                        state.servo_open = True
                        self.add_log(f"AUTO OPEN (weight: {state.weight_kg:.2f}kg)")
                elif state.paper_detected_count < OPEN_FRAMES_REQUIRED:
                    if state.servo_open and (time.time() - state.last_no_paper_time >= CLOSE_DELAY):
                        arduino.write(b"close\n")
                        state.servo_open = False
                        self.add_log("AUTO CLOSE")
                    if not state.servo_open:
                        state.last_no_paper_time = time.time()
            
            # Update UI elements
            self.update_stats()
            
            # Calculate FPS
            state.frame_count += 1
            if time.time() - state.last_fps_time >= 1.0:
                state.fps = state.frame_count
                state.frame_count = 0
                state.last_fps_time = time.time()
            
            # Convert frame for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.video_canvas.imgtk = imgtk
        
        if state.running:
            self.root.after(50, self.update_frame)
    
    def update_stats(self):
        # Update confidence
        conf_percent = int(state.last_detection_conf * 100)
        self.conf_label.config(text=f"{conf_percent}%")
        self.conf_progress['value'] = conf_percent
        
        # Update frames
        self.frames_label.config(text=f"{state.paper_detected_count} / {OPEN_FRAMES_REQUIRED}")
        self.frames_progress['value'] = (state.paper_detected_count / OPEN_FRAMES_REQUIRED) * 100
        
        # Update weight
        self.weight_label.config(text=f"{state.weight_kg:.2f} kg")
        
        # Update servo status
        if state.servo_open:
            self.servo_status_label.config(text="OPEN", bg='#27AE60')
        else:
            self.servo_status_label.config(text="CLOSED", bg='#E74C3C')
        
        # Update FPS
        self.fps_label.config(text=f"FPS: {state.fps}")

# =========================
# Main
# =========================
def main():
    print("[INFO] YOLO GUI detection started")
    
    root = tk.Tk()
    app = PaperDetectionGUI(root)
    
    def on_closing():
        state.running = False
        cap.release()
        if arduino:
            arduino.close()
        print("[INFO] Camera released, Serial closed")
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
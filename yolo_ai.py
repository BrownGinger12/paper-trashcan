from ultralytics import YOLO
import cv2
import time
import serial
import tkinter as tk
from tkinter import font as tkfont

# =========================
# CONFIGURATION
# =========================
CONF_THRESHOLD = 0.70
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600
OPEN_FRAMES_REQUIRED = 1
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
# Camera setup (headless — detection only, no display)
# =========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)
cap.set(cv2.CAP_PROP_FPS, 10)


# =========================
# Global State
# =========================
class SystemState:
    def __init__(self):
        self.servo_open = False
        self.fill_level = 0.0        # % from ultrasonic
        self.battery_percent = 100
        self.paper_detected_count = 0
        self.last_detection_conf = 0.0
        self.last_detection_class = None
        self.last_no_paper_time = 0
        self.running = True

state = SystemState()


# =========================
# Theme Colors
# =========================
BG          = "#0D0F14"
PANEL       = "#13161E"
ACCENT_GRN  = "#00E5A0"
ACCENT_RED  = "#FF4560"
ACCENT_YLW  = "#FFB830"
ACCENT_BLU  = "#4D9EFF"
TEXT_PRI    = "#EAEEF5"
TEXT_SEC    = "#5A6070"
BORDER      = "#1E2330"


# =========================
# GUI Class
# =========================
class PaperTrashcanGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Paper Trashcan Monitor")
        self.root.configure(bg=BG)
        self.root.overrideredirect(True)
        self.root.attributes('-fullscreen', True)
        self.root.geometry('800x480+0+0')
        self.root.focus_set()
        self.root.config(cursor="none")

        self._build_ui()
        self.update_frame()

    # --------------------------------------------------
    def _build_ui(self):
        root = self.root

        # ── Header bar ─────────────────────────────────
        header = tk.Frame(root, bg=PANEL, height=52)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        tk.Label(header, text="● SMART PAPER BIN",
                 font=("Courier", 13, "bold"),
                 bg=PANEL, fg=ACCENT_GRN).pack(side=tk.LEFT, padx=20, pady=14)

        self.time_label = tk.Label(header, text="",
                                   font=("Courier", 11),
                                   bg=PANEL, fg=TEXT_SEC)
        self.time_label.pack(side=tk.RIGHT, padx=20)

        # ── Body ───────────────────────────────────────
        body = tk.Frame(root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=16)

        # Left column — Detection panel
        left = tk.Frame(body, bg=PANEL, bd=0, highlightthickness=1,
                        highlightbackground=BORDER)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        tk.Label(left, text="DETECTION STATUS",
                 font=("Courier", 9, "bold"),
                 bg=PANEL, fg=TEXT_SEC).pack(anchor="w", padx=20, pady=(18, 0))

        # Large detection icon
        self.icon_label = tk.Label(left, text="📄",
                                   font=("Arial", 64),
                                   bg=PANEL)
        self.icon_label.pack(pady=(10, 0))

        # Main status text
        self.status_label = tk.Label(left, text="SCANNING...",
                                     font=("Courier", 22, "bold"),
                                     bg=PANEL, fg=TEXT_SEC)
        self.status_label.pack(pady=(4, 0))

        # Confidence badge
        self.conf_frame = tk.Frame(left, bg=PANEL)
        self.conf_frame.pack(pady=(8, 0))

        tk.Label(self.conf_frame, text="ACCURACY",
                 font=("Courier", 8), bg=PANEL, fg=TEXT_SEC).pack()

        self.conf_label = tk.Label(self.conf_frame, text="— %",
                                   font=("Courier", 32, "bold"),
                                   bg=PANEL, fg=TEXT_SEC)
        self.conf_label.pack()

        # Servo status chip
        self.servo_chip = tk.Label(left, text="  LID: CLOSED  ",
                                   font=("Courier", 9, "bold"),
                                   bg=TEXT_SEC, fg=BG, padx=6, pady=3)
        self.servo_chip.pack(pady=(12, 18))

        # Right column — Metrics
        right = tk.Frame(body, bg=BG)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(0, 0))
        right.config(width=220)
        right.pack_propagate(False)

        # ── Capacity card ──────────────────────────────
        cap_card = tk.Frame(right, bg=PANEL, bd=0,
                            highlightthickness=1, highlightbackground=BORDER)
        cap_card.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        tk.Label(cap_card, text="BIN CAPACITY",
                 font=("Courier", 9, "bold"),
                 bg=PANEL, fg=TEXT_SEC).pack(anchor="w", padx=16, pady=(14, 4))

        self.cap_value = tk.Label(cap_card, text="0%",
                                  font=("Courier", 38, "bold"),
                                  bg=PANEL, fg=ACCENT_BLU)
        self.cap_value.pack(padx=16, anchor="w")

        # Progress bar background
        bar_bg = tk.Frame(cap_card, bg=BORDER, height=12)
        bar_bg.pack(fill=tk.X, padx=16, pady=(6, 4))
        bar_bg.pack_propagate(False)

        self.cap_bar = tk.Frame(bar_bg, bg=ACCENT_BLU, height=12)
        self.cap_bar.place(x=0, y=0, relheight=1.0, relwidth=0.0)

        self.cap_status = tk.Label(cap_card, text="EMPTY",
                                   font=("Courier", 9),
                                   bg=PANEL, fg=ACCENT_BLU)
        self.cap_status.pack(anchor="w", padx=16, pady=(0, 14))

        # ── Battery card ───────────────────────────────
        bat_card = tk.Frame(right, bg=PANEL, bd=0,
                            highlightthickness=1, highlightbackground=BORDER)
        bat_card.pack(fill=tk.BOTH, expand=True)

        tk.Label(bat_card, text="BATTERY",
                 font=("Courier", 9, "bold"),
                 bg=PANEL, fg=TEXT_SEC).pack(anchor="w", padx=16, pady=(14, 4))

        self.bat_value = tk.Label(bat_card, text="100%",
                                  font=("Courier", 38, "bold"),
                                  bg=PANEL, fg=ACCENT_GRN)
        self.bat_value.pack(padx=16, anchor="w")

        # Battery bar background
        bat_bg = tk.Frame(bat_card, bg=BORDER, height=12)
        bat_bg.pack(fill=tk.X, padx=16, pady=(6, 4))
        bat_bg.pack_propagate(False)

        self.bat_bar = tk.Frame(bat_bg, bg=ACCENT_GRN, height=12)
        self.bat_bar.place(x=0, y=0, relheight=1.0, relwidth=1.0)

        self.bat_status = tk.Label(bat_card, text="NOMINAL",
                                   font=("Courier", 9),
                                   bg=PANEL, fg=ACCENT_GRN)
        self.bat_status.pack(anchor="w", padx=16, pady=(0, 14))

        # ── Footer ─────────────────────────────────────
        footer = tk.Frame(root, bg=BORDER, height=1)
        footer.pack(fill=tk.X, side=tk.BOTTOM)

        self.root.bind('<Escape>', lambda e: self.on_closing())

    # --------------------------------------------------
    def update_frame(self):
        ret, frame = cap.read()
        if ret:
            results = model(frame, imgsz=416, conf=CONF_THRESHOLD,
                            device="cpu", verbose=False)

            paper_detected = False
            plastic_detected = False
            mixed_detected = False
            max_conf = 0.0
            detected_class = None

            for r in results:
                if r.boxes:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = model.names[cls_id].lower()
                        if conf >= CONF_THRESHOLD:
                            if class_name == "paper":
                                paper_detected = True
                            elif class_name == "plastic":
                                plastic_detected = True

                            if conf > max_conf:
                                max_conf = conf
                                detected_class = class_name

            if paper_detected and plastic_detected:
                mixed_detected = True

            if paper_detected:
                state.paper_detected_count += 1
            else:
                state.paper_detected_count = 0

            if detected_class:
                state.last_detection_class = "mixed" if mixed_detected else detected_class
                state.last_detection_conf = max_conf
            else:
                state.last_detection_class = None
                state.last_detection_conf = 0.0

            # ── Read serial from Arduino ────────────────
            if arduino:
                try:
                    while arduino.in_waiting:
                        raw = arduino.readline().decode('utf-8', errors='ignore').strip().lower()

                        # "fill level: 45.3 %" → extract 45.3
                        if raw.startswith("fill level:"):
                            val = raw.replace("fill level:", "").replace("%", "").strip()
                            try:
                                state.fill_level = float(val)
                            except:
                                pass

                        # "battery: 87 %" → extract 87
                        elif raw.startswith("battery:"):
                            val = raw.replace("battery:", "").replace("%", "").strip()
                            try:
                                state.battery_percent = int(float(val))
                            except:
                                pass
                except:
                    pass

            # ── Servo logic ─────────────────────────────
            if arduino:
                if (state.paper_detected_count >= OPEN_FRAMES_REQUIRED
                        and state.fill_level < 100.0
                        and not plastic_detected):
                    if not state.servo_open:
                        arduino.write(b"open\n")
                        state.servo_open = True
                        print(f"[OPEN] fill:{state.fill_level:.1f}%")
                else:
                    if state.servo_open and (time.time() - state.last_no_paper_time > CLOSE_DELAY):
                        arduino.write(b"close\n")
                        state.servo_open = False
                        print("[CLOSE]")
                    if not state.servo_open:
                        state.last_no_paper_time = time.time()

            self._refresh_ui()

        if state.running:
            self.root.after(50, self.update_frame)

    # --------------------------------------------------
    def _refresh_ui(self):
        # ── Clock ───────────────────────────────────────
        self.time_label.config(text=time.strftime("%H:%M:%S"))

        # ── Detection panel ─────────────────────────────
        detected = (state.last_detection_class == "paper"
                    and state.paper_detected_count >= OPEN_FRAMES_REQUIRED)

        if state.last_detection_class == "plastic":
            self.status_label.config(text="PLASTIC DETECTED", fg=ACCENT_RED)
            self.icon_label.config(text="🧴", fg=ACCENT_RED)
            conf_pct = f"{state.last_detection_conf * 100:.1f}%"
            self.conf_label.config(text=conf_pct, fg=ACCENT_RED)
        elif state.last_detection_class == "mixed":
            self.status_label.config(text="PAPER + PLASTIC", fg=ACCENT_RED)
            self.icon_label.config(text="⚠️", fg=ACCENT_RED)
            conf_pct = f"{state.last_detection_conf * 100:.1f}%"
            self.conf_label.config(text=conf_pct, fg=ACCENT_RED)
        elif detected:
            self.status_label.config(text="PAPER DETECTED", fg=ACCENT_GRN)
            self.icon_label.config(text="📄", fg=ACCENT_GRN)
            conf_pct = f"{state.last_detection_conf * 100:.1f}%"
            self.conf_label.config(text=conf_pct, fg=ACCENT_GRN)
        elif state.paper_detected_count > 0:
            self.status_label.config(text="DETECTING PAPER...", fg=ACCENT_YLW)
            self.icon_label.config(text="📄", fg=ACCENT_YLW)
            conf_pct = f"{state.last_detection_conf * 100:.1f}%"
            self.conf_label.config(text=conf_pct, fg=ACCENT_YLW)
        else:
            self.status_label.config(text="NO PAPER", fg=TEXT_SEC)
            self.icon_label.config(text="📄", fg=TEXT_SEC)
            self.conf_label.config(text="— %", fg=TEXT_SEC)

        # Servo chip
        if state.servo_open:
            self.servo_chip.config(text="  LID: OPEN  ", bg=ACCENT_GRN, fg=BG)
        else:
            self.servo_chip.config(text="  LID: CLOSED  ", bg=BORDER, fg=TEXT_SEC)

        # ── Capacity ────────────────────────────────────
        fill = state.fill_level
        fill_ratio = fill / 100.0

        if fill >= 100:
            cap_color = ACCENT_RED
            cap_text = "BIN IS FULL"
            cap_font = ("Courier", 16, "bold")
        elif fill >= 90:
            cap_color = ACCENT_RED
            cap_text  = "FULL — EMPTY BIN"
            cap_font = ("Courier", 9)
        elif fill >= 70:
            cap_color = ACCENT_YLW
            cap_text  = "GETTING FULL"
            cap_font = ("Courier", 9)
        else:
            cap_color = ACCENT_BLU
            cap_text  = "OK"
            cap_font = ("Courier", 9)

        self.cap_value.config(text=f"{fill:.0f}%", fg=cap_color)
        self.cap_bar.config(bg=cap_color)
        self.cap_bar.place(relwidth=fill_ratio)
        self.cap_status.config(text=cap_text, fg=cap_color, font=cap_font)

        # ── Battery ─────────────────────────────────────
        bat = state.battery_percent
        bat_ratio = bat / 100.0

        if bat > 50:
            bat_color = ACCENT_GRN
            bat_text  = "NOMINAL"
        elif bat > 20:
            bat_color = ACCENT_YLW
            bat_text  = "LOW — CHARGE SOON"
        else:
            bat_color = ACCENT_RED
            bat_text  = "CRITICAL"

        self.bat_value.config(text=f"{bat}%", fg=bat_color)
        self.bat_bar.config(bg=bat_color)
        self.bat_bar.place(relwidth=bat_ratio)
        self.bat_status.config(text=bat_text, fg=bat_color)

    # --------------------------------------------------
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
    print("[INFO] Starting Paper Trashcan GUI")
    root = tk.Tk()
    app = PaperTrashcanGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
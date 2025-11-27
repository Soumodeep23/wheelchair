
"""
camera_grid_robot.py
Full GUI + 3x3 grid camera feed + TTS + mapping (raw camera) + IK + serial commands.

Usage:
- Edit SERIAL_PORT to your Arduino port (e.g., 'COM3' on Windows or '/dev/ttyUSB0' on Linux).
- Calibrate PIXEL_TO_CM and motor deg/sec on Arduino sketch before running real moves.
- Place arm in initial pose (bottom-center) and set HOME_ANGLES if needed.
"""

import cv2
import tkinter as tk
from PIL import Image, ImageTk
import pyttsx3
import threading
import math
import time

# Serial is optional; if unavailable the script will still run and print commands.
try:
    import serial
except Exception:
    serial = None

# ---------------- CONFIG ----------------
PIXEL_TO_CM = 0.12   # cm per pixel (calibrate)
L1 = 13.0            # cm (shoulder -> elbow)
L2 = 15.0            # cm (elbow -> end effector)

SERIAL_PORT = '/dev/ttyUSB0'   # Replace 'COM3' on Windows
SERIAL_BAUD = 115200

# Limits & safety
MAX_STEP_DEG = 60.0  # max commanded delta per joint per click
# ----------------------------------------

# ---------------- Serial helper ----------------
def open_serial(port, baud):
    if serial is None:
        print("pyserial not available, running in dry-run mode.")
        return None
    try:
        s = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  # allow Arduino to reset
        print("Opened serial port", port)
        return s
    except Exception as e:
        print("Warning: cannot open serial port:", e)
        return None

# Open serial (global)
_ser = open_serial(SERIAL_PORT, SERIAL_BAUD)

def send_to_robot_delta(d_base_deg, d_shoulder_deg, d_elbow_deg, gripper=None):
    """Send ASCII delta command to robot. Example: B:+30.0 S:-10.0 E:+5.0 G:1\n"""
    msg = f"B:{d_base_deg:+.1f} S:{d_shoulder_deg:+.1f} E:{d_elbow_deg:+.1f}"
    if gripper is not None:
        msg += f" G:{1 if gripper else 0}"
    msg += "\n"
    if _ser:
        try:
            _ser.write(msg.encode('utf-8'))
            # optional: read ack
            # ack = _ser.readline().decode().strip()
            # print("ACK:", ack)
        except Exception as e:
            print("Serial write error:", e)
    else:
        print("[DRY RUN] Would send:", msg.strip())

# ---------------- Math / IK helpers ----------------
def planar_2link_ik(x, y, L1, L2):
    """
    Solve planar 2-link IK for links L1, L2.
    Returns (theta1, theta2) in radians.
    theta1: shoulder angle relative to forward axis
    theta2: elbow angle (internal joint angle)
    """
    r2 = x*x + y*y
    r = math.sqrt(r2)
    if r > (L1 + L2) + 1e-6:
        raise ValueError("Out of reach")
    if r < abs(L1 - L2) - 1e-6:
        raise ValueError("Too close")
    cos_q2 = (r2 - L1*L1 - L2*L2) / (2 * L1 * L2)
    cos_q2 = max(-1.0, min(1.0, cos_q2))
    q2 = math.acos(cos_q2)
    k1 = L1 + L2 * math.cos(q2)
    k2 = L2 * math.sin(q2)
    q1 = math.atan2(y, x) - math.atan2(k2, k1)
    return q1, q2

def rad_to_deg(r): return r * 180.0 / math.pi
def deg_to_rad(d): return d * math.pi / 180.0
def clamp(val, low, high): return max(low, min(high, val))

# ---------------- GUI class ----------------
def draw_grid(frame, grid_size=(3, 3), color=(0, 255, 0), thickness=2):
    h, w, _ = frame.shape
    dh, dw = h // grid_size[0], w // grid_size[1]
    for i in range(1, grid_size[0]):
        cv2.line(frame, (0, i * dh), (w, i * dh), color, thickness)
    for j in range(1, grid_size[1]):
        cv2.line(frame, (j * dw, 0), (j * dw, h), color, thickness)
    return frame

class CameraGridUI:
    def __init__(self, camera_id=0):
        # Camera capture (raw resolution)
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera id {camera_id}")

        # GUI
        self.root = tk.Tk()
        self.root.title("3x3 Grid Camera Control")
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg="black")
        self.root.bind("<Escape>", lambda e: self.close())

        self.label = tk.Label(self.root, bg="black")
        self.label.pack(fill="both", expand=True)
        # handle both mouse and touch events (touch acts like mouse)
        self.label.bind("<Button-1>", self.on_click)

        self.grid_rows = 3
        self.grid_cols = 3

        # TTS engine config
        self.tts_rate = 160
        self.tts_volume = 1.0

        # assumed current joint angles (deg) - must be set to real initial pose
        # By convention, set initial pose to bottom-center grid (2,1) before running.
        self.current_base_deg = 0.0
        self.current_shoulder_deg = 0.0
        self.current_elbow_deg = 0.0

        # camera->robot offsets (cm) -- set if camera is not colocated.
        # Example: if camera is 10 cm behind robot base along forward axis set yb_cm = 10.
        self.xb_cm = 0.0
        self.yb_cm = 0.0

        # Start
        self.update_frame()
        self.root.mainloop()

    # ---------- TTS ----------
    def speak(self, text):
        threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()

    def _speak_thread(self, text):
        engine = pyttsx3.init()
        engine.setProperty('rate', self.tts_rate)
        engine.setProperty('volume', self.tts_volume)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    # ---------- Camera display ----------
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Use raw frame but flip horizontally so it feels natural (mirror)
            frame = cv2.flip(frame, 1)

            # Resize to screen for display only (we will use raw camera resolution for mapping)
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
            frame_disp = cv2.resize(frame, (screen_w, screen_h))

            # Draw grid on display frame
            frame_disp = draw_grid(frame_disp, (self.grid_rows, self.grid_cols))

            # Convert for Tk
            img = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(img))
            self.label.imgtk = img
            self.label.configure(image=img)

        self.root.after(10, self.update_frame)

    # ---------- Click handler ----------
    def on_click(self, event):
        # screen coords where user clicked
        w_screen = self.label.winfo_width()
        h_screen = self.label.winfo_height()
        col_width = w_screen / self.grid_cols
        row_height = h_screen / self.grid_rows
        col = int(event.x // col_width)
        row = int(event.y // row_height)
        print(f"[CLICK] screen ({event.x},{event.y}) -> grid row={row}, col={col}")
        # call movement routine
        self.send_arm_command(row, col)

    # ---------- Movement / IK / Serial ----------
    def send_arm_command(self, row, col):
        # 1) raw camera resolution (not display)
        cam_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cell_w = cam_w / 3.0
        cell_h = cam_h / 3.0
        # center of clicked grid in raw pixel coords
        target_px = col * cell_w + (cell_w / 2.0)
        target_py = row * cell_h + (cell_h / 2.0)

        # 2) map pixel -> world cm (camera coordinate)
        x_cam_cm = (target_px - (cam_w / 2.0)) * PIXEL_TO_CM   # left/right (cm)
        y_cam_cm = (cam_h - target_py) * PIXEL_TO_CM           # forward (cm) from camera bottom

        # apply camera->robot offsets (if camera is not at robot base)
        target_x_robot = x_cam_cm - self.xb_cm
        target_y_robot = y_cam_cm - self.yb_cm

        print(f"[MAPPING] pixel ({target_px:.1f},{target_py:.1f}) -> world cm ({target_x_robot:.1f},{target_y_robot:.1f})")

        # 3) base yaw and planar distance
        # base yaw rotates to face lateral offset (x) vs forward (y)
        theta_base_rad = math.atan2(target_x_robot, max(1e-6, target_y_robot))
        base_deg = rad_to_deg(theta_base_rad)

        planar_dist = math.hypot(target_x_robot, target_y_robot)  # forward distance from base

        # 4) Planar IK (2-link) - we treat arm plane after base rotation; assume target height is in-plane.
        x_arm = planar_dist
        y_arm = 0.0   # If you want vertical reach add a height estimate here

        try:
            q1, q2 = planar_2link_ik(x_arm, y_arm, L1, L2)
        except ValueError as e:
            print("[IK] error:", e)
            # clamp to max reach
            x_arm = L1 + L2 - 0.1
            q1, q2 = planar_2link_ik(x_arm, y_arm, L1, L2)

        shoulder_deg = rad_to_deg(q1)
        elbow_deg = rad_to_deg(q2)

        # 5) compute deltas relative to assumed current angles
        delta_base = base_deg - self.current_base_deg
        delta_shoulder = shoulder_deg - self.current_shoulder_deg
        delta_elbow = elbow_deg - self.current_elbow_deg

        # 6) Safety clamp
        delta_base = clamp(delta_base, -MAX_STEP_DEG, MAX_STEP_DEG)
        delta_shoulder = clamp(delta_shoulder, -MAX_STEP_DEG, MAX_STEP_DEG)
        delta_elbow = clamp(delta_elbow, -MAX_STEP_DEG, MAX_STEP_DEG)

        # 7) Speak + send
        labels = ["top-left","top","top-right","left","center","right","bottom-left","bottom","bottom-right"]
        idx = max(0, min(8, row*3 + col))
        self.speak(f"Moving to {labels[idx]}")
        send_to_robot_delta(delta_base, delta_shoulder, delta_elbow, gripper=None)

        # 8) update assumed angles
        self.current_base_deg += delta_base
        self.current_shoulder_deg += delta_shoulder
        self.current_elbow_deg += delta_elbow

        print("Assumed angles updated: base={:.1f}, sh={:.1f}, el={:.1f}".format(
            self.current_base_deg, self.current_shoulder_deg, self.current_elbow_deg))

    def close(self):
        print("Closing...")
        try:
            self.cap.release()
        except:
            pass
        try:
            if _ser:
                _ser.close()
        except:
            pass
        self.root.destroy()

# ---------------- Main ----------------
if __name__ == "__main__":
    # print("Calibration image (for reference): file:///mnt/data/IMG_20251124_145848.jpg")
    CameraGridUI(camera_id=1)  # change camera_id if needed

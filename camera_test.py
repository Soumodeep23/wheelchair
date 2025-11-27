"""
camera_grid_robot_with_pickdrop.py
Full GUI + 3x3 grid + dwell pick/drop + 9-point calibration + smoothing + IK + serial delta commands.

Features added on top of previous script:
- 9-point calibration (affine) to map screen pixels -> robot world cm (better than PIXEL_TO_CM)
- Dwell-based selection (500 ms) for pick/drop (touch-friendly)
- GUI PICK and DROP buttons and automatic pick_and_drop(source,dest)
- Smoothing: split large angle deltas into segments per paper (<=30°, 30-60° => 2 segments, >60° => 3)
- Motion segments sent sequentially to Arduino via existing ASCII protocol
- Uses raw camera frame for mapping (not stretched display)
- Keeps TTS in background threads
- Includes helper calibration utilities (collect calibration points)
References: paper (sections 3.6.2 and 3.6.3). :contentReference[oaicite:1]{index=1}
"""

import cv2
import tkinter as tk
from PIL import Image, ImageTk
import pyttsx3
import threading
import math
import time
import numpy as np

try:
    import serial
except Exception:
    serial = None

# ---------------- CONFIG ----------------
PIXEL_TO_CM = 0.12   # fallback cm per pixel (if not calibrated)
L1 = 13.0            # cm (shoulder -> elbow)
L2 = 15.0            # cm (elbow -> end effector)
SERIAL_PORT = '/dev/ttyUSB0'   # adjust to your system (COMx on Windows)
SERIAL_BAUD = 115200
DWELL_MS = 500      # milliseconds to consider a dwell selection
SEGMENT_DELAY = 0.25  # seconds delay between motion segments (smoothing)

# Safety: maximum allowed single-step (degrees) for any joint before clamping
MAX_STEP_DEG = 45.0
# ----------------------------------------

# Serial helper
def open_serial(port, baud):
    if serial is None:
        print("pyserial not available; dry-run mode.")
        return None
    try:
        s = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        print("Opened serial", port)
        return s
    except Exception as e:
        print("Warning: cannot open serial port:", e)
        return None

_ser = open_serial(SERIAL_PORT, SERIAL_BAUD)

def send_to_robot_delta(d_base_deg, d_shoulder_deg, d_elbow_deg, gripper=None):
    msg = f"B:{d_base_deg:+.1f} S:{d_shoulder_deg:+.1f} E:{d_elbow_deg:+.1f}"
    if gripper is not None:
        msg += f" G:{1 if gripper else 0}"
    msg += "\n"
    if _ser:
        try:
            _ser.write(msg.encode('utf-8'))
        except Exception as e:
            print("Serial write error:", e)
    else:
        print("[DRY RUN] send:", msg.strip())

# ---------------- Math / IK helpers ----------------
def planar_2link_ik(x, y, L1, L2):
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
def clamp(v,a,b): return max(a,min(b,v))

# ---------------- Calibration utilities ----------------
# We'll allow user to collect up to 9 (pixel -> world_cm) pairs and compute an affine map:
# world = A * [px, py, 1]^T  where A is 2x3 affine. Solve with least squares.

def estimate_affine_from_pairs(pixel_pts, world_pts):
    # pixel_pts: Nx2, world_pts: Nx2
    # Solve for 2x3 matrix M minimizing || M*[px,py,1]^T - world ||
    N = len(pixel_pts)
    if N < 3:
        raise ValueError("Need at least 3 points for affine fit")
    P = np.hstack([pixel_pts, np.ones((N,1))])  # Nx3
    X = world_pts[:,0]; Y = world_pts[:,1]
    # Solve for Mx: P @ m_x = X  -> m_x = (P^T P)^-1 P^T X
    m_x,_,_,_ = np.linalg.lstsq(P, X, rcond=None)
    m_y,_,_,_ = np.linalg.lstsq(P, Y, rcond=None)
    M = np.vstack([m_x, m_y])  # 2x3
    return M

def apply_affine(M, pixel):
    # pixel = (px,py)
    v = np.array([pixel[0], pixel[1], 1.0])
    world = M.dot(v)
    return float(world[0]), float(world[1])

# ---------------- GUI / Main class ----------------
def draw_grid(frame, grid_size=(3,3), color=(0,255,0), thickness=2):
    h,w,_ = frame.shape
    dh, dw = h//grid_size[0], w//grid_size[1]
    for i in range(1, grid_size[0]):
        cv2.line(frame, (0,i*dh),(w,i*dh), color, thickness)
    for j in range(1, grid_size[1]):
        cv2.line(frame, (j*dw,0),(j*dw,h), color, thickness)
    return frame

class CameraGridUI:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera", camera_id)

        # mapping model (affine) initially None -> fallback to PIXEL_TO_CM linear mapping
        self.mapping_M = None
        self.calib_pairs = []  # list of (pixel, world_cm)

        # GUI
        self.root = tk.Tk()
        self.root.title("3x3 Grid Camera Control (Pick & Drop)")
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg="black")
        self.root.bind("<Escape>", lambda e: self.close())

        self.label = tk.Label(self.root, bg="black")
        self.label.pack(fill='both', expand=True)
        self.label.bind("<Button-1>", self.on_touch_press)
        self.label.bind("<ButtonRelease-1>", self.on_touch_release)

        # top buttons: pick/drop, calibrate, set home
        btn_frame = tk.Frame(self.root, bg='black')
        btn_frame.place(relx=0.02, rely=0.02, anchor='nw')

        tk.Button(btn_frame, text="PICK (manual)", command=self.close_gripper, width=10).pack(side='left', padx=6)
        tk.Button(btn_frame, text="DROP (manual)", command=self.open_gripper, width=10).pack(side='left', padx=6)
        tk.Button(btn_frame, text="Auto Pick→Drop", command=self.ask_pick_drop, width=12).pack(side='left', padx=6)
        tk.Button(btn_frame, text="Calibrate (9 pts)", command=self.start_calibration, width=14).pack(side='left', padx=6)
        tk.Button(btn_frame, text="Set HOME", command=self.set_home_pose, width=8).pack(side='left', padx=6)

        # status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(self.root, textvariable=self.status_var, bg='black', fg='white', font=('Arial',14))
        self.status_label.place(relx=0.02, rely=0.92, anchor='sw')

        # dwell handling
        self._press_time = None
        self._press_pos = None

        # TTS
        self.tts_rate = 160
        self.tts_volume = 1.0

        # assumed angles (must be set to real home)
        self.current_base_deg = 0.0
        self.current_shoulder_deg = 0.0
        self.current_elbow_deg = 0.0

        # camera->robot offsets (if camera not at base). User can set after calibration.
        self.xb_cm = 0.0
        self.yb_cm = 0.0

        # smoothing thresholds per paper
        self.smooth_thresh_1 = 30.0
        self.smooth_thresh_2 = 60.0

        # start
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

    # ---------- Display ----------
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # mirror for natural touch
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
            frame_disp = cv2.resize(frame, (screen_w, screen_h))
            frame_disp = draw_grid(frame_disp, (3,3))
            img = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(img))
            self.label.imgtk = img
            self.label.configure(image=img)
        self.root.after(20, self.update_frame)

    # ---------- Touch / Dwell ----------
    def on_touch_press(self, event):
        # start dwell timer
        self._press_time = time.time()
        self._press_pos = (event.x, event.y)
        # schedule dwell check
        self.root.after(DWELL_MS, self._check_dwell, event.x, event.y)

    def on_touch_release(self, event):
        # if released before dwell timeout: treat as simple click -> move
        if self._press_time is None:
            return
        dt = (time.time() - self._press_time)*1000.0
        self._press_time = None
        if dt < DWELL_MS:
            # simple tap
            self._simple_tap(event.x, event.y)
        else:
            # releasing after dwell: ignore (dwell already handled)
            pass

    def _check_dwell(self, x, y):
        if self._press_time is None:
            return
        # if finger hasn't moved much and still holding, treat as dwell selection
        dx = abs(self._press_pos[0] - x)
        dy = abs(self._press_pos[1] - y)
        if dx < 10 and dy < 10:
            # interpret as dwell select
            self._dwell_select(x,y)

    def _simple_tap(self, sx, sy):
        # map screen coords to grid cell and call send_arm_command
        w_screen = self.label.winfo_width()
        h_screen = self.label.winfo_height()
        col_width = w_screen / 3.0
        row_height = h_screen / 3.0
        col = int(sx // col_width)
        row = int(sy // row_height)
        self.speak("Moving")
        self.send_arm_command(row, col)

    def _dwell_select(self, sx, sy):
        # dwell selection: treat as pick or drop depending on mode
        # for now: single dwell triggers "pick" (close gripper) if object present -> user can then tap dest
        w_screen = self.label.winfo_width()
        h_screen = self.label.winfo_height()
        col = int(sx // (w_screen/3.0))
        row = int(sy // (h_screen/3.0))
        self.speak(f"Selected {row+1},{col+1}")
        # toggle pick/drop: if gripper open -> pick, else -> drop
        # For simplicity, assume pick flow: move -> lower -> close -> raise
        self.pick_sequence(row, col)

    # ---------- Gripper helpers ----------
    def close_gripper(self):
        # send gripper close (G:1)
        send_to_robot_delta(0,0,0, gripper=True)
        self.speak("Gripper closed")

    def open_gripper(self):
        send_to_robot_delta(0,0,0, gripper=False)
        self.speak("Gripper opened")

    # ---------- Calibration workflow ----------
    def start_calibration(self):
        # start interactive 9-point calibration: user will click on display for each calibration pixel
        self.speak("Calibration mode: follow instructions in terminal")
        self.status_var.set("Calibration: click 9 screen points; for each provide world X,Y (cm) in terminal")
        self.calib_pairs = []
        self._calib_count = 0
        self._collect_calib_point()

    def _collect_calib_point(self):
        if self._calib_count >= 9:
            # compute affine
            pixel_pts = np.array([p for p,w in self.calib_pairs], dtype=np.float64)
            world_pts = np.array([w for p,w in self.calib_pairs], dtype=np.float64)
            M = estimate_affine_from_pairs(pixel_pts, world_pts)
            self.mapping_M = M
            self.speak("Calibration done")
            self.status_var.set("Calibration complete")
            print("Estimated affine M:\n", M)
            return

        n = self._calib_count + 1
        self.speak(f"Calibration point {n}: tap screen where robot is positioned")
        self.status_var.set(f"Calibration: tap point {n} then enter world coords in terminal.")
        # wait for next tap; we'll capture next simple tap handler to store pixel and then ask for coords
        # We'll reuse a temporary handler
        def calib_tap(event):
            # capture pixel on display
            sx, sy = event.x, event.y
            # transform to raw camera pixel coordinates
            cam_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cam_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            screen_w = self.label.winfo_width()
            screen_h = self.label.winfo_height()
            # map back to raw pixel coords (inverse of display resize)
            px = (sx / screen_w) * cam_w
            py = (sy / screen_h) * cam_h
            print(f"Captured pixel (raw cam): ({px:.1f},{py:.1f})")
            # ask user to type world coords (cm) in terminal
            try:
                wx = float(input("Enter world X (cm) for this position (relative to robot base): "))
                wy = float(input("Enter world Y (cm) for this position (forward from base): "))
            except Exception as e:
                print("Invalid input; skipping point.")
                self.status_var.set("Calibration cancelled or invalid input")
                self.label.unbind("<Button-1>", (calib_id))
                return
            self.calib_pairs.append(((px,py),(wx,wy)))
            self._calib_count += 1
            self.label.unbind("<Button-1>")
            # continue to next point
            self._collect_calib_point()

        calib_id = self.label.bind("<Button-1>", calib_tap)
        # binding will be removed after tap

    # ---------- Home / pick & drop flows ----------
    def set_home_pose(self):
        # user ensures arm is physically at home; then update assumed angles
        try:
            b = float(input("Enter current base angle (deg) as home: "))
            s = float(input("Enter current shoulder angle (deg) as home: "))
            e = float(input("Enter current elbow angle (deg) as home: "))
        except Exception:
            print("Invalid input. Home not set.")
            return
        self.current_base_deg = b
        self.current_shoulder_deg = s
        self.current_elbow_deg = e
        self.speak("Home pose set")
        self.status_var.set(f"Home set: base {b:.1f} sh {s:.1f} el {e:.1f}")

    def ask_pick_drop(self):
        # ask user for source and dest grid (via terminal) or use simple popup
        try:
            src = int(input("Source grid index (0..8) row-major e.g. 7 for row3 col1: "))
            dst = int(input("Destination grid index (0..8): "))
            sr = src // 3; sc = src % 3
            dr = dst // 3; dc = dst % 3
        except Exception:
            print("Invalid indices.")
            return
        # run automatic pick and drop in background so UI not blocked
        threading.Thread(target=self.pick_and_drop, args=(sr,sc,dr,dc), daemon=True).start()

    # automated pick/drop sequence (open-loop, robustify with delays)
    def pick_and_drop(self, source_row, source_col, dest_row, dest_col):
        # Move to source
        self.speak("Moving to source")
        self.send_arm_command(source_row, source_col)
        time.sleep(0.5)

        # Lower: emulate vertical lowering by relative joint motion
        self.speak("Lowering for pick")
        # tuning values; you must calibrate these empirically for your arm
        lower_shoulder = +8.0
        lower_elbow = -12.0
        self._send_smoothed_delta(0.0, lower_shoulder, lower_elbow)

        time.sleep(0.4)

        # close gripper
        send_to_robot_delta(0,0,0, gripper=True)
        time.sleep(0.4)

        # raise
        self._send_smoothed_delta(0.0, -lower_shoulder, -lower_elbow)
        time.sleep(0.4)

        # Move to destination
        self.speak("Moving to destination")
        self.send_arm_command(dest_row, dest_col)
        time.sleep(0.5)

        # lower and drop
        self.speak("Lowering to drop")
        self._send_smoothed_delta(0.0, lower_shoulder, lower_elbow)
        time.sleep(0.3)
        send_to_robot_delta(0,0,0, gripper=False)
        time.sleep(0.4)
        self._send_smoothed_delta(0.0, -lower_shoulder, -lower_elbow)
        self.speak("Pick and drop complete")

    # ---------- Motion smoothing & segmented send ----------
    def _split_and_send(self, delta_deg, send_fn):
        """Given delta_deg (scalar), split into segments per smoothing rules.
           send_fn(segment_value) sends a single segment delta (synchronous).
        """
        absd = abs(delta_deg)
        sign = 1.0 if delta_deg >= 0 else -1.0
        if absd <= self.smooth_thresh_1:
            # single
            send_fn(delta_deg)
        elif absd <= self.smooth_thresh_2:
            # split into 2
            seg = absd / 2.0
            send_fn(sign * seg)
            time.sleep(SEGMENT_DELAY)
            send_fn(sign * seg)
        else:
            # split into 3
            seg = absd / 3.0
            send_fn(sign * seg)
            time.sleep(SEGMENT_DELAY)
            send_fn(sign * seg)
            time.sleep(SEGMENT_DELAY)
            send_fn(sign * seg)

    def _send_smoothed_delta(self, d_base, d_shoulder, d_elbow, gripper=None):
        """Sends deltas to robot but segments large moves using smoothing rules.
           We send three independent segmented streams in sequence to reduce concurrency complexity:
           1) base segments
           2) shoulder segments
           3) elbow segments
           This serializes the motion so motors don't fight; small overhead but safer for DC open-loop.
        """
        # base
        def send_b(seg):
            send_to_robot_delta(seg, 0.0, 0.0, gripper=None)
            # update assumed
            self.current_base_deg += seg
        def send_s(seg):
            send_to_robot_delta(0.0, seg, 0.0, gripper=None)
            self.current_shoulder_deg += seg
        def send_e(seg):
            send_to_robot_delta(0.0, 0.0, seg, gripper=None)
            self.current_elbow_deg += seg

        # clamp segments to MAX_STEP (safety)
        d_base = clamp(d_base, -MAX_STEP_DEG, MAX_STEP_DEG)
        d_shoulder = clamp(d_shoulder, -MAX_STEP_DEG, MAX_STEP_DEG)
        d_elbow = clamp(d_elbow, -MAX_STEP_DEG, MAX_STEP_DEG)

        # send in base->shoulder->elbow order
        self._split_and_send(d_base, send_b)
        self._split_and_send(d_shoulder, send_s)
        self._split_and_send(d_elbow, send_e)

        # gripper if requested (single command)
        if gripper is not None:
            send_to_robot_delta(0,0,0, gripper=gripper)

    # ---------- Movement / IK that uses mapping and smoothing ----------
    def send_arm_command(self, row, col):
        # get raw camera dims
        cam_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cell_w = cam_w / 3.0
        cell_h = cam_h / 3.0
        target_px = col * cell_w + (cell_w / 2.0)
        target_py = row * cell_h + (cell_h / 2.0)

        # pixel -> world cm using affine mapping if available
        if self.mapping_M is not None:
            tx, ty = apply_affine(self.mapping_M, (target_px, target_py))
        else:
            # fallback linear scale (camera center => x=0)
            tx = (target_px - (cam_w / 2.0)) * PIXEL_TO_CM
            ty = (cam_h - target_py) * PIXEL_TO_CM

        # apply camera->robot offsets
        target_x_robot = tx - self.xb_cm
        target_y_robot = ty - self.yb_cm

        print(f"[MAPPING] pixel ({target_px:.1f},{target_py:.1f}) -> world cm ({target_x_robot:.1f},{target_y_robot:.1f})")

        # base yaw (radians) and planar distance
        theta_base_rad = math.atan2(target_x_robot, max(1e-6, target_y_robot))
        base_deg = rad_to_deg(theta_base_rad)
        planar_dist = math.hypot(target_x_robot, target_y_robot)

        # planar IK (treat planar_dist as forward x, assume same plane)
        x_arm = planar_dist
        y_arm = 0.0
        try:
            q1, q2 = planar_2link_ik(x_arm, y_arm, L1, L2)
        except ValueError as e:
            print("[IK] error:", e)
            x_arm = L1 + L2 - 0.1
            q1, q2 = planar_2link_ik(x_arm, y_arm, L1, L2)

        shoulder_deg = rad_to_deg(q1)
        elbow_deg = rad_to_deg(q2)

        # compute deltas
        delta_base = base_deg - self.current_base_deg
        delta_shoulder = shoulder_deg - self.current_shoulder_deg
        delta_elbow = elbow_deg - self.current_elbow_deg

        # safety clamp
        delta_base = clamp(delta_base, -MAX_STEP_DEG, MAX_STEP_DEG)
        delta_shoulder = clamp(delta_shoulder, -MAX_STEP_DEG, MAX_STEP_DEG)
        delta_elbow = clamp(delta_elbow, -MAX_STEP_DEG, MAX_STEP_DEG)

        # speak and send using smoothed segmented sends
        labels = ["top-left","top","top-right","left","center","right","bottom-left","bottom","bottom-right"]
        idx = max(0, min(8, row*3 + col))
        self.speak(f"Moving to {labels[idx]}")
        # send segmented deltas (threaded to avoid blocking UI)
        threading.Thread(target=self._send_smoothed_delta, args=(delta_base, delta_shoulder, delta_elbow, None), daemon=True).start()

    def close(self):
        print("Closing UI...")
        try: self.cap.release()
        except: pass
        try:
            if _ser: _ser.close()
        except: pass
        self.root.destroy()

# ----------------- MAIN -----------------
if __name__ == "__main__":
    CameraGridUI(camera_id=1)  # change to your camera ID

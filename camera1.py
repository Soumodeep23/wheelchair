
"""
camera_grid_visual_servo.py

Fullscreen 3x3 camera UI + dwell/tap + visual servo movement (DC motor friendly).
By default it's a DRY_RUN that prints motor letters (A/B/C/D etc).
Supports simple color-marker detection for the arm tip; falls back to last-known cell (2,1).
"""

import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
import numpy as np

# ---------------- USER CONFIG ----------------
CAMERA_ID = 1               # change if needed
DWELL_MS = 500               # dwell time to trigger pick (ms)
DRY_RUN = True               # if True -> only print motor chars; if False -> actually send via serial
SERIAL_PORT = "com11" # e.g., 'COM3' on Windows
SERIAL_BAUD = 9600

# Motor-letter mapping (already determined by you via trials)
CMD_UP    = "A 500"   # letter to move "up" (toward smaller row index)
CMD_DOWN  = "B 500"   # letter to move "down"
CMD_LEFT  = "C 500"   # letter to move "left"
CMD_RIGHT = "D 500"   # letter to move "right"
CMD_Shoulder_UP   = "E 500"
CMD_Shoulder_DOWN = "F 500"
CMD_GRIP_CLOSE = "G 500"  # example, adjust if needed
CMD_GRIP_OPEN  = "H 500"

# Movement step delay (time between command cycles). For DRY_RUN this is just a pace value.
STEP_DELAY = 0.35  # seconds

# Tip detection HSV color range (tune for your marker color). Default is bright green.
# You can change these ranges to match your marker color.
TIP_HSV_MIN = (40, 80, 80)
TIP_HSV_MAX = (90, 255, 255)

# ----------------------------------------------

try:
    import serial
except Exception:
    serial = None

# Serial helper (only used if DRY_RUN=False)
_ser = None
def open_serial(port, baud):
    global _ser
    if serial is None:
        print("pyserial not installed; running in DRY_RUN mode.")
        return
    try:
        _ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        print("Serial opened:", port)
    except Exception as e:
        print("Failed to open serial port:", e)
        _ser = None

def send_char(c):
    """Send a single character to Arduino or print (DRY_RUN)."""
    if DRY_RUN:
        print("[DRY RUN] Would send:", c)
        return
    if _ser:
        try:
            _ser.write(c.encode('utf-8'))
            print("[SERIAL] Sent:", c)
        except Exception as e:
            print("Serial write error:", e)
    else:
        print("[SERIAL] _ser not available, would send:", c)


# ----------------- Utility grid helpers -----------------
def px_to_grid(px_x, px_y, cam_w, cam_h):
    """Convert raw camera pixel to grid cell (row, col)."""
    col = int(px_x * 3 // cam_w)
    row = int(px_y * 3 // cam_h)
    # clamp
    col = max(0, min(2, col))
    row = max(0, min(2, row))
    return row, col

# ---------------- Main GUI / Controller -----------------
class CameraGridUI:
    def __init__(self, camera_id=0):
        # camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera id", camera_id)

        # last-known tip cell (start at bottom-center as you requested)
        self.last_known_cell = (2, 1)

        # GUI
        self.root = tk.Tk()
        self.root.title("3x3 Visual Servo (DRY_RUN)" if DRY_RUN else "3x3 Visual Servo")
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg="black")
        self.root.bind("<Escape>", lambda e: self.close())

        # display label
        self.label = tk.Label(self.root, bg="black")
        self.label.pack(fill="both", expand=True)
        self.label.bind("<Button-1>", self.on_press)
        self.label.bind("<ButtonRelease-1>", self.on_release)

        # top controls
        ctrl = tk.Frame(self.root, bg="black")
        ctrl.place(relx=0.02, rely=0.02, anchor="nw")
        tk.Button(ctrl, text="PICK (manual)", command=self.manual_pick, width=10).pack(side="left", padx=6)
        tk.Button(ctrl, text="DROP (manual)", command=self.manual_drop, width=10).pack(side="left", padx=6)
        tk.Button(ctrl, text="Auto Pick→Drop", command=self.ask_pick_drop, width=12).pack(side="left", padx=6)
        tk.Button(ctrl, text="Calibrate (no-op)", command=lambda: print("Calibration - use full script for 9-point"), width=12).pack(side="left", padx=6)
        tk.Button(ctrl, text="SET HOME", command=self.set_home_prompt, width=8).pack(side="left", padx=6)

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(self.root, textvariable=self.status_var, bg="black", fg="white")
        self.status_label.place(relx=0.02, rely=0.92, anchor="sw")

        # dwell handling
        self._press_time = None
        self._press_pos = None

        # start serial if requested
        if not DRY_RUN:
            open_serial(SERIAL_PORT, SERIAL_BAUD)

        # camera loop
        self._running = True
        self.update_frame()
        self.root.mainloop()

    # ---------- camera display ----------
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # mirror for natural touch
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
            frame_disp = cv2.resize(frame, (screen_w, screen_h))
            frame_disp = self.draw_grid(frame_disp)
            img = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(img))
            self.label.imgtk = img
            self.label.configure(image=img)
        if self._running:
            self.root.after(20, self.update_frame)

    def draw_grid(self, frame):
        h,w,_ = frame.shape
        dh = h//3; dw = w//3
        for i in range(1,3):
            cv2.line(frame, (0,i*dh),(w,i*dh),(0,255,0),2)
            cv2.line(frame, (i*dw,0),(i*dw,h),(0,255,0),2)
        # optionally draw marker detection overlay
        tip = self._last_detected_px if hasattr(self, "_last_detected_px") else None
        if tip is not None:
            tx, ty = tip
            # convert raw px -> display coords
            cam_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cam_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            disp_x = int((tx / cam_w) * w)
            disp_y = int((ty / cam_h) * h)
            cv2.circle(frame, (disp_x, disp_y), 10, (0,0,255), -1)
        return frame

    # ---------- touch / dwell ----------
    def on_press(self, event):
        self._press_time = time.time()
        self._press_pos = (event.x, event.y)
        self.root.after(DWELL_MS, self._check_dwell, event.x, event.y)

    def on_release(self, event):
        if self._press_time is None:
            return
        dt = (time.time() - self._press_time) * 1000.0
        self._press_time = None
        if dt < DWELL_MS:
            self._simple_tap(event.x, event.y)
        # else dwell handled

    def _check_dwell(self, x, y):
        if self._press_time is None:
            return
        dx = abs(self._press_pos[0] - x)
        dy = abs(self._press_pos[1] - y)
        if dx < 10 and dy < 10:
            self._dwell_select(x, y)

    def _simple_tap(self, sx, sy):
        row, col = self.screen_to_grid(sx, sy)
        print("Tap -> target cell:", (row, col))
        self.status_var.set(f"Moving to {row},{col}")
        threading.Thread(target=self.move_to_cell, args=(row, col), daemon=True).start()

    def _dwell_select(self, sx, sy):
        row, col = self.screen_to_grid(sx, sy)
        print("Dwell select -> target cell:", (row, col))
        # treat dwell as pick sequence start
        threading.Thread(target=self.pick_sequence, args=(row, col), daemon=True).start()

    def screen_to_grid(self, sx, sy):
        w = self.label.winfo_width()
        h = self.label.winfo_height()
        col = int(sx // (w/3))
        row = int(sy // (h/3))
        col = max(0, min(2, col)); row = max(0, min(2, row))
        return row, col

    # ---------- simple tip detection (color marker) ----------
    def detect_tip_pixel(self):
        """
        Try to detect a colored marker on the arm tip using HSV filtering.
        Returns raw camera pixel (px_x, px_y) or None if not found.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)
        # use raw resolution
        cam_h, cam_w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array(TIP_HSV_MIN, dtype=np.uint8)
        upper = np.array(TIP_HSV_MAX, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        # morphological clean
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # find largest contour
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < 50:   # too small
            return None
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # store for overlay
        self._last_detected_px = (cx, cy)
        return (cx, cy)

    def detect_current_cell(self):
        """
        Return the grid cell (row,col) where the end-effector is believed to be.
        Strategy:
         - Try color-marker detection
         - Fallback to last_known_cell
        """
        tip = self.detect_tip_pixel()
        if tip is None:
            # fallback
            return self.last_known_cell
        cam_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        r,c = px_to_grid(tip[0], tip[1], cam_w, cam_h)
        # update last-known
        self.last_known_cell = (r, c)
        return (r, c)

    # ---------- movement primitive ----------
    def move_to_cell(self, target_row, target_col):
        """
        Visual-servo loop to move arm to target grid cell.
        Prints motor letters (or sends them if DRY_RUN=False).
        Runs on a background thread.
        """
        print(f"Starting move to {target_row},{target_col}")
        max_iters = 60  # safety cap
        it = 0
        while it < max_iters:
            it += 1
            cur_row, cur_col = self.detect_current_cell()
            print(f"Iteration {it}: current cell = {cur_row},{cur_col}  target = {target_row},{target_col}")

            if (cur_row, cur_col) == (target_row, target_col):
                print("Reached target cell.")
                self.status_var.set(f"Reached {target_row},{target_col}")
                return True

            # decide vertical movement first (rows), then horizontal (cols)
            if target_row < cur_row:
                print("Action -> UP  :", CMD_UP)
                send_char(CMD_UP)
            elif target_row > cur_row:
                print("Action -> DOWN:", CMD_DOWN)
                send_char(CMD_DOWN)

            if target_col < cur_col:
                print("Action -> LEFT:", CMD_LEFT)
                send_char(CMD_LEFT)
            elif target_col > cur_col:
                print("Action -> RIGHT:", CMD_RIGHT)
                send_char(CMD_RIGHT)

            # allow the arm to move a step; detection will observe change
            time.sleep(STEP_DELAY)

        print("Move aborted: max iterations reached.")
        self.status_var.set("Move aborted")
        return False

    # ---------- pick/drop flows ----------
    def manual_pick(self):
        print("Manual pick (gripper close) -> would send:", CMD_GRIP_CLOSE)
        send_char(CMD_GRIP_CLOSE)
        self.status_var.set("Manual pick")

    def manual_drop(self):
        print("Manual drop (gripper open) -> would send:", CMD_GRIP_OPEN)
        send_char(CMD_GRIP_OPEN)
        self.status_var.set("Manual drop")

    def ask_pick_drop(self):
        # terminal based quick input (non-blocking wrapper)
        def _ask():
            try:
                src = int(input("Source grid idx (0..8): "))
                dst = int(input("Dest grid idx (0..8): "))
            except Exception:
                print("Invalid indices")
                return
            sr, sc = src // 3, src % 3
            dr, dc = dst // 3, dst % 3
            threading.Thread(target=self.pick_and_drop, args=(sr,sc,dr,dc), daemon=True).start()
        threading.Thread(target=_ask, daemon=True).start()

    def pick_sequence(self, row, col):
        """Simple local pick routine: move to cell, lower (approx), close, raise."""
        print("pick_sequence start:", (row, col))
        ok = self.move_to_cell(row, col)
        if not ok:
            print("pick_sequence: failed to reach source")
            return
        # lower (approx) - we simulate as sending DOWN small pulses (adjust as needed)
        # For DC motor systems you would send dedicated letters; here we print them.
        print("Lowering toward object (simulated)")
        # example: send small down movements (these letters should map to actual lowering on your arm)
        send_char(CMD_DOWN)
        time.sleep(0.3)
        send_char(CMD_DOWN)
        time.sleep(0.2)

        # close gripper
        send_char(CMD_GRIP_CLOSE)
        time.sleep(0.4)

        # raise back
        send_char(CMD_UP)
        time.sleep(0.3)
        print("Picked (simulated).")

    def pick_and_drop(self, source_row, source_col, dest_row, dest_col):
        # automated sequence
        self.status_var.set("Auto pick/drop running")
        print(f"Auto pick/drop: {source_row,source_col} -> {dest_row,dest_col}")
        self.pick_sequence(source_row, source_col)
        time.sleep(0.5)
        # move to dest
        self.move_to_cell(dest_row, dest_col)
        time.sleep(0.2)
        # lower and drop
        send_char(CMD_DOWN); time.sleep(0.3)
        send_char(CMD_GRIP_OPEN); time.sleep(0.3)
        send_char(CMD_UP); time.sleep(0.2)
        self.status_var.set("Auto pick/drop done")
        print("Auto pick/drop done.")

    # ---------- home / utils ----------
    def set_home_prompt(self):
        def _ask():
            try:
                b = float(input("Enter base angle (deg) assumed at HOME (not used for visual servo): "))
            except Exception:
                print("Invalid")
                return
            # This is informational only because we rely on camera for pose
            print("Home set (informational).")
            self.status_var.set("Home set (informational)")
        threading.Thread(target=_ask, daemon=True).start()

    def close(self):
        print("Shutting down...")
        self._running = False
        try: self.cap.release()
        except: pass
        try:
            if _ser:
                _ser.close()
        except: pass
        self.root.destroy()


# ----------------- MAIN -----------------
if __name__ == "__main__":
    CameraGridUI(camera_id=CAMERA_ID)

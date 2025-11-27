#!/usr/bin/env python3
"""
camera_grid_visual_servo.py

Visual-servo 3×3 grid robot controller for DC-motor robotic arm.
Updated with YOUR MOTOR MAPPINGS:

A → Arm Forward (Up)
B → Arm Backward (Down)
C → Shoulder Forward
D → Shoulder Backward
E → Base Rotate Left
F → Base Rotate Right
G → Gripper Close
H → Gripper Open
"""

import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
import numpy as np

# ---------------- USER CONFIG ----------------
CAMERA_ID = 1
DWELL_MS = 500

DRY_RUN = False          # Set False → send to Arduino
SERIAL_PORT = "COM11"
SERIAL_BAUD = 9600

# ----------- YOUR DC MOTOR MAPPINGS -----------
CMD_UP    = "A"   
CMD_DOWN  = "B"   
CMD_SHOULDER_UP   = "C"
CMD_SHOULDER_DOWN = "D"
CMD_LEFT  = "E"
CMD_RIGHT = "F"
CMD_GRIP_CLOSE = "G"
CMD_GRIP_OPEN  = "H"

STEP_DELAY = 0.35

# Color detection (optional)
TIP_HSV_MIN = (40, 80, 80)
TIP_HSV_MAX = (90, 255, 255)

try:
    import serial
except:
    serial = None

_ser = None
def open_serial(port, baud):
    global _ser
    if DRY_RUN:
        print("DRY RUN mode → no serial connection.")
        return
    try:
        _ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        print("Serial connected on", port)
    except Exception as e:
        print("Serial connect ERROR:", e)
        _ser = None

def send_char(c):
    """Send a single command letter to Arduino."""
    if DRY_RUN:
        print("[DRY RUN] Would send:", c)
        return
    if _ser:
        try:
            _ser.write(c.encode('utf-8'))
            print("[SERIAL] Sent:", c)
        except Exception as e:
            print("Serial write ERROR:", e)
    else:
        print("[SERIAL] Serial not connected, cannot send:", c)


# ---------------- GRID HELPERS ----------------
def px_to_grid(px_x, px_y, cam_w, cam_h):
    col = int(px_x * 3 // cam_w)
    row = int(px_y * 3 // cam_h)
    col = max(0, min(2, col))
    row = max(0, min(2, row))
    return row, col


# ---------------- MAIN CLASS -----------------
class CameraGridUI:

    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera", camera_id)

        self.last_known_cell = (2,1)  # initial arm tip position

        self.root = tk.Tk()
        self.root.title("3×3 Visual Servo Robot")
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg="black")
        self.root.bind("<Escape>", lambda e: self.close())

        self.label = tk.Label(self.root, bg="black")
        self.label.pack(fill="both", expand=True)
        self.label.bind("<Button-1>", self.on_press)
        self.label.bind("<ButtonRelease-1>", self.on_release)

        ctrl = tk.Frame(self.root, bg="black")
        ctrl.place(relx=0.02, rely=0.02, anchor="nw")

        tk.Button(ctrl, text="PICK", command=self.manual_pick, width=8).pack(side="left", padx=6)
        tk.Button(ctrl, text="DROP", command=self.manual_drop, width=8).pack(side="left", padx=6)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(self.root, textvariable=self.status_var, bg="black", fg="white")
        self.status_label.place(relx=0.02, rely=0.92, anchor="sw")

        self._press_time = None
        self._press_pos = None

        open_serial(SERIAL_PORT, SERIAL_BAUD)

        self._running = True
        self.update_frame()
        self.root.mainloop()

    # CAMERA DISPLAY LOGIC
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            frame_disp = cv2.resize(frame, (sw, sh))
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
        return frame


    # TOUCH / DWELL DETECTION
    def on_press(self, event):
        self._press_time = time.time()
        self._press_pos = (event.x, event.y)
        self.root.after(DWELL_MS, self.check_dwell, event.x, event.y)

    def on_release(self, event):
        if self._press_time is None:
            return
        elapsed = (time.time() - self._press_time)*1000
        self._press_time = None
        if elapsed < DWELL_MS:
            self.simple_tap(event.x, event.y)

    def check_dwell(self, x, y):
        if self._press_time is None:
            return
        dx = abs(self._press_pos[0] - x)
        dy = abs(self._press_pos[1] - y)
        if dx < 10 and dy < 10:
            self.dwell_select(x,y)

    def simple_tap(self, sx, sy):
        row,col = self.screen_to_grid(sx, sy)
        print("Tapped cell:", (row,col))
        threading.Thread(target=self.move_to_cell, args=(row,col), daemon=True).start()

    def dwell_select(self, sx, sy):
        row,col = self.screen_to_grid(sx, sy)
        print("Dwell selected cell:", (row,col))
        threading.Thread(target=self.pick_sequence, args=(row,col), daemon=True).start()

    def screen_to_grid(self, sx, sy):
        w = self.label.winfo_width()
        h = self.label.winfo_height()
        col = int(sx // (w/3))
        row = int(sy // (h/3))
        return max(0,min(2,row)), max(0,min(2,col))


    # TIP DETECTION (optional)
    def detect_tip_pixel(self):
        ret, frame = self.cap.read()
        if not ret: return None
        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, TIP_HSV_MIN, TIP_HSV_MAX)
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 40:
            return None
        x,y,w,h = cv2.boundingRect(cnt)
        return (x+w//2, y+h//2)

    def detect_current_cell(self):
        tip = self.detect_tip_pixel()
        if tip is None:
            return self.last_known_cell
        cam_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        row,col = px_to_grid(tip[0], tip[1], cam_w, cam_h)
        self.last_known_cell = (row,col)
        return (row,col)


    # MOVEMENT CONTROL (UP/DOWN/LEFT/RIGHT)
    def move_to_cell(self, target_row, target_col):
        print(f"Move to ({target_row},{target_col})")
        iters = 0
        while iters < 40:
            iters += 1
            r,c = self.detect_current_cell()
            print("Current cell:",(r,c))

            if (r,c) == (target_row,target_col):
                print("Reached target cell!")
                self.status_var.set("Reached target")
                return True

            if target_row < r:
                send_char(CMD_UP)
            elif target_row > r:
                send_char(CMD_DOWN)

            if target_col < c:
                send_char(CMD_LEFT)
            elif target_col > c:
                send_char(CMD_RIGHT)

            time.sleep(STEP_DELAY)

        print("Movement timeout")
        return False


    # GRIPPER & PICK/DROP
    def manual_pick(self):
        send_char(CMD_GRIP_CLOSE)

    def manual_drop(self):
        send_char(CMD_GRIP_OPEN)

    def pick_sequence(self, row, col):
        print("Pick at:",(row,col))
        if not self.move_to_cell(row,col):
            return
        send_char(CMD_DOWN)
        time.sleep(0.3)
        send_char(CMD_GRIP_CLOSE)
        time.sleep(0.3)
        send_char(CMD_UP)

    def close(self):
        self._running = False
        try:
            self.cap.release()
        except: pass
        if _ser:
            _ser.close()
        self.root.destroy()


# ---------------- MAIN ----------------
if __name__ == "__main__":
    CameraGridUI(camera_id=CAMERA_ID)

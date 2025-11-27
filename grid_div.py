import cv2
import tkinter as tk
from PIL import Image, ImageTk

def draw_grid(frame, grid_size=(3, 3), color=(0, 255, 0), thickness=2):
    """Draws a 3x3 grid on the given frame."""
    h, w, _ = frame.shape
    dh, dw = h // grid_size[0], w // grid_size[1]

    for i in range(1, grid_size[0]):
        cv2.line(frame, (0, i * dh), (w, i * dh), color, thickness)
    for j in range(1, grid_size[1]):
        cv2.line(frame, (j * dw, 0), (j * dw, h), color, thickness)

    return frame

class CameraGridUI:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)

        # --- Setup Tkinter window ---
        self.root = tk.Tk()
        self.root.title("3x3 Grid Camera Control")
        self.root.attributes('-fullscreen', True)  # Full screen
        self.root.configure(bg="black")

        # Bind ESC key to exit
        self.root.bind("<Escape>", lambda e: self.close())

        self.label = tk.Label(self.root, bg="black")
        self.label.pack(fill="both", expand=True)
        self.label.bind("<Button-1>", self.on_click)

        self.grid_rows = 3
        self.grid_cols = 3

        self.update_frame()
        self.root.mainloop()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)

            # Get screen size
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()

            # Resize camera frame to full screen
            frame = cv2.resize(frame, (screen_width, screen_height))

            # Draw grid overlay
            frame = draw_grid(frame, (self.grid_rows, self.grid_cols))

            # Convert for Tkinter
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(img))
            self.label.imgtk = img
            self.label.configure(image=img)

        # Refresh frame every 10 ms
        self.root.after(10, self.update_frame)

    def on_click(self, event):
        w = self.label.winfo_width()
        h = self.label.winfo_height()
        col_width = w / self.grid_cols
        row_height = h / self.grid_rows

        col = int(event.x // col_width)
        row = int(event.y // row_height)

        print(f"Clicked grid: Row {row+1}, Col {col+1}")
        self.send_arm_command(row, col)

    def send_arm_command(self, row, col):
        grid_to_command = {
            (0,0): "move_top_left",
            (0,1): "move_up",
            (0,2): "move_top_right",
            (1,0): "move_left",
            (1,1): "center",
            (1,2): "move_right",
            (2,0): "move_bottom_left",
            (2,1): "move_down",
            (2,2): "move_bottom_right",
        }
        command = grid_to_command.get((row, col), "center")
        print("Sending command:", command)
        # TODO: integrate with robotic arm control API

    def close(self):
        """Release camera and close window."""
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    CameraGridUI()

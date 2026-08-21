import time
import threading
import tkinter as tk
from pynput import keyboard
import pyautogui
import pystray
from PIL import Image, ImageDraw
import ctypes

ES_CONTINUOUS = 0x80000000
ES_DISPLAY_REQUIRED = 0x00000002
ES_SYSTEM_REQUIRED = 0x00000001

# ===============================
# 全局状态
# ===============================
running = False
last_input_time = time.time()
INTERVAL = 45


# ===============================
# 检测键盘输入
# ===============================
def on_press(key):
    global last_input_time
    last_input_time = time.time()

keyboard.Listener(on_press=on_press).start()


# ===============================
# 防黑屏逻辑
# ===============================
def keep_awake():
    while True:
        if running:
            # 告诉系统：我在用，不要休眠/黑屏
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_DISPLAY_REQUIRED | ES_SYSTEM_REQUIRED
            )
        else:
            # 恢复系统默认行为
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)

        time.sleep(10)


# ===============================
# GUI
# ===============================
def start_action():
    global running
    running = True
    status_label.config(text="状态：运行中", fg="green")

def stop_action():
    global running
    running = False
    status_label.config(text="状态：已停止", fg="red")


def quit_app(icon=None, item=None):
    icon.stop()
    root.destroy()


def hide_window():
    root.withdraw()


def show_window(icon=None, item=None):
    root.deiconify()


# ===============================
# 托盘图标
# ===============================
def create_image():
    img = Image.new('RGB', (64, 64), color=(0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle((16, 16, 48, 48), fill=(0, 255, 0))
    return img


def setup_tray():
    icon = pystray.Icon("keep_awake")
    icon.icon = create_image()
    icon.title = "Keep Awake"

    icon.menu = pystray.Menu(
        pystray.MenuItem("显示窗口", show_window),
        pystray.MenuItem("开始", lambda: start_action()),
        pystray.MenuItem("停止", lambda: stop_action()),
        pystray.MenuItem("退出", quit_app)
    )

    icon.run()


# ===============================
# 主程序
# ===============================
threading.Thread(target=keep_awake, daemon=True).start()
threading.Thread(target=setup_tray, daemon=True).start()

root = tk.Tk()
root.title("防黑屏工具")
root.geometry("300x150")

status_label = tk.Label(root, text="状态：已停止", fg="red")
status_label.pack(pady=10)

start_btn = tk.Button(root, text="开始", command=start_action, width=10)
start_btn.pack(pady=5)

stop_btn = tk.Button(root, text="停止", command=stop_action, width=10)
stop_btn.pack(pady=5)

hide_btn = tk.Button(root, text="最小化到托盘", command=hide_window)
hide_btn.pack(pady=5)

root.mainloop()

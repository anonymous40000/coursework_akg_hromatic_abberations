
import os

os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

root = tk.Tk()
root.title("ТЕСТ РЕНДЕРА: красный = ImageTk, зелёный = PPM")

red = np.zeros((200, 200, 3), np.uint8)
red[:, :, 2] = 255  
imgtk = ImageTk.PhotoImage(Image.fromarray(red), master=root)
tk.Label(root, image=imgtk).pack()

green = np.zeros((200, 200, 3), np.uint8)
green[:, :, 1] = 255  
ok, buf = cv2.imencode(".ppm", green)  
ppm = tk.PhotoImage(data=buf.tobytes(), format="ppm", master=root)
tk.Label(root, image=ppm).pack()

tk.Label(root, text="Красный сверху — ImageTk. Зелёный снизу — PPM.").pack()
root.mainloop()

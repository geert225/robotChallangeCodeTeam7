import os
import time
import mmap
import struct
import fcntl
import numpy as np
import cv2
from picamera2 import Picamera2
from libcamera import Transform

# ============================================================
# CONFIG
# ============================================================

WIDTH = 320
HEIGHT = 240
CHANNELS = 3

FRAME_SIZE = WIDTH * HEIGHT * CHANNELS

# timestamp + detected + cx + cy + frame
TOTAL_SIZE = 8 + 1 + 4 + 4 + FRAME_SIZE

SHM_PATH = "/dev/shm/vision_frame"

# HSV startwaarden (jouw tuning)
h_min, h_max = 158, 168
s_min, s_max = 124, 255
v_min, v_max = 100, 205

alpha = 0.1  # learning speed

# ============================================================
# FILE + MMAP
# ============================================================

fd = os.open(SHM_PATH, os.O_CREAT | os.O_RDWR)
os.ftruncate(fd, TOTAL_SIZE)

mm = mmap.mmap(fd, TOTAL_SIZE, mmap.MAP_SHARED, mmap.PROT_WRITE)

# ============================================================
# CAMERA
# ============================================================

picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={"format": "RGB888", "size": (WIDTH, HEIGHT)},
    controls={"FrameRate": 40},
    transform=Transform(hflip=1, vflip=1)
)

picam2.configure(config)
picam2.start()

time.sleep(1)

print("Adaptive vision module gestart")

# ============================================================
# HELPER
# ============================================================

def get_hsv_stats(hsv, mask):
    pixels = hsv[mask > 0]
    if len(pixels) < 50:
        return None
    return np.mean(pixels, axis=0), np.std(pixels, axis=0)

# ============================================================
# LOOP
# ============================================================

detection_counter = 0

while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv, lower, upper)

    # morphology
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_area = WIDTH * HEIGHT
    detected = False
    cx, cy = -1, -1

    best_cnt = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if not (0.0005 < area/frame_area < 0.2):
            continue

        x,y,w,h = cv2.boundingRect(cnt)

        rect_area = w*h
        if rect_area == 0:
            continue

        extent = area / rect_area

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue

        solidity = area / hull_area

        if extent > 0.4 and solidity > 0.8:
            if area > best_area:
                best_area = area
                best_cnt = cnt

    # ============================================================
    # BEST DETECTION
    # ============================================================

    if best_cnt is not None:
        detection_counter += 1
    else:
        detection_counter = max(0, detection_counter - 1)

    confirmed = detection_counter > 3

    if confirmed and best_cnt is not None:
        detected = True

        x,y,w,h = cv2.boundingRect(best_cnt)
        cx = x + w//2
        cy = y + h//2

        # adaptive HSV
        temp_mask = np.zeros_like(mask)
        cv2.drawContours(temp_mask, [best_cnt], -1, 255, -1)

        stats = get_hsv_stats(hsv, temp_mask)

        if stats is not None:
            mean, std = stats

            new_lower = mean - 2 * std
            new_upper = mean + 2 * std

            h_min = int((1-alpha)*h_min + alpha*new_lower[0])
            h_max = int((1-alpha)*h_max + alpha*new_upper[0])

            s_min = int((1-alpha)*s_min + alpha*new_lower[1])
            s_max = int((1-alpha)*s_max + alpha*new_upper[1])

            v_min = int((1-alpha)*v_min + alpha*new_lower[2])
            v_max = int((1-alpha)*v_max + alpha*new_upper[2])

            # clamp
            h_min = max(0, min(179, h_min))
            h_max = max(0, min(179, h_max))
            s_min = max(50, min(255, s_min))
            v_min = max(50, min(255, v_min))

    # ============================================================
    # WRITE TO SHARED MEMORY
    # ============================================================

    fcntl.flock(fd, fcntl.LOCK_EX)

    try:
        mm.seek(0)

        mm.write(struct.pack('d', time.time()))
        mm.write(struct.pack('b', int(detected)))
        mm.write(struct.pack('i', cx))
        mm.write(struct.pack('i', cy))

        mm.write(frame.tobytes())

    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
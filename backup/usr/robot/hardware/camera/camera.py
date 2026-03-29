import os
import time
import mmap
import struct
import fcntl
import numpy as np
from picamera2 import Picamera2
from libcamera import Transform

# ============================================================
# CONFIG
# ============================================================

WIDTH = 320
HEIGHT = 240
CHANNELS = 3
FRAME_SIZE = WIDTH * HEIGHT * CHANNELS
TOTAL_SIZE = 8 + FRAME_SIZE  # timestamp + frame
SHM_PATH = "/dev/shm/vision_frame"


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
    transform=Transform(hflip=1, vflip=1)  # 180° rotatie
)

picam2.configure(config)
picam2.start()

time.sleep(1)

print("mmap writer gestart")

# ============================================================
# LOOP
# ============================================================

while True:
    frame = picam2.capture_array()

    fcntl.flock(fd, fcntl.LOCK_EX)

    try:
        mm.seek(0)

        # timestamp
        mm.write(struct.pack('d', time.time()))

        # frame
        mm.write(frame.tobytes())

    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
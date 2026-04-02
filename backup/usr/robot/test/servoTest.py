import time
import mmap
import struct
import os
import fcntl

SERVO_PATH = "/dev/shm/servo"
SERVO_FORMAT = "<2B"

def open_shm(path, fmt):
    size = struct.calcsize(fmt)

    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(b"\x00" * size)

    f = open(path, "r+b")
    shm = mmap.mmap(f.fileno(), size)
    return shm, f

def shm_write(shm, fd, fmt, values):
    fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
    shm.seek(0)
    shm.write(struct.pack(fmt, *values))
    fcntl.flock(fd.fileno(), fcntl.LOCK_UN)

# open shared memory
shm, fd = open_shm(SERVO_PATH, SERVO_FORMAT)

print("Start servo sweep...")

try:
    while True:
        # van 0 → 180
        for angle in range(0, 181, 2):
            shm_write(shm, fd, SERVO_FORMAT, (angle, angle))
            time.sleep(0.1)

        # van 180 → 0
        for angle in range(180, -1, -2):
            shm_write(shm, fd, SERVO_FORMAT, (angle, angle))
            time.sleep(0.1)

except KeyboardInterrupt:
    print("Stopped")
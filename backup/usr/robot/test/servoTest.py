import time
import mmap
import struct
import os
import fcntl

SERVO_PATH = "/dev/shm/servo"
SERVO_FORMAT = "<B"

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
    shm.write(struct.pack(fmt, values))
    fcntl.flock(fd.fileno(), fcntl.LOCK_UN)

# open shared memory
shm, fd = open_shm(SERVO_PATH, SERVO_FORMAT)

print("Start servo sweep...")

try:
    while True:
        shm_write(shm, fd, SERVO_FORMAT, (1))
        print("dicht")
        time.sleep(5)
        shm_write(shm, fd, SERVO_FORMAT, (0))
        print("open")
        time.sleep(5)

except KeyboardInterrupt:
    print("Stopped")
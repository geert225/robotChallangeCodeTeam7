import time
import mmap
import struct
import fcntl
import os

# ================= CONFIG =================
ULTRA_PATH = "/dev/shm/ultrasoon"
ULTRA_FORMAT = "<ddHH"  # (update_rate, timestamp, sensor1, sensor2)
DEFAULT_UPDATE_RATE = 10  # Hz

# ================= SHM HELPERS =================
def create_or_open_shm(path, fmt):
    size = struct.calcsize(fmt)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(b"\x00" * size)
    f = open(path, "r+b")
    shm = mmap.mmap(f.fileno(), size)
    return shm, f

def shm_read(shm, fd, fmt):
    fcntl.flock(fd.fileno(), fcntl.LOCK_SH)
    shm.seek(0)
    data = struct.unpack(fmt, shm.read())
    fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
    return data

def shm_write(shm, fd, fmt, values):
    fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
    shm.seek(0)
    shm.write(struct.pack(fmt, *values))
    fcntl.flock(fd.fileno(), fcntl.LOCK_UN)

# ================= INIT =================
shm_ultra, fd_ultra = create_or_open_shm(ULTRA_PATH, ULTRA_FORMAT)

# Lees huidige update_rate
update_rate, timestamp, d1, d2 = shm_read(shm_ultra, fd_ultra, ULTRA_FORMAT)

# Als update_rate nog nul is, schrijf een default
if update_rate == 0:
    shm_write(shm_ultra, fd_ultra, ULTRA_FORMAT, (DEFAULT_UPDATE_RATE, timestamp, d1, d2))
    update_rate = DEFAULT_UPDATE_RATE
    print(f"Update rate ingesteld op {DEFAULT_UPDATE_RATE} Hz")

print("Lezen van ultrasoon SHM. Ctrl+C om te stoppen.")

try:
    while True:
        update_rate, timestamp, d1, d2 = shm_read(shm_ultra, fd_ultra, ULTRA_FORMAT)
        print(f"Update rate: {update_rate:.1f}, Tijd: {timestamp:.3f}, Sensor1: {d1}, Sensor2: {d2}")
        time.sleep(0.5)
except KeyboardInterrupt:
    print("Test gestopt.")
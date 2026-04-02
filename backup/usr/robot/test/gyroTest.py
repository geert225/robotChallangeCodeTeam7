import time
import mmap
import struct
import os
import fcntl

# ================= FORMATS =================
GYRO_FORMAT  = "<dddd"   # (timestamp, gx, gy, gz)
ULTRA_FORMAT = "<ddHH"   # (update_rate, last_time, d1, d2)

GYRO_PATH  = "/dev/shm/gyro"
ULTRA_PATH = "/dev/shm/ultrasoon"


# ================= SHM HELPERS =================

def open_shm(path, fmt):
    size = struct.calcsize(fmt)

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} bestaat niet")

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

shm_gyro, fd_gyro = open_shm(GYRO_PATH, GYRO_FORMAT)
shm_ultra, fd_ultra = open_shm(ULTRA_PATH, ULTRA_FORMAT)


# ================= CONFIG =================

# gewenste update rate (Hz) voor ultrasoon loop (optioneel)
UPDATE_RATE = 0.0

# schrijf update rate naar shm (behoud rest)
ultra_values = shm_read(shm_ultra, fd_ultra, ULTRA_FORMAT)
shm_write(
    shm_ultra,
    fd_ultra,
    ULTRA_FORMAT,
    (UPDATE_RATE, ultra_values[1], ultra_values[2], ultra_values[3])
)

print(f"Ultrasoon update rate ingesteld op {UPDATE_RATE} Hz\n")


# ================= LOOP =================

last_ts = None

while True:
    ts, gx, gy, gz = shm_read(shm_gyro, fd_gyro, GYRO_FORMAT)

    if last_ts is not None:
        dt = ts - last_ts
        freq = 1.0 / dt if dt > 0 else 0
    else:
        dt = 0
        freq = 0

    last_ts = ts

    print(
        f"t={ts:.3f} | "
        f"gx={gx:8.2f} gy={gy:8.2f} gz={gz:8.2f} | "
        f"dt={dt:.4f}s ({freq:6.1f} Hz)"
    )

    time.sleep(0.05)
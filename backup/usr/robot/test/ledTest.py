import time
import mmap
import struct
import fcntl
import os

# ================= SHM CONFIG =================
LED_FORMAT = "<7B"  # mode, r1,g1,b1, r2,g2,b2
SHM_PATH = "/dev/shm/led_ctrl"

# ================= SHM HELPERS =================
def create_or_open_shm(path, fmt):
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

# ================= INIT SHM =================
shm_led, fd_led = create_or_open_shm(SHM_PATH, LED_FORMAT)

# ================= KLEUREN =================
VERBLINDING = 255

colors = [
    #(0,0,0),
    #(VERBLINDING, 0, 0),    # rood
    #(0, VERBLINDING, 0),    # groen
    #(0, 0, VERBLINDING),    # blauw
    #(VERBLINDING, VERBLINDING, 0),  # geel
    (VERBLINDING, 0, VERBLINDING),  # magenta
    #(0, VERBLINDING, VERBLINDING),  # cyan
    #(VERBLINDING, VERBLINDING, VERBLINDING) # wit
]

# ================= LOOP =================
idx = 0
#while True:

r, g, b = colors[idx % len(colors)]
# modus 1 = direct kleur
shm_write(shm_led, fd_led, LED_FORMAT, (2, 0, VERBLINDING, VERBLINDING, 0, 0, 0))
print(f"Set LED color: R={r} G={g} B={b}")
idx += 1
time.sleep(0.1)
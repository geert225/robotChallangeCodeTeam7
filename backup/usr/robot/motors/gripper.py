import os
import mmap
import struct
import time
import fcntl

from motor import set_rad, rem_rad

# =========================
# CONFIG
# =========================
ROTATIES = 3
ENC_PPR = 1238  # encoder pulses per rotation

# Modes
MODE_IDLE = 0
MODE_AUTO = 1
MODE_JOG  = 2

# =========================
# ENCODER SHARED MEMORY
# =========================
num_encoders = 5
ENC_SHM_PATH = "/dev/shm/encoder_positions"

fd_enc = os.open(ENC_SHM_PATH, os.O_CREAT | os.O_RDWR)
os.ftruncate(fd_enc, num_encoders * 8)
enc_mem = mmap.mmap(fd_enc, num_encoders * 8, mmap.MAP_SHARED, mmap.PROT_READ)
os.close(fd_enc)

def read_encoder():
    offset = 4 * 8
    data = enc_mem[offset:offset + 8]
    return struct.unpack("q", data)[0]

# =========================
# COMMAND SHARED MEMORY
# =========================
# Format: mode (int32) + speed (double)
#   mode 0 = IDLE  : motor stop
#   mode 1 = AUTO  : standaard afloop (ROTATIES slagen, met vertraging)
#   mode 2 = JOG   : handmatige jog, speed > 0 open, speed < 0 dicht
CMD_SHM_PATH = "/dev/shm/gripper_cmd"
CMD_FORMAT = "<id"  # int32 + double = 12 bytes
CMD_SIZE = struct.calcsize(CMD_FORMAT)

def create_or_open_shm(path, size):
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(b"\x00" * size)
    f = open(path, "r+b")
    shm = mmap.mmap(f.fileno(), size)
    return shm, f

cmd_shm, cmd_fd = create_or_open_shm(CMD_SHM_PATH, CMD_SIZE)

def read_command():
    fcntl.flock(cmd_fd.fileno(), fcntl.LOCK_SH)
    cmd_shm.seek(0)
    data = cmd_shm.read(CMD_SIZE)
    fcntl.flock(cmd_fd.fileno(), fcntl.LOCK_UN)
    mode, speed = struct.unpack(CMD_FORMAT, data)
    return mode, speed

def write_command(mode, speed=0.0):
    fcntl.flock(cmd_fd.fileno(), fcntl.LOCK_EX)
    cmd_shm.seek(0)
    cmd_shm.write(struct.pack(CMD_FORMAT, mode, speed))
    fcntl.flock(cmd_fd.fileno(), fcntl.LOCK_UN)

# =========================
# GRIPPER FUNCTIES
# =========================

def run_auto():
    """Standaard afloop: draait ROTATIES slagen, vertraagt op het einde."""
    start_enc = read_encoder()
    end_enc = start_enc + (ROTATIES * ENC_PPR)
    slow = False

    set_rad(75)

    while True:
        # Controleer of een ander commando binnenkomt
        mode, speed = read_command()
        if mode != MODE_AUTO:
            rem_rad()
            return

        act_enc = read_encoder()

        if not slow and (end_enc - act_enc) < ((ENC_PPR / 3) * 2):
            slow = True
            set_rad(30)

        if (end_enc - act_enc) < 10:
            break

        time.sleep(0.01)

    rem_rad()
    time.sleep(1)
    set_rad(0)

    act_enc = read_encoder()
    print(f"[gripper] auto klaar, rest encoderfout: {end_enc - act_enc}")

    # Terug naar IDLE na voltooiing
    write_command(MODE_IDLE)


def run_jog(speed):
    """Handmatige jog: rij op gegeven snelheid (PWM %). Positief = open, negatief = dicht."""
    pwm = max(-100, min(100, int(speed)))
    set_rad(abs(pwm))
    if pwm == 0:
        rem_rad()


def run_idle():
    """Stop de gripper motor."""
    rem_rad()
    set_rad(0)


# =========================
# HOOFD LOOP
# =========================
prev_mode = None
prev_speed = None

print("[gripper] gestart, luistert op", CMD_SHM_PATH)

while True:
    mode, speed = read_command()

    if mode == MODE_IDLE:
        if prev_mode != MODE_IDLE:
            run_idle()
        prev_mode = mode
        time.sleep(0.05)

    elif mode == MODE_AUTO:
        if prev_mode != MODE_AUTO:
            print("[gripper] auto modus gestart")
        run_auto()  # blokkeert tot klaar of onderbroken
        prev_mode = MODE_IDLE

    elif mode == MODE_JOG:
        if prev_mode != MODE_JOG or prev_speed != speed:
            print(f"[gripper] jog snelheid={speed:.1f}")
            run_jog(speed)
        prev_mode = mode
        prev_speed = speed
        time.sleep(0.05)

    else:
        # Onbekende mode, stop veilig
        run_idle()
        prev_mode = MODE_IDLE
        time.sleep(0.05)

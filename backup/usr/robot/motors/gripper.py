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
ENC_PPR = 1848  # encoder pulses per rotation

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

# =========================
# STATE SHARED MEMORY
# =========================
# Schrijft de huidige gripper-toestand terug zodat fullBrain.py
# weet wanneer de cyclus klaar is.
#   0 = IDLE  (klaar / gestopt)
#   1 = BUSY  (auto-cyclus bezig)
STATE_SHM_PATH = "/dev/shm/gripper_state"
STATE_FORMAT   = "<i"   # int32
STATE_SIZE     = struct.calcsize(STATE_FORMAT)

state_shm, state_fd = create_or_open_shm(STATE_SHM_PATH, STATE_SIZE)

GRIPPER_STATE_IDLE = 0
GRIPPER_STATE_BUSY = 1

def write_state(state: int):
    fcntl.flock(state_fd.fileno(), fcntl.LOCK_EX)
    state_shm.seek(0)
    state_shm.write(struct.pack(STATE_FORMAT, state))
    fcntl.flock(state_fd.fileno(), fcntl.LOCK_UN)

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
    """Standaard afloop: draait ROTATIES slagen, vertraagt op het einde.

    Snelheid wordt gelezen uit de command-SHM (speed veld, 0–100).
    Fases worden proprotioneel geschaald:
      opstartfase  = speed * 0.55  (rustig aanlopen)
      volledig     = speed          (ingesteld setpoint)
      vertraagfase = speed * 0.40  (afremmen vlak voor einde)
    """
    global encoder_offset

    _, spd = read_command()
    speed     = max(10.0, min(100.0, spd))   # begrenzen 10–100
    spd_start = max(10.0, speed * 0.55)
    spd_slow  = max(10.0, speed * 0.40)
    print(f"[gripper] auto: setpoint={speed:.0f}  start={spd_start:.0f}  slow={spd_slow:.0f}")

    write_state(GRIPPER_STATE_BUSY)
    print("[gripper] state → BUSY")

    start_enc = read_encoder()
    end_enc = (start_enc - encoder_offset) + (ROTATIES * ENC_PPR)
    slow = False

    set_rad(int(spd_start))
    time.sleep(0.5)
    set_rad(int(speed))

    while True:
        act_enc = read_encoder()

        if not slow and (end_enc - act_enc) < ((ENC_PPR / 3) * 2):
            slow = True
            set_rad(int(spd_slow))

        if (end_enc - act_enc) < 10:
            break

        time.sleep(0.01)

    rem_rad()
    time.sleep(1)
    set_rad(0)

    time.sleep(1)
    act_enc = read_encoder()
    print(f"[gripper] auto klaar, rest encoderfout: {end_enc - act_enc}")
    encoder_offset = act_enc - end_enc

    # Terugkoppeling: cyclus klaar
    write_state(GRIPPER_STATE_IDLE)
    print("[gripper] state → IDLE")

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
encoder_offset = 0

print("[gripper] gestart, luistert op", CMD_SHM_PATH)

while True:
    mode, speed = read_command()

    if mode == MODE_IDLE:
        if prev_mode != MODE_IDLE:
            run_idle()
        prev_mode = mode
        time.sleep(0.05)
        #encoder_offset = 0

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
        encoder_offset = 0

    else:
        # Onbekende mode, stop veilig
        run_idle()
        prev_mode = MODE_IDLE
        time.sleep(0.05)
        encoder_offset = 0

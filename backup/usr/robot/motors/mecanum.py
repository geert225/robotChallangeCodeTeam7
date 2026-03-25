import os
import mmap
import struct
import time

from pwm import set_motor

# =========================
# CONFIG
# =========================
ENC_PPR = 540
DT = 0.01

KP = 0.25
KI = 0.8
INTEGRAL_LIMIT = 50
MAX_PWM = 100
ALPHA = 0.3  # low-pass filter

# Robot geometry
L = 0.190
W = 0.210
LW = L + W
R = 0.08

num_encoders = 4
encoder_signs = [1, -1, 1, 1]

# =========================
# ENCODER SHARED MEMORY
# =========================
shm_file_path = "/dev/shm/encoder_positions"
fd = os.open(shm_file_path, os.O_CREAT | os.O_RDWR)
os.ftruncate(fd, num_encoders*8)
shared_mem = mmap.mmap(fd, num_encoders*8, mmap.MAP_SHARED, mmap.PROT_READ)
os.close(fd)

def read_encoder(i):
    offset = i*8
    data = shared_mem[offset:offset+8]
    value = struct.unpack("q", data)[0]
    return value * encoder_signs[i]

# =========================
# COMMAND SHARED MEMORY
# =========================
CMD_SHM_PATH = "/dev/shm/robot_cmd"
CMD_SIZE = 32  # 4 doubles

fd_cmd = os.open(CMD_SHM_PATH, os.O_CREAT | os.O_RDWR)
os.ftruncate(fd_cmd, CMD_SIZE)
cmd_mem = mmap.mmap(fd_cmd, CMD_SIZE, mmap.MAP_SHARED, mmap.PROT_READ)
os.close(fd_cmd)

def read_command():
    data = cmd_mem[:CMD_SIZE]
    vx, vy, omega, timestamp = struct.unpack("dddd", data)
    return vx, vy, omega, timestamp

# =========================
# FEEDFORWARD
# =========================
def feedforward_formula(rpm):
    DEADZONE_RPM = 27.0
    OFFSET = 20.0
    SLOPE_FORWARD = 3.3
    SLOPE_BACKWARD = 3.0
    
    if abs(rpm) < DEADZONE_RPM:
        return 0
    
    sign = 1 if rpm > 0 else -1
    slope = SLOPE_FORWARD if rpm > 0 else SLOPE_BACKWARD
    pwm = OFFSET + abs(rpm)/slope
    return pwm * sign

# =========================
# MECANUM
# =========================
def mecanum_kinematics(vx, vy, omega):
    w0 = (vx - vy - omega*LW)/R
    w1 = (vx + vy + omega*LW)/R
    w2 = (vx + vy - omega*LW)/R
    w3 = (vx - vy + omega*LW)/R
    wheels = [w0, w1, w2, w3]
    return [w*60/(2*3.1416) for w in wheels]

# =========================
# SMOOTH ACCELERATION
# =========================
MAX_ACCEL = 8.0  # m/s^2
MAX_OMEGA_ACCEL = 5.0

vx_cmd = 0.0
vy_cmd = 0.0
omega_cmd = 0.0

def slew(current, target, rate):
    delta = target - current
    max_delta = rate * DT
    if delta > max_delta:
        delta = max_delta
    elif delta < -max_delta:
        delta = -max_delta
    return current + delta

# =========================
# CONTROLLER STATE
# =========================
integrals = [0]*num_encoders
prev_pos = [read_encoder(i) for i in range(num_encoders)]
rpm_filtered = [0]*num_encoders

TIMEOUT = 0.2

# =========================
# LOOP
# =========================
while True:
    # --- read command ---
    vx_target, vy_target, omega_target, ts = read_command()
    now = time.time()

    # failsafe
    if now - ts > TIMEOUT:
        vx_target, vy_target, omega_target = 0.0, 0.0, 0.0

    # smooth acceleration
    vx_cmd = slew(vx_cmd, vx_target, MAX_ACCEL)
    vy_cmd = slew(vy_cmd, vy_target, MAX_ACCEL)
    omega_cmd = slew(omega_cmd, omega_target, MAX_OMEGA_ACCEL)

    target_rpms = mecanum_kinematics(vx_cmd, vy_cmd, omega_cmd)

    # normalize
    max_rpm = max(abs(r) for r in target_rpms)
    MAX_RPM = 221.78
    if max_rpm > MAX_RPM:
        scale = MAX_RPM / max_rpm
        target_rpms = [r*scale for r in target_rpms]

    rpms = []
    pwms = []

    for i in range(num_encoders):
        pos = read_encoder(i)
        delta = pos - prev_pos[i]
        prev_pos[i] = pos

        rpm_raw = (delta/ENC_PPR)/DT*60
        rpm_filtered[i] = ALPHA*rpm_raw + (1-ALPHA)*rpm_filtered[i]
        rpm = rpm_filtered[i]
        rpms.append(rpm)

    avg_rpm = sum(rpms)/num_encoders

    for i in range(num_encoders):
        error = target_rpms[i] - rpm_filtered[i]
        sync_error = rpm_filtered[i] - avg_rpm
        total_error = error - 0.5*sync_error

        integrals[i] += total_error*DT
        integrals[i] = max(-INTEGRAL_LIMIT, min(INTEGRAL_LIMIT, integrals[i]))

        pwm_ff = feedforward_formula(target_rpms[i])
        pwm_pi = KP*total_error + KI*integrals[i]
        pwm = pwm_ff + pwm_pi
        pwm = max(-MAX_PWM, min(MAX_PWM, pwm))

        pwms.append(pwm)
        set_motor(i, pwm)

    status = "  ".join(
        f"M{i}: rpm:{rpms[i]:6.1f} tgt:{target_rpms[i]:6.1f} pwm:{pwms[i]:6.1f}"
        for i in range(num_encoders)
    )
    #print(f"\r{status}", end="", flush=True)

    time.sleep(DT)
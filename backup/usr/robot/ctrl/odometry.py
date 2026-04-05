#!/usr/bin/env python3
"""
odometry.py  —  Positiebepaling voor mecanum robot (50 Hz)

Fusie-strategie:
  - Heading (theta)  : Madgwick-yaw uit /dev/shm/gyro      (i2c.py)
  - Translatie fusie : complementary filter van twee bronnen:
      • Encoder-snelheid : mecanum forward-kinematics        (slip = overrapportage)
      • Accel-snelheid   : integratie van /dev/shm/mpu_accel (drift, maar slip-onafhankelijk)
    Instelbaar via ACCEL_TRUST (0.0 = puur encoder, 0.3 = aanbevolen bij slip)

Uitvoer: /dev/shm/robot_pose  →  3 doubles  (x [m], y [m], theta [rad])
         Nulpunt = positie bij opstarten, theta=0 = rijrichting bij opstarten.

Start als apart proces:
    python3 odometry.py

Vereisten:
  - encoders.py draait  → /dev/shm/encoder_positions
  - i2c.py draait       → /dev/shm/gyro  en  /dev/shm/mpu_accel
"""

import os, mmap, struct, time, math

# =============================================================================
# ROBOT PARAMETERS  (zelfde als mecanum.py)
# =============================================================================
ENC_PPR = 1080        # encoder pulsen per wiel-omwenteling
R       = 0.04       # wielstraal [m]
L       = 0.190      # halve wielafstand voor-achter [m]
W       = 0.210      # halve wielafstand links-rechts [m]
LW      = L + W      # = 0.400 m

# Tekens per wiel  (zelfde als mecanum.py)
#   0 = links-voor  (neg)   1 = rechts-voor  (pos)
#   2 = links-achter(neg)   3 = rechts-achter(pos)
ENCODER_SIGNS = [-1, 1, -1, 1]

# =============================================================================
# KALIBRATIEWAARDEN  —  pas aan als de kaart-schaal niet klopt
# =============================================================================
# Slip-compensatie op encoder-pad (gaat mee in gewogen fusie):
#   stel in op SLIP_VX = werkelijke_afstand / encoder_afstand  (< 1.0 bij slip)
# Meting: rij 1 m vooruit, kijk wat de kaart toont, pas dienovereenkomstig aan.
SLIP_VX = 1  # gemeten: robot reed 0.48 m, kaart toonde 1.80 m → 0.48/1.80
SLIP_VY = 0.8  # gemeten: robot reed 0.20 m, kaart toonde 0.94 m → 0.20/0.94

# ── Complementary filter: encoder vs accel ────────────────────────────────
# ACCEL_TRUST = gewicht van de accel-snelheid in de fusie (0.0 .. 1.0)
#   0.0  → puur encoder-odometrie (stabiel, maar slip telt volledig mee)
#   0.3  → aanbevolen beginpunt: accel corrigeert grofste slip
#   0.7+ → accel domineert; noisy maar slip-onafhankelijk
ACCEL_TRUST = 0.3

# Drempelwaarde [m/s²] waaronder versnelling als nul beschouwd wordt.
# Voorkomt dat accel-snelheid doorloopt als de robot stilstaat.
ACCEL_DEADZONE = 0.15

# Vervalcoëfficiënt voor accel-snelheid per stap (dichter bij 1.0 = langzamer verval).
# Corrigeert langzaam opgebouwde accel-drift als robot stilstaat.
ACCEL_VEL_DECAY = 0.90

# =============================================================================
# SHARED MEMORY — ENCODERS  (geschreven door encoders.py)
# =============================================================================
ENC_SHM_PATH = "/dev/shm/encoder_positions"
_NUM_ENC_ALL = 5   # 4 rijwielen + 1 gripper

fd_enc  = os.open(ENC_SHM_PATH, os.O_CREAT | os.O_RDWR)
os.ftruncate(fd_enc, _NUM_ENC_ALL * 8)
enc_mem = mmap.mmap(fd_enc, _NUM_ENC_ALL * 8, mmap.MAP_SHARED, mmap.PROT_READ)
os.close(fd_enc)

def read_enc(i: int) -> int:
    enc_mem.seek(i * 8)
    return struct.unpack('q', enc_mem.read(8))[0] * ENCODER_SIGNS[i]

# =============================================================================
# SHARED MEMORY — ACCEL  (geschreven door i2c.py)
# Formaat: "<ddd"  →  (timestamp, ax_world [m/s²], ay_world [m/s²])
# Zwaartekracht al afgetrokken, wereldframe, via Madgwick-quaternion.
# =============================================================================
ACCEL_SHM_PATH = "/dev/shm/mpu_accel"
ACCEL_FMT      = "<ddd"
ACCEL_SIZE     = struct.calcsize(ACCEL_FMT)   # 24 bytes

fd_accel  = os.open(ACCEL_SHM_PATH, os.O_CREAT | os.O_RDWR)
os.ftruncate(fd_accel, ACCEL_SIZE)
accel_mem = mmap.mmap(fd_accel, ACCEL_SIZE, mmap.MAP_SHARED, mmap.PROT_READ)
os.close(fd_accel)

_last_accel_ts = 0.0

def read_accel_world():
    """Leest (ax, ay) in m/s² uit SHM. Geeft (0,0) als data niet ververst."""
    global _last_accel_ts
    accel_mem.seek(0)
    ts, ax, ay = struct.unpack(ACCEL_FMT, accel_mem.read(ACCEL_SIZE))
    if ts == _last_accel_ts:          # geen nieuwe meting
        return 0.0, 0.0
    _last_accel_ts = ts
    return ax, ay

# =============================================================================
# SHARED MEMORY — GYRO  (geschreven door i2c.py, Madgwick-filter)
# Formaat: "<dddd"  →  (timestamp, roll, pitch, yaw)   yaw in graden
# =============================================================================
GYRO_SHM_PATH = "/dev/shm/gyro"
GYRO_FMT      = "<dddd"
GYRO_SIZE     = struct.calcsize(GYRO_FMT)   # 32 bytes

fd_gyro  = os.open(GYRO_SHM_PATH, os.O_CREAT | os.O_RDWR)
os.ftruncate(fd_gyro, GYRO_SIZE)
gyro_mem = mmap.mmap(fd_gyro, GYRO_SIZE, mmap.MAP_SHARED, mmap.PROT_READ)
os.close(fd_gyro)

def read_yaw_rad() -> float:
    """Leest Madgwick-yaw uit SHM, geeft terug in radialen."""
    gyro_mem.seek(0)
    _, _, _, yaw_deg = struct.unpack(GYRO_FMT, gyro_mem.read(GYRO_SIZE))
    return math.radians(yaw_deg)

# =============================================================================
# SHARED MEMORY — POSE OUTPUT
# Formaat: "ddd"  →  x [m], y [m], theta [rad]
# =============================================================================
POSE_SHM_PATH = "/dev/shm/robot_pose"
POSE_FMT      = "ddd"
POSE_SIZE     = struct.calcsize(POSE_FMT)   # 24 bytes

fd_pose  = os.open(POSE_SHM_PATH, os.O_CREAT | os.O_RDWR)
os.ftruncate(fd_pose, POSE_SIZE)
pose_mem = mmap.mmap(fd_pose, POSE_SIZE, mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ)
os.close(fd_pose)

def write_pose(x: float, y: float, theta: float):
    pose_mem.seek(0)
    pose_mem.write(struct.pack(POSE_FMT, x, y, theta))

# =============================================================================
# WACHT TOT GYRO SHM BESCHIKBAAR IS
# (i2c.py moet al gestart zijn en minimaal 1 meting geschreven hebben)
# =============================================================================
print("[odometry] Wacht op /dev/shm/gyro van i2c.py...")
deadline = time.time() + 10.0
while time.time() < deadline:
    gyro_mem.seek(0)
    ts, _, _, _ = struct.unpack(GYRO_FMT, gyro_mem.read(GYRO_SIZE))
    if ts > 0.0:
        break
    time.sleep(0.05)
else:
    print("[odometry] WAARSCHUWING: /dev/shm/gyro is leeg — is i2c.py gestart?")

# =============================================================================
# INITIEEL REFERENTIE-YAW
# theta=0 = rijrichting bij opstarten  →  sla de start-yaw op als offset
# =============================================================================
yaw_offset = read_yaw_rad()
print(f"[odometry] Yaw-offset bij start: {math.degrees(yaw_offset):.2f}°")

def _delta_angle(a: float, b: float) -> float:
    """Kortste hoek van b naar a (wrap-safe), in radialen."""
    return math.atan2(math.sin(a - b), math.cos(a - b))

# =============================================================================
# POSITIE STATE
# =============================================================================
x     = 0.0
y     = 0.0
theta = 0.0

# Accel-snelheidsschatting (wereld frame) — wordt los bijgehouden en gefused
vx_accel = 0.0
vy_accel = 0.0

prev_enc   = [read_enc(i) for i in range(4)]
prev_yaw   = read_yaw_rad()
prev_t     = time.perf_counter()

DT_TARGET = 0.02   # 50 Hz

write_pose(x, y, theta)
print("[odometry] Gestart. Schrijft naar /dev/shm/robot_pose  (50 Hz)")
print(f"[odometry] ACCEL_TRUST={ACCEL_TRUST}  (0=puur encoder, 1=puur accel)")
print("[odometry] Heading via Madgwick-yaw, translatie via encoder+accel fusie")

# =============================================================================
# MAIN LOOP
# =============================================================================
try:
    while True:
        t0 = time.perf_counter()
        dt = t0 - prev_t
        prev_t = t0
        if dt <= 0:
            dt = DT_TARGET

        # ── Encoders: delta in meters per wiel ───────────────────────────
        cur_enc  = [read_enc(i) for i in range(4)]
        delta    = [cur_enc[i] - prev_enc[i] for i in range(4)]
        prev_enc = cur_enc

        # ticks → meters  (1 omw = ENC_PPR ticks = 2πR meter)
        d = [(delta[i] / ENC_PPR) * 2.0 * math.pi * R for i in range(4)]

        # Mecanum forward kinematics in robotframe
        #   d[0]=LV, d[1]=RV, d[2]=LA, d[3]=RA
        dx_robot = (d[0] + d[1] + d[2] + d[3]) * 0.25 * SLIP_VX
        dy_robot = (-d[0] + d[1] + d[2] - d[3]) * 0.25 * SLIP_VY

        # ── Heading: Madgwick-yaw uit i2c.py SHM ────────────────────────
        raw_yaw = read_yaw_rad()
        theta   = _delta_angle(raw_yaw, yaw_offset)
        dtheta  = _delta_angle(raw_yaw, prev_yaw)
        prev_yaw = raw_yaw

        # ── Robot-frame → wereld-frame (halve-stap rotatie) ──────────────
        mid   = theta - dtheta * 0.5
        cos_m = math.cos(mid)
        sin_m = math.sin(mid)

        dx_world = dx_robot * cos_m - dy_robot * sin_m
        dy_world = dx_robot * sin_m + dy_robot * cos_m

        # Encoder-snelheid [m/s] in wereldframe
        vx_enc = dx_world / dt
        vy_enc = dy_world / dt

        # ── Accel: snelheidsintegratie ────────────────────────────────────
        ax_w, ay_w = read_accel_world()

        # Dode zone: kleine versnellingen → ruis, niet integreren
        if abs(ax_w) < ACCEL_DEADZONE:
            ax_w = 0.0
        if abs(ay_w) < ACCEL_DEADZONE:
            ay_w = 0.0

        vx_accel += ax_w * dt
        vy_accel += ay_w * dt

        # Verval als er geen versnelling is gemeten (drift-preventie)
        if ax_w == 0.0:
            vx_accel *= ACCEL_VEL_DECAY
        if ay_w == 0.0:
            vy_accel *= ACCEL_VEL_DECAY

        # ── Complementary filter: fuseer encoder + accel snelheid ─────────
        vx_fused = (1.0 - ACCEL_TRUST) * vx_enc + ACCEL_TRUST * vx_accel
        vy_fused = (1.0 - ACCEL_TRUST) * vy_enc + ACCEL_TRUST * vy_accel

        # Trek accel-state bij naar gefused resultaat (voorkomt divergentie)
        vx_accel = vx_fused
        vy_accel = vy_fused

        # ── Positie-update ────────────────────────────────────────────────
        x += vx_fused * dt
        y += vy_fused * dt

        write_pose(x, y, theta)

        # ── Debug (optioneel: haal commentaar weg) ───────────────────────
        # print(f"\rx={x:6.3f}m  y={y:6.3f}m  θ={math.degrees(theta):7.2f}°"
        #       f"  venc=({vx_enc:.2f},{vy_enc:.2f})"
        #       f"  vacc=({vx_accel:.2f},{vy_accel:.2f})", end="")

        # ── Slaap tot volgende cyclus ────────────────────────────────────
        elapsed = time.perf_counter() - t0
        rem = DT_TARGET - elapsed
        if rem > 0:
            time.sleep(rem)

except KeyboardInterrupt:
    print("\n[odometry] Gestopt.")
finally:
    enc_mem.close()
    gyro_mem.close()
    accel_mem.close()
    pose_mem.close()

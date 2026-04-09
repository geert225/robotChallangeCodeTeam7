# fullBrain.py
#
# Mini-states (robot_mode):
#   MANUAL  - bestuur via webinterface + gripper jog/trigger
#   AUTO    - autonome visie-logica (upgradedBasicBrain), webinterface blijft actief
#
# WebSocket commando's:
#   Rijden    : up / down / left / right / rot_left / rot_right / stop
#   Modus     : set_mode:manual  /  set_mode:auto
#   Gripper   : gripper_jog:<speed>  /  gripper_stop  /  gripper_auto
#
# Shared Memory:
#   /dev/shm/robot_cmd      <- mecanum rijcommando's  (4x double)
#   /dev/shm/gripper_cmd    <- gripper mode+speed     (<id>)
#   /dev/shm/vision_frame   <- camera frame           (WIDTH + HEIGHT + timestamp + RGB)
#   /dev/shm/led_ctrl       <- LED kleuren            (<7B>)

import os
import time
import mmap
import struct
import fcntl
import asyncio
import io
import base64
import hashlib
import math
import json
from PIL import Image
import numpy as np
import cv2
from obstacle_map import ObstacleMap

# ============================================================
# CALIBRATIE  (HSV blob-detectie — wordt opgeslagen op schijf)
# ============================================================
_CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "calibration.json")

def _load_calib_hsv():
    try:
        with open(_CALIB_PATH) as f:
            d = json.load(f)
        return np.array(d["hsv_lower"]), np.array(d["hsv_upper"])
    except Exception:
        return np.array([130, 140,  90]), np.array([150, 255, 220])

def _load_calib_yellow():
    try:
        with open(_CALIB_PATH) as f:
            d = json.load(f)
        return np.array(d["yellow_lower"]), np.array(d["yellow_upper"])
    except Exception:
        return np.array([18, 80, 80]), np.array([38, 255, 255])

def _load_calib_exclude():
    try:
        with open(_CALIB_PATH) as f:
            d = json.load(f)
        if "exclude_lower" in d and "exclude_upper" in d:
            return np.array(d["exclude_lower"]), np.array(d["exclude_upper"])
        return None, None
    except Exception:
        return None, None

def _save_calib():
    try:
        data = {
            "hsv_lower":    LOWER_HSV.tolist(),
            "hsv_upper":    UPPER_HSV.tolist(),
            "yellow_lower": YELLOW_HSV_LOWER.tolist(),
            "yellow_upper": YELLOW_HSV_UPPER.tolist(),
        }
        if EXCLUDE_HSV_LOWER is not None and EXCLUDE_HSV_UPPER is not None:
            data["exclude_lower"] = EXCLUDE_HSV_LOWER.tolist()
            data["exclude_upper"] = EXCLUDE_HSV_UPPER.tolist()
        with open(_CALIB_PATH, "w") as f:
            json.dump(data, f)
        print("[calib] opgeslagen")
    except Exception as e:
        print(f"[calib] fout: {e}")

# ============================================================
# ROBOT MODUS
# ============================================================
MODE_MANUAL = "manual"
MODE_AUTO   = "auto"
MODE_HOME   = "home"
robot_mode  = MODE_MANUAL          # beginstand = MANUAL

# ============================================================
# AUTO MODUS STATES
# ============================================================
AUTO_IDLE               = "idle"        # in deze state wachten op start commando
AUTO_INIT               = "init"        # in deze state alle waardes initalizeren (klep dicht en teller ressetten)
AUTO_SEARCH             = "zoeken"      # rondjes draaien tot target gevonden is
AUTO_DRIVE_TRAGET       = "goTraget"    # rijden naar gevonden target met obstakel ontwijking (als tragert kwijt dan naar zoeken behalve als onder rad dan naar gopickup)
AUTO_DRIVE_PICKUP       = "goPickup"    # als target (paars beker) onder het rad (geel blok) komt een standaard op pak procudure uitvoeren
AUTO_CENTER_PICKUP      = "centerPickup"  # draai robot zodat beker perfect gecentreerd staat voordat pickup begint
AUTO_PICKUP             = "pickup"      # opraap routine starten (optellen aantal bekers)
AUTO_TO_START           = "toStartPos"  # terug naar start positie rijden -> als robot x aantal bekers opgepakt heeft
AUTO_LOSSEN             = "lossen"      # laad klep open zetten
AUTO_PAUSED             = "paused"      # state machine op pauze (rijdt niet)
AUTO_SEARCH_HOME        = "zoekHome"    # draai zoekend naar gele paal (startpositie)
AUTO_DRIVE_HOME         = "rijHome"     # rij op gele paal af → klep open bij aankomst
auto_state              = AUTO_IDLE
_auto_paused            = False         # pauze-vlag (True = bevroren)

# ============================================================
# AUTO STATE MACHINE — gedeelde data met vision_loop
# ============================================================
_auto_bekers: list  = []    # laatste beker-detecties (gevuld door vision_loop)
_auto_frame_w: int  = 320   # frame breedte (gevuld door vision_loop)
_auto_frame_h: int  = 240   # frame hoogte (gevuld door vision_loop)
_auto_cup_count: int = 0    # aantal opgepakte bekers in huidige cyclus
AUTO_MAX_CUPS        = 5    # instelbaar via web (set_max_cups:N), min 1 max 10

# ============================================================
# SHARED MEMORY — SERVO (laadklep)
# Format: <B  →  0 = open, 1 = dicht
# ============================================================
SERVO_SHM_PATH = "/dev/shm/servo"
SERVO_FMT      = "<B"
SERVO_SIZE     = struct.calcsize(SERVO_FMT)

def _open_servo_shm():
    if not os.path.exists(SERVO_SHM_PATH):
        with open(SERVO_SHM_PATH, "wb") as f:
            f.write(b"\x00" * SERVO_SIZE)
    fh  = open(SERVO_SHM_PATH, "r+b")
    shm = mmap.mmap(fh.fileno(), SERVO_SIZE)
    return shm, fh

servo_shm, servo_fh = _open_servo_shm()

def _servo_write(value: int):
    """Schrijf servo-positie: 0 = open, 1 = dicht."""
    fcntl.flock(servo_fh.fileno(), fcntl.LOCK_EX)
    servo_shm.seek(0)
    servo_shm.write(struct.pack(SERVO_FMT, value))
    fcntl.flock(servo_fh.fileno(), fcntl.LOCK_UN)

# ============================================================
# LAADKLEP — sturing via servo SHM
# ============================================================
def _klep_open():
    """Laadklep openen (servo = 0)."""
    _servo_write(0)
    print("[klep] OPEN")

def _klep_dicht():
    """Laadklep sluiten (servo = 1)."""
    _servo_write(1)
    print("[klep] DICHT")

# ============================================================
# SHARED MEMORY — HELPERS
# ============================================================
def _create_or_open_shm(path, size):
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(b"\x00" * size)
    fh = open(path, "r+b")
    shm = mmap.mmap(fh.fileno(), size)
    return shm, fh

# ============================================================
# SHARED MEMORY — RIJCOMMANDO'S
# ============================================================
CMD_SHM_PATH = "/dev/shm/robot_cmd"
CMD_SIZE     = 32                  # 4x double

fd_cmd = os.open(CMD_SHM_PATH, os.O_CREAT | os.O_RDWR)
os.ftruncate(fd_cmd, CMD_SIZE)
cmd_mem = mmap.mmap(fd_cmd, CMD_SIZE, mmap.MAP_SHARED, mmap.PROT_WRITE)
os.close(fd_cmd)

def write_drive_cmd(vx, vy, omega):
    cmd_mem[:CMD_SIZE] = struct.pack("dddd", vx, vy, omega, time.time())

# ============================================================
# SHARED MEMORY — GRIPPER
# ============================================================
GRIPPER_IDLE     = 0
GRIPPER_AUTO_CMD = 1
GRIPPER_JOG      = 2

GRIPPER_SHM_PATH = "/dev/shm/gripper_cmd"
GRIPPER_FMT      = "<id"
GRIPPER_SIZE     = struct.calcsize(GRIPPER_FMT)

gripper_shm, gripper_fh = _create_or_open_shm(GRIPPER_SHM_PATH, GRIPPER_SIZE)

# Gripper state terugkoppeling (geschreven door gripper.py)
#   0 = IDLE (klaar), 1 = BUSY (bezig)
_GRIPPER_STATE_SHM_PATH = "/dev/shm/gripper_state"
_GRIPPER_STATE_FMT      = "<i"
_GRIPPER_STATE_SIZE     = struct.calcsize(_GRIPPER_STATE_FMT)

gripper_state_shm, gripper_state_fh = _create_or_open_shm(
    _GRIPPER_STATE_SHM_PATH, _GRIPPER_STATE_SIZE)

def _read_gripper_state() -> int:
    """Leest de toestand van gripper.py: 0=IDLE, 1=BUSY."""
    fcntl.flock(gripper_state_fh.fileno(), fcntl.LOCK_SH)
    gripper_state_shm.seek(0)
    raw = gripper_state_shm.read(_GRIPPER_STATE_SIZE)
    fcntl.flock(gripper_state_fh.fileno(), fcntl.LOCK_UN)
    return struct.unpack(_GRIPPER_STATE_FMT, raw)[0]

# Gripper snelheden (aanpasbaar via web, 0–100)
gripper_jog_speed  = 30   # handmatige jog
gripper_auto_speed = 100   # auto cyclus
gripper_home_speed = 20   # vasthouden tijdens HOME

# Encoder SHM (voor gripper-feedback, index 4)
_ENC_SHM_PATH = "/dev/shm/encoder_positions"
_fd_enc_g     = os.open(_ENC_SHM_PATH, os.O_CREAT | os.O_RDWR)
os.ftruncate(_fd_enc_g, 5 * 8)
_enc_mem_g    = mmap.mmap(_fd_enc_g, 5 * 8, mmap.MAP_SHARED, mmap.PROT_READ)
os.close(_fd_enc_g)

def _read_gripper_enc() -> int:
    _enc_mem_g.seek(4 * 8)
    return struct.unpack('q', _enc_mem_g.read(8))[0]

# Gripper auto-cyclus state
_GRIPPER_AUTO_TARGET_TICKS = 400   # standaard doelrotatie in encoder-ticks
_gripper_overshoot         = 0     # ticks te veel bij vorige auto-cyclus
_gripper_manual_jogged     = False # True als er handmatig gejogged is na laatste auto
_gripper_pending_auto      = False # getriggerd vanuit WS

def write_gripper_cmd(mode, speed=0.0):
    fcntl.flock(gripper_fh.fileno(), fcntl.LOCK_EX)
    gripper_shm.seek(0)
    gripper_shm.write(struct.pack(GRIPPER_FMT, mode, speed))
    fcntl.flock(gripper_fh.fileno(), fcntl.LOCK_UN)

# ============================================================
# SHARED MEMORY — LED
# LED_FORMAT = "<7B"  →  mode, r1, g1, b1, r2, g2, b2
# mode 1 = solid beide LEDs
# mode 2 = idle/speciaal
# mode 3 = gecombineerd
# ============================================================
LED_SHM_PATH = "/dev/shm/led_ctrl"
LED_FMT      = "<7B"
LED_SIZE     = struct.calcsize(LED_FMT)

led_shm, led_fh = _create_or_open_shm(LED_SHM_PATH, LED_SIZE)

def write_led(mode, r1, g1, b1, r2, g2, b2):
    fcntl.flock(led_fh.fileno(), fcntl.LOCK_EX)
    led_shm.seek(0)
    led_shm.write(struct.pack(LED_FMT, mode, r1, g1, b1, r2, g2, b2))
    fcntl.flock(led_fh.fileno(), fcntl.LOCK_UN)

# LED-kleuren constanten
# MANUAL — zelfde als mecanum gebruikte
def led_manual_update(vx, vy, omega):
    avx = abs(vx)
    avy = abs(vy)
    if avx > 0.2 and avy > 0.2:
        write_led(2, 255, 255,   0,   0, 255,   0)   # vooruit + zijwaarts: geel + groen
    elif avx > 0.2:
        write_led(0,   0, 255,   0,   0,   0,   0)   # vooruit/achteruit: groen
    elif avy > 0.2:
        write_led(0, 255, 255,   0,   0, 255,   0)   # zijwaarts: geel + groen
    elif abs(omega) > 0.5:
        write_led(0,   0, 150, 255,   0, 150, 255)   # roteren: blauw
    else:
        write_led(1,   0, 255, 255,   0,   0,   0)   # idle: cyaan

# AUTO — paars: zoekend knippert, gevonden solide
_led_blink_state = False

def led_auto_update(has_target: bool):
    global _led_blink_state
    if has_target:
        # solide paars
        write_led(0, 160, 0, 200, 160, 0, 200)
    else:
        # knipperend paars — toggle elke aanroep (vision_loop ~20ms, update elke ~500ms via teller)
        write_led(1, 160, 0, 200, 160, 0, 200)

# ============================================================
# SHARED MEMORY — ULTRASOON
# Format: <ddHH  →  update_rate (double) + timestamp (double) + d1 (uint16) + d2 (uint16)
# Sensororiëntatie (aanpassen op fysieke montage):
#   d1 = voor,  d2 = achter
# Afstanden in cm. Waarde 0 = geen data → geen beperking.
# ============================================================
# ============================================================
# SHARED MEMORY — POSE  (geschreven door odometry.py)
# Format: ddd  →  x [m], y [m], theta [rad]
# ============================================================
POSE_SHM_PATH = "/dev/shm/robot_pose"
POSE_FMT      = "ddd"
POSE_SIZE     = struct.calcsize(POSE_FMT)   # 24 bytes

pose_shm, pose_fh = _create_or_open_shm(POSE_SHM_PATH, POSE_SIZE)

def _read_pose():
    """Leest (x, y, theta) uit shared memory van odometry.py. Geeft (0,0,0) bij fout."""
    try:
        pose_shm.seek(0)
        raw = pose_shm.read(POSE_SIZE)
        return struct.unpack(POSE_FMT, raw)
    except Exception:
        return 0.0, 0.0, 0.0

# ============================================================
# SHARED MEMORY — GYRO RAW  (geschreven door i2c.py, Madgwick-filter)
# Formaat: "<dddd"  →  (timestamp, roll, pitch, yaw)   yaw in graden
# ============================================================
GYRO_SHM_PATH = "/dev/shm/gyro"
GYRO_FMT      = "<dddd"
GYRO_SIZE     = struct.calcsize(GYRO_FMT)

gyro_shm, gyro_fh = _create_or_open_shm(GYRO_SHM_PATH, GYRO_SIZE)

def _read_gyro_yaw() -> float:
    """Leest ruwe Madgwick-yaw uit /dev/shm/gyro, geeft terug in radialen."""
    try:
        gyro_shm.seek(0)
        _, _, _, yaw_deg = struct.unpack(GYRO_FMT, gyro_shm.read(GYRO_SIZE))
        return math.radians(yaw_deg)
    except Exception:
        return _read_pose()[2]   # fallback naar odometrie

# ============================================================
# SHARED MEMORY — ULTRASOON
# Format: <ddHH  →  update_rate (double) + timestamp (double) + d1 (uint16) + d2 (uint16)
# Sensororiëntatie (aanpassen op fysieke montage):
#   d1 = voor,  d2 = achter
# Afstanden in cm. Waarde 0 = geen data → geen beperking.
# ============================================================
ULTRA_SHM_PATH = "/dev/shm/ultrasoon"
ULTRA_FMT      = "<ddHH"
ULTRA_SIZE     = struct.calcsize(ULTRA_FMT)

ultra_shm, ultra_fh = _create_or_open_shm(ULTRA_SHM_PATH, ULTRA_SIZE)

_ultra_d1: int = 0    # laatste meting sensor1 (voor)
_ultra_d2: int = 0    # laatste meting sensor2 (achter)

def _read_ultra():
    fcntl.flock(ultra_fh.fileno(), fcntl.LOCK_SH)
    ultra_shm.seek(0)
    raw = ultra_shm.read(ULTRA_SIZE)
    fcntl.flock(ultra_fh.fileno(), fcntl.LOCK_UN)
    _, _, d1, d2 = struct.unpack(ULTRA_FMT, raw)
    return int(d1), int(d2)

# ---- Obstacle avoidance drempelwaarden (cm) ----
# d1 = rechter sensor, d2 = linker sensor (fysiek omgedraaid).
ULTRA_SAFE_CM        = 15    # hard stop — blokkeer vooruitrijden
ULTRA_WARN_CM        = 30    # vertraagzone — vx lineair afschalen
ULTRA_DODGE_CM       = 20    # ontwijkdrempel — start zijdelingse dodge
ULTRA_DODGE_SPD      = 0.6   # [m/s] zijdelingse ontwijksnelheid
ULTRA_DODGE_VX_SCALE = 0.3   # vx-reductie factor tijdens dodge (ongebruikt tijdens schuin-achteruit)
ULTRA_DODGE_VX_BACK  = 0.1#0.15  # [m/s] achteruitsnelheid tijdens dodge (schuin naar achter)
ULTRA_CLEAR_TIME     = 2.5   # [s] BEIDE sensoren vrij voor vrijgave richting
ULTRA_STUCK_TIME     = 0.5   # [s] wacht voor stuck-check na starten dodge
ULTRA_STUCK_MIN_TICKS = 30   # min. encoder strafe-ticks in ULTRA_STUCK_TIME (anders = vast)

# Obstakeldetectie tijdens rotatie-zoeken (SEARCH + SEARCH_HOME)
ROT_ULTRA_STOP_CM  = 20    # [cm] drempel: stop draaien en rij achteruit
ROT_ULTRA_BACK_SPD = 0.15   # [m/s] achteruitsnelheid tot sensoren vrij zijn

_ultra_dodge_hold_dir  = 0    # 1=links, -1=rechts, 0=inactief
_ultra_dodge_clear_t   = 0.0  # tijdstip waarop beide sensoren voor het eerst vrij waren
_ultra_dodge_start_t   = 0.0  # tijdstip waarop huidige dodge-richting gekozen is
_ultra_dodge_enc_snap  = 0    # encoder strafe-waarde bij start van stuck-check interval

# Encoder-tekens (identiek aan odometry.py)
_ENC_DRIVE_SIGNS = [-1, 1, -1, 1]

def _read_strafe_enc() -> int:
    """Leest strafe-component uit rijwiel-encoders (raw ticks, signed).
    strafe ∝ (raw[0] + raw[1] - raw[2] - raw[3])  na sign-correctie.
    Positief = strafe rechts, negatief = strafe links.
    """
    vals = []
    for i in range(4):
        _enc_mem_g.seek(i * 8)
        raw = struct.unpack('q', _enc_mem_g.read(8))[0]
        vals.append(raw * _ENC_DRIVE_SIGNS[i])
    # strafe = -d0 + d1 + d2 - d3
    return -vals[0] + vals[1] + vals[2] - vals[3]

def _read_total_enc() -> int:
    """Som van absolute encoder-waarden voor alle 4 rijwielen (maat voor totale beweging)."""
    total = 0
    for i in range(4):
        _enc_mem_g.seek(i * 8)
        raw = struct.unpack('q', _enc_mem_g.read(8))[0]
        total += abs(raw)
    return total

def _read_rotation_enc() -> int:
    """Rotatie-component uit rijwiel-encoders (signed ticks).
    rotatie ∝ -d0 + d1 - d2 + d3  (na sign-correctie).
    Positief = CCW, negatief = CW.
    """
    vals = []
    for i in range(4):
        _enc_mem_g.seek(i * 8)
        raw = struct.unpack('q', _enc_mem_g.read(8))[0]
        vals.append(raw * _ENC_DRIVE_SIGNS[i])
    return -vals[0] + vals[1] - vals[2] + vals[3]

# ---- Motor stuck detectie ----
# Motor-stuck detectie drempelwaarden (translatie)
MOTOR_STUCK_TIME      = 0.8    # [s] meetvenster per check
MOTOR_STUCK_MIN_TICKS = 80     # min. totale encoder-ticks per venster (anders = vast)
MOTOR_STUCK_CMD_MIN   = 0.25   # min. gecombineerde |vx|+|vy| om check te activeren
MOTOR_STUCK_CONFIRMS  = 2      # aantal opeenvolgende "vast"-vensters voor trigger

_motor_stuck_t      = 0.0
_motor_stuck_snap   = 0
_motor_stuck_count  = 0        # aaneengesloten "vast"-vensters teller

# Motor-stuck detectie drempelwaarden (rotatie) — gyroscoop-gebaseerd
ROT_STUCK_TIME      = 0.8    # [s] meetvenster per check
ROT_STUCK_MIN_RAD   = 0.06   # min. gyroverandering [rad] per venster (anders = vast)
ROT_STUCK_CMD_MIN   = 0.3    # min. |omega| om check te activeren
ROT_STUCK_CONFIRMS  = 2      # aantal opeenvolgende "vast"-vensters voor trigger

_rot_stuck_t     = 0.0
_rot_stuck_snap  = 0.0        # yaw-snapshot [rad] bij start van venster
_rot_stuck_count = 0           # aaneengesloten "vast"-vensters teller

def _motor_stuck_update(vx: float, vy: float) -> bool:
    """Geeft True als de robot MOTOR_STUCK_CONFIRMS opeenvolgende vensters vast zit."""
    global _motor_stuck_t, _motor_stuck_snap, _motor_stuck_count

    if (abs(vx) + abs(vy)) <= MOTOR_STUCK_CMD_MIN:
        _motor_stuck_t     = 0.0
        _motor_stuck_snap  = _read_total_enc()
        _motor_stuck_count = 0
        return False

    now = time.time()
    if _motor_stuck_t == 0.0:
        _motor_stuck_t     = now
        _motor_stuck_snap  = _read_total_enc()
        _motor_stuck_count = 0
        return False

    if now - _motor_stuck_t < MOTOR_STUCK_TIME:
        return False

    delta = abs(_read_total_enc() - _motor_stuck_snap)
    _motor_stuck_t    = now
    _motor_stuck_snap = _read_total_enc()

    if delta < MOTOR_STUCK_MIN_TICKS:
        _motor_stuck_count += 1
        print(f"[motor-stuck] venster {_motor_stuck_count}/{MOTOR_STUCK_CONFIRMS}"
              f"  Δticks={delta}")
        if _motor_stuck_count >= MOTOR_STUCK_CONFIRMS:
            _motor_stuck_count = 0
            return True
    else:
        _motor_stuck_count = 0
    return False

def _motor_stuck_update_rot(omega: float) -> bool:
    """Controleer of de robot draait terwijl een omega-commando actief is.

    Geeft True als de gyroscoop-yaw nauwelijks wijzigt terwijl omega actief is
    (robot vastzittend bij draaien). Moet elke control-tick (~50 ms) aangeroepen worden.
    """
    global _rot_stuck_t, _rot_stuck_snap, _rot_stuck_count

    cur_yaw = _read_pose()[2]   # theta [rad] uit odometry/Madgwick SHM

    if abs(omega) < ROT_STUCK_CMD_MIN:
        _rot_stuck_t     = 0.0
        _rot_stuck_snap  = cur_yaw
        _rot_stuck_count = 0
        return False

    now = time.time()
    if _rot_stuck_t == 0.0:
        _rot_stuck_t     = now
        _rot_stuck_snap  = cur_yaw
        _rot_stuck_count = 0
        return False

    if now - _rot_stuck_t < ROT_STUCK_TIME:
        return False

    # Wrap-veilige yaw-delta
    delta = abs(math.atan2(math.sin(cur_yaw - _rot_stuck_snap),
                            math.cos(cur_yaw - _rot_stuck_snap)))
    _rot_stuck_t    = now
    _rot_stuck_snap = cur_yaw

    if delta < ROT_STUCK_MIN_RAD:
        _rot_stuck_count += 1
        print(f"[rot-stuck] venster {_rot_stuck_count}/{ROT_STUCK_CONFIRMS}"
              f"  Δyaw={math.degrees(delta):.2f}°")
        if _rot_stuck_count >= ROT_STUCK_CONFIRMS:
            _rot_stuck_count = 0
            return True
    else:
        _rot_stuck_count = 0
    return False

def _apply_avoidance(vx: float, vy: float, omega: float):
    """Schaal vx terug o.b.v. ultrasoonwaarden. 0 = geen data → geen ingreep."""
    if vx > 0:
        s1 = _ultra_d1 if _ultra_d1 > 0 else 9999
        s2 = _ultra_d2 if _ultra_d2 > 0 else 9999
        front = min(s1, s2)
        if front <= ULTRA_SAFE_CM:
            vx = 0.0
        elif front < ULTRA_WARN_CM:
            vx *= (front - ULTRA_SAFE_CM) / (ULTRA_WARN_CM - ULTRA_SAFE_CM)
    return vx, vy, omega

def _ultra_dodge_vy(cup_cx: int | None = None, frame_w: int = 320) -> float:
    """Zijdelingse ontwijksnelheid op basis van ultrasoon + camera.

    Richting-keuze (prioriteit):
      1. Camera: obstakel zit tussen robot en cup → dodge NAAR de cup
         (cup links → obstakel links → dodge rechts, en vice versa)
      2. Sensor: ga naar de kant met de MEESTE ruimte (grootste waarde).
      3. Default: rechts.

    Opmerking: d1 = RECHTER sensor, d2 = LINKER sensor (fysiek omgedraaid).

    Gekozen richting vasthouden totdat BEIDE sensoren vrij zijn
    gedurende ULTRA_CLEAR_TIME seconden.

    Returns:
      vy [m/s], positief = links, negatief = rechts, 0.0 = geen dodge.
    """
    global _ultra_dodge_hold_dir, _ultra_dodge_clear_t
    global _ultra_dodge_start_t, _ultra_dodge_enc_snap

    # Sensoren zijn fysiek omgedraaid: d1=rechts, d2=links
    s_right = _ultra_d1 if _ultra_d1 > 0 else 9999
    s_left  = _ultra_d2 if _ultra_d2 > 0 else 9999
    both_clear = s_right > ULTRA_DODGE_CM and s_left > ULTRA_DODGE_CM

    if _ultra_dodge_hold_dir != 0:
        # ── Al een richting actief ────────────────────────────────────────

        # Stuck-check: na ULTRA_STUCK_TIME meten of robot daadwerkelijk bewoog
        now = time.time()
        if (now - _ultra_dodge_start_t) >= ULTRA_STUCK_TIME:
            strafe_now  = _read_strafe_enc()
            delta_ticks = abs(strafe_now - _ultra_dodge_enc_snap)
            if delta_ticks < ULTRA_STUCK_MIN_TICKS:
                # Robot beweegt niet → vast tegen zijkant, draai richting om
                _ultra_dodge_hold_dir = -_ultra_dodge_hold_dir
                _ultra_dodge_enc_snap = strafe_now
                _ultra_dodge_start_t  = now
                _ultra_dodge_clear_t  = 0.0
                print(f"[dodge] VAST gedetecteerd (Δticks={delta_ticks}) → "
                      f"omgekeerd naar {'links' if _ultra_dodge_hold_dir>0 else 'rechts'}")
            else:
                # Wel bewogen: snapshot vernieuwen voor volgende check
                _ultra_dodge_enc_snap = strafe_now
                _ultra_dodge_start_t  = now

        # Vrijgave-timer: alleen vrijgeven als BEIDE sensoren lang genoeg vrij
        if both_clear:
            if _ultra_dodge_clear_t == 0.0:
                _ultra_dodge_clear_t = time.time()
            elif time.time() - _ultra_dodge_clear_t >= ULTRA_CLEAR_TIME:
                _ultra_dodge_hold_dir = 0
                _ultra_dodge_clear_t  = 0.0
                return 0.0
        else:
            _ultra_dodge_clear_t = 0.0
        return ULTRA_DODGE_SPD if _ultra_dodge_hold_dir > 0 else -ULTRA_DODGE_SPD

    # ── Geen actieve richting ─────────────────────────────────────────────
    if both_clear:
        _ultra_dodge_clear_t = 0.0
        return 0.0

    # Obstakel gedetecteerd: kies richting
    # 1. Camera-hint: cup is het doel, dodge NAAR de cup toe
    if cup_cx is not None and frame_w > 0:
        cup_left = cup_cx < frame_w // 2   # cup links in beeld
        # obstakel staat voor de cup → dodge naar kant van de cup
        dir_choice = -1 if cup_left else 1   # cup links → dodge rechts (-1), cup rechts → links (+1)
        reden = f"camera (cup {'links' if cup_left else 'rechts'})"
    # 2. Sensor-hint: ga naar kant met meeste ruimte
    elif s_left != s_right:
        dir_choice = 1 if s_left > s_right else -1   # meer ruimte links → links (+1)
        reden = f"sensor (links={s_left}cm rechts={s_right}cm)"
    else:
        dir_choice = -1   # default: rechts
        reden = "default"

    _ultra_dodge_hold_dir = dir_choice
    _ultra_dodge_clear_t  = 0.0
    _ultra_dodge_start_t  = time.time()
    _ultra_dodge_enc_snap = _read_strafe_enc()
    print(f"[dodge] {'links' if dir_choice>0 else 'rechts'} o.b.v. {reden}")
    return ULTRA_DODGE_SPD if _ultra_dodge_hold_dir > 0 else -ULTRA_DODGE_SPD

# ============================================================
# OBSTACLE MAP  (positiegebaseerde obstakelskaart)
# Wordt gevuld door obstacle_map_loop() en gebruikt door home_loop().
# ============================================================
obstacle_map = ObstacleMap()

# Maximale snelheid na het samenvoegen van HOME-vector + afstoting [m/s]
_HOME_MAX_SPD_REPULSE = 1.1   # iets boven _HOME_MAX_SPD zodat afstoting ruimte heeft

# ============================================================
# SHARED MEMORY — VISION FRAME
# Layout: [WIDTH:H][HEIGHT:H] [timestamp:d] [frame:RGB...]
# WIDTH en HEIGHT worden dynamisch gelezen uit de SHM header (geschreven door camera.py)
# ============================================================
VISION_META_FMT  = 'HH'
VISION_META_SIZE = struct.calcsize(VISION_META_FMT)   # 4 bytes: WIDTH, HEIGHT
CHANNELS         = 3
VISION_FRAME_PATH = "/dev/shm/vision_frame"

# Open vision frame SHM — camera.py schrijft de dimensies in het header
if not os.path.exists(VISION_FRAME_PATH):
    _default_frame_size = 320 * 240 * CHANNELS
    with open(VISION_FRAME_PATH, "wb") as f:
        f.write(b"\x00" * (VISION_META_SIZE + 8 + _default_frame_size))
vision_fh  = open(VISION_FRAME_PATH, "r+b")
vision_shm = mmap.mmap(vision_fh.fileno(), 0)   # 0 = map hele file

latest_jpeg = None

# Selecteerbare vision debug-stap (instelbaar via WebSocket vanuit HTML)
# Opties: "raw" | "hsv" | "mask" | "bekers" | "final"
_vision_debug_step = "final"

# Verbonden control-websocket clients (voor broadcast van ultrasoon-data)
_control_clients: set = set()

# ============================================================
# LOG INTERCEPTOR  — stuurt print()-output naar WS-clients
# ============================================================
import sys as _sys

_log_queue: list = []          # wachtrij voor log-regels (max 200)
_LOG_QUEUE_MAX = 200

class _WsBroadcastLogger:
    """Vervangt sys.stdout: schrijft naar console én naar _log_queue."""
    def __init__(self, original):
        self._orig = original

    def write(self, text: str):
        self._orig.write(text)
        stripped = text.strip()
        if stripped:
            if len(_log_queue) >= _LOG_QUEUE_MAX:
                _log_queue.pop(0)
            _log_queue.append(stripped)

    def flush(self):
        self._orig.flush()

    def fileno(self):
        return self._orig.fileno()

_sys.stdout = _WsBroadcastLogger(_sys.stdout)

# ============================================================
# MANUAL RIJSTATUS
# ============================================================
drive_state = {"vx": 0.0, "vy": 0.0, "omega": 0.0}
SPEED     = 1.5
ROT_SPEED = 3.0

# ============================================================
# AUTO — VISIE PARAMETERS  (uit upgradedBasicBrain)
# ============================================================
LOWER_HSV, UPPER_HSV         = _load_calib_hsv()
YELLOW_HSV_LOWER, YELLOW_HSV_UPPER = _load_calib_yellow()
EXCLUDE_HSV_LOWER, EXCLUDE_HSV_UPPER = _load_calib_exclude()   # None = uitgeschakeld
MIN_BEKER_AREA  = 50
MAX_BEKER_AREA  = 300_000
MIN_OBST_AREA   = 250

K_OMEGA         = 2.8
MAX_OMEGA       = 4.0
STOP_AREA       = 9_000
SEARCH_OMEGA    = 2.0
ERROR_DEADBAND  = 0.02
DRIVE_SPEED     = 0.7
DRIVE_MIN_OMEGA = 1.5
MAX_TRACK_DIST  = 80

# Persistent tracking — hoeveel frames het laatste target bewaard blijft als er
# geen detectie is (voorkomt dat de robot zoekend gaat draaien bij korte occlussie)
MAX_TARGET_LOST_FRAMES = 15   # ~300 ms bij 20 fps

# Gele gripper-zone detectie (HSV, OpenCV-schaal: H 0–180) — geladen uit calibration.json
MIN_YELLOW_AREA  = 400   # kleinste gele blob die als gripper-vlak telt

# Pickup-event drempelwaarden
PICKUP_OVERLAP_RATIO = 0.25  # fractie van de beker-bbox die het gele vlak moet overlappen
PICKUP_MIN_CUP_AREA  = 2500  # beker moet groot genoeg zijn (dichtbij) om event te triggeren

# Bovenkant van het frame negeren (% van frame hoogte, 0 = alles checken)
# Bekers waarvan het middelpunt boven deze lijn valt worden genegeerd.
CUP_IGNORE_TOP_PCT = 27   # instelbaar via web (set_ignore_top_pct:XX)

# Onderkant-trigger: beker wordt als "gevangen" beschouwd zodra de onderkant
# van de bbox deze lijn bereikt (% van frame hoogte vanaf bovenkant).
CUP_BOTTOM_TRIGGER_PCT = 85  # instelbaar via web (set_bottom_trigger_pct:XX)

_last_dir_pos  = False
_last_target   = None   # (cx, cy) van vorige frame
_target_lost_n = 0      # frames achtereen zonder detectie

# Gedeeld resultaat vanuit vision_loop → auto_loop
_yellow_zone   = None   # (x, y, w, h) van grootste gele blob, of None
_pickup_ready  = False  # True wanneer beker voldoende overlapt met geel vlak

_cup_detection_enabled = True  # instelbaar via web (set_cup_detection:1/0)
_cup_touching_gripper  = False  # True zodra cup-bbox het gele vlak raakt (reset bij INIT)


def _classify_zone(cx, frame_w):
    if cx < frame_w * 0.33:
        return "LINKS"
    elif cx < frame_w * 0.66:
        return "MIDDEN"
    return "RECHTS"


def _compute_velocity(bekers, frame_w):
    """Niet langer gebruikt door auto_loop — staat nog in voor backward compat."""
    global _last_dir_pos, _last_target, _target_lost_n
    if not bekers:
        _target_lost_n += 1
        if _last_target is not None and _target_lost_n <= MAX_TARGET_LOST_FRAMES:
            cx    = _last_target[0]
            error = (cx - frame_w // 2) / (frame_w // 2)
            omega = 0.0 if abs(error) < ERROR_DEADBAND else max(-MAX_OMEGA, min(MAX_OMEGA, -K_OMEGA * error))
            return DRIVE_SPEED, 0.0, omega
        _last_target = None
        return (0.0, 0.0, -SEARCH_OMEGA) if _last_dir_pos else (0.0, 0.0, SEARCH_OMEGA)
    _target_lost_n = 0
    if _last_target is not None:
        best = min(bekers, key=lambda b: (b[0]+b[2]//2-_last_target[0])**2+(b[1]+b[3]//2-_last_target[1])**2)
    else:
        best = max(bekers, key=lambda b: b[4])
    x, y, w, h, _ = best
    cx = x + w // 2
    _last_target  = (cx, y + h // 2)
    _last_dir_pos = cx >= frame_w // 2
    error = (cx - frame_w // 2) / (frame_w // 2)
    omega = 0.0 if abs(error) < ERROR_DEADBAND else max(-MAX_OMEGA, min(MAX_OMEGA, -K_OMEGA * error))
    return DRIVE_SPEED, 0.0, omega

# ============================================================
# VISION LOOP  (altijd actief, schrijft rij+LED-commando afhankelijk van modus)
# ============================================================
_blink_counter = 0
BLINK_TICKS    = 25   # ~500 ms bij 20 ms sleep

async def vision_loop():
    global latest_jpeg, _blink_counter, _auto_bekers, _auto_frame_w, _auto_frame_h
    global _yellow_zone, _pickup_ready, _cup_touching_gripper

    while True:
        # frame lezen
        fcntl.flock(vision_fh.fileno(), fcntl.LOCK_SH)
        try:
            vision_shm.seek(0)
            frame_w, frame_h = struct.unpack(VISION_META_FMT, vision_shm.read(VISION_META_SIZE))
            vision_shm.read(8)                     # timestamp (overgeslagen)
            frame_bytes = vision_shm.read(frame_w * frame_h * CHANNELS)
        finally:
            fcntl.flock(vision_fh.fileno(), fcntl.LOCK_UN)

        # Wacht op geldig frame (camera.py heeft nog niet geschreven bij opstarten)
        expected = frame_w * frame_h * CHANNELS
        if frame_w == 0 or frame_h == 0 or len(frame_bytes) < expected:
            await asyncio.sleep(0.05)
            continue

        frame     = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((frame_h, frame_w, CHANNELS))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # camera geeft RGB, omzetten naar BGR voor CV

        raw_bgr = frame_bgr.copy()   # stap "raw": origineel frame voor debug

        # bekerdetectie
        hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # Uitsluitkleur: verwijder pixels die overeenkomen met het exclude-bereik
        if EXCLUDE_HSV_LOWER is not None and EXCLUDE_HSV_UPPER is not None:
            ex_mask = cv2.inRange(hsv, EXCLUDE_HSV_LOWER, EXCLUDE_HSV_UPPER)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(ex_mask))

        # Bereken de Y-cutoff voor de topzone (pixels vanaf bovenkant)
        _ignore_top_y = int(frame_h * CUP_IGNORE_TOP_PCT / 100)

        bekers = []
        if _cup_detection_enabled:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if MIN_BEKER_AREA < area < MAX_BEKER_AREA:
                    x, y, w, h = cv2.boundingRect(c)
                    cy_center = y + h // 2
                    if cy_center < _ignore_top_y:
                        continue   # beker zit in genegeerde topzone
                    bekers.append((x, y, w, h, area))
                    cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Teken topzone-grens als stippellijn (alleen als % > 0)
        if _ignore_top_y > 0:
            for xi in range(0, frame_w, 12):
                cv2.line(frame_bgr, (xi, _ignore_top_y), (min(xi+6, frame_w), _ignore_top_y),
                         (0, 180, 255), 1)

        cups_bgr = frame_bgr.copy()  # stap "bekers": na cup-boxes, vóór gele zone

        # ── Gele gripper-zone detectie ────────────────────────────────────────
        yellow_mask = cv2.inRange(hsv, YELLOW_HSV_LOWER, YELLOW_HSV_UPPER)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN,
                                       np.ones((5, 5), np.uint8))
        ycnt, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        best_yellow = None
        best_yellow_area = 0
        for c in ycnt:
            a = cv2.contourArea(c)
            if a >= MIN_YELLOW_AREA and a > best_yellow_area:
                x, y, w, h = cv2.boundingRect(c)
                cy_center = y + h // 2
                if cy_center < _ignore_top_y:
                    continue   # gele zone zit in genegeerde topzone
                best_yellow = (x, y, w, h)
                best_yellow_area = a
        _yellow_zone = best_yellow

        # Visualiseer gele zone met stippellijn rand
        if _yellow_zone is not None:
            yx, yy, yw, yh = _yellow_zone
            # dikke gele rand
            cv2.rectangle(frame_bgr, (yx, yy), (yx + yw, yy + yh),
                          (0, 220, 220), 2)
            cv2.putText(frame_bgr, "GRIPPER", (yx + 2, yy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 220), 1)

        # ── Bodem-trigger: beker onderkant bereikt trigger-lijn ──────────────
        # Kies de gevolgde beker (dichtstbij _last_target, anders grootste).
        _trigger_y = int(frame_h * CUP_BOTTOM_TRIGGER_PCT / 100)

        # Teken trigger-lijn (rood stippel)
        for xi in range(0, frame_w, 12):
            cv2.line(frame_bgr, (xi, _trigger_y), (min(xi + 6, frame_w), _trigger_y),
                     (0, 0, 255), 1)

        # Teken home-trigger lijn (groen stippel) alleen tijdens terug-naar-start states
        if auto_state in (AUTO_TO_START, AUTO_SEARCH_HOME, AUTO_DRIVE_HOME):
            _home_trig_y = int(frame_h * HOME_POLE_ARRIVED_PCT / 100)
            for xi in range(0, frame_w, 12):
                cv2.line(frame_bgr, (xi, _home_trig_y), (min(xi + 6, frame_w), _home_trig_y),
                         (0, 255, 0), 2)

        _pickup_ready = False
        if bekers:
            if _last_target is not None:
                tracked = min(bekers,
                              key=lambda b: (b[0] + b[2]//2 - _last_target[0])**2
                                          + (b[1] + b[3]//2 - _last_target[1])**2)
            else:
                tracked = max(bekers, key=lambda b: b[4])

            bx, by, bw, bh, barea = tracked
            cup_bottom = by + bh   # onderkant bbox in pixels

            if cup_bottom >= _trigger_y:
                _pickup_ready = True
                # visuele feedback: rode bbox + label
                cv2.rectangle(frame_bgr, (bx, by), (bx + bw, by + bh), (0, 0, 255), 3)
                cv2.putText(frame_bgr, "PICKUP READY",
                            (bx, by - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 80, 255), 1)

        # ── Cup captured (eenmaal True, reset in AUTO_INIT) ──────────────────
        if not _cup_touching_gripper and _pickup_ready:
            _cup_touching_gripper = True
        if _cup_touching_gripper:
            cv2.putText(frame_bgr, "CAPTURED", (5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # obstakeldetectie (onderste helft)
        #roi   = frame_bgr[int(frame_h * 0.55):frame_h, :]
        #gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        #edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
        #edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        #cnt_obs, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #for c in cnt_obs:
        #    if cv2.contourArea(c) < MIN_OBST_AREA:
        #        continue
        #    ox, oy, ow, oh = cv2.boundingRect(c)
        #    if ow < 12 or oh < 6:
        #        continue
        #    cv2.rectangle(roi, (ox, oy), (ox+ow, oy+oh), (0, 0, 255), 2)
        #frame_bgr[int(frame_h * 0.55):frame_h, :] = roi

        # rijcommando + LED afhankelijk van modus
        if robot_mode == MODE_AUTO:
            # Deel beker-data met auto_loop state machine (die schrijft rijcommando's)
            _auto_bekers = bekers
            _auto_frame_w = frame_w
            _auto_frame_h = frame_h

            # LED: elke BLINK_TICKS ticks de blink-toestand wisselen
            _blink_counter += 1
            if _blink_counter >= BLINK_TICKS:
                _blink_counter = 0
                led_auto_update(has_target=bool(bekers))

            label = f"AUTO [{auto_state}]  bekers:{len(bekers)}"
        else:
            label = f"MANUAL  bekers:{len(bekers)}"

        cv2.putText(frame_bgr, label, (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # JPEG bouwen — debug-stap selecteren
        _debug_frames = {
            "raw":    raw_bgr,
            "hsv":    cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR),
            "mask":   cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
            "bekers": cups_bgr,
            "final":  frame_bgr,
        }
        display_frame = _debug_frames.get(_vision_debug_step, frame_bgr)
        img = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        latest_jpeg = buf.getvalue()

        await asyncio.sleep(0.02)

# ============================================================
# WEBSOCKET HELPERS
# ============================================================
GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

def _ws_handshake(request):
    key = None
    for line in request.split("\r\n"):
        if "Sec-WebSocket-Key" in line:
            key = line.split(":", 1)[1].strip()
    accept = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
    return (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
    )

async def _ws_read_frame(reader):
    header = await reader.read(2)
    if not header:
        return None
    payload_len = header[1] & 127
    if payload_len == 126:
        payload_len = int.from_bytes(await reader.read(2), "big")
    elif payload_len == 127:
        payload_len = int.from_bytes(await reader.read(8), "big")
    mask    = await reader.read(4)
    payload = await reader.read(payload_len)
    return bytes(b ^ mask[i % 4] for i, b in enumerate(payload)).decode(errors="ignore")

async def _ws_send_binary(writer, data: bytes):
    header = bytearray([0x82])
    length = len(data)
    if length < 126:
        header.append(length)
    elif length < 65536:
        header.append(126)
        header += length.to_bytes(2, "big")
    else:
        header.append(127)
        header += length.to_bytes(8, "big")
    writer.write(header + data)
    await writer.drain()

async def _ws_send_text(writer, text: str):
    data   = text.encode()
    header = bytearray([0x81])
    length = len(data)
    if length < 126:
        header.append(length)
    elif length < 65536:
        header.append(126)
        header += length.to_bytes(2, "big")
    else:
        header.append(127)
        header += length.to_bytes(8, "big")
    writer.write(header + data)
    await writer.drain()

# ============================================================
# CONTROLE WEBSOCKET
# ============================================================
async def handle_control_ws(reader, writer, request):
    global robot_mode, drive_state, _blink_counter
    global gripper_jog_speed, gripper_auto_speed, gripper_home_speed
    global LOWER_HSV, UPPER_HSV, YELLOW_HSV_LOWER, YELLOW_HSV_UPPER
    global EXCLUDE_HSV_LOWER, EXCLUDE_HSV_UPPER
    global home_strategy, _gripper_pending_auto, _gripper_manual_jogged
    global _vision_debug_step, auto_state, _cup_detection_enabled
    global CUP_IGNORE_TOP_PCT, CUP_BOTTOM_TRIGGER_PCT, AUTO_MAX_CUPS
    global _auto_paused, _motor_stuck_t, _motor_stuck_snap

    writer.write(_ws_handshake(request).encode())
    await writer.drain()

    _control_clients.add(writer)

    # stuur huidige staat direct na verbinding
    await _ws_send_text(writer, f"mode:{robot_mode}")
    await _ws_send_text(writer, f"ultra:{_ultra_d1},{_ultra_d2}")
    await _ws_send_text(writer, f"gripper_enc:{_read_gripper_enc()}")
    px, py, pth = _read_pose()
    await _ws_send_text(writer, f"pose:{px:.4f},{py:.4f},{pth:.4f}")
    await _ws_send_text(writer, f"home_strat:{home_strategy}")
    await _ws_send_text(writer, f"gripper_speeds:{gripper_jog_speed},{gripper_auto_speed},{gripper_home_speed}")
    lo, hi = LOWER_HSV.tolist(), UPPER_HSV.tolist()
    await _ws_send_text(writer, f"hsv:{lo[0]},{lo[1]},{lo[2]},{hi[0]},{hi[1]},{hi[2]}")
    yl, yh = YELLOW_HSV_LOWER.tolist(), YELLOW_HSV_UPPER.tolist()
    await _ws_send_text(writer, f"yellow_hsv:{yl[0]},{yl[1]},{yl[2]},{yh[0]},{yh[1]},{yh[2]}")
    if EXCLUDE_HSV_LOWER is not None and EXCLUDE_HSV_UPPER is not None:
        el, eh = EXCLUDE_HSV_LOWER.tolist(), EXCLUDE_HSV_UPPER.tolist()
        await _ws_send_text(writer, f"hsv_exclude:{el[0]},{el[1]},{el[2]},{eh[0]},{eh[1]},{eh[2]}")
    else:
        await _ws_send_text(writer, "hsv_exclude:off")
    await _ws_send_text(writer, f"vision_step:{_vision_debug_step}")
    await _ws_send_text(writer, f"cup_detection:{'1' if _cup_detection_enabled else '0'}")
    await _ws_send_text(writer, f"ignore_top_pct:{CUP_IGNORE_TOP_PCT}")
    await _ws_send_text(writer, f"bottom_trigger_pct:{CUP_BOTTOM_TRIGGER_PCT}")
    await _ws_send_text(writer, f"max_cups:{AUTO_MAX_CUPS}")

    # Stuur de laatste 50 log-regels als history bij verbinding
    history = _log_queue[-50:] if len(_log_queue) > 50 else list(_log_queue)
    for line in history:
        await _ws_send_text(writer, "log:" + line)

    try:
        while True:
            msg = await _ws_read_frame(reader)
            if msg is None:
                break

            # ---- MODUS WISSELEN ----
            if msg == "set_mode:manual":
                robot_mode  = MODE_MANUAL
                drive_state = {"vx": 0.0, "vy": 0.0, "omega": 0.0}
                _blink_counter = 0
                write_drive_cmd(0.0, 0.0, 0.0)
                led_manual_update(0.0, 0.0, 0.0)
                print("[brain] → MANUAL")
                await _ws_send_text(writer, "mode:manual")

            elif msg == "set_mode:auto":
                robot_mode  = MODE_AUTO
                auto_state  = AUTO_INIT    # altijd initialiseren bij wisselen naar AUTO
                drive_state = {"vx": 0.0, "vy": 0.0, "omega": 0.0}
                _blink_counter = 0
                print("[brain] → AUTO (INIT)")
                await _ws_send_text(writer, "mode:auto")
                await _ws_send_text(writer, f"auto_state:{auto_state}")

            elif msg == "auto_start":
                # Start vanuit IDLE (als robot al in AUTO staat)
                if robot_mode == MODE_AUTO and auto_state == AUTO_IDLE:
                    auto_state = AUTO_INIT
                    print("[brain] auto_start → INIT")
                    await _ws_send_text(writer, f"auto_state:{auto_state}")

            elif msg == "auto_stop":
                # Zet state terug naar IDLE en stop rijden
                if robot_mode == MODE_AUTO:
                    auto_state   = AUTO_IDLE
                    _auto_paused = False
                    write_drive_cmd(0.0, 0.0, 0.0)
                    print("[brain] auto_stop → IDLE")
                    await _ws_send_text(writer, f"auto_state:{auto_state}")
                    await _ws_send_text(writer, "paused:false")

            elif msg == "auto_pause":
                if robot_mode == MODE_AUTO and not _auto_paused:
                    _auto_paused = True
                    write_drive_cmd(0.0, 0.0, 0.0)
                    print("[brain] auto PAUZE")
                    await _broadcast("paused:true")

            elif msg == "auto_resume":
                if robot_mode == MODE_AUTO and _auto_paused:
                    drive_state["vx"] = 0.0
                    drive_state["vy"] = 0.0
                    drive_state["omega"] = 0.0
                    write_drive_cmd(0.0, 0.0, 0.0)
                    _auto_paused = False
                    print("[brain] auto HERVAT")
                    await _broadcast("paused:false")

            elif msg == "set_mode:home":
                robot_mode  = MODE_HOME
                drive_state = {"vx": 0.0, "vy": 0.0, "omega": 0.0}
                _blink_counter = 0
                write_drive_cmd(0.0, 0.0, 0.0)
                write_led(0, 0, 100, 255, 0, 100, 255)   # solide blauw tijdens HOME
                print("[brain] → HOME")
                await _ws_send_text(writer, "mode:home")

            # ---- INSTELLINGEN ----
            elif msg.startswith("set_home_strat:"):
                home_strategy = msg.split(":", 1)[1].strip()
                print(f"[brain] home strategie → {home_strategy}")
                await _broadcast(f"home_strat:{home_strategy}")

            elif msg.startswith("set_gripper_speeds:"):
                try:
                    parts = msg.split(":", 1)[1].split(",")
                    gripper_jog_speed  = int(parts[0])
                    gripper_auto_speed = int(parts[1])
                    gripper_home_speed = int(parts[2])
                    print(f"[brain] gripper speeds: jog={gripper_jog_speed} auto={gripper_auto_speed} home={gripper_home_speed}")
                except Exception:
                    pass

            elif msg.startswith("set_vision_step:"):
                _vision_debug_step = msg.split(":", 1)[1].strip()
                print(f"[brain] vision stap → {_vision_debug_step}")

            elif msg == "reset_obstacles":
                obstacle_map.clear()
                print("[brain] obstacle_map gewist")
                await _ws_send_text(writer, "obstacles:0:")

            elif msg.startswith("set_hsv:"):
                try:
                    v = [int(x) for x in msg.split(":", 1)[1].split(",")]
                    LOWER_HSV = np.array(v[0:3])
                    UPPER_HSV = np.array(v[3:6])
                    _save_calib()
                    print(f"[brain] HSV → lower={LOWER_HSV.tolist()} upper={UPPER_HSV.tolist()}")
                except Exception:
                    pass

            elif msg.startswith("set_yellow_hsv:"):
                try:
                    v = [int(x) for x in msg.split(":", 1)[1].split(",")]
                    YELLOW_HSV_LOWER = np.array(v[0:3])
                    YELLOW_HSV_UPPER = np.array(v[3:6])
                    _save_calib()
                    print(f"[brain] Yellow HSV → lower={YELLOW_HSV_LOWER.tolist()} upper={YELLOW_HSV_UPPER.tolist()}")
                except Exception:
                    pass

            elif msg.startswith("set_hsv_exclude:"):
                try:
                    payload = msg.split(":", 1)[1]
                    if payload == "off":
                        EXCLUDE_HSV_LOWER = None
                        EXCLUDE_HSV_UPPER = None
                        print("[brain] uitsluitkleur → uitgeschakeld")
                    else:
                        v = [int(x) for x in payload.split(",")]
                        EXCLUDE_HSV_LOWER = np.array(v[0:3])
                        EXCLUDE_HSV_UPPER = np.array(v[3:6])
                        print(f"[brain] uitsluitkleur → lower={EXCLUDE_HSV_LOWER.tolist()} upper={EXCLUDE_HSV_UPPER.tolist()}")
                    _save_calib()
                except Exception:
                    pass

            elif msg.startswith("set_cup_detection:"):
                _cup_detection_enabled = msg.split(":", 1)[1].strip() == "1"
                print(f"[brain] cup detectie → {'aan' if _cup_detection_enabled else 'uit'}")
                await _broadcast(f"cup_detection:{'1' if _cup_detection_enabled else '0'}")

            elif msg.startswith("set_ignore_top_pct:"):
                try:
                    CUP_IGNORE_TOP_PCT = max(0, min(80, int(msg.split(":", 1)[1])))
                    print(f"[brain] topzone negeren → {CUP_IGNORE_TOP_PCT}%")
                    await _broadcast(f"ignore_top_pct:{CUP_IGNORE_TOP_PCT}")
                except ValueError:
                    pass

            elif msg.startswith("set_bottom_trigger_pct:"):
                try:
                    CUP_BOTTOM_TRIGGER_PCT = max(50, min(100, int(msg.split(":", 1)[1])))
                    print(f"[brain] bodem-trigger → {CUP_BOTTOM_TRIGGER_PCT}%")
                    await _broadcast(f"bottom_trigger_pct:{CUP_BOTTOM_TRIGGER_PCT}")
                except ValueError:
                    pass

            elif msg.startswith("set_max_cups:"):
                try:
                    AUTO_MAX_CUPS = max(1, min(10, int(msg.split(":", 1)[1])))
                    print(f"[brain] max bekers → {AUTO_MAX_CUPS}")
                    await _broadcast(f"max_cups:{AUTO_MAX_CUPS}")
                except ValueError:
                    pass

            # ---- LAADKLEP (manual bediening, werkt in alle modi) ----
            elif msg == "klep_open":
                _klep_open()
                await _broadcast("klep:open")

            elif msg == "klep_dicht":
                _klep_dicht()
                await _broadcast("klep:dicht")

            # ---- RIJDEN (MANUAL of AUTO-PAUSED) ----
            elif robot_mode == MODE_MANUAL or (robot_mode == MODE_AUTO and _auto_paused):
                if msg == "up":
                    drive_state["vx"] = SPEED
                elif msg == "down":
                    drive_state["vx"] = -SPEED
                elif msg == "left":
                    drive_state["vy"] = SPEED
                elif msg == "right":
                    drive_state["vy"] = -SPEED
                elif msg == "rot_left":
                    drive_state["omega"] = ROT_SPEED
                elif msg == "rot_right":
                    drive_state["omega"] = -ROT_SPEED
                elif msg == "stop":
                    drive_state = {"vx": 0.0, "vy": 0.0, "omega": 0.0}

                # ---- GRIPPER (alleen in MANUAL) ----
                elif msg.startswith("gripper_jog:"):
                    try:
                        direction = float(msg.split(":", 1)[1])  # +1 of -1
                        write_gripper_cmd(GRIPPER_JOG, direction * gripper_jog_speed)
                        _gripper_manual_jogged = True
                    except ValueError:
                        pass

                elif msg == "gripper_stop":
                    write_gripper_cmd(GRIPPER_IDLE)

                elif msg == "gripper_auto":
                    write_gripper_cmd(GRIPPER_AUTO_CMD, gripper_auto_speed)
                    print("[brain] gripper AUTO gestart")

                write_drive_cmd(drive_state["vx"], drive_state["vy"], drive_state["omega"])

    except Exception as e:
        print(f"[control WS] fout: {e}")
    finally:
        _control_clients.discard(writer)

    # verbinding verloren → stop robot veilig
    write_drive_cmd(0.0, 0.0, 0.0)
    write_gripper_cmd(GRIPPER_IDLE)
    led_manual_update(0.0, 0.0, 0.0)
    writer.close()

# ============================================================
# CAMERA WEBSOCKET
# ============================================================
async def handle_camera_ws(reader, writer, request):
    writer.write(_ws_handshake(request).encode())
    await writer.drain()

    try:
        while True:
            if latest_jpeg is None:
                await asyncio.sleep(0.02)
                continue
            await _ws_send_binary(writer, latest_jpeg)
            await asyncio.sleep(0.05)
    except Exception:
        pass

    writer.close()

# ============================================================
# HTTP HANDLER  —  laadt HTML uit web/index.html
# ============================================================
_WEB_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
_HTML_CACHE: bytes | None = None

def _load_html() -> bytes:
    global _HTML_CACHE
    if _HTML_CACHE is None:
        path = os.path.join(_WEB_DIR, "index.html")
        with open(path, "rb") as f:
            body = f.read()
        _HTML_CACHE = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/html; charset=utf-8\r\n"
            b"Content-Length: " + str(len(body)).encode() + b"\r\n"
            b"\r\n" + body
        )
    return _HTML_CACHE

async def handle_client(reader, writer):
    data    = await reader.read(2048)
    request = data.decode(errors="ignore")

    if "Upgrade: websocket" in request:
        if "/cam" in request:
            await handle_camera_ws(reader, writer, request)
        else:
            await handle_control_ws(reader, writer, request)
        return

    writer.write(_load_html())
    await writer.drain()
    writer.close()

# ============================================================
# HEARTBEAT  (refresht timestamp in MANUAL + update LED)
# ============================================================
async def heartbeat():
    while True:
        if robot_mode == MODE_MANUAL or (robot_mode == MODE_AUTO and _auto_paused):
            write_drive_cmd(drive_state["vx"], drive_state["vy"], drive_state["omega"])
            led_manual_update(drive_state["vx"], drive_state["vy"], drive_state["omega"])
        await asyncio.sleep(0.05)

# ============================================================
# LOG BROADCAST LOOP  (stuurt nieuwe log-regels naar WS-clients, 10 Hz)
# ============================================================
_log_sent_idx = 0   # hoeveel regels al verstuurd

async def log_loop():
    global _log_sent_idx
    while True:
        if _control_clients and _log_sent_idx < len(_log_queue):
            lines = _log_queue[_log_sent_idx:]
            _log_sent_idx = len(_log_queue)
            for line in lines:
                msg = "log:" + line
                for w in list(_control_clients):
                    try:
                        await _ws_send_text(w, msg)
                    except Exception:
                        _control_clients.discard(w)
        await asyncio.sleep(0.1)

# ============================================================
# ULTRASOON LOOP  (leest SHM en broadcast naar alle WS-clients, 10 Hz)
# ============================================================
_ultra_obstacle_led_state = False   # huidige blink-toestand voor obstakel-LED

async def ultra_loop():
    global _ultra_d1, _ultra_d2, _ultra_obstacle_led_state

    while True:
        d1, d2 = _read_ultra()
        _ultra_d1, _ultra_d2 = d1, d2

        # ── Obstakel-LED: rood knipperen (mode 2) als dodge actief ───────────
        s1 = d1 if d1 > 0 else 9999
        s2 = d2 if d2 > 0 else 9999
        obstacle_active = (s1 <= ULTRA_DODGE_CM or s2 <= ULTRA_DODGE_CM)

        if obstacle_active:
            _ultra_obstacle_led_state = not _ultra_obstacle_led_state
            if _ultra_obstacle_led_state:
                write_led(2, 255, 0, 0, 255, 0, 0)   # rood aan
            else:
                write_led(2, 0, 0, 0, 0, 0, 0)        # uit

        if _control_clients:
            enc = _read_gripper_enc()
            ultra_msg = f"ultra:{d1},{d2}"
            enc_msg   = f"gripper_enc:{enc}"
            for w in list(_control_clients):
                try:
                    await _ws_send_text(w, ultra_msg)
                    await _ws_send_text(w, enc_msg)
                except Exception:
                    _control_clients.discard(w)

        await asyncio.sleep(0.1)

# ============================================================
# POSE LOOP  (leest odometry SHM en broadcast naar WS clients, 5 Hz)
# ============================================================
async def pose_loop():
    while True:
        if _control_clients:
            px, py, pth = _read_pose()
            msg = f"pose:{px:.4f},{py:.4f},{pth:.4f}"
            for w in list(_control_clients):
                try:
                    await _ws_send_text(w, msg)
                except Exception:
                    _control_clients.discard(w)
        await asyncio.sleep(0.2)   # 5 Hz is genoeg voor visualisatie

# ============================================================
# OBSTACLE MAP LOOP  (5 Hz)
# Leest huidige pose + ultrasoonmeting en voegt mogelijke obstakels
# toe aan de kaart. Draait in ALLE modi zodat de kaart tijdens de
# verkenning (MANUAL / AUTO) al wordt opgebouwd.
# ============================================================
_ULTRA_TRIGGER_CM = 30   # cm — log obstakel als sensor < deze waarde meet

async def obstacle_map_loop():
    """Vult de obstacle_map met sensordata (5 Hz)."""
    _broadcast_counter = 0

    while True:
        px, py, pth = _read_pose()

        # beide sensoren zitten aan de voorzijde
        if 0 < _ultra_d1 <= _ULTRA_TRIGGER_CM:
            obstacle_map.add_reading(px, py, pth, _ultra_d1, direction_offset=0.0)
        if 0 < _ultra_d2 <= _ULTRA_TRIGGER_CM:
            obstacle_map.add_reading(px, py, pth, _ultra_d2, direction_offset=0.0)

        # Stuur obstakelkaart elke ~2 seconden naar clients (10 ticks × 0.2 s)
        _broadcast_counter += 1
        if _broadcast_counter >= 10 and _control_clients:
            _broadcast_counter = 0
            obs = obstacle_map.get_obstacles()
            if obs:
                pts = "|".join(f"{ox:.3f},{oy:.3f}" for ox, oy in obs)
                await _broadcast(f"obstacles:{len(obs)}:{pts}")
            else:
                await _broadcast("obstacles:0:")

        await asyncio.sleep(0.2)   # 5 Hz

# ============================================================
# HOME LOOP  — stuurt robot terug naar positie (0, 0, 0)
# Rijdt op 20 Hz; stopt automatisch en schakelt naar MANUAL.
# ============================================================

# ── Navigatiestrategie ───────────────────────────────────────────────────────
# "forward" (default): fase 1 = draaien naar home-richting,
#                      fase 2 = recht vooruit rijden,
#                      fase 3 = draaien naar theta=0
# "holonomic": rijden + draaien tegelijk (kan oscilleren bij hoge gains)
home_strategy = "forward"

# ── Deadband omega: kleine hoekfouten worden genegeerd (vermindert oscillatie)
_HOME_OMEGA_DEADBAND = 0.05  # rad

# ── P-regelaar instelling (laag gehouden om oscillaties te vermijden) ────────
_HOME_K_POS    = 0.6   # [m/s per m]   positiefout → rijsnelheid
_HOME_K_OMEGA  = 1.5   # [rad/s per rad] hoekfout → rotatiesnelheid
_HOME_MAX_SPD  = 0.5   # maximale rijsnelheid [m/s]
_HOME_MAX_OMG  = 1.2   # maximale rotatiesnelheid [rad/s]
_HOME_FINAL_OMG = 0.5  # maximale rotatiesnelheid eindcorrectie [rad/s]
_HOME_DONE_XY  = 0.15  # aankomstdrempel positie [m]  (15 cm)
_HOME_DONE_TH  = 0.10  # aankomstdrempel heading [rad] (~6°)

# ── Gele paal (startpositie) detectie voor terugrijden via camera ─────────────
HOME_POLE_MIN_AREA       = 1000   # min. bbox-oppervlak gele blob om als paal te tellen [px²]
HOME_POLE_SEARCH_OMEGA   = 2.0    # [rad/s] zoekdraaisnelheid
HOME_POLE_DRIVE_SPEED    = 0.5    # [m/s] rijsnelheid naar paal
HOME_POLE_K_OMEGA        = 2.8    # bijstuur-gain (zelfde als bekervolger)
HOME_POLE_MAX_OMEGA      = 3.0    # [rad/s] max bijstuursnelheid
HOME_POLE_ERROR_DEADBAND = 0.05   # bijstuur-deadband (relatief framewidth)
HOME_POLE_ARRIVED_PCT    = 80     # [%] onderkant gele blob t.o.v. framehoogte = "thuis"

# Escape maneuver bij vastzitten tijdens zoek-draaiing (SEARCH + SEARCH_HOME)
HOME_STUCK_ROT_TIME      = 0.6    # [s] terugdraaien bij vastzitten (legacy alias)
HOME_STUCK_FWD_DIST      = 0.25   # [m] (legacy alias)
HOME_STUCK_FWD_SPEED     = 0.3    # [m/s] (legacy alias)
SEARCH_STUCK_ROT_TIME    = 0.6    # [s] terugdraaien bij vastzitten tijdens zoeken
SEARCH_STUCK_DIST        = 0.25   # [m] voor-/achteruit rijden na terugdraaien
SEARCH_STUCK_SPD         = 0.3    # [m/s] rijsnelheid escape

def _angle_wrap(a: float) -> float:
    """Brengt hoek terug naar [-π, π]."""
    return math.atan2(math.sin(a), math.cos(a))

async def _broadcast(msg: str):
    """Stuurt tekst naar alle verbonden WebSocket-clients."""
    for w in list(_control_clients):
        try:
            await _ws_send_text(w, msg)
        except Exception:
            _control_clients.discard(w)

async def home_loop():
    global robot_mode, drive_state, _blink_counter
    _was_home = False

    while True:
        if robot_mode != MODE_HOME:
            _was_home = False
            await asyncio.sleep(0.05)
            continue

        px, py, pth = _read_pose()
        dist   = math.sqrt(px*px + py*py)
        th_err = _angle_wrap(-pth)   # fout t.o.v. theta=0

        # ── Diagnostiek bij eerste tick in HOME-modus ─────────────────────
        if not _was_home:
            _was_home = True
            print(f"[home] START — pose: x={px:.3f} y={py:.3f} θ={math.degrees(pth):.1f}° dist={dist:.3f}m")

        # ── Guard: als pose (0,0,0) is, is odometrie waarschijnlijk niet actief ─
        if dist < 0.001 and abs(pth) < 0.001:
            print("[home] WAARSCHUWING: pose is (0,0,0) — odometrie actief?")
            await asyncio.sleep(0.5)
            continue

        # ── Klaar? (positie EN heading binnen drempel) ────────────────────
        if dist < _HOME_DONE_XY and abs(th_err) < _HOME_DONE_TH:
            write_drive_cmd(0.0, 0.0, 0.0)
            write_gripper_cmd(GRIPPER_IDLE)
            robot_mode  = MODE_MANUAL
            drive_state = {"vx": 0.0, "vy": 0.0, "omega": 0.0}
            led_manual_update(0.0, 0.0, 0.0)
            print("[brain] HOME bereikt → MANUAL")
            await _broadcast("mode:manual")
            await _broadcast("home:done")
            await asyncio.sleep(0.05)
            continue

        # ── Holonomische P-regelaar: world-frame → robot-frame ───────────────
        # Gewenste snelheid richting (0,0) in world-frame
        speed    = min(_HOME_MAX_SPD, _HOME_K_POS * dist)
        vx_w     = (-px / dist) * speed
        vy_w     = (-py / dist) * speed

        # Roteer world-frame naar robot-frame via huidige heading
        cos_h = math.cos(pth)
        sin_h = math.sin(pth)
        vx_r  = vx_w * cos_h + vy_w * sin_h
        vy_r  = -vx_w * sin_h + vy_w * cos_h

        # Heading-correctie naar theta=0
        omega = 0.0 if abs(th_err) < _HOME_OMEGA_DEADBAND else \
                max(-_HOME_MAX_OMG, min(_HOME_MAX_OMG, _HOME_K_OMEGA * th_err))

        write_drive_cmd(vx_r, vy_r, omega)

        await _broadcast(f"home:dist:{dist:.3f},{math.degrees(pth):.1f}")
        await asyncio.sleep(0.05)   # 20 Hz

# ============================================================
# AUTO — TO_START constanten
# ============================================================
_TS_TIMEOUT       = 45.0   # [s]  maximale tijd voor terugrit; daarna direct LOSSEN
_TS_MAX_CORR_OMG  = 0.8    # [rad/s] max bijstuur-omega (ongebruikt na vereenvoudiging)
_ts_t_start       = 0.0    # starttijd TO_START (wordt gezet bij binnenkomst state)

# ── DRIVE_PICKUP rechte-lijn parameters ─────────────────────
PICKUP_APPROACH_DIST = 0.25   # [m]   afstand rechtdoor bij DRIVE_PICKUP
PICKUP_APPROACH_SPD  = 0.4    # [m/s] snelheid tijdens rechte-lijn aanrijden

async def _drive_straight(distance: float, speed: float):
    """Rij een vaste afstand [m] rechtdoor in de huidige heading.

    Gebruikt odometrie voor afstandsmeting. Kleine omega-bijsturing
    houdt de heading constant. Stopt als de gewenste afstand bereikt is
    of het maximale aantal iteraties voorbij is.
    """
    px0, py0, pth0 = _read_pose()
    travelled = 0.0
    DT = 0.05   # 20 Hz

    while travelled < distance:
        px, py, pth, = _read_pose()
        travelled = math.sqrt((px - px0)**2 + (py - py0)**2)

        # Houd de starthoek aan — kleine omega-correctie
        heading_err = _angle_wrap(pth0 - pth)
        omega = max(-_TS_MAX_CORR_OMG, min(_TS_MAX_CORR_OMG,
                    _HOME_K_OMEGA * heading_err))
        write_drive_cmd(speed, 0.0, omega)
        await asyncio.sleep(DT)

# ============================================================
# AUTO LOOP  — state machine voor autonoom rijden (20 Hz)
# _auto_bekers en _auto_frame_w worden gevuld door vision_loop.
# ============================================================
async def _rot_stuck_escape(search_omega: float):
    """Escape-maneuver bij vastzitten tijdens rotatie-zoeken (SEARCH + SEARCH_HOME).

    1. Draai terug (tegengestelde richting) voor SEARCH_STUCK_ROT_TIME.
    2. Controleer ultrasoon-sensoren:
       - Beide sensoren vrij  → rij vooruit  (weg van het obstakel achter ons).
       - Obstakel gedetecteerd → rij achteruit (obstakel zit voor ons).
    3. Stop, wissel zoekrichting.
    """
    global _last_dir_pos
    # Check sensoren vóór de maneuver: s1=rechts, s2=links (fysiek omgedraaid)
    s1 = _ultra_d1 if _ultra_d1 > 0 else 9999
    s2 = _ultra_d2 if _ultra_d2 > 0 else 9999
    front_clear = s1 > ULTRA_SAFE_CM and s2 > ULTRA_SAFE_CM

    # Stap 1: draai terug
    write_drive_cmd(0.0, 0.0, -search_omega)
    await asyncio.sleep(SEARCH_STUCK_ROT_TIME)
    write_drive_cmd(0.0, 0.0, 0.0)
    await asyncio.sleep(0.1)

    # Stap 2: rij voor- of achteruit
    if front_clear:
        print("[rot-escape] sensoren vrij → rij vooruit")
        await _drive_straight(SEARCH_STUCK_DIST, SEARCH_STUCK_SPD)
    else:
        print("[rot-escape] obstakel voor → rij achteruit")
        await _drive_straight(SEARCH_STUCK_DIST, -SEARCH_STUCK_SPD)

    write_drive_cmd(0.0, 0.0, 0.0)
    _last_dir_pos = not _last_dir_pos


async def auto_loop():
    global auto_state, _auto_bekers, _auto_frame_w, _auto_frame_h, _auto_cup_count, robot_mode, _cup_touching_gripper
    global _last_target, _target_lost_n, _last_dir_pos
    global _ultra_dodge_hold_dir, _ultra_dodge_clear_t, _ultra_dodge_start_t, _ultra_dodge_enc_snap
    global _auto_paused, _rot_stuck_t, _rot_stuck_snap, _motor_stuck_t, _motor_stuck_snap
    _prev_state = None

    while True:
        if robot_mode != MODE_AUTO:
            _prev_state = None
            _ultra_dodge_hold_dir = 0; _ultra_dodge_clear_t = 0.0; _ultra_dodge_start_t = 0.0
            _rot_stuck_t = 0.0; _motor_stuck_t = 0.0
            await asyncio.sleep(0.05)
            continue

        # ── Pauze ────────────────────────────────────────────────────────────
        if _auto_paused:
            write_drive_cmd(0.0, 0.0, 0.0)
            await asyncio.sleep(0.05)
            continue

        # Broadcast state-wijziging naar alle WS-clients
        if auto_state != _prev_state:
            await _broadcast(f"auto_state:{auto_state}")
            _prev_state = auto_state

        # ── IDLE ─────────────────────────────────────────────────────────────
        # Wacht op start-commando (via WebSocket: "auto_start")
        if auto_state == AUTO_IDLE:
            write_drive_cmd(0.0, 0.0, 0.0)
            await asyncio.sleep(0.05)

        # ── INIT ─────────────────────────────────────────────────────────────
        # Reset tellers, sluit laadklep, dan direct naar SEARCH
        elif auto_state == AUTO_INIT:
            write_drive_cmd(0.0, 0.0, 0.0)
            _auto_cup_count       = 0
            _cup_touching_gripper = False
            _last_target          = None   # tracking schoon starten
            _target_lost_n        = 0
            _last_dir_pos         = False
            _ultra_dodge_hold_dir = 0; _ultra_dodge_clear_t = 0.0; _ultra_dodge_start_t = 0.0
            _rot_stuck_t = 0.0; _motor_stuck_t = 0.0
            _klep_dicht()
            print("[auto] INIT: tellers gereset, klep dicht → SEARCH")
            auto_state = AUTO_SEARCH
            await asyncio.sleep(0.1)

        # ── SEARCH ───────────────────────────────────────────────────────────
        # Draai op vaste snelheid (SEARCH_OMEGA) tot een beker gedetecteerd is.
        # Reset tracking zodat DRIVE_TARGET schoon begint.
        elif auto_state == AUTO_SEARCH:
            if _auto_bekers:
                _last_target   = None   # tracking schoon beginnen
                _target_lost_n = 0
                print("[auto] SEARCH: beker gevonden → DRIVE_TARGET")
                auto_state = AUTO_DRIVE_TRAGET
            else:
                # Obstakel check: stop draaien en rij achteruit tot sensoren vrij
                _s1 = _ultra_d1 if _ultra_d1 > 0 else 9999
                _s2 = _ultra_d2 if _ultra_d2 > 0 else 9999
                if min(_s1, _s2) < ROT_ULTRA_STOP_CM:
                    write_drive_cmd(-ROT_ULTRA_BACK_SPD, 0.0, 0.0)
                    await asyncio.sleep(0.01)
                    continue

                search_omega = SEARCH_OMEGA if not _last_dir_pos else -SEARCH_OMEGA
                write_drive_cmd(0.0, 0.0, search_omega)
                if _motor_stuck_update_rot(search_omega):
                    print("[auto] SEARCH: vast bij draaien → escape maneuver")
                    await _rot_stuck_escape(search_omega)
                    print("[auto] SEARCH: escape klaar → hervatten zoeken")
            await asyncio.sleep(0.05)

        # ── DRIVE TARGET ─────────────────────────────────────────────────────
        # Rij altijd recht vooruit op DRIVE_SPEED.
        # Omega corrigeert de kop zodat de beker gecentreerd blijft —
        # maar de robot stopt NIET om te draaien, hij rijdt altijd door.
        # Zodra beker-onderkant de rode bodemzone raakt → direct stoppen → CENTER_PICKUP.
        # Target > MAX_TARGET_LOST_FRAMES frames kwijt → SEARCH.
        elif auto_state == AUTO_DRIVE_TRAGET:

            # ── Bodemtrigger: direct stoppen en centreren ─────────────────────
            if _cup_touching_gripper:
                write_drive_cmd(0.0, 0.0, 0.0)
                print("[auto] DRIVE_TARGET: beker in rode bodemzone → CENTER_PICKUP")
                auto_state = AUTO_CENTER_PICKUP
                await asyncio.sleep(0.05)
                continue

            # ── Beker selectie + tracking ─────────────────────────────────────
            if _auto_bekers:
                _target_lost_n = 0
                # Locked tracking: altijd de beker die het dichtst bij het
                # laatste bekende punt zit. Pas bij geen vorig punt → grootste.
                if _last_target is not None:
                    best = min(_auto_bekers,
                               key=lambda b: (b[0]+b[2]//2 - _last_target[0])**2
                                           + (b[1]+b[3]//2 - _last_target[1])**2)
                else:
                    best = max(_auto_bekers, key=lambda b: b[4])

                bx, by, bw, bh, _ = best
                cx = bx + bw // 2
                _last_target  = (cx, by + bh // 2)
                _last_dir_pos = cx >= _auto_frame_w // 2

            elif _last_target is not None:
                # Korte occlussie: stuur op laatste bekende positie
                _target_lost_n += 1
                if _target_lost_n > MAX_TARGET_LOST_FRAMES:
                    print("[auto] DRIVE_TARGET: target kwijt → SEARCH")
                    _last_target = None
                    auto_state   = AUTO_SEARCH
                    await asyncio.sleep(0.05)
                    continue
                cx = _last_target[0]

            else:
                # Geen target en geen history → SEARCH
                print("[auto] DRIVE_TARGET: geen target → SEARCH")
                auto_state = AUTO_SEARCH
                await asyncio.sleep(0.05)
                continue

            # ── Rijcommando: altijd vooruit, omega corrigeert kop ────────────
            cx    = _last_target[0]
            error = (cx - _auto_frame_w // 2) / (_auto_frame_w // 2)
            omega = 0.0 if abs(error) < ERROR_DEADBAND else \
                    max(-MAX_OMEGA, min(MAX_OMEGA, -K_OMEGA * error))
            vx    = DRIVE_SPEED

            # Obstakel veiligheidscheck (kap vx bij obstakel voor)
            vx, _, omega = _apply_avoidance(vx, 0.0, omega)

            # Ultrasoon zijdelingse dodge (camera-hint meegeven voor richting)
            vy_dodge = _ultra_dodge_vy(cup_cx=cx, frame_w=_auto_frame_w)
            if vy_dodge != 0.0:
                # Schuin naar achter: negatieve vx zodat robot achteruitwijkt
                vx = -ULTRA_DODGE_VX_BACK
            vy = vy_dodge

            write_drive_cmd(vx, vy, omega)

            # Motor-stuck check: rijcommando maar wielen staan stil → kies andere kant
            if _motor_stuck_update(vx, vy):
                if _ultra_dodge_hold_dir != 0:
                    _ultra_dodge_hold_dir = -_ultra_dodge_hold_dir
                    _ultra_dodge_start_t  = time.time()
                    _ultra_dodge_enc_snap = _read_strafe_enc()
                    _ultra_dodge_clear_t  = 0.0
                    print(f"[motor-stuck] dodge omgedraaid → "
                          f"{'links' if _ultra_dodge_hold_dir>0 else 'rechts'}")
                else:
                    # Nog geen dodge actief: start er één richting cup
                    _ultra_dodge_hold_dir = -1 if (cx > _auto_frame_w // 2) else 1
                    _ultra_dodge_start_t  = time.time()
                    _ultra_dodge_enc_snap = _read_strafe_enc()
                    _ultra_dodge_clear_t  = 0.0
                    print(f"[motor-stuck] nieuw dodge gestart → "
                          f"{'links' if _ultra_dodge_hold_dir>0 else 'rechts'}")

            await asyncio.sleep(0.05)

        # ── CENTER PICKUP ─────────────────────────────────────────────────────
        # Robot staat stil. Draai op de plek totdat de gevolgde beker exact
        # horizontaal gecentreerd staat (error < ERROR_DEADBAND).
        # Daarna pas verder naar DRIVE_PICKUP.
        elif auto_state == AUTO_CENTER_PICKUP:
            if _auto_bekers:
                # Gebruik de gevolgde beker (dichtst bij _last_target),
                # niet de grootste — zodat bij meerdere cups de juiste blijft gekozen.
                if _last_target is not None:
                    best = min(_auto_bekers,
                               key=lambda b: (b[0] + b[2]//2 - _last_target[0])**2
                                           + (b[1] + b[3]//2 - _last_target[1])**2)
                else:
                    best = max(_auto_bekers, key=lambda b: b[4])

                bx, by, bw, bh, _ = best
                cx    = bx + bw // 2
                error = (cx - _auto_frame_w // 2) / (_auto_frame_w // 2)

                if abs(error) < ERROR_DEADBAND:
                    write_drive_cmd(0.0, 0.0, 0.0)
                    print("[auto] CENTER_PICKUP: beker gecentreerd → DRIVE_PICKUP")
                    auto_state = AUTO_DRIVE_PICKUP
                else:
                    # Alleen draaien op de plek, niet rijden
                    omega = max(-MAX_OMEGA * 0.5, min(MAX_OMEGA * 0.5, -K_OMEGA * error))
                    write_drive_cmd(0.0, 0.0, omega)
                    if _motor_stuck_update_rot(omega):
                        write_drive_cmd(0.0, 0.0, 0.0)
                        print("[auto] CENTER_PICKUP: vast bij draaien → toch DRIVE_PICKUP")
                        auto_state = AUTO_DRIVE_PICKUP
            else:
                # Beker niet meer zichtbaar (bijv. te dichtbij) → toch doorgaan
                write_drive_cmd(0.0, 0.0, 0.0)
                print("[auto] CENTER_PICKUP: beker niet zichtbaar → DRIVE_PICKUP")
                auto_state = AUTO_DRIVE_PICKUP
            await asyncio.sleep(0.05)

        # ── DRIVE PICKUP ─────────────────────────────────────────────────────
        # Rij een vaste afstand rechtdoor in de huidige heading-richting,
        # dan direct naar PICKUP.
        elif auto_state == AUTO_DRIVE_PICKUP:
            print("[auto] DRIVE_PICKUP: rechte lijn rijden")
            await _drive_straight(PICKUP_APPROACH_DIST, PICKUP_APPROACH_SPD)
            write_drive_cmd(0.0, 0.0, 0.0)
            print("[auto] DRIVE_PICKUP: klaar → PICKUP")
            auto_state = AUTO_PICKUP
            await asyncio.sleep(0.1)

        # ── PICKUP ───────────────────────────────────────────────────────────
        # Start gripper auto-cyclus en wacht tot het rad klaar is (encoder).
        # Veiligheids-timeout van 5 seconden als encoder niet reageert.
        elif auto_state == AUTO_PICKUP:
            write_drive_cmd(0.0, 0.0, 0.0)
            write_gripper_cmd(GRIPPER_AUTO_CMD, gripper_auto_speed)
            print(f"[auto] PICKUP: gripper gestart (speed={gripper_auto_speed}), wacht op state=IDLE van gripper.py")

            PICKUP_TIMEOUT = 10.0   # veiligheids-timeout [s]
            _t_start = time.time()
            while True:
                await asyncio.sleep(0.05)
                if _read_gripper_state() == 0:   # 0 = IDLE → cyclus klaar
                    print("[auto] PICKUP: gripper meldt IDLE → klaar")
                    break
                if time.time() - _t_start >= PICKUP_TIMEOUT:
                    print("[auto] PICKUP: timeout, gripper.py reageerde niet → doorgaan")
                    break

            write_gripper_cmd(GRIPPER_IDLE)

            _auto_cup_count += 1
            print(f"[auto] PICKUP: beker #{_auto_cup_count} opgepakt")
            await _broadcast(f"auto_cups:{_auto_cup_count}")

            if _auto_cup_count >= AUTO_MAX_CUPS:
                print(f"[auto] PICKUP: {AUTO_MAX_CUPS} bekers vol → TO_START")
                auto_state = AUTO_TO_START
            else:
                _cup_touching_gripper = False   # reset voor volgende beker
                auto_state = AUTO_SEARCH
            await asyncio.sleep(0.1)

        # ── TO START ─────────────────────────────────────────────────────────
        # Startpunt: stap direct over naar visuele zoekfase (gele paal).
        elif auto_state == AUTO_TO_START:
            write_drive_cmd(0.0, 0.0, 0.0)
            _last_dir_pos = False
            _last_target  = None
            print("[auto] TO_START → SEARCH_HOME (zoek gele paal)")
            auto_state = AUTO_SEARCH_HOME
            await asyncio.sleep(0.05)

        # ── SEARCH HOME ──────────────────────────────────────────────────────
        # Draai op de plek totdat de gele paal zichtbaar is.
        # Bij vastzitten: draai terug + escape vooruit.
        elif auto_state == AUTO_SEARCH_HOME:
            if _yellow_zone is not None:
                yx, yy, yw, yh = _yellow_zone
                if yw * yh >= HOME_POLE_MIN_AREA:
                    _last_target = None
                    print("[auto] SEARCH_HOME: gele paal gevonden → DRIVE_HOME")
                    auto_state = AUTO_DRIVE_HOME
                    await asyncio.sleep(0.05)
                    continue

            # Obstakel check: stop draaien en rij achteruit tot sensoren vrij
            _s1 = _ultra_d1 if _ultra_d1 > 0 else 9999
            _s2 = _ultra_d2 if _ultra_d2 > 0 else 9999
            if min(_s1, _s2) < ROT_ULTRA_STOP_CM:
                write_drive_cmd(-ROT_ULTRA_BACK_SPD, 0.0, 0.0)
                await asyncio.sleep(0.01)
                continue

            search_omega = HOME_POLE_SEARCH_OMEGA if not _last_dir_pos else -HOME_POLE_SEARCH_OMEGA
            write_drive_cmd(0.0, 0.0, search_omega)

            if _motor_stuck_update_rot(search_omega):
                print("[auto] SEARCH_HOME: vast bij draaien → escape maneuver")
                await _rot_stuck_escape(search_omega)
                print("[auto] SEARCH_HOME: escape klaar → hervatten zoeken")

            await asyncio.sleep(0.05)

        # ── DRIVE HOME ───────────────────────────────────────────────────────
        # Rij op de gele paal af. Zodra de paal de groene trigger-lijn bereikt
        # (HOME_POLE_ARRIVED_PCT) is de robot thuis: klep open → LOSSEN.
        elif auto_state == AUTO_DRIVE_HOME:
            if _yellow_zone is None:
                print("[auto] DRIVE_HOME: gele paal kwijt → SEARCH_HOME")
                auto_state = AUTO_SEARCH_HOME
                await asyncio.sleep(0.05)
                continue

            yx, yy, yw, yh = _yellow_zone
            pole_cx     = yx + yw // 2
            pole_bottom = yy + yh
            arrived_y   = int(_auto_frame_h * HOME_POLE_ARRIVED_PCT / 100) if _auto_frame_h > 0 else 9999

            # Thuis: onderkant paal voorbij de groene trigger-lijn
            if pole_bottom >= arrived_y:
                write_drive_cmd(0.0, 0.0, 0.0)
                print("[auto] DRIVE_HOME: gele paal bereikt (HOME) → LOSSEN")
                auto_state = AUTO_LOSSEN
                await asyncio.sleep(0.05)
                continue

            # Rij naar paal, omega corrigeert heading
            error = (pole_cx - _auto_frame_w // 2) / max(_auto_frame_w // 2, 1)
            omega = 0.0 if abs(error) < HOME_POLE_ERROR_DEADBAND else \
                    max(-HOME_POLE_MAX_OMEGA, min(HOME_POLE_MAX_OMEGA, -HOME_POLE_K_OMEGA * error))
            vx = HOME_POLE_DRIVE_SPEED

            # Obstakel veiligheidscheck: kap vx terug bij obstakel voor
            vx, _, omega = _apply_avoidance(vx, 0.0, omega)

            # Zijdelingse dodge (geen camera-hint voor paal, sensoren bepalen richting)
            vy_dodge = _ultra_dodge_vy(cup_cx=None, frame_w=_auto_frame_w)
            if vy_dodge != 0.0:
                vx = -ULTRA_DODGE_VX_BACK   # schuin achteruit tijdens dodge

            write_drive_cmd(vx, vy_dodge, omega)

            # Motor stuck tijdens rijden → terug naar zoekfase
            if _motor_stuck_update(HOME_POLE_DRIVE_SPEED, 0.0):
                write_drive_cmd(0.0, 0.0, 0.0)
                print("[auto] DRIVE_HOME: vast → SEARCH_HOME")
                auto_state = AUTO_SEARCH_HOME

            await asyncio.sleep(0.05)

        # ── LOSSEN ───────────────────────────────────────────────────────────
        # Draai 180°, open laadklep, wacht tot bekers zijn gevallen, sluit klep → IDLE.
        elif auto_state == AUTO_LOSSEN:
            write_drive_cmd(0.0, 0.0, 0.0)

            # ── 180° draaien via ruwe gyro (heading-gebaseerd) ───────────────
            pth_start = _read_gyro_yaw()
            target_heading = _angle_wrap(pth_start + math.pi)   # +180°
            TURN_OMEGA  = 2.0   # [rad/s] draaisnelheid
            TURN_DONE   = 0.08  # [rad] deadband voor "klaar"
            TURN_TIMEOUT = 20.0  # [s] veiligheids-timeout
            _t_turn = time.time()
            print(f"[auto] LOSSEN: 180° draaien via gyro (van {math.degrees(pth_start):.1f}° naar {math.degrees(target_heading):.1f}°)")
            while True:
                await asyncio.sleep(0.05)
                pth_now = _read_gyro_yaw()
                err = _angle_wrap(target_heading - pth_now)
                if abs(err) < TURN_DONE or (time.time() - _t_turn) > TURN_TIMEOUT:
                    break
                omega = max(-TURN_OMEGA, min(TURN_OMEGA, 3.0 * err))
                write_drive_cmd(0.0, 0.0, omega)
            write_drive_cmd(0.0, 0.0, 0.0)
            print("[auto] LOSSEN: 180° gedraaid → klep open")
            await asyncio.sleep(0.2)

            _klep_open()
            print("[auto] LOSSEN: klep open, bekers lossen...")
            await asyncio.sleep(15.0)   # TODO: pas duur aan op basis van hardware
            _klep_dicht()
            print("[auto] LOSSEN: klep dicht → IDLE")
            auto_state = AUTO_IDLE
            await asyncio.sleep(0.1)

        else:
            await asyncio.sleep(0.05)

# ============================================================
# MAIN
# ============================================================
async def main():
    server = await asyncio.start_server(handle_client, "0.0.0.0", 8765)
    print("[brain] Server actief op http://ROBOT_IP:8765")
    print("[brain] Start in MANUAL modus")

    # beginstand LED
    led_manual_update(0.0, 0.0, 0.0)

    async with server:
        await asyncio.gather(
            server.serve_forever(),
            vision_loop(),
            heartbeat(),
            ultra_loop(),
            pose_loop(),
            obstacle_map_loop(),
            home_loop(),
            auto_loop(),
            log_loop()
        )

if __name__ == "__main__":
    asyncio.run(main())

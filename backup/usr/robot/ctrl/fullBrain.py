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

def _save_calib():
    try:
        with open(_CALIB_PATH, "w") as f:
            json.dump({"hsv_lower": LOWER_HSV.tolist(),
                       "hsv_upper": UPPER_HSV.tolist()}, f)
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

# Gripper snelheden (aanpasbaar via web, 0–100)
gripper_jog_speed  = 30   # handmatige jog
gripper_auto_speed = 60   # auto cyclus
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
# Beide sensoren zitten aan de voorzijde.
ULTRA_SAFE_CM = 15   # hard stop  — blokkeer vooruitrijden
ULTRA_WARN_CM = 30   # vertraagzone — snelheid lineair afschalen

def _apply_avoidance(vx: float, vy: float, omega: float):
    """Past vx aan o.b.v. actuele ultrasoonwaarden (alleen AUTO). 0 = geen data → geen ingreep."""
    if vx > 0:   # vooruit → check beide voor-sensoren, neem de kortste afstand
        s1 = _ultra_d1 if _ultra_d1 > 0 else 9999
        s2 = _ultra_d2 if _ultra_d2 > 0 else 9999
        front = min(s1, s2)
        if front <= ULTRA_SAFE_CM:
            vx = 0.0
        elif front < ULTRA_WARN_CM:
            vx *= (front - ULTRA_SAFE_CM) / (ULTRA_WARN_CM - ULTRA_SAFE_CM)
    return vx, vy, omega

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
# MANUAL RIJSTATUS
# ============================================================
drive_state = {"vx": 0.0, "vy": 0.0, "omega": 0.0}
SPEED     = 1.5
ROT_SPEED = 3.0

# ============================================================
# AUTO — VISIE PARAMETERS  (uit upgradedBasicBrain)
# ============================================================
LOWER_HSV, UPPER_HSV = _load_calib_hsv()
MIN_BEKER_AREA  = 50
MAX_BEKER_AREA  = 300_000
MIN_OBST_AREA   = 250

K_OMEGA         = 2.8
MAX_OMEGA       = 4.0
STOP_AREA       = 9_000
SEARCH_OMEGA    = 4.0
ERROR_DEADBAND  = 0.02
DRIVE_SPEED     = 2.0
DRIVE_MIN_OMEGA = 1.5
MAX_TRACK_DIST  = 80

_last_dir_pos = False
_last_target  = None          # (cx, cy) van vorige frame


def _classify_zone(cx, frame_w):
    if cx < frame_w * 0.33:
        return "LINKS"
    elif cx < frame_w * 0.66:
        return "MIDDEN"
    return "RECHTS"


def _compute_velocity(bekers, frame_w):
    global _last_dir_pos, _last_target

    if not bekers:
        _last_target = None
        return (0.0, 0.0, -SEARCH_OMEGA) if _last_dir_pos else (0.0, 0.0, SEARCH_OMEGA)

    # target tracking
    if _last_target is not None:
        hits = []
        for b in bekers:
            x, y, w, h, area = b
            cx, cy = x + w // 2, y + h // 2
            d2 = (cx - _last_target[0])**2 + (cy - _last_target[1])**2
            if d2 < MAX_TRACK_DIST**2:
                hits.append((d2, b))
        best = min(hits, key=lambda t: t[0])[1] if hits else max(bekers, key=lambda b: b[4])
    else:
        best = max(bekers, key=lambda b: b[4])

    x, y, w, h, area = best
    cx = x + w // 2
    _last_target  = (cx, y + h // 2)
    _last_dir_pos = cx >= frame_w // 2

    error = (cx - frame_w // 2) / (frame_w // 2)
    omega = 0.0 if abs(error) < ERROR_DEADBAND else max(-MAX_OMEGA, min(MAX_OMEGA, -K_OMEGA * error))
    vx    = DRIVE_SPEED if abs(omega) < DRIVE_MIN_OMEGA else 0.0

    if area > STOP_AREA:
        return 0.0, 0.0, 0.0

    return vx, 0.0, omega

# ============================================================
# VISION LOOP  (altijd actief, schrijft rij+LED-commando afhankelijk van modus)
# ============================================================
_blink_counter = 0
BLINK_TICKS    = 25   # ~500 ms bij 20 ms sleep

async def vision_loop():
    global latest_jpeg, _blink_counter

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

        frame     = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((frame_h, frame_w, CHANNELS))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # camera geeft RGB, omzetten naar BGR voor CV

        raw_bgr = frame_bgr.copy()   # stap "raw": origineel frame voor debug

        # bekerdetectie
        hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bekers = []
        for c in contours:
            area = cv2.contourArea(c)
            if MIN_BEKER_AREA < area < MAX_BEKER_AREA:
                x, y, w, h = cv2.boundingRect(c)
                bekers.append((x, y, w, h, area))
                cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cups_bgr = frame_bgr.copy()  # stap "bekers": na cup-boxes, vóór obstakeldetectie

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
            vx, vy, omega = _compute_velocity(bekers, frame_w)
            vx, vy, omega = _apply_avoidance(vx, vy, omega)
            write_drive_cmd(vx, vy, omega)

            # LED: elke BLINK_TICKS ticks de blink-toestand wisselen
            _blink_counter += 1
            if _blink_counter >= BLINK_TICKS:
                _blink_counter = 0
                led_auto_update(has_target=bool(bekers))

            label = f"AUTO  vx:{vx:.2f} vy:{vy:.2f} w:{omega:.2f} bekers:{len(bekers)}"
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
    global LOWER_HSV, UPPER_HSV
    global home_strategy, _gripper_pending_auto, _gripper_manual_jogged
    global _vision_debug_step

    writer.write(_ws_handshake(request).encode())
    await writer.drain()

    _control_clients.add(writer)

    # stuur huidige staat direct na verbinding
    await _ws_send_text(writer, f"mode:{robot_mode}")
    await _ws_send_text(writer, f"ultra:{_ultra_d1},{_ultra_d2}")
    px, py, pth = _read_pose()
    await _ws_send_text(writer, f"pose:{px:.4f},{py:.4f},{pth:.4f}")
    await _ws_send_text(writer, f"home_strat:{home_strategy}")
    await _ws_send_text(writer, f"gripper_speeds:{gripper_jog_speed},{gripper_auto_speed},{gripper_home_speed}")
    lo, hi = LOWER_HSV.tolist(), UPPER_HSV.tolist()
    await _ws_send_text(writer, f"hsv:{lo[0]},{lo[1]},{lo[2]},{hi[0]},{hi[1]},{hi[2]}")
    await _ws_send_text(writer, f"vision_step:{_vision_debug_step}")

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
                drive_state = {"vx": 0.0, "vy": 0.0, "omega": 0.0}
                _blink_counter = 0
                print("[brain] → AUTO")
                await _ws_send_text(writer, "mode:auto")

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

            # ---- RIJDEN (alleen in MANUAL) ----
            elif robot_mode == MODE_MANUAL:
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
                    write_gripper_cmd(GRIPPER_AUTO_CMD)
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
        if robot_mode == MODE_MANUAL:
            write_drive_cmd(drive_state["vx"], drive_state["vy"], drive_state["omega"])
            led_manual_update(drive_state["vx"], drive_state["vy"], drive_state["omega"])
        await asyncio.sleep(0.05)

# ============================================================
# ULTRASOON LOOP  (leest SHM en broadcast naar alle WS-clients, 10 Hz)
# ============================================================
async def ultra_loop():
    global _ultra_d1, _ultra_d2

    while True:
        d1, d2 = _read_ultra()
        _ultra_d1, _ultra_d2 = d1, d2

        if _control_clients:
            msg = f"ultra:{d1},{d2}"
            for w in list(_control_clients):
                try:
                    await _ws_send_text(w, msg)
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
# "holonomic" : rij in elke richting (mecanum, tegelijk draaien)
# "forward"   : eerst draaien, dan rechtdoor rijden, dan draaien naar theta=0
home_strategy = "holonomic"

_FORWARD_ALIGN_TH = 0.12   # rad (~7°): acceptabele kopfout voordat vooruitrijden begint

# ── P-regelaar instelling ────────────────────────────────────────────────────
_HOME_K_POS    = 1.2   # versterkingsfactor positie [m/s per m fout]
_HOME_K_OMEGA  = 4.0   # versterkingsfactor heading [rad/s per rad fout]
_HOME_MAX_SPD  = 0.8   # maximale rijsnelheid [m/s]
_HOME_MAX_OMG  = 2.5   # maximale rotatiesnelheid tijdens rijfase [rad/s]
_HOME_FINAL_OMG = 0.8  # maximale rotatiesnelheid tijdens eindcorrectie [rad/s]
_HOME_DONE_XY  = 0.12  # aankomstdrempel positie [m]  (12 cm)
_HOME_DONE_TH  = 0.08  # aankomstdrempel heading [rad] (~5°)

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

    while True:
        if robot_mode != MODE_HOME:
            await asyncio.sleep(0.05)
            continue

        px, py, pth = _read_pose()
        dist   = math.sqrt(px*px + py*py)
        th_err = _angle_wrap(-pth)   # fout t.o.v. theta=0

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

        if home_strategy == "forward":
            # ── Strategie FORWARD: draaien → rechtdoor → draaien ─────────
            if dist > _HOME_DONE_XY:
                # Hoek vanuit huidige positie richting oorsprong (wereldframe)
                angle_to_home = math.atan2(-py, -px)
                align_err = _angle_wrap(angle_to_home - pth)

                if abs(align_err) > _FORWARD_ALIGN_TH:
                    # Fase 1: draai om oorsprong in te richten, niet rijden
                    omega = _HOME_K_OMEGA * align_err
                    omega = max(-_HOME_MAX_OMG, min(_HOME_MAX_OMG, omega))
                    write_drive_cmd(0.0, 0.0, omega)
                else:
                    # Fase 2: rechtdoor rijden + kleine kopscorrectie + obstakelafstoting
                    vx_robot = min(_HOME_MAX_SPD, _HOME_K_POS * dist)
                    omega    = _HOME_K_OMEGA * align_err
                    omega    = max(-_HOME_MAX_OMG * 0.4, min(_HOME_MAX_OMG * 0.4, omega))
                    # Obstakelafstoting: alleen laterale component (vy) toepassen
                    # zodat de vooruitsnelheid niet geblokkeerd raakt door de regelaar
                    _, vy_rep = obstacle_map.apply_repulsion(px, py, pth, 0.0, 0.0)
                    vy_clamped = max(-_HOME_MAX_SPD * 0.6, min(_HOME_MAX_SPD * 0.6, vy_rep))
                    write_drive_cmd(vx_robot, vy_clamped, omega)
            else:
                # Fase 3: op positie, draai naar theta=0
                omega = _HOME_K_OMEGA * th_err
                omega = max(-_HOME_FINAL_OMG, min(_HOME_FINAL_OMG, omega))
                write_drive_cmd(0.0, 0.0, omega)

        else:
            # ── Strategie HOLONOMIC: rij in elke richting tegelijk ────────
            omega_lim = _HOME_MAX_OMG if dist > _HOME_DONE_XY else _HOME_FINAL_OMG
            omega = _HOME_K_OMEGA * th_err
            omega = max(-omega_lim, min(omega_lim, omega))

            vx_w = _HOME_K_POS * (-px)
            vy_w = _HOME_K_POS * (-py)
            spd  = math.sqrt(vx_w*vx_w + vy_w*vy_w)
            max_spd = _HOME_MAX_SPD if dist > _HOME_DONE_XY else min(_HOME_MAX_SPD, dist * 2.0)
            if spd > max_spd:
                vx_w *= max_spd / spd
                vy_w *= max_spd / spd

            cos_t    = math.cos(pth)
            sin_t    = math.sin(pth)
            vx_robot =  vx_w * cos_t + vy_w * sin_t
            vy_robot = -vx_w * sin_t + vy_w * cos_t

            # ── Obstakelafstoting op HOME-pad ─────────────────────────────
            vx_robot, vy_robot = obstacle_map.apply_repulsion(
                px, py, pth, vx_robot, vy_robot)
            # Begrens totale snelheid na samenvoegen HOME-vector + afstoting
            combined_spd = math.sqrt(vx_robot**2 + vy_robot**2)
            if combined_spd > _HOME_MAX_SPD_REPULSE:
                factor = _HOME_MAX_SPD_REPULSE / combined_spd
                vx_robot *= factor
                vy_robot *= factor

            write_drive_cmd(vx_robot, vy_robot, omega)

        await _broadcast(f"home:dist:{dist:.3f},{math.degrees(pth):.1f}")
        await asyncio.sleep(0.05)   # 20 Hz

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
            home_loop()
        )

if __name__ == "__main__":
    asyncio.run(main())

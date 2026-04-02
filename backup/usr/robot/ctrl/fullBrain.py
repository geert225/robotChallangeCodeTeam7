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
#   /dev/shm/vision_frame   <- camera frame           (timestamp + RGB)
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
from PIL import Image
import numpy as np
import cv2

# ============================================================
# ROBOT MODUS
# ============================================================
MODE_MANUAL = "manual"
MODE_AUTO   = "auto"
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
        write_led(3, 255, 255,   0,   0, 255,   0)   # vooruit + zijwaarts: geel + groen
    elif avx > 0.2:
        write_led(1,   0, 255,   0,   0,   0,   0)   # vooruit/achteruit: groen
    elif avy > 0.2:
        write_led(1, 255, 255,   0,   0, 255,   0)   # zijwaarts: geel + groen
    elif abs(omega) > 0.5:
        write_led(1,   0, 150, 255,   0, 150, 255)   # roteren: blauw
    else:
        write_led(2,   0, 255, 255,   0,   0,   0)   # idle: cyaan

# AUTO — paars: zoekend knippert, gevonden solide
_led_blink_state = False

def led_auto_update(has_target: bool):
    global _led_blink_state
    if has_target:
        # solide paars
        write_led(1, 160, 0, 200, 160, 0, 200)
    else:
        # knipperend paars — toggle elke aanroep (vision_loop ~20ms, update elke ~500ms via teller)
        if _led_blink_state:
            write_led(1, 160, 0, 200, 160, 0, 200)
        else:
            write_led(1,   0, 0,   0,   0, 0,   0)   # uit
        _led_blink_state = not _led_blink_state

# ============================================================
# SHARED MEMORY — VISION FRAME
# ============================================================
WIDTH, HEIGHT, CHANNELS = 320, 240, 3
FRAME_SIZE         = WIDTH * HEIGHT * CHANNELS
VISION_FRAME_TOTAL = 8 + FRAME_SIZE
VISION_FRAME_PATH  = "/dev/shm/vision_frame"

vision_shm, vision_fh = _create_or_open_shm(VISION_FRAME_PATH, VISION_FRAME_TOTAL)

latest_jpeg = None

# ============================================================
# MANUAL RIJSTATUS
# ============================================================
drive_state = {"vx": 0.0, "vy": 0.0, "omega": 0.0}
SPEED     = 1.5
ROT_SPEED = 3.0

# ============================================================
# AUTO — VISIE PARAMETERS  (uit upgradedBasicBrain)
# ============================================================
LOWER_HSV       = np.array([130, 140,  90])
UPPER_HSV       = np.array([150, 255, 220])
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
            vision_shm.read(8)                     # timestamp (overgeslagen)
            frame_bytes = vision_shm.read(FRAME_SIZE)
        finally:
            fcntl.flock(vision_fh.fileno(), fcntl.LOCK_UN)

        frame     = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((HEIGHT, WIDTH, CHANNELS))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

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

        # obstakeldetectie (onderste helft)
        roi   = frame_bgr[int(HEIGHT * 0.55):HEIGHT, :]
        gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cnt_obs, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnt_obs:
            if cv2.contourArea(c) < MIN_OBST_AREA:
                continue
            ox, oy, ow, oh = cv2.boundingRect(c)
            if ow < 12 or oh < 6:
                continue
            cv2.rectangle(roi, (ox, oy), (ox+ow, oy+oh), (0, 0, 255), 2)
        frame_bgr[int(HEIGHT * 0.55):HEIGHT, :] = roi

        # rijcommando + LED afhankelijk van modus
        if robot_mode == MODE_AUTO:
            vx, vy, omega = _compute_velocity(bekers, WIDTH)
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

        # JPEG bouwen
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
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

    writer.write(_ws_handshake(request).encode())
    await writer.drain()

    # stuur huidige modus direct na verbinding
    await _ws_send_text(writer, f"mode:{robot_mode}")

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
                        spd = float(msg.split(":", 1)[1])
                        write_gripper_cmd(GRIPPER_JOG, spd)
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
        )

if __name__ == "__main__":
    asyncio.run(main())
